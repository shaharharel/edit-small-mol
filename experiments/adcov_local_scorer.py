#!/usr/bin/env python3
"""AD-CovDock --local_only scorer for ZAP70 Cys346.

Per QA expert: --local_only (Vina BFGS local minimization on the tethered pose)
is the rank-correlated covalent-aware scoring, not --score_only.

Pipeline per mol:
  RDKit ETKDG embed → meeko CovalentBuilder tether at Cys346 SG →
  Vina --local_only on tethered pose → output score (kcal/mol)

CLI: adcov_local_scorer.py <input.smi> <output.csv> [N_WORKERS=16]
"""
import os, sys, csv, time, subprocess as sp
from pathlib import Path
from multiprocessing import Pool

# Redirect per-worker stdout to stderr so meeko's "CovalentBuilder>" prints don't
# break the parent process's JSON output handling (same pattern as Vina scorer).
def _redirect_worker_stdout():
    if not getattr(_redirect_worker_stdout, "_done", False):
        os.dup2(2, 1)
        _redirect_worker_stdout._done = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

from experiments.run_covalent_docking import build_cov_tethered_pdbqt, RECEPTOR_PDBQT_STRIPPED, BOX_CENTER, BOX_SIZE_ADCOV, VINA_BIN
# AD-CovDock --local_only needs the LARGER 40Å box — tethered ligands extend
# beyond the 20Å eval box because the warhead anchor pulls the molecule.
BOX_SIZE = BOX_SIZE_ADCOV

VINA = str(VINA_BIN) if VINA_BIN.exists() else "/usr/bin/vina"
if not Path(VINA).exists():
    for alt in ("/usr/local/bin/vina",
                "/opt/miniconda3/envs/quris/bin/vina",
                "/home/shaharh_quris_ai/miniconda3/envs/quris/bin/vina"):
        if Path(alt).exists():
            VINA = alt
            break


def vina_local(lig_pdbqt: Path) -> tuple[float, str]:
    """Run vina --local_only on the tethered pose (Vina BFGS minimization)."""
    cx, cy, cz = BOX_CENTER
    sx, sy, sz = BOX_SIZE
    out_pdbqt = lig_pdbqt.with_suffix(".local.pdbqt")
    cmd = [VINA,
           "--receptor", str(RECEPTOR_PDBQT_STRIPPED),
           "--ligand", str(lig_pdbqt),
           "--out", str(out_pdbqt),
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
           "--local_only", "--cpu", "1"]
    try:
        res = sp.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            return float("nan"), f"vina_err:{res.returncode}"
        for line in res.stdout.splitlines():
            if "Estimated Free Energy" in line or "REMARK VINA RESULT" in line:
                for tok in line.split():
                    try:
                        s = float(tok)
                        if -50 < s < 200:
                            return s, "ok"
                    except ValueError:
                        continue
        if out_pdbqt.exists():
            for line in out_pdbqt.read_text().splitlines():
                if line.startswith("REMARK VINA RESULT"):
                    for tok in line.split():
                        try:
                            s = float(tok)
                            if -50 < s < 200:
                                return s, "ok_pdbqt"
                        except ValueError:
                            continue
        return float("nan"), "no_score"
    except sp.TimeoutExpired:
        return float("nan"), "timeout"
    except Exception as e:
        return float("nan"), f"exc:{type(e).__name__}"
    finally:
        try: out_pdbqt.unlink(missing_ok=True)
        except Exception: pass


_WORK_DIR_GLOBAL = None


def _init(work_dir_str):
    global _WORK_DIR_GLOBAL
    _WORK_DIR_GLOBAL = Path(work_dir_str)
    _redirect_worker_stdout()


def score_one(args):
    row_id, smi = args
    work = _WORK_DIR_GLOBAL
    lig = work / f"lig_{os.getpid()}_{row_id}.pdbqt"
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return row_id, smi, float("nan"), "bad_smi"
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, randomSeed=42, maxAttempts=20) < 0:
            return row_id, smi, float("nan"), "embed_fail"
        try: AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception: pass
        prep = build_cov_tethered_pdbqt(m, smi, lig)
        if not prep["ok"]:
            return row_id, smi, float("nan"), f"prep:{prep.get('msg','?')[:30]}"
        score, status = vina_local(lig)
        return row_id, smi, score, status
    finally:
        try: lig.unlink(missing_ok=True)
        except Exception: pass


def main():
    inp = Path(sys.argv[1]); out = Path(sys.argv[2]); N_WORKERS = int(sys.argv[3]) if len(sys.argv)>3 else 16
    tasks = []
    with open(inp) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if not parts or not parts[0]: continue
            smi = parts[0]
            row_id = parts[1] if len(parts) > 1 else str(i)
            tasks.append((row_id, smi))
    n = len(tasks)
    print(f"[adcov-local] {n:,} mols  ·  {N_WORKERS} workers  ·  receptor={RECEPTOR_PDBQT_STRIPPED.name}", flush=True)
    print(f"  box center: {BOX_CENTER}  ·  size: {BOX_SIZE}  ·  vina: {VINA}", flush=True)

    work = Path("/dev/shm/adcov_work" if Path("/dev/shm").exists() else "/tmp/adcov_work")
    work.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w", newline="")
    w = csv.writer(f); w.writerow(["row_id","smiles","adcov_local_kcalmol","status"]); f.flush()
    t0 = time.time(); ok = 0; n_done = 0; last = t0
    with Pool(N_WORKERS, initializer=_init, initargs=(str(work),)) as pool:
        for row_id, smi, score, status in pool.imap_unordered(score_one, tasks, chunksize=4):
            w.writerow([row_id, smi, "" if score != score else score, status])
            f.flush(); n_done += 1
            if status.startswith("ok"): ok += 1
            now = time.time()
            if now - last > 30:
                rate = n_done / (now - t0)
                eta = (n - n_done) / max(1e-3, rate) / 60
                print(f"  ...{n_done:,}/{n:,}  ({(now-t0):.0f}s, {rate:.2f}/s, ETA {eta:.1f}min, ok={ok})", flush=True)
                last = now
    f.close()
    dt = time.time() - t0
    print(f"\n[adcov-local] done {dt:.0f}s ({n/dt:.2f}/s)  ·  ok: {ok:,}/{n:,}", flush=True)


if __name__ == "__main__":
    main()
