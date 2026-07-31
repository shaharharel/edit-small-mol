"""Generic SMILES→β-C Fukui f+ batch via xTB GFN2 --vfukui.

Reads a SMI file (row_id\tSMILES per line — cov-Vina format), detects the
electrophilic β-C for one of several covalent warhead classes, and writes a
CSV [row_id, smiles, warhead_class, beta_c_idx, f_plus, status].

Usage:
  python fukui_smi_batch.py <input.smi> <output.csv> [N_WORKERS=16] [--limit N]
"""
from __future__ import annotations
import argparse, csv, json, os, re, subprocess, sys, tempfile, time
import multiprocessing as mp
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

# Warhead SMARTS + which match-atom is the electrophilic β-C.
# Order matters: first match wins.
WARHEADS = [
    # (name, smarts, beta_idx_in_match)
    ("acryl",           "[CH2;X3]=[CH;X3][C;X3](=O)[N]",         0),  # β = terminal CH2
    ("chloroacetamide", "[Cl][CH2][C](=O)[N]",                    1),  # β = α-C (leaving-group side)
    ("bromoacetamide",  "[Br][CH2][C](=O)[N]",                    1),
    ("fluoroacetamide", "[F][CH2][C](=O)[N]",                     1),
    ("vinylsulfone",    "[CH2]=[CH][S](=O)(=O)",                  0),
    ("nitrile",         "[C]#[N;X1]",                             0),  # C of nitrile
    ("aldehyde",        "[CH1;X3](=O)[!O;!N]",                   0),
    ("β-lactam",        "O=C1[C][C][N]1",                         1),  # ring α-C
]
WARHEAD_SMARTS = [(n, Chem.MolFromSmarts(s), b) for (n, s, b) in WARHEADS]

FUKUI_LINE_RE = re.compile(r"^\s*(\d+)([A-Za-z]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


def find_beta_c(mol):
    """Return (warhead_class, beta_atom_idx) or (None, None)."""
    for name, smarts, bidx in WARHEAD_SMARTS:
        matches = mol.GetSubstructMatches(smarts)
        if matches:
            return name, matches[0][bidx]
    return None, None


def _xtb_one(args):
    row_id, smi = args
    result = {"row_id": row_id, "smiles": smi, "warhead_class": "",
              "beta_c_idx": -1, "f_plus": float("nan"), "status": ""}
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            result["status"] = "smi_parse_fail"
            return result
        wclass, beta_rdkit = find_beta_c(m)
        if wclass is None:
            result["status"] = "no_warhead"
            return result
        result["warhead_class"] = wclass
        result["beta_c_idx"] = int(beta_rdkit)
        mH = Chem.AddHs(m)
        params = AllChem.ETKDGv3(); params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mH, params)
        if cid < 0:
            result["status"] = "embed_fail"
            return result
        try:
            AllChem.MMFFOptimizeMolecule(mH, maxIters=200)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "mol.xyz"
            conf = mH.GetConformer(cid)
            n = mH.GetNumAtoms()
            lines = [str(n), ""]
            for i in range(n):
                a = mH.GetAtomWithIdx(i)
                p = conf.GetAtomPosition(i)
                lines.append(f"{a.GetSymbol():<3s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
            xyz.write_text("\n".join(lines))
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"; env["MKL_NUM_THREADS"] = "1"
            try:
                res = subprocess.run(
                    ["xtb", str(xyz), "--gfn", "2", "--vfukui", "--iterations", "150"],
                    cwd=tmp, env=env, capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired:
                result["status"] = "timeout"; return result
            if res.returncode != 0:
                result["status"] = f"xtb_rc={res.returncode}"; return result
            in_block = False; table = {}
            for ln in res.stdout.splitlines():
                if not in_block:
                    if ln.strip().startswith("Fukui functions"):
                        in_block = True
                    continue
                if ln.strip().startswith("---"): break
                m2 = FUKUI_LINE_RE.match(ln)
                if m2 is None: continue
                table[int(m2.group(1))] = (float(m2.group(3)),
                                            float(m2.group(4)),
                                            float(m2.group(5)))
            key = beta_rdkit + 1  # xtb 1-based
            if key not in table:
                result["status"] = "beta_missing_in_fukui"; return result
            fp, _, _ = table[key]
            result["f_plus"] = fp
            result["status"] = "ok"
            return result
    except Exception as e:
        result["status"] = f"exc_{type(e).__name__}"
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_smi", help="TSV: row_id\\tSMILES per line")
    ap.add_argument("output_csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    jobs = []
    with open(args.input_smi) as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line: continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 2:
                row_id, smi = parts[0], parts[1]
            else:
                # Single-column SMI: use line number as row_id.
                row_id, smi = str(lineno), parts[0]
            jobs.append((row_id, smi))
    if args.limit:
        jobs = jobs[:args.limit]
    n_total = len(jobs)
    print(f"[fukui] {n_total} jobs, {args.workers} workers", flush=True)
    t0 = time.time()

    with open(args.output_csv, "w") as out:
        w = csv.writer(out)
        w.writerow(["row_id", "smiles", "warhead_class", "beta_c_idx", "f_plus", "status"])
        with mp.Pool(processes=args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_xtb_one, jobs, chunksize=2)):
                w.writerow([r["row_id"], r["smiles"], r["warhead_class"],
                            r["beta_c_idx"], f"{r['f_plus']:.4f}", r["status"]])
                out.flush()
                if (k + 1) % 25 == 0 or (k + 1) == n_total:
                    dt = time.time() - t0
                    rate = (k + 1) / max(dt, 1e-6)
                    eta = (n_total - k - 1) / max(rate, 1e-6)
                    print(f"[{k+1}/{n_total}] {rate:.2f}/s  ETA {eta/60:.1f}min",
                          flush=True)
    print(f"[fukui] done in {(time.time()-t0)/60:.1f}min → {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
