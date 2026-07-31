"""Iteration driver: pull Boltz cofolds from 6 GCP machines, convert CIFs,
run Vina --score_only, heartbeat to /tmp/vina_rescore_progress.json.

Pulls ONLY the cif + lig sdf (if present) + confidence json, skips the huge
npz files (PAE/PDE/PLDDT). Then converts cif -> full PDB locally with obabel,
and reuses experiments/vina_rescore_cofolds.rescore_cofold() which already
splits the full PDB into protein + ligand.

Output:
  data/boltz_results/cohort_3597_full/from_<letter>/<id>/<id>_model_0.cif
  data/boltz_results/cohort_3597_full/from_<letter>/<id>/<id>_model_0.pdb (converted)
  data/boltz_results/cohort_3597_full/from_<letter>/<id>/confidence_<id>_model_0.json
  data/tier4_scored/vina_rescore_boltz_intermediate.csv  (rolling)

Heartbeat:
  /tmp/vina_rescore_progress.json
  /tmp/vina_rescore_progress_log.txt  (one line per iteration)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vina_rescore_cofolds import rescore_cofold  # noqa: E402

COHORT_ROOT = PROJECT_ROOT / "data" / "boltz_results" / "cohort_3597_full"
OUT_CSV = PROJECT_ROOT / "data" / "tier4_scored" / "vina_rescore_boltz_intermediate.csv"
PROGRESS_JSON = Path("/tmp/vina_rescore_progress.json")
PROGRESS_LOG = Path("/tmp/vina_rescore_progress_log.txt")

MACHINES = [
    # (letter, instance, zone)
    ("a", "ai-gpu-a100", "us-central1-a"),
    ("b", "ai-gpu-a100-b", "us-central1-b"),
    ("c", "ai-gpu-a100-c", "us-central1-c"),
    ("d", "ai-gpu-a100-d", "us-east1-b"),
    ("e", "ai-gpu-a100-e", "us-west1-b"),
    ("f", "ai-gpu-a100-f", "us-west4-b"),
]

CSV_FIELDS = [
    "yaml_name",
    "row_id",
    "vina_affinity_kcalmol",
    "vina_inter_kcalmol",
    "vina_intra_kcalmol",
    "vina_torsions_kcalmol",
    "vina_unbound_kcalmol",
    "success_flag",
    "error",
    "boltz_machine",
    "ts",
]

OBABEL = "/opt/miniconda3/envs/quris/bin/obabel"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_heartbeat(payload: dict) -> None:
    PROGRESS_JSON.write_text(json.dumps(payload, indent=2))


def log_line(line: str) -> None:
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Remote listing + pull
# ---------------------------------------------------------------------------

def remote_list_ids(instance: str, zone: str) -> list[str]:
    """Get list of cofold ids that have completed on the remote (have a CIF)."""
    cmd = [
        "gcloud", "compute", "ssh", instance, f"--zone={zone}",
        "--command",
        # Print just the inner id whose CIF exists; results layout:
        #   ~/boltz_run/results/boltz_results_<id>/predictions/<id>/<id>_model_0.cif
        "for d in ~/boltz_run/results/boltz_results_*/predictions/*/; do "
        "id=$(basename \"$d\"); "
        "if [ -f \"$d/${id}_model_0.cif\" ]; then echo \"$id\"; fi; "
        "done",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        return []
    return [s for s in res.stdout.strip().split("\n") if s]


def local_done_ids(letter: str) -> set[str]:
    """Ids already pulled locally (have a CIF or converted PDB)."""
    base = COHORT_ROOT / f"from_{letter}"
    if not base.exists():
        return set()
    out = set()
    for d in base.iterdir():
        if not d.is_dir():
            continue
        cif = d / f"{d.name}_model_0.cif"
        if cif.exists() and cif.stat().st_size > 0:
            out.add(d.name)
    return out


def pull_ids(instance: str, zone: str, letter: str, ids: list[str]) -> int:
    """Pull cif + confidence json for the given ids via tar-over-ssh.

    Returns the number of ids successfully pulled.
    """
    if not ids:
        return 0
    dest = COHORT_ROOT / f"from_{letter}"
    dest.mkdir(parents=True, exist_ok=True)

    # Build a remote tar that streams just the small files (cif + confidence json).
    # We pull at most ~50 at a time to keep tar argv reasonable.
    BATCH = 50
    total = 0
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i+BATCH]
        # Remote shell: build a list of files for these ids, tar them.
        find_expr = " -o ".join(
            f"-path '*/predictions/{x}/{x}_model_0.cif' -o "
            f"-path '*/predictions/{x}/confidence_{x}_model_0.json'"
            for x in batch
        )
        remote_cmd = (
            "cd ~/boltz_run/results && "
            f"find . \\( {find_expr} \\) -print0 | "
            "tar --null -czf - --files-from=-"
        )
        cmd = [
            "gcloud", "compute", "ssh", instance, f"--zone={zone}",
            "--command", remote_cmd,
        ]
        # Stream tar to local extract; we strip the leading
        # ./boltz_results_<batchid>/predictions/<id>/  =>  <id>/
        # by extracting then renaming. Simpler: extract to a staging dir.
        staging = dest / "_staging"
        staging.mkdir(exist_ok=True)
        proc_remote = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            tar = subprocess.run(
                ["tar", "-xzf", "-", "-C", str(staging)],
                stdin=proc_remote.stdout,
                capture_output=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            proc_remote.kill()
            continue
        proc_remote.wait(timeout=10)
        if tar.returncode != 0:
            print(f"  tar extract failed for {letter} batch: {tar.stderr[:200]}")
            continue

        # Move extracted predictions/<id>/* into from_<letter>/<id>/
        for boltz_dir in staging.glob("boltz_results_*"):
            pred = boltz_dir / "predictions"
            if not pred.exists():
                continue
            for id_dir in pred.iterdir():
                if not id_dir.is_dir():
                    continue
                target = dest / id_dir.name
                target.mkdir(parents=True, exist_ok=True)
                for f in id_dir.iterdir():
                    target_f = target / f.name
                    if not target_f.exists():
                        os.replace(f, target_f)
                # Track success only if CIF arrived.
                if (target / f"{id_dir.name}_model_0.cif").exists():
                    total += 1
        # Clean staging.
        subprocess.run(["rm", "-rf", str(staging)], check=False)

    return total


# ---------------------------------------------------------------------------
# CIF -> PDB conversion
# ---------------------------------------------------------------------------

def cif_to_pdb(cif: Path, pdb: Path) -> bool:
    """Convert mmCIF to PDB with obabel, then relabel ligand records as HETATM.

    Boltz writes the ligand as resname 'LIG1' on chain B. obabel collapses to
    'LIG' but emits ATOM (not HETATM). vina_rescore_cofolds.split_pdb_to_protein_and_ligand
    expects HETATM with resname 'LIG'. We fix this in-place after conversion.
    """
    if pdb.exists() and pdb.stat().st_size > 0:
        return True
    cmd = [OBABEL, str(cif), "-O", str(pdb)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not pdb.exists() or pdb.stat().st_size == 0:
        return False
    # Relabel chain-B LIG ATOM records as HETATM.
    fixed_lines = []
    n_lig = 0
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM") and len(line) >= 22:
                resname = line[17:20].strip()
                chain = line[21:22]
                if resname == "LIG" and chain == "B":
                    line = "HETATM" + line[6:]
                    n_lig += 1
            fixed_lines.append(line)
    if n_lig == 0:
        # No LIG found — something off; leave as-is for caller to error.
        return True
    with open(pdb, "w") as f:
        f.writelines(fixed_lines)
    return True


def _convert_worker(cif_path_str: str) -> tuple[str, bool]:
    cif = Path(cif_path_str)
    pdb = cif.with_suffix(".pdb")
    ok = cif_to_pdb(cif, pdb)
    return cif_path_str, ok


def convert_all_cifs(workers: int = 8) -> tuple[int, int]:
    """Find all CIFs without a sibling PDB and convert. Returns (ok, fail)."""
    to_convert = []
    for letter in "abcdef":
        base = COHORT_ROOT / f"from_{letter}"
        if not base.exists():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            cif = d / f"{d.name}_model_0.cif"
            pdb = d / f"{d.name}_model_0.pdb"
            if cif.exists() and not pdb.exists():
                to_convert.append(str(cif))
    n_ok = n_fail = 0
    if not to_convert:
        return n_ok, n_fail
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futs = {exc.submit(_convert_worker, p): p for p in to_convert}
        for fut in as_completed(futs):
            _p, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
    return n_ok, n_fail


# ---------------------------------------------------------------------------
# Vina rescore
# ---------------------------------------------------------------------------

def load_done_csv() -> set[str]:
    """Return set of yaml_name already successfully scored."""
    if not OUT_CSV.exists():
        return set()
    done = set()
    with open(OUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("success_flag") == "1":
                done.add(row["yaml_name"])
    return done


def cofolds_to_rescore() -> list[tuple[Path, str]]:
    """Return list of (cofold_dir, machine_letter) for all local cofolds with PDB."""
    out = []
    for letter in "abcdef":
        base = COHORT_ROOT / f"from_{letter}"
        if not base.exists():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            pdb = d / f"{d.name}_model_0.pdb"
            if pdb.exists() and pdb.stat().st_size > 0:
                out.append((d, letter))
    return out


def _rescore_worker(args: tuple[str, str]) -> dict:
    cofold_dir_str, letter = args
    cofold_dir = Path(cofold_dir_str)
    try:
        res = rescore_cofold(cofold_dir)
    except Exception as e:
        res = {
            "name": cofold_dir.name,
            "success_flag": 0,
            "error": f"uncaught:{type(e).__name__}:{e}",
            "vina_kcalmol": None,
            "vina_inter_kcalmol": None,
            "vina_intra_kcalmol": None,
        }
    # Map to our CSV schema.
    name = res.get("name", cofold_dir.name)
    row_id = None
    try:
        row_id = int(name.rsplit("_", 1)[-1]) if "_" in name else int(name)
    except ValueError:
        try:
            row_id = int(name)
        except ValueError:
            row_id = None
    return {
        "yaml_name": name,
        "row_id": row_id,
        "vina_affinity_kcalmol": res.get("vina_kcalmol"),
        "vina_inter_kcalmol": res.get("vina_inter_kcalmol"),
        "vina_intra_kcalmol": res.get("vina_intra_kcalmol"),
        "vina_torsions_kcalmol": res.get("vina_torsion_kcalmol"),
        "vina_unbound_kcalmol": res.get("vina_unbound_kcalmol"),
        "success_flag": res.get("success_flag", 0),
        "error": res.get("error"),
        "boltz_machine": letter,
        "ts": now_iso(),
    }


def rescore_all(workers: int = 8) -> tuple[int, int, int]:
    """Run rescore on all local cofolds not yet in the CSV.
    Returns (n_new, n_ok, n_fail).
    """
    done = load_done_csv()
    tasks = [(str(d), letter) for d, letter in cofolds_to_rescore() if d.name not in done]
    if not tasks:
        return 0, 0, 0

    new_file = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    fout = open(OUT_CSV, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        fout.flush()

    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futs = {exc.submit(_rescore_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:
                t = futs[fut]
                row = {
                    "yaml_name": Path(t[0]).name,
                    "row_id": None,
                    "vina_affinity_kcalmol": None,
                    "vina_inter_kcalmol": None,
                    "vina_intra_kcalmol": None,
                    "vina_torsions_kcalmol": None,
                    "vina_unbound_kcalmol": None,
                    "success_flag": 0,
                    "error": f"future:{type(e).__name__}:{e}",
                    "boltz_machine": t[1],
                    "ts": now_iso(),
                }
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})
            fout.flush()
            if row.get("success_flag") == 1:
                n_ok += 1
            else:
                n_fail += 1
    fout.close()
    return len(tasks), n_ok, n_fail


# ---------------------------------------------------------------------------
# Iteration loop
# ---------------------------------------------------------------------------

def count_local_cofolds() -> dict:
    counts = {}
    total = 0
    for letter in "abcdef":
        base = COHORT_ROOT / f"from_{letter}"
        if not base.exists():
            counts[letter] = 0
            continue
        n = sum(1 for d in base.iterdir() if d.is_dir() and (d / f"{d.name}_model_0.cif").exists())
        counts[letter] = n
        total += n
    return {"counts": counts, "total": total}


def csv_counts() -> tuple[int, int, int]:
    """Return (total rows, n_ok, n_fail) in the rolling CSV."""
    if not OUT_CSV.exists():
        return 0, 0, 0
    total = ok = fail = 0
    with open(OUT_CSV) as f:
        for row in csv.DictReader(f):
            total += 1
            if row.get("success_flag") == "1":
                ok += 1
            else:
                fail += 1
    return total, ok, fail


def run_iteration(iteration: int, workers: int, sleep_sec: int) -> dict:
    t0 = time.time()
    log_line(f"--- iter {iteration} START @ {now_iso()}")

    # Phase: pulling
    write_heartbeat({
        "ts": now_iso(), "iteration": iteration, "phase": "pulling",
        **count_local_cofolds_payload(),
    })
    pulled_per_machine = {}
    for letter, instance, zone in MACHINES:
        try:
            remote_ids = remote_list_ids(instance, zone)
        except Exception as e:
            log_line(f"  iter {iteration} list-fail {letter}: {e}")
            remote_ids = []
        local_done = local_done_ids(letter)
        new_ids = [x for x in remote_ids if x not in local_done]
        log_line(f"  iter {iteration} machine {letter}: remote={len(remote_ids)} local={len(local_done)} new={len(new_ids)}")
        if not new_ids:
            pulled_per_machine[letter] = 0
            continue
        try:
            n_pulled = pull_ids(instance, zone, letter, new_ids)
        except Exception as e:
            log_line(f"  iter {iteration} pull-fail {letter}: {e}")
            n_pulled = 0
        pulled_per_machine[letter] = n_pulled

    # Phase: converting
    write_heartbeat({
        "ts": now_iso(), "iteration": iteration, "phase": "converting",
        **count_local_cofolds_payload(),
    })
    n_conv_ok, n_conv_fail = convert_all_cifs(workers=workers)
    log_line(f"  iter {iteration} convert: ok={n_conv_ok} fail={n_conv_fail}")

    # Phase: rescoring
    write_heartbeat({
        "ts": now_iso(), "iteration": iteration, "phase": "rescoring",
        **count_local_cofolds_payload(),
    })
    n_new, n_ok, n_fail = rescore_all(workers=workers)
    log_line(f"  iter {iteration} rescore: new={n_new} ok={n_ok} fail={n_fail}")

    # Phase: done
    csv_total, csv_ok, csv_fail = csv_counts()
    dur = time.time() - t0
    rate = csv_ok / max(dur / 3600.0, 1e-9) if n_new else 0.0
    payload = {
        "ts": now_iso(),
        "iteration": iteration,
        "phase": "done",
        **count_local_cofolds_payload(),
        "rescored_total": csv_total,
        "vanilla_done": csv_ok,
        "covalent_done": 0,
        "errors": csv_fail,
        "rate_per_hr": round(rate, 1),
        "boltz_machine_counts": count_local_cofolds()["counts"],
        "pulled_this_iter": pulled_per_machine,
        "iter_duration_sec": round(dur, 1),
        "next_iteration_in_sec": sleep_sec,
    }
    write_heartbeat(payload)
    log_line(
        f"iter {iteration} DONE in {dur:.1f}s | local={payload['local_cofolds_total']} "
        f"rescored={csv_total} ok={csv_ok} fail={csv_fail} pulled={pulled_per_machine}"
    )
    return payload


def count_local_cofolds_payload() -> dict:
    info = count_local_cofolds()
    return {
        "local_cofolds_total": info["total"],
        "local_cofolds_per_machine": info["counts"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--sleep-sec", type=int, default=1800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-total", type=int, default=3597)
    args = ap.parse_args()

    PROGRESS_LOG.write_text("")  # reset log
    for i in range(1, args.iterations + 1):
        try:
            payload = run_iteration(i, workers=args.workers, sleep_sec=args.sleep_sec)
        except Exception as e:
            err = f"iter {i} BLOCKED: {type(e).__name__}: {e}"
            log_line(err)
            import traceback
            tb = traceback.format_exc()
            write_heartbeat({
                "ts": now_iso(),
                "iteration": i,
                "phase": "BLOCKED",
                "error": err,
                "traceback": tb,
            })
            return 2
        if payload.get("local_cofolds_total", 0) >= args.target_total:
            log_line(f"reached target {args.target_total}, stopping after iter {i}")
            break
        if i < args.iterations:
            time.sleep(args.sleep_sec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
