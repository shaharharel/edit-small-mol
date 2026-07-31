#!/usr/bin/env python3
"""Resume-aware wrapper around vanilla_vina_scorer.py / adcov_local_scorer.py.

Why: SPOT instances can be preempted at any moment. The base scorers open the
output CSV in "w" mode, so a partial run is lost on restart. This wrapper:
  1. Reads existing output CSV (if present) and collects already-done row_ids.
  2. Filters the input .smi to skip those rows.
  3. Writes a filtered .smi to a temp file.
  4. Invokes the base scorer, redirecting output to a temp CSV, then APPENDS
     rows (skipping the header) to the final output CSV.
  5. Installs a SIGTERM handler that flushes & exits cleanly so already-written
     rows are durable.

Usage:
  python resume_scorer_wrapper.py vanilla <input.smi> <output.csv> [N_WORKERS]
  python resume_scorer_wrapper.py adcov   <input.smi> <output.csv> [N_WORKERS]

Designed to be run in an infinite-retry loop:
  while ! python resume_scorer_wrapper.py vanilla in.smi out.csv 32; do sleep 5; done
"""
import csv
import os
import signal
import subprocess as sp
import sys
import tempfile
import time
from pathlib import Path

SCORER_MAP = {
    "vanilla": "experiments/vanilla_vina_scorer.py",
    "adcov":   "experiments/adcov_local_scorer.py",
}


def load_done_row_ids(out_csv: Path) -> set[str]:
    done: set[str] = set()
    if not out_csv.exists():
        return done
    try:
        with open(out_csv) as f:
            r = csv.DictReader(f)
            for row in r:
                rid = row.get("row_id")
                if rid is not None and rid != "":
                    done.add(rid)
    except Exception as e:
        print(f"[wrapper] warn: could not parse existing {out_csv}: {e}", file=sys.stderr)
    return done


def filter_input(inp: Path, done: set[str], filtered_path: Path) -> tuple[int, int]:
    total = 0
    kept = 0
    with open(inp) as fin, open(filtered_path, "w") as fout:
        for i, line in enumerate(fin):
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            total += 1
            rid = parts[1] if len(parts) > 1 else str(i)
            if rid in done:
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    return total, kept


def append_temp_to_final(temp_csv: Path, final_csv: Path, write_header: bool) -> int:
    n = 0
    with open(temp_csv) as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            return 0
        mode = "w" if write_header else "a"
        with open(final_csv, mode, newline="") as fout:
            w = csv.writer(fout)
            if write_header:
                w.writerow(header)
            for row in rdr:
                w.writerow(row)
                n += 1
    return n


def main():
    if len(sys.argv) < 4:
        print("usage: resume_scorer_wrapper.py {vanilla|adcov} <input.smi> <output.csv> [N_WORKERS]", file=sys.stderr)
        sys.exit(2)
    kind = sys.argv[1]
    if kind not in SCORER_MAP:
        print(f"unknown kind: {kind}", file=sys.stderr); sys.exit(2)
    inp = Path(sys.argv[2]).resolve()
    out = Path(sys.argv[3]).resolve()
    nw = sys.argv[4] if len(sys.argv) > 4 else "32"

    repo_root = Path(__file__).resolve().parent.parent
    scorer = repo_root / SCORER_MAP[kind]
    if not scorer.exists():
        print(f"scorer not found: {scorer}", file=sys.stderr); sys.exit(3)

    out.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_row_ids(out)
    print(f"[wrapper] kind={kind}  input={inp.name}  output={out.name}", flush=True)
    print(f"[wrapper] already-done rows: {len(done):,}", flush=True)

    # Write filtered .smi to a temp file
    with tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False) as tf:
        filtered = Path(tf.name)
    total, kept = filter_input(inp, done, filtered)
    print(f"[wrapper] input total={total:,}  remaining={kept:,}  skipping={total-kept:,}", flush=True)

    if kept == 0:
        print("[wrapper] nothing to do — all rows already scored.", flush=True)
        filtered.unlink(missing_ok=True)
        sys.exit(0)

    # Temp output CSV for this run
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf2:
        tmp_out = Path(tf2.name)

    # Forward SIGTERM to child so it exits cleanly (its per-row writes are flushed)
    child_proc = {"p": None}

    def _on_term(signum, _frame):
        print(f"[wrapper] caught signal {signum} — forwarding to scorer", file=sys.stderr, flush=True)
        p = child_proc["p"]
        if p is not None:
            try: p.send_signal(signum)
            except Exception: pass

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    cmd = [sys.executable, str(scorer), str(filtered), str(tmp_out), str(nw)]
    print(f"[wrapper] cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    child_proc["p"] = sp.Popen(cmd, cwd=str(repo_root))
    rc = child_proc["p"].wait()
    dt = time.time() - t0
    print(f"[wrapper] scorer exited rc={rc} in {dt:.0f}s", flush=True)

    # Append whatever the scorer wrote (even partial) to the final CSV
    write_header = not out.exists() or out.stat().st_size == 0
    n_appended = 0
    if tmp_out.exists():
        n_appended = append_temp_to_final(tmp_out, out, write_header)
    print(f"[wrapper] appended {n_appended:,} rows → {out}", flush=True)

    # Cleanup
    filtered.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)

    # If preempted (non-zero rc OR fewer-than-expected rows), exit non-zero so
    # the outer loop retries. Otherwise exit 0.
    if rc != 0 or n_appended < kept:
        print(f"[wrapper] INCOMPLETE: rc={rc}, appended={n_appended}/{kept}. Outer loop should retry.", flush=True)
        sys.exit(rc if rc != 0 else 1)
    print(f"[wrapper] COMPLETE.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
