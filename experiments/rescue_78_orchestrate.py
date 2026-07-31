"""Rescue-78 final orchestrator: runs Phase 3 + 1.6 + 4 after Boltz cofolds are pulled.

Assumes:
  - data/boltz_rescue_78/<rid>/<rid>_model_0.cif files exist for all 78 mols
  - data/tier4_scored/rescue_78_working.csv exists with Phase 1.1-1.5 done

Runs:
  Phase 3.main      — boltz_iptm/pde/mPAE/d_SG/BD/n_h_bonds/.../vina/propka/strain
  Phase 3.7         — MM-GBSA (slow; may time out)
  Phase 1.6         — composite scores (P_potency etc + desirability_score)
  Phase 4.1         — write rescue_78_full.csv with column-complete schema

Phase 4.2 (backend) and 4.3 (light report) are already patched in source; the
remote deploy (4.4) is a separate manual SCP step the orchestrator triggers.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
LOG = ROOT / "data/tier4_scored/rescue_78_progress.log"
STATE = ROOT / "data/tier4_scored/rescue_78_state.json"


def now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"[{now_z()}] {msg}\n")


def run(name, cmd, timeout=14400):
    append_log(f"--> {name}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc = r.returncode
        append_log(f"<-- {name}: rc={rc} wall={time.time()-t0:.1f}s")
        if rc != 0:
            tail = (r.stderr or r.stdout)[-500:]
            append_log(f"    stderr tail: {tail}")
        return rc == 0
    except subprocess.TimeoutExpired:
        append_log(f"<-- {name}: TIMEOUT after {timeout}s")
        return False


def main():
    py = "/opt/miniconda3/envs/quris/bin/python"

    # Phase 3 main (boltz metrics + pose extras)
    ok_3 = run("phase_3_main", [py, str(ROOT / "experiments/rescue_78_phase3.py"), "3.main"],
              timeout=3600)

    # Phase 3.7 MM-GBSA — slow
    ok_37 = run("phase_3_7_mmgbsa", [py, str(ROOT / "experiments/rescue_78_phase3.py"), "3.7"],
                timeout=14400)

    # Phase 1.6 composite scores (needs Boltz cols)
    ok_16 = run("phase_1_6_composite", [py, str(ROOT / "experiments/rescue_78_phase3.py"), "1.6"],
                timeout=600)

    # Phase 4.1 write rescue_78_full.csv
    ok_41 = run("phase_4_1_full_csv", [py, str(ROOT / "experiments/rescue_78_phase4.py")],
                timeout=300)

    append_log(f"orchestrate summary: 3.main={ok_3} 3.7={ok_37} 1.6={ok_16} 4.1={ok_41}")


if __name__ == "__main__":
    main()
