"""Build retrieval prefix JSON files for K in {1, 5, 10, 20}.

Runs build_retrieval_prefix.py once with K=20 then slices it into K=1, 5, 10.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v5_retrievalK"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Build top-20 by running the existing script.
    top20_path = out_dir / "retrieval_top20.json"
    cmd = ["python",
             str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/cfg_retrieval/"
                  "build_retrieval_prefix.py"),
             "--k", "20",
             "--out", str(top20_path)]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    d = json.loads(top20_path.read_text())
    top20 = d.get("top_k", [])
    print(f"Full top-20 loaded ({len(top20)})", flush=True)
    for K in [1, 5, 10, 20]:
        sub = dict(d); sub["top_k"] = top20[:K]
        p = out_dir / f"retrieval_top{K}.json"
        p.write_text(json.dumps(sub, indent=2))
        print(f"Wrote {p} (K={K})", flush=True)


if __name__ == "__main__":
    main()
