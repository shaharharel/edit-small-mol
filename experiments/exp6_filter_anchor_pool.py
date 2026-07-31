"""EXP6 Phase 0: filter anchor pools for mol2mol covalent_ft prior vocab.

The 100-anchor Strategy B pools contain SMILES with charged atoms / tokens
that the covalent_ft mol2mol prior cannot tokenize ([S+], [P], [C-]#[N+] ...).
Every B_iter1 staged_learning run fails because of this.

We use the SAME _is_supported_smi heuristic from experiments/exp6_rl_driver.py
to filter each target's anchor_pool_strategy_b.csv -> anchor_pool_strategy_b_filtered.csv.

Output: per-target stats (n_total, n_kept, n_drop, dropped reasons).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

EXP6_ROOT = Path(__file__).resolve().parent.parent / "data" / "exp6_retrospective"
TARGETS = ("egfr_t790m", "btk", "kras_g12c")

# This MUST stay identical to experiments/exp6_rl_driver.py::_is_supported_smi
ALLOWED_BRACKETS = {
    "N+", "N@+", "N@@+", "O-", "nH", "n+",
    "C@H", "C@@H", "C@", "C@@",
    "O", "*",
    "18F", "19F", "76Br", "123I", "11c", "11C", "11CH3",
    "S@", "S@@",
}

BARE_BAD_RE = re.compile(r"(?<![A-Za-z])(?:P|p|B)(?![a-z])")


def is_supported(smi: str) -> tuple[bool, str]:
    if not smi:
        return False, "empty"
    if BARE_BAD_RE.search(smi):
        return False, "bare_P_or_B"
    for m in re.findall(r"\[([^\]]+)\]", smi):
        if m not in ALLOWED_BRACKETS:
            return False, f"bracket:{m}"
    return True, ""


def canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def filter_pool(target: str) -> dict:
    src = EXP6_ROOT / target / "anchor_pool_strategy_b.csv"
    dst = EXP6_ROOT / target / "anchor_pool_strategy_b_filtered.csv"
    df = pd.read_csv(src)
    n_total = len(df)
    rows_keep = []
    drop_reasons: dict[str, int] = {}
    for _i, row in df.iterrows():
        raw = str(row["smiles"])
        canon = canonicalize(raw)
        if canon is None:
            drop_reasons["unparseable"] = drop_reasons.get("unparseable", 0) + 1
            continue
        # check BOTH the raw and the canonical form against vocab — RDKit
        # may canonicalize away a charge or rearrange brackets.
        for cand in (raw, canon):
            ok, reason = is_supported(cand)
            if not ok:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                break
        else:
            new_row = dict(row)
            new_row["smiles"] = canon
            rows_keep.append(new_row)
    out_df = pd.DataFrame(rows_keep)
    out_df.to_csv(dst, index=False)
    return {
        "target": target,
        "src": str(src),
        "dst": str(dst),
        "n_total": n_total,
        "n_kept": len(out_df),
        "n_dropped": n_total - len(out_df),
        "drop_reasons": drop_reasons,
    }


def main():
    results = []
    for tgt in TARGETS:
        r = filter_pool(tgt)
        results.append(r)
        print(f"[{tgt}] n_total={r['n_total']}  n_kept={r['n_kept']}  n_dropped={r['n_dropped']}")
        for reason, count in sorted(r["drop_reasons"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"   drop[{reason}] = {count}")
    out_json = EXP6_ROOT / "_rl" / "phase0_anchor_pool_filter_stats.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote stats -> {out_json}")
    return results


if __name__ == "__main__":
    main()
