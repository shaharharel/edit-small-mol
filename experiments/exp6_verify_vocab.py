"""EXP6 Phase 0 verify: tokenize each filtered SMILES against the covalent_ft
mol2mol prior's actual token set. Anything containing an unknown token is dropped.

The prior's vocab dict has 128 tokens. We tokenize via REINVENT4's SMILES regex
(reinvent.models.mol2mol.dataset.dataset.Tokenizer) and check every token is in vocab.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

PROJ = Path(__file__).resolve().parent.parent
EXP6_ROOT = PROJ / "data" / "exp6_retrospective"
PRIOR = PROJ / "models" / "reinvent4_mol2mol_covalent_ft.prior"
TARGETS = ("egfr_t790m", "btk", "kras_g12c")

# REINVENT4 mol2mol SMILES regex — from reinvent.models.transformer.core.dataset
SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def tokenize(smi: str) -> list[str]:
    return SMI_REGEX.findall(smi)


def main():
    obj = torch.load(PRIOR, map_location="cpu", weights_only=False)
    vocab = set(obj["vocabulary"]["tokens"].keys())
    print(f"vocab size: {len(vocab)}")

    results = []
    for tgt in TARGETS:
        filt = EXP6_ROOT / tgt / "anchor_pool_strategy_b_filtered.csv"
        df = pd.read_csv(filt)
        n_total = len(df)
        kept_idx = []
        drop_reasons: dict[str, int] = {}
        for i, smi in enumerate(df["smiles"].astype(str).tolist()):
            toks = tokenize(smi)
            bad = [t for t in toks if t not in vocab]
            if bad:
                key = f"unknown_token:{bad[0]}"
                drop_reasons[key] = drop_reasons.get(key, 0) + 1
                continue
            kept_idx.append(i)
        out = df.iloc[kept_idx].reset_index(drop=True)
        out.to_csv(filt, index=False)
        results.append({
            "target": tgt,
            "n_total": n_total,
            "n_kept": len(out),
            "n_dropped": n_total - len(out),
            "drop_reasons": drop_reasons,
        })
        print(f"[{tgt}] n_total={n_total}  n_kept={len(out)}  n_dropped={n_total - len(out)}")
        for r, c in sorted(drop_reasons.items(), key=lambda kv: -kv[1])[:5]:
            print(f"   {r}: {c}")

    out_stats = EXP6_ROOT / "_rl" / "phase0_vocab_verify_stats.json"
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(results, indent=2))
    print(f"\nWrote -> {out_stats}")


if __name__ == "__main__":
    main()
