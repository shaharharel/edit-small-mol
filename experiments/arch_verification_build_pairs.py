"""Build training-pair CSVs for arch_verification recipes 5, 7, 8.

Recipe 5 (v2_pairs_random_scaffold_pairing):
  Take v2 SMILES pool, build pairs by scaffold-hopping (any-Tc)
  instead of Tc-based nearest-neighbor. n ~= 1286.

Recipe 7 (v1_covaFT + v2_pairs + Mol1_anchored_augmentation):
  v2 pairs + 500 additional pairs where src or tgt is
  Mol1 or a Mol1 substructure fragment.

Recipe 8 (Massive Mol1-augmented):
  v2 pairs + 5000 pairs (src, Mol1) where src is a molecule at Tc>=0.5 to Mol1.

Outputs go to data/arch_verification/pairs/.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
V2_PAIRS_CSV = REPO / "data" / "optionA" / "v2_pairs_scheme_A_1286.csv"
V2_TRIPLES = REPO / "data" / "m1a_triples_v2" / "covindb_v2_triples_smarts_fixed.parquet"
OUT_DIR = REPO / "data" / "arch_verification" / "pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s)
    except Exception:
        return None


def build_recipe5(seed=0):
    """Random scaffold-hop pairs: for each v2 tgt scaffold, pair with a v2 tgt
    of a DIFFERENT scaffold. Keep n ~= 1286."""
    rng = random.Random(seed)
    df = pd.read_csv(V2_PAIRS_CSV)
    # Pool: unique v2 targets (267 unique per Claude's log)
    pool = sorted(set(df["tgt"].tolist()))
    print(f"[recipe5] target pool size = {len(pool)}", flush=True)
    scafs = {smi: scaffold(smi) for smi in pool}
    valid_pool = [s for s in pool if scafs[s] is not None]
    # For each src (there are 1286 unique srcs), pair with a random tgt of DIFFERENT scaffold from src's scaffold
    out_rows = []
    for _, row in df.iterrows():
        s_src = scaffold(row["src"])
        candidates = [t for t in valid_pool if scafs[t] != s_src]
        if not candidates:
            continue
        t = rng.choice(candidates)
        out_rows.append({"src": row["src"], "tgt": t})
    out_df = pd.DataFrame(out_rows)
    out_csv = OUT_DIR / "recipe5_scaffold_hop_pairs.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[recipe5] wrote {len(out_df)} pairs -> {out_csv}", flush=True)
    return out_csv


def build_recipe7(seed=0):
    """v2 pairs + 500 pairs where TGT = Mol1 and SRC = a v2-pool molecule
    (any Tc), or SRC = Mol1 substructure fragment."""
    rng = random.Random(seed)
    df = pd.read_csv(V2_PAIRS_CSV)
    triples = pd.read_parquet(V2_TRIPLES)
    v2_smis = sorted(set(triples["canon_smi"].tolist()))
    # 500 aug pairs: src <- random v2 mol, tgt <- Mol1
    aug_srcs = rng.sample(v2_smis, k=min(500, len(v2_smis)))
    aug_pairs = pd.DataFrame({"src": aug_srcs, "tgt": [MOL1_SMI] * len(aug_srcs)})
    combined = pd.concat([df, aug_pairs], ignore_index=True)
    out_csv = OUT_DIR / "recipe7_v2_plus_mol1_aug.csv"
    combined.to_csv(out_csv, index=False)
    print(f"[recipe7] base={len(df)} + aug={len(aug_pairs)} = {len(combined)} pairs -> {out_csv}", flush=True)
    return out_csv


def build_recipe8(seed=0, aug_n=5000):
    """v2 pairs + N pairs (src = mol most-similar-to-Mol1 in pool, tgt=Mol1).

    Original spec said Tc>=0.5, but v2 pool has 0 mols at Tc>=0.5 (v2 is
    generic covalent structures, Mol1 is a specific ZAP70 acrylamide).
    We instead take top-N Tc mols regardless of threshold, replicating with
    replacement if pool is smaller than aug_n. This is the STRONGEST possible
    Mol1-anchored augmentation: 5000 pairs all with tgt=Mol1.
    """
    rng = random.Random(seed)
    df = pd.read_csv(V2_PAIRS_CSV)
    # Merge pools: v2 targets/sources from the pairs CSV PLUS v2 raw triples
    triples = pd.read_parquet(V2_TRIPLES)
    v2_smis = sorted(set(triples["canon_smi"].tolist()) |
                     set(df["src"].tolist()) |
                     set(df["tgt"].tolist()))
    mol1_fp = morgan_fp(MOL1_SMI)
    tc_pairs = []
    for smi in v2_smis:
        fp = morgan_fp(smi)
        if fp is None or smi == MOL1_SMI:
            continue
        tc = tanimoto(fp, mol1_fp)
        tc_pairs.append((smi, tc))
    tc_pairs.sort(key=lambda x: -x[1])
    print(f"[recipe8] pool size = {len(tc_pairs)}; top Tc = "
          f"{[round(p[1],3) for p in tc_pairs[:10]]}", flush=True)
    # Take top-N by Tc; if pool smaller than aug_n, replicate with replacement
    picks = []
    top_pool = [p[0] for p in tc_pairs[: min(len(tc_pairs), aug_n)]]
    picks.extend(top_pool)
    while len(picks) < aug_n:
        picks.append(rng.choice(top_pool))
    picks = picks[:aug_n]
    aug_pairs = pd.DataFrame({"src": picks, "tgt": [MOL1_SMI] * len(picks)})
    combined = pd.concat([df, aug_pairs], ignore_index=True)
    out_csv = OUT_DIR / "recipe8_v2_plus_massive_mol1.csv"
    combined.to_csv(out_csv, index=False)
    print(f"[recipe8] base={len(df)} + aug={len(aug_pairs)} = {len(combined)} pairs -> {out_csv}", flush=True)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=["5", "7", "8", "all"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.recipe in ("5", "all"):
        build_recipe5(seed=args.seed)
    if args.recipe in ("7", "all"):
        build_recipe7(seed=args.seed)
    if args.recipe in ("8", "all"):
        build_recipe8(seed=args.seed)


if __name__ == "__main__":
    main()
