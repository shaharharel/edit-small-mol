"""Exp-P6 new pair-mining schemes for peer-variant Phase-1 architectures.

Extends P5 schemes with the following:
  scheme=dir_wh_planar   → tgt strictly better planar than src; SAME warhead + SAME struct_id
  scheme=diverse_wh      → tgt has scaffold-Tc<0.7 to src; SAME warhead + SAME struct_id
  scheme=within_wh_only  → within struct_id; same warhead class filter (no directionality)

All schemes emit a pkl compatible with train_m1a_v2_pairs.py.

Usage:
  python expP6_mine_pairs.py --scheme dir_wh_planar \\
      --cache data/m1a_triples_v2/esm2_cache_posefix_v3.npz \\
      --planar_csv data/exp_P5/pairs_P5-A_planar.csv \\
      --pairs_out data/exp_P6/pairs_dir_wh_planar.pkl
"""
from __future__ import annotations
import argparse, pickle, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

# Warhead SMARTS — must match expP5_mine_and_train.py.
ACRYL  = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
HALO   = Chem.MolFromSmarts("[Cl,Br,I][CH2]C(=O)N")
VINSF  = Chem.MolFromSmarts("[CH2]=[CH]S(=O)(=O)")
NITRIL = Chem.MolFromSmarts("[C]#[N]")
ALDEH  = Chem.MolFromSmarts("[CH]=O")
BLACT  = Chem.MolFromSmarts("[NX3R]1C(=O)[CX4R]1")


def warhead_class(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None: return "invalid"
    if m.GetSubstructMatches(ACRYL):  return "acryl"
    if m.GetSubstructMatches(HALO):   return "haloacetamide"
    if m.GetSubstructMatches(VINSF):  return "vinylsulfone"
    if m.GetSubstructMatches(NITRIL): return "nitrile"
    if m.GetSubstructMatches(ALDEH):  return "aldehyde"
    if m.GetSubstructMatches(BLACT):  return "beta_lactam"
    return "none"


def murcko_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf.GetNumAtoms() == 0: return None
        return AllChem.GetMorganFingerprintAsBitVect(scaf, 2, 1024)
    except Exception:
        return None


def load_cache(cache_path: str):
    d = np.load(cache_path, allow_pickle=True)
    return {
        'smiles':     [str(s) for s in d['smiles']],
        'struct_ids': [str(s) for s in d['struct_ids']],
    }


def mine_dir_wh_planar(cache, planar_csv: str, min_gap_deg: float = 1.0):
    """Directional peer pairs: same struct_id + same warhead class + tgt planar better by ≥min_gap."""
    import pandas as pd
    df = pd.read_csv(planar_csv)
    # Accept common column names.
    idx_col = 'idx' if 'idx' in df.columns else 'row_idx'
    val_col = 'planar_dev_deg' if 'planar_dev_deg' in df.columns else 'planar_dev'
    idx_to_planar = dict(zip(df[idx_col], df[val_col]))

    smis, sids = cache['smiles'], cache['struct_ids']
    print(f"[mine] classifying {len(smis)} warheads...", flush=True)
    whs = [warhead_class(s) for s in smis]

    # Group by (struct_id, warhead_class), keep only those in planar table.
    groups = defaultdict(list)
    for i, (sid, w) in enumerate(zip(sids, whs)):
        if w in ('none', 'invalid'): continue
        if i not in idx_to_planar: continue
        groups[(sid, w)].append(i)

    pairs = []
    for (sid, w), idxs in groups.items():
        if len(idxs) < 2: continue
        for src in idxs:
            for tgt in idxs:
                if src == tgt: continue
                if smis[src] == smis[tgt]: continue
                if idx_to_planar[src] - idx_to_planar[tgt] < min_gap_deg: continue
                pairs.append({'src_idx': src, 'tgt_idx': tgt, 'sid': sid, 'wh': w,
                              'src_planar': float(idx_to_planar[src]),
                              'tgt_planar': float(idx_to_planar[tgt])})
    print(f"[mine] {len(pairs)} directional pairs (same wh + planar gap ≥{min_gap_deg}°)", flush=True)
    return pairs


def mine_diverse_wh(cache, max_scaf_tc: float = 0.7):
    """Diverse peer pairs: same struct_id + same warhead + scaffold Tanimoto ≤ max_scaf_tc."""
    smis, sids = cache['smiles'], cache['struct_ids']
    whs = [warhead_class(s) for s in smis]
    print(f"[mine] computing Murcko FPs for {len(smis)} mols...", flush=True)
    fps = [murcko_fp(s) for s in smis]

    groups = defaultdict(list)
    for i, (sid, w, fp) in enumerate(zip(sids, whs, fps)):
        if w in ('none', 'invalid'): continue
        if fp is None: continue
        groups[(sid, w)].append(i)

    pairs = []
    for (sid, w), idxs in groups.items():
        if len(idxs) < 2: continue
        for src in idxs:
            for tgt in idxs:
                if src == tgt: continue
                if smis[src] == smis[tgt]: continue
                tc = DataStructs.TanimotoSimilarity(fps[src], fps[tgt])
                if tc > max_scaf_tc: continue
                pairs.append({'src_idx': src, 'tgt_idx': tgt, 'sid': sid, 'wh': w,
                              'scaf_tc': float(tc)})
    print(f"[mine] {len(pairs)} scaffold-diverse pairs (Tc ≤ {max_scaf_tc})", flush=True)
    return pairs


def mine_within_wh_only(cache):
    """Within struct_id + same warhead class (no directionality)."""
    smis, sids = cache['smiles'], cache['struct_ids']
    whs = [warhead_class(s) for s in smis]
    groups = defaultdict(list)
    for i, (sid, w) in enumerate(zip(sids, whs)):
        if w in ('none', 'invalid'): continue
        groups[(sid, w)].append(i)
    pairs = []
    for (sid, w), idxs in groups.items():
        if len(idxs) < 2: continue
        for src in idxs:
            for tgt in idxs:
                if src == tgt: continue
                if smis[src] == smis[tgt]: continue
                pairs.append({'src_idx': src, 'tgt_idx': tgt, 'sid': sid, 'wh': w})
    print(f"[mine] {len(pairs)} within-wh pairs", flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', required=True,
                     choices=['dir_wh_planar', 'diverse_wh', 'within_wh_only'])
    ap.add_argument('--cache', default=str(PROJECT_ROOT / 'data/m1a_triples_v2/esm2_cache_posefix_v3.npz'))
    ap.add_argument('--pairs_out', required=True)
    ap.add_argument('--planar_csv', help='required for dir_wh_planar')
    ap.add_argument('--min_gap_deg', type=float, default=1.0)
    ap.add_argument('--max_scaf_tc', type=float, default=0.7)
    ap.add_argument('--max_pairs', type=int, default=10000)
    args = ap.parse_args()

    print(f"[mine] loading cache {args.cache}...", flush=True)
    cache = load_cache(args.cache)

    if args.scheme == 'dir_wh_planar':
        assert args.planar_csv, "--planar_csv required for dir_wh_planar"
        pairs = mine_dir_wh_planar(cache, args.planar_csv, args.min_gap_deg)
    elif args.scheme == 'diverse_wh':
        pairs = mine_diverse_wh(cache, args.max_scaf_tc)
    elif args.scheme == 'within_wh_only':
        pairs = mine_within_wh_only(cache)

    if len(pairs) > args.max_pairs:
        random.seed(42)
        pairs = random.sample(pairs, args.max_pairs)
        print(f"[mine] capped to {len(pairs)}", flush=True)

    Path(args.pairs_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pairs_out, 'wb') as f:
        pickle.dump({'pairs': pairs, 'scheme': args.scheme, 'cache_path': args.cache}, f)
    print(f"[mine] wrote {args.pairs_out} ({len(pairs)} pairs)", flush=True)

    stats = Path(args.pairs_out).with_suffix('.stats.json')
    stats.write_text(json.dumps({'scheme': args.scheme, 'n_pairs': len(pairs)}, indent=2))


if __name__ == '__main__':
    main()
