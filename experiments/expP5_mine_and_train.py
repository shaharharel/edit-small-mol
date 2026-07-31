"""Exp-P5 unified: mine pairs from ESM cache + train v2_cond variant.

Supports multiple pair-selection schemes:
  scheme=within_struct_id   → pair within same PDB (baseline)
  scheme=warhead_swap       → pair mols with matching de-warheaded scaffold but different warhead
  scheme=geom_improvement   → pair (worst_planar, best_planar) within struct_id
  scheme=covvina_graded     → pair (weak_covvina, strong_covvina) within struct_id  [requires --covvina_csv]

Writes a filtered pairs.npz that the modified training loop expects.

Then trains v2_cond with src≠tgt (src=paired src SMILES, tgt=paired tgt SMILES).

Usage:
  python expP5_mine_and_train.py --scheme warhead_swap \\
      --cache data/m1a_triples_v2/esm2_cache_posefix_v3.npz \\
      --out_dir data/exp_P5/models/P5-4 \\
      --pairs_out data/exp_P5/pairs_warhead_swap.pkl \\
      --epochs 30 --lr 1e-4 --bs 16 --seed 42
"""
from __future__ import annotations
import argparse, pickle, sys, time, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
HALO  = Chem.MolFromSmarts("[Cl,Br,I][CH2]C(=O)N")
VINSF = Chem.MolFromSmarts("[CH2]=[CH]S(=O)(=O)")
NITRIL= Chem.MolFromSmarts("[C]#[N]")
ALDEH = Chem.MolFromSmarts("[CH]=O")
BLACT = Chem.MolFromSmarts("[NX3R]1C(=O)[CX4R]1")
WARHEAD_PATTERNS = [ACRYL, HALO, VINSF, NITRIL, ALDEH, BLACT]


def get_warhead_class(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None: return "invalid"
    if m.GetSubstructMatches(ACRYL): return "acryl"
    if m.GetSubstructMatches(HALO): return "haloacetamide"
    if m.GetSubstructMatches(VINSF): return "vinylsulfone"
    if m.GetSubstructMatches(NITRIL): return "nitrile"
    if m.GetSubstructMatches(ALDEH): return "aldehyde"
    if m.GetSubstructMatches(BLACT): return "beta_lactam"
    return "none"


def strip_warhead(smi: str) -> str | None:
    """Remove atoms matching any warhead pattern; return SMILES of remainder."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    edit = Chem.RWMol(m)
    to_del = set()
    for patt in WARHEAD_PATTERNS:
        for match in m.GetSubstructMatches(patt):
            to_del.update(match)
    if not to_del:
        return Chem.MolToSmiles(m)  # nothing to strip
    for i in sorted(to_del, reverse=True):
        edit.RemoveAtom(i)
    try:
        Chem.SanitizeMol(edit)
        return Chem.MolToSmiles(edit)
    except Exception:
        return None


def de_warheaded_murcko(smi: str) -> str | None:
    """Murcko scaffold of the de-warheaded molecule."""
    dw = strip_warhead(smi)
    if dw is None: return None
    m = Chem.MolFromSmiles(dw)
    if m is None: return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf) if scaf.GetNumAtoms() > 0 else None
    except Exception:
        return None


def load_cache(cache_path: str):
    d = np.load(cache_path, allow_pickle=True)
    return {
        'smiles': [str(s) for s in d['smiles']],
        'struct_ids': [str(s) for s in d['struct_ids']],
        'sources': [str(s) for s in d['sources']],
        'poses': d['poses'],
        'poses_unnorm': d['poses_unnorm'],
        'row_seq_idx': d['row_seq_idx'],
        'residues_emb': d['residues_emb'],
        'residues_mask': d['residues_mask'],
        'seq_hashes': [str(s) for s in d['seq_hashes']],
        'pose_mean': d['pose_mean'],
        'pose_std': d['pose_std'],
    }


def mine_within_struct_id(cache):
    """Scheme 1: pair mols within same struct_id (same PDB)."""
    smis = cache['smiles']; sids = cache['struct_ids']
    groups = defaultdict(list)
    for i, sid in enumerate(sids):
        groups[sid].append(i)
    pairs = []
    for sid, idxs in groups.items():
        smset = list({smis[i] for i in idxs})
        if len(smset) < 2: continue
        # enumerate ordered pairs of distinct smiles
        for a in idxs:
            for b in idxs:
                if a == b: continue
                if smis[a] == smis[b]: continue
                pairs.append({'src_idx': a, 'tgt_idx': b, 'sid': sid})
    return pairs


def mine_warhead_swap(cache):
    """Scheme 4: pair mols sharing de-warheaded scaffold but different warhead class."""
    smis = cache['smiles']
    print(f"[mine] computing de-warheaded scaffolds for {len(smis)} mols...", flush=True)
    scaf_map = defaultdict(list)  # scaffold → list of (idx, warhead_class)
    for i, smi in enumerate(smis):
        s = de_warheaded_murcko(smi)
        if s is None or s == '': continue
        w = get_warhead_class(smi)
        if w in ('none', 'invalid'): continue
        scaf_map[s].append((i, w))
    print(f"[mine] {len(scaf_map)} unique scaffolds", flush=True)
    pairs = []
    for scaf, entries in scaf_map.items():
        if len(entries) < 2: continue
        # pair entries with different warhead class OR different exact SMILES
        for i, (a, wa) in enumerate(entries):
            for b, wb in entries:
                if a == b: continue
                if smis[a] == smis[b]: continue
                # Include if warheads differ OR if same-warhead but different molecule
                # (we want warhead-swap primarily, but same-warhead-different-scaffold is useful too)
                pairs.append({'src_idx': a, 'tgt_idx': b, 'scaffold': scaf,
                              'wh_src': wa, 'wh_tgt': wb})
    print(f"[mine] {len(pairs)} warhead-swap pairs mined", flush=True)
    return pairs


def mine_geom_improvement(cache, planar_csv: str):
    """Scheme A: pair (worst_planar, best_planar) — first try within-struct-id,
    then fall back to same-sequence group (row_seq_idx), then finally same-warhead-class."""
    import pandas as pd
    df = pd.read_csv(planar_csv)
    idx_to_planar = dict(zip(df['idx'], df['planar_dev_deg']))
    smis = cache['smiles']; sids = cache['struct_ids']
    row_seq_idx = cache['row_seq_idx']
    pairs = []
    seen = set()

    # Tier 1: same struct_id, planar gap >= 0 (any improvement)
    g1 = defaultdict(list)
    for i, sid in enumerate(sids):
        if i in idx_to_planar: g1[sid].append(i)
    for sid, idxs in g1.items():
        if len(idxs) < 2: continue
        idxs_sorted = sorted(idxs, key=lambda i: idx_to_planar[i])
        best_i, worst_i = idxs_sorted[0], idxs_sorted[-1]
        if idx_to_planar[worst_i] - idx_to_planar[best_i] < 0.5: continue
        if smis[worst_i] == smis[best_i]: continue
        key = (worst_i, best_i)
        if key in seen: continue
        seen.add(key)
        pairs.append({'src_idx': worst_i, 'tgt_idx': best_i, 'tier': 'struct_id',
                      'src_planar': idx_to_planar[worst_i], 'tgt_planar': idx_to_planar[best_i]})

    # Tier 2: same seq_idx (same pocket sequence — could be different PDB/struct)
    g2 = defaultdict(list)
    for i in idx_to_planar:
        g2[int(row_seq_idx[i])].append(i)
    for sq, idxs in g2.items():
        if len(idxs) < 2: continue
        idxs_sorted = sorted(idxs, key=lambda i: idx_to_planar[i])
        # take top 3 worst paired with top 3 best (or all if small)
        n = min(3, len(idxs_sorted) // 2)
        for j in range(n):
            src, tgt = idxs_sorted[-1-j], idxs_sorted[j]
            if idx_to_planar[src] - idx_to_planar[tgt] < 3.0: continue
            if smis[src] == smis[tgt]: continue
            key = (src, tgt)
            if key in seen: continue
            seen.add(key)
            pairs.append({'src_idx': src, 'tgt_idx': tgt, 'tier': 'same_seq',
                          'src_planar': idx_to_planar[src], 'tgt_planar': idx_to_planar[tgt]})

    # Tier 3: cross-cache — pair random-ish (worst, best) globally, capped
    if len(pairs) < 2000:
        all_sorted = sorted(idx_to_planar.keys(), key=lambda i: idx_to_planar[i])
        n_take = min(1000, len(all_sorted) // 4)
        worsts = all_sorted[-n_take:]
        bests = all_sorted[:n_take]
        import random; random.seed(42); random.shuffle(worsts); random.shuffle(bests)
        for src, tgt in zip(worsts, bests):
            if smis[src] == smis[tgt]: continue
            if idx_to_planar[src] - idx_to_planar[tgt] < 5.0: continue
            key = (src, tgt)
            if key in seen: continue
            seen.add(key)
            pairs.append({'src_idx': src, 'tgt_idx': tgt, 'tier': 'cross_global',
                          'src_planar': idx_to_planar[src], 'tgt_planar': idx_to_planar[tgt]})
    return pairs


def mine_covvina_graded(cache, covvina_csv: str):
    """Scheme C: pair (weak, strong) cov-Vina within struct_id.
    Requires CSV with [idx, covvina_kcal]."""
    import pandas as pd
    df = pd.read_csv(covvina_csv)
    idx_to_v = dict(zip(df['idx'], df['covvina_kcal']))
    smis = cache['smiles']; sids = cache['struct_ids']
    groups = defaultdict(list)
    for i, sid in enumerate(sids):
        if i in idx_to_v: groups[sid].append(i)
    pairs = []
    for sid, idxs in groups.items():
        if len(idxs) < 2: continue
        # sort by cov-Vina (strongest = most negative)
        idxs_sorted = sorted(idxs, key=lambda i: idx_to_v[i])
        strong, weak = idxs_sorted[0], idxs_sorted[-1]  # most negative first
        if idx_to_v[weak] - idx_to_v[strong] < 1.0: continue  # need ≥1 kcal gap
        pairs.append({'src_idx': weak, 'tgt_idx': strong, 'sid': sid,
                      'src_covvina': idx_to_v[weak], 'tgt_covvina': idx_to_v[strong]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', required=True, choices=['within_struct_id','warhead_swap','geom_improvement','covvina_graded'])
    ap.add_argument('--cache', default=str(PROJECT_ROOT / 'data/m1a_triples_v2/esm2_cache_posefix_v3.npz'))
    ap.add_argument('--pairs_out', required=True)
    ap.add_argument('--planar_csv', help='required for geom_improvement')
    ap.add_argument('--covvina_csv', help='required for covvina_graded')
    ap.add_argument('--dry_run', action='store_true', help='just mine pairs, do not launch train')
    ap.add_argument('--max_pairs', type=int, default=10000, help='cap pair count to avoid overtraining')
    args = ap.parse_args()

    print(f"[mine] loading cache {args.cache}...", flush=True)
    cache = load_cache(args.cache)

    if args.scheme == 'within_struct_id':
        pairs = mine_within_struct_id(cache)
    elif args.scheme == 'warhead_swap':
        pairs = mine_warhead_swap(cache)
    elif args.scheme == 'geom_improvement':
        assert args.planar_csv, "--planar_csv required"
        pairs = mine_geom_improvement(cache, args.planar_csv)
    elif args.scheme == 'covvina_graded':
        assert args.covvina_csv, "--covvina_csv required"
        pairs = mine_covvina_graded(cache, args.covvina_csv)

    print(f"[mine] {len(pairs)} pairs total", flush=True)
    if len(pairs) > args.max_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, args.max_pairs)
        print(f"[mine] capped to {len(pairs)}", flush=True)

    Path(args.pairs_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pairs_out, 'wb') as f:
        pickle.dump({'pairs': pairs, 'scheme': args.scheme, 'cache_path': args.cache}, f)
    print(f"[mine] wrote {args.pairs_out}", flush=True)

    stats_out = Path(args.pairs_out).with_suffix('.stats.json')
    stats_out.write_text(json.dumps({'scheme': args.scheme, 'n_pairs': len(pairs)}, indent=2))
    print(f"[done] mining complete. To train, use train_m1a_v2_pairs.py --pairs_pkl {args.pairs_out}", flush=True)


if __name__ == '__main__':
    main()
