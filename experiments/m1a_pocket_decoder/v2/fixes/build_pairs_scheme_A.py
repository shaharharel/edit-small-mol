"""Scheme A — Same-Pocket Neighbor Edit (MMP-on-pocket).

Offline pair builder. For every pocket with >= 2 distinct ligands, form
directed pairs whose pairwise Tanimoto lies in [0.3, 0.85] (excluding
identical and completely-unrelated pairs). Yields both directions
(A->B and B->A) so the model must read the pose token to know which
neighbor is the current target.

Emits `data/m1a_triples_v2/pairs_scheme_A.npz` with fields:
  src_smi (str, one per pair)
  tgt_smi (str, one per pair)          - canonical
  residues_emb_idx (int32)             - index into ESM cache (U)
  residues_mask_idx (int32)            - same index
  pose (float32, (N, 3))               - z-scored tgt pose
  tc (float32, (N,))
  source_a (object, (N,))              - source-tag of src row (audit)
  source_b (object, (N,))              - source-tag of tgt row (audit)

Plus `pairs_scheme_A_stats.json`.

Design decisions:
  * We use "distinct ligands" = distinct canonical SMILES. Two rows with
    the SAME canonical SMILES on the same pocket are treated as one node.
    (Prevents forming pairs like C=CC(=O)... -> C=CC(=O)... where the
    identity edit teaches the model nothing.)
  * We DO NOT restrict by warhead class here. Scheme A's point is edit
    diversity; warhead-switching pairs are welcome and their pose deltas
    are exactly the signal we want the pose token to carry.
  * Directional: (a -> b) uses pose_b, (b -> a) uses pose_a. Same
    residues_emb_idx.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

TC_MIN = 0.3
TC_MAX = 0.85
FP_RADIUS = 2
FP_NBITS = 2048


def mol_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS)


def canonicalize(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def randomize_smi(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=False, doRandom=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--out_npz", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/pairs_scheme_A.npz"))
    ap.add_argument("--out_stats", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/pairs_scheme_A_stats.json"))
    ap.add_argument("--tc_min", type=float, default=TC_MIN)
    ap.add_argument("--tc_max", type=float, default=TC_MAX)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only process the first --limit source rows. "
                         "For unit tests.")
    ap.add_argument("--max_pairs_per_pocket", type=int, default=200,
                    help="Cap directed pairs per pocket. Prevents a single "
                         "large pocket (e.g. the boltz_zap70 449-mol block) "
                         "from dominating.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cache_path = Path(args.cache)
    print(f"[Scheme A] Loading cache: {cache_path}", flush=True)
    d = np.load(cache_path, allow_pickle=True)
    sources = d["sources"]
    smiles = d["smiles"]
    row_seq_idx = d["row_seq_idx"].astype(np.int32)
    poses = d["poses"].astype(np.float32)

    rows = np.arange(len(smiles))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[Scheme A] considering {len(rows)} rows", flush=True)

    # ---- Canonicalize, group by (pocket_idx, canonical_smi) ----
    # We pick ONE representative row per (pocket, canon_smi). Preference:
    # smallest row index (deterministic). Store the row's pose so the target
    # side is consistent.
    per_pocket_nodes: dict[int, dict[str, int]] = defaultdict(dict)
    canon_of: dict[int, str] = {}
    for i in rows:
        smi = str(smiles[i])
        cs = canonicalize(smi)
        if cs is None:
            continue
        canon_of[int(i)] = cs
        pocket_idx = int(row_seq_idx[i])
        if cs not in per_pocket_nodes[pocket_idx]:
            per_pocket_nodes[pocket_idx][cs] = int(i)

    # Report pocket coverage
    n_pockets = len(per_pocket_nodes)
    n_active = sum(1 for m in per_pocket_nodes.values() if len(m) >= 2)
    print(f"[Scheme A] pockets seen: {n_pockets}  with >=2 distinct ligs: "
           f"{n_active}", flush=True)

    # ---- Fingerprint cache (per unique row we'll use) ----
    fp_cache: dict[int, object] = {}
    for pocket_idx, m in per_pocket_nodes.items():
        if len(m) < 2:
            continue
        for canon, row in m.items():
            if row in fp_cache:
                continue
            fp = mol_fp(canon)
            if fp is not None:
                fp_cache[row] = fp

    # ---- Form pairs ----
    src_smi_list: List[str] = []
    tgt_smi_list: List[str] = []
    res_idx_list: List[int] = []
    pose_list: List[np.ndarray] = []
    tc_list: List[float] = []
    source_a_list: List[str] = []
    source_b_list: List[str] = []
    tc_bins = Counter()
    pockets_used = 0

    for pocket_idx, node_map in per_pocket_nodes.items():
        rows_here = [r for r in node_map.values() if r in fp_cache]
        if len(rows_here) < 2:
            continue
        # All directed pairs (both directions).
        candidates: List[tuple[int, int, float]] = []
        for i_row in rows_here:
            for j_row in rows_here:
                if i_row == j_row:
                    continue
                tc = DataStructs.TanimotoSimilarity(fp_cache[i_row], fp_cache[j_row])
                if not (args.tc_min <= tc <= args.tc_max):
                    continue
                candidates.append((i_row, j_row, float(tc)))
        if not candidates:
            continue
        # Cap per pocket to keep the dataset balanced.
        if len(candidates) > args.max_pairs_per_pocket:
            idx_sel = rng.choice(len(candidates),
                                 size=args.max_pairs_per_pocket, replace=False)
            candidates = [candidates[k] for k in idx_sel]
        for i_row, j_row, tc in candidates:
            src_smi = randomize_smi(canon_of[i_row])
            tgt_smi = canon_of[j_row]
            src_smi_list.append(src_smi)
            tgt_smi_list.append(tgt_smi)
            res_idx_list.append(pocket_idx)
            pose_list.append(poses[j_row])
            tc_list.append(tc)
            source_a_list.append(str(sources[i_row]))
            source_b_list.append(str(sources[j_row]))
            tc_bins[round(tc, 1)] += 1
        pockets_used += 1

    n_pairs = len(src_smi_list)
    print(f"[Scheme A] formed {n_pairs} directed pairs "
           f"(on {pockets_used} pockets)", flush=True)

    if n_pairs == 0:
        print("[Scheme A] WARNING: no pairs formed; writing empty NPZ.",
               flush=True)

    src_arr = np.array(src_smi_list, dtype=object)
    tgt_arr = np.array(tgt_smi_list, dtype=object)
    res_arr = np.array(res_idx_list, dtype=np.int32)
    pose_arr = (np.stack(pose_list).astype(np.float32)
                if pose_list else np.zeros((0, 3), dtype=np.float32))
    tc_arr = np.array(tc_list, dtype=np.float32)
    src_source_arr = np.array(source_a_list, dtype=object)
    tgt_source_arr = np.array(source_b_list, dtype=object)

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             src_smi=src_arr,
             tgt_smi=tgt_arr,
             residues_emb_idx=res_arr,
             residues_mask_idx=res_arr.copy(),
             pose=pose_arr,
             tc=tc_arr,
             source_a=src_source_arr,
             source_b=tgt_source_arr,
             cache_path=str(cache_path),
             scheme="A")
    print(f"[Scheme A] wrote {out_path}", flush=True)

    stats = {
        "scheme": "A",
        "n_pairs": int(n_pairs),
        "n_source_rows": int(len(rows)),
        "n_pockets_with_2plus_ligands": int(n_active),
        "n_pockets_producing_pairs": int(pockets_used),
        "tc_bins": {str(k): int(v) for k, v in sorted(tc_bins.items())},
        "filter": {
            "tc_min": args.tc_min,
            "tc_max": args.tc_max,
            "max_pairs_per_pocket": args.max_pairs_per_pocket,
        },
    }
    Path(args.out_stats).write_text(json.dumps(stats, indent=2))
    print(f"[Scheme A] wrote stats: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
