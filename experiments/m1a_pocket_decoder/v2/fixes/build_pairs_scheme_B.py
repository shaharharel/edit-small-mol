"""Scheme B — Boltz Self-Play Pose Steering (ZAP70 block).

Offline pair builder for the M1a v2-cond decoder. Restricts to the
`boltz_zap70` cohort (1,546 rows on 124 pockets), buckets ligands per pocket
by warhead class, ranks them by pose "productivity" q, and forms directed
pairs (mol_low_q, mol_high_q) with Tc >= 0.4.

Emits `data/m1a_triples_v2/pairs_scheme_B.npz` with fields:
  src_smi (str, one per pair)       - non-canonical source (low-q neighbor SMILES)
  tgt_smi (str, one per pair)       - canonical target (high-q neighbor SMILES)
  residues_emb_idx (int32)          - index into the ESM pocket cache (U)
  residues_mask_idx (int32)         - same index (kept as a separate field so
                                       downstream code doesn't need to know they
                                       share the axis)
  pose (float32, (N, 3))            - z-scored tgt (high-q) pose from the cache
  tc (float32, (N,))                - Tanimoto similarity of the pair
  warhead_class (object, (N,))      - shared warhead class
  q_src (float32, (N,))             - productivity score of src (lower = worse)
  q_tgt (float32, (N,))             - productivity score of tgt (lower = better after ranking)

Plus `pairs_scheme_B_stats.json`.

PRODUCTIVITY q — LIMITATION (documented per spec).
We do not have per-row pIC50 or Vina for the boltz_zap70 cohort, so we
fall back on the invariant pose channel exactly as recommended: a rank on
`pose[0]` (d_warhead-Cys after z-scoring, smaller = better) with tie-break
on |pose[1]| (θ_BD deviation after z-scoring, smaller = better).

NOTE ON PAIRING: for each pocket*warhead_class bucket we form ALL directed
low->high pairs whose Tc is within [TC_MIN, TC_MAX] and whose q ranks
differ by at least MIN_Q_GAP (rank slots — not a raw q difference). This
gives ~O(N^2) pairs per bucket but the buckets are small (median size ~5).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Path shims — works whether invoked locally (Users/shaharharel) or on a100.
LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

# We reuse the SMARTS priority table from build_triples_v2.py so that the
# warhead class we bucket on is derived from the SMILES exactly the same way
# as the original triples curator.
V2_DIR = PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))
from build_triples_v2 import WARHEAD_SMARTS_V2, find_warhead_atoms_smarts  # noqa: E402

# ---- Filter constants ----
TC_MIN = 0.4       # per spec
TC_MAX = 1.0 - 1e-6  # exclude identical (tautomer-collapsed) mols but keep Tc<1 exact hits
MIN_Q_GAP = 1      # at least 1 rank slot apart
BOLTZ_TAG = "boltz_zap70"
MAX_PAIRS_PER_POCKET = 1500  # raised from 200: top pocket has 449 rows → up to ~100k pairs
MIN_POCKET_ROWS = 4          # drop pockets with <N rows to prevent per-ligand memorization

FP_RADIUS = 2
FP_NBITS = 2048

# Warhead-class match mode.  "acrylamide" = strict; "michael_acceptor_family"
# maps both 'acrylamide' and 'michael_acceptor' SMARTS-derived labels to a
# common family label so pairs can cross the acrylamide↔generic α,β-unsat
# carbonyl boundary within a pocket.
MICHAEL_FAMILY_LABELS = {"acrylamide", "michael_acceptor"}
MICHAEL_FAMILY_TAG = "michael_acceptor_family"


def mol_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS)


def infer_warhead_class(smi: str) -> str | None:
    """Priority-ordered SMARTS match on the SMILES (same order as
    build_triples_v2.py). Returns None if no SMARTS matches — the row cannot
    be bucketed and is dropped.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    info = find_warhead_atoms_smarts(m, None)
    if info is None:
        return None
    return info["name"]


def canonicalize(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def randomize_smi(smi: str) -> str:
    """Non-canonical random walk (same helper as train_m1a_v2.py). Kept
    lightweight so downstream training can re-randomize per-epoch if desired,
    but we also freeze one non-canonical form here so the offline NPZ is
    reproducible.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=False, doRandom=True)


def rank_by_productivity(poses_z: np.ndarray) -> np.ndarray:
    """poses_z: (n, 3) z-scored (d_b_nuc, angle, dihedral).
    Ranks ascending on pose[0] (d_warhead-Cys, smaller = better),
    tie-break on |pose[1]|.

    Returns int array of ranks 0..n-1 where 0 = best (most productive).
    """
    # Primary key: z-scored d
    primary = poses_z[:, 0]
    secondary = np.abs(poses_z[:, 1])
    order = np.lexsort((secondary, primary))
    ranks = np.empty(len(poses_z), dtype=np.int32)
    for r, idx in enumerate(order):
        ranks[idx] = r
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--out_npz", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/pairs_scheme_B.npz"))
    ap.add_argument("--out_stats", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/pairs_scheme_B_stats.json"))
    ap.add_argument("--tc_min", type=float, default=TC_MIN)
    ap.add_argument("--tc_max", type=float, default=TC_MAX)
    ap.add_argument("--min_q_gap", type=int, default=MIN_Q_GAP)
    ap.add_argument("--max_pairs_per_pocket", type=int,
                    default=MAX_PAIRS_PER_POCKET,
                    help="Cap directed pairs emitted per pocket_idx (across "
                         "all warhead-class buckets in that pocket). Raised "
                         "from 200 → 1500 to accommodate the top ZAP70 "
                         "pocket (449 rows).")
    ap.add_argument("--min_pocket_rows", type=int, default=MIN_POCKET_ROWS,
                    help="Drop any pocket whose total row count (across all "
                         "warhead classes, after class inference) is below "
                         "this threshold. Prevents singletons + tiny buckets "
                         "from encouraging per-ligand pocket-memorization.")
    ap.add_argument("--warhead_match", choices=["acrylamide",
                                                 "michael_acceptor_family"],
                    default="michael_acceptor_family",
                    help="How strictly two ligands in the same pocket must "
                         "match on warhead class. 'acrylamide' = strict "
                         "SMARTS name equality; 'michael_acceptor_family' "
                         "collapses acrylamide + generic α,β-unsat carbonyl "
                         "under a common family label so pairs can cross "
                         "that boundary. Boltz self-play against ZAP70 is "
                         "~100%% acrylamide so this is a minor loosening.")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only process the first --limit source rows "
                         "(after the boltz_zap70 filter). For unit tests.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    cache_path = Path(args.cache)
    print(f"[Scheme B] Loading cache: {cache_path}", flush=True)
    d = np.load(cache_path, allow_pickle=True)
    sources = d["sources"]
    smiles = d["smiles"]
    row_seq_idx = d["row_seq_idx"].astype(np.int32)
    poses = d["poses"].astype(np.float32)   # z-scored

    # ---- Restrict to boltz_zap70 ----
    mask = sources == BOLTZ_TAG
    rows = np.where(mask)[0]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[Scheme B] boltz_zap70 rows: {len(rows)}", flush=True)

    # ---- Infer warhead class per row (SMARTS priority) ----
    # `row_raw_class` keeps the SMARTS-derived label; `row_class` is the
    # (possibly family-collapsed) label used for bucketing.
    def _to_bucket_label(raw: str) -> str:
        if args.warhead_match == "michael_acceptor_family" and \
                raw in MICHAEL_FAMILY_LABELS:
            return MICHAEL_FAMILY_TAG
        return raw

    row_raw_class: dict[int, str] = {}
    row_class: dict[int, str] = {}
    n_no_class = 0
    for i in rows:
        smi = str(smiles[i])
        wc = infer_warhead_class(smi)
        if wc is None:
            n_no_class += 1
            continue
        row_raw_class[int(i)] = wc
        row_class[int(i)] = _to_bucket_label(wc)
    print(f"[Scheme B] rows with a SMARTS warhead class: "
           f"{len(row_class)} (skipped {n_no_class})  "
           f"match_mode={args.warhead_match}", flush=True)

    # ---- Enforce --min_pocket_rows (drop pockets with < N classified rows) ----
    pocket_row_counts: Counter = Counter()
    for i in row_class:
        pocket_row_counts[int(row_seq_idx[i])] += 1
    kept_pockets = {p for p, n in pocket_row_counts.items()
                    if n >= args.min_pocket_rows}
    dropped_pockets = len(pocket_row_counts) - len(kept_pockets)
    print(f"[Scheme B] pockets after --min_pocket_rows={args.min_pocket_rows}: "
           f"{len(kept_pockets)} kept, {dropped_pockets} dropped "
           f"(total classified rows kept: "
           f"{sum(pocket_row_counts[p] for p in kept_pockets)})",
           flush=True)

    # ---- Bucket by (pocket, warhead_class) ----
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, wc in row_class.items():
        pocket_idx = int(row_seq_idx[i])
        if pocket_idx not in kept_pockets:
            continue
        buckets[(pocket_idx, wc)].append(i)

    bucket_sizes = [len(v) for v in buckets.values()]
    print(f"[Scheme B] total buckets: {len(buckets)}  "
           f"buckets with >=2 rows: {sum(1 for v in bucket_sizes if v>=2)}",
           flush=True)

    # ---- Pre-compute fingerprints & canonical SMILES ----
    fp_cache: dict[int, object] = {}
    canon_cache: dict[int, str] = {}
    for i in row_class:
        smi = str(smiles[i])
        fp = mol_fp(smi)
        if fp is None:
            continue
        fp_cache[i] = fp
        canon_cache[i] = canonicalize(smi) or smi

    # ---- Form directed low->high pairs ----
    src_smi_list: List[str] = []
    tgt_smi_list: List[str] = []
    res_idx_list: List[int] = []
    pose_list: List[np.ndarray] = []
    tc_list: List[float] = []
    class_list: List[str] = []
    qsrc_list: List[float] = []
    qtgt_list: List[float] = []
    tc_bins = Counter()
    per_pocket_bucket_sizes: dict[int, list[int]] = defaultdict(list)
    per_pocket_pair_counts: Counter = Counter()
    per_pocket_capped: set[int] = set()

    # Group buckets by pocket so we can enforce per-pocket cap across all
    # warhead-class buckets in that pocket, and can shuffle bucket order
    # deterministically so no one class monopolises the cap.
    pocket_to_buckets: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for (pocket_idx, wc), members in buckets.items():
        pocket_to_buckets[pocket_idx].append((wc, members))
    rng_pockets = np.random.default_rng(args.seed)

    for pocket_idx, wc_members in pocket_to_buckets.items():
        for wc, members in wc_members:
            per_pocket_bucket_sizes[pocket_idx].append(len(members))
        if per_pocket_pair_counts[pocket_idx] >= args.max_pairs_per_pocket:
            per_pocket_capped.add(pocket_idx)
            continue
        for wc, members in wc_members:
            if len(members) < 2:
                continue
            members = [i for i in members if i in fp_cache]
            if len(members) < 2:
                continue
            member_poses = poses[members]  # (m, 3)
            ranks = rank_by_productivity(member_poses)  # 0 = best
            # deterministic shuffle of candidate (a, b) pairs so the cap
            # doesn't systematically favour early row indices.
            cand = []
            for a_pos, i_row in enumerate(members):
                for b_pos, j_row in enumerate(members):
                    if i_row == j_row:
                        continue
                    if not (ranks[a_pos] > ranks[b_pos] + args.min_q_gap - 1):
                        continue
                    cand.append((a_pos, b_pos, i_row, j_row))
            rng_pockets.shuffle(cand)
            for a_pos, b_pos, i_row, j_row in cand:
                if per_pocket_pair_counts[pocket_idx] >= args.max_pairs_per_pocket:
                    per_pocket_capped.add(pocket_idx)
                    break
                tc = DataStructs.TanimotoSimilarity(fp_cache[i_row], fp_cache[j_row])
                if not (args.tc_min <= tc <= args.tc_max):
                    continue
                # bucketize tc in 0.1 bins for the stats file
                tc_bins[round(tc, 1)] += 1
                src_smi = randomize_smi(canon_cache[i_row])
                tgt_smi = canon_cache[j_row]
                src_smi_list.append(src_smi)
                tgt_smi_list.append(tgt_smi)
                res_idx_list.append(pocket_idx)
                pose_list.append(poses[j_row])
                tc_list.append(float(tc))
                class_list.append(wc)
                qsrc_list.append(float(ranks[a_pos]))
                qtgt_list.append(float(ranks[b_pos]))
                per_pocket_pair_counts[pocket_idx] += 1
            if per_pocket_pair_counts[pocket_idx] >= args.max_pairs_per_pocket:
                break

    n_pairs = len(src_smi_list)
    print(f"[Scheme B] formed {n_pairs} directed pairs", flush=True)

    if n_pairs == 0:
        print("[Scheme B] WARNING: no pairs formed; writing empty NPZ.",
               flush=True)

    src_arr = np.array(src_smi_list, dtype=object)
    tgt_arr = np.array(tgt_smi_list, dtype=object)
    res_arr = np.array(res_idx_list, dtype=np.int32)
    pose_arr = (np.stack(pose_list).astype(np.float32)
                if pose_list else np.zeros((0, 3), dtype=np.float32))
    tc_arr = np.array(tc_list, dtype=np.float32)
    class_arr = np.array(class_list, dtype=object)
    qsrc_arr = np.array(qsrc_list, dtype=np.float32)
    qtgt_arr = np.array(qtgt_list, dtype=np.float32)

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             src_smi=src_arr,
             tgt_smi=tgt_arr,
             residues_emb_idx=res_arr,
             residues_mask_idx=res_arr.copy(),  # same axis
             pose=pose_arr,
             tc=tc_arr,
             warhead_class=class_arr,
             q_src=qsrc_arr,
             q_tgt=qtgt_arr,
             cache_path=str(cache_path),
             scheme="B")
    print(f"[Scheme B] wrote {out_path}", flush=True)

    per_pocket_summary = {
        str(pk): {"n_buckets": len(sizes), "sizes": sorted(sizes, reverse=True)}
        for pk, sizes in per_pocket_bucket_sizes.items()
    }

    # Distribution of raw (pre-family-collapse) SMARTS labels among source rows
    raw_class_counts = Counter(row_raw_class.values())

    stats = {
        "scheme": "B",
        "n_pairs": int(n_pairs),
        "n_source_rows": int(len(rows)),
        "n_rows_with_class": int(len(row_class)),
        "n_buckets": int(len(buckets)),
        "n_active_pockets": int(len(per_pocket_bucket_sizes)),
        "n_kept_pockets": int(len(kept_pockets)),
        "n_dropped_pockets_min_rows": int(dropped_pockets),
        "n_pockets_capped": int(len(per_pocket_capped)),
        "tc_bins": {str(k): int(v) for k, v in sorted(tc_bins.items())},
        "warhead_class_counts": {
            wc: int(sum(1 for x in class_list if x == wc))
            for wc in sorted(set(class_list))
        },
        "raw_warhead_class_counts_source_rows": {
            wc: int(n) for wc, n in raw_class_counts.most_common()
        },
        "per_pocket_bucket_sizes": per_pocket_summary,
        "per_pocket_pair_counts": {
            str(pk): int(n)
            for pk, n in per_pocket_pair_counts.most_common()
        },
        "filter": {
            "tc_min": args.tc_min,
            "tc_max": args.tc_max,
            "min_q_gap": args.min_q_gap,
            "max_pairs_per_pocket": args.max_pairs_per_pocket,
            "min_pocket_rows": args.min_pocket_rows,
            "warhead_match": args.warhead_match,
        },
        "productivity_ranking": "primary=z(d_b_nuc) asc; tiebreak=|z(bd_angle)| asc",
        "limitation": (
            "No per-row pIC50 or Vina available for the boltz_zap70 cohort; "
            "productivity is derived from the invariant pose only. This ranks "
            "geometric hookup quality of the docked pose but does not "
            "distinguish weakly vs strongly binding scaffolds when the "
            "cofolder placed both correctly. Consider replacing q with an "
            "external quality signal (Vina, xTB k_inact) if available."
        ),
    }
    Path(args.out_stats).write_text(json.dumps(stats, indent=2))
    print(f"[Scheme B] wrote stats: {args.out_stats}", flush=True)
    if per_pocket_summary:
        top = sorted(per_pocket_bucket_sizes.items(),
                     key=lambda kv: -sum(kv[1]))[:5]
        print("[Scheme B] top 5 pockets by total bucket size:", flush=True)
        for pk, sizes in top:
            print(f"    pocket_idx={pk}  buckets={sizes}", flush=True)


if __name__ == "__main__":
    main()
