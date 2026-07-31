"""Scheme B-max — Boltz-expanded self-play pairs (ZAP70 block).

User's hypothesis (see supervisor prompt): Scheme B failed the causal test
because its productivity ranking (rank on z-scored d_warhead-Cys) was itself
derived from the invariant pose channel — the model could learn a shortcut
where "go closer to warhead" was baked into the target selection rather than
being disambiguated by the pose conditioning input. Removing the ranking and
widening the Tc floor should:

  * give the model more pairs (30k-80k vs Scheme B's 13k)
  * let pose_of_tgt be the *sole* disambiguator between (src, pose_A) and
    (src, pose_B) — no ranking bias
  * still keep chemical similarity as a natural bucket (all boltz_zap70 rows
    are ZAP70-anchored acrylamides so the population is already narrow)

Concrete differences vs `build_pairs_scheme_B.py`:

  1. NO productivity ranking. All ordered pairs (a, b) with a != b are emitted.
  2. Tc floor lowered from 0.4 to 0.2 (user spec: "very permissive").
  3. Bidirectional by construction — both (a→b, pose_b) AND (b→a, pose_a).
  4. Per-pocket cap raised 1500 → 3000 (user spec: "all the pairs").
  5. Same pocket + warhead-class family bucketing (michael_acceptor_family).
  6. Same min_pocket_rows = 4 (removes 60 singletons per prompt).

Emits `data/paper_pair_training/scheme_Bmax/pairs_scheme_Bmax.npz` with the
same schema as Scheme B so `train_m1a_pairs.py` and `eval_scheme_B_clamps.py`
can be reused unchanged. The `q_src`/`q_tgt` columns are kept for schema
compatibility but filled with sentinel -1 (there is no ranking).
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

# Path shims — works whether invoked locally (Users/shaharharel) or on V100.
LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

# Reuse the SMARTS priority table from build_triples_v2.py (same as Scheme B).
V2_DIR = PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))
from build_triples_v2 import WARHEAD_SMARTS_V2, find_warhead_atoms_smarts  # noqa: E402

# ---- Filter constants (differ from Scheme B where noted) ----
TC_MIN = 0.2       # Bmax: lowered from Scheme B's 0.4
TC_MAX = 1.0 - 1e-6
BOLTZ_TAG = "boltz_zap70"
MAX_PAIRS_PER_POCKET = 3000  # Bmax: raised from Scheme B's 1500
MIN_POCKET_ROWS = 4          # same as Scheme B

FP_RADIUS = 2
FP_NBITS = 2048

MICHAEL_FAMILY_LABELS = {"acrylamide", "michael_acceptor"}
MICHAEL_FAMILY_TAG = "michael_acceptor_family"


def mol_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS)


def infer_warhead_class(smi: str) -> str | None:
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
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=False, doRandom=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--out_npz", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/scheme_Bmax/pairs_scheme_Bmax.npz"))
    ap.add_argument("--out_stats", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/scheme_Bmax/pairs_scheme_Bmax_stats.json"))
    ap.add_argument("--tc_min", type=float, default=TC_MIN)
    ap.add_argument("--tc_max", type=float, default=TC_MAX)
    ap.add_argument("--max_pairs_per_pocket", type=int,
                    default=MAX_PAIRS_PER_POCKET,
                    help="Cap unordered pairs per pocket_idx BEFORE the "
                         "bidirectional expansion (so total emitted rows per "
                         "pocket is up to 2*max_pairs_per_pocket).")
    ap.add_argument("--min_pocket_rows", type=int, default=MIN_POCKET_ROWS)
    ap.add_argument("--warhead_match", choices=["acrylamide",
                                                 "michael_acceptor_family"],
                    default="michael_acceptor_family")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    cache_path = Path(args.cache)
    print(f"[Scheme B-max] Loading cache: {cache_path}", flush=True)
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
    print(f"[Scheme B-max] boltz_zap70 rows: {len(rows)}", flush=True)

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
    print(f"[Scheme B-max] rows with a SMARTS warhead class: "
           f"{len(row_class)} (skipped {n_no_class})  "
           f"match_mode={args.warhead_match}", flush=True)

    # ---- Enforce --min_pocket_rows ----
    pocket_row_counts: Counter = Counter()
    for i in row_class:
        pocket_row_counts[int(row_seq_idx[i])] += 1
    kept_pockets = {p for p, n in pocket_row_counts.items()
                    if n >= args.min_pocket_rows}
    dropped_pockets = len(pocket_row_counts) - len(kept_pockets)
    print(f"[Scheme B-max] pockets after --min_pocket_rows={args.min_pocket_rows}: "
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
    print(f"[Scheme B-max] total buckets: {len(buckets)}  "
           f"buckets with >=2 rows: {sum(1 for v in bucket_sizes if v>=2)}",
           flush=True)

    # ---- Fingerprints & canonical SMILES ----
    fp_cache: dict[int, object] = {}
    canon_cache: dict[int, str] = {}
    for i in row_class:
        smi = str(smiles[i])
        fp = mol_fp(smi)
        if fp is None:
            continue
        fp_cache[i] = fp
        canon_cache[i] = canonicalize(smi) or smi

    # ---- Form ALL bidirectional pairs (no ranking) ----
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
    per_pocket_pair_counts: Counter = Counter()   # unordered pairs kept
    per_pocket_capped: set[int] = set()

    pocket_to_buckets: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for (pocket_idx, wc), members in buckets.items():
        pocket_to_buckets[pocket_idx].append((wc, members))
    rng_pockets = np.random.default_rng(args.seed)

    n_tc_reject = 0
    for pocket_idx, wc_members in pocket_to_buckets.items():
        for wc, members in wc_members:
            per_pocket_bucket_sizes[pocket_idx].append(len(members))
        for wc, members in wc_members:
            if len(members) < 2:
                continue
            members = [i for i in members if i in fp_cache]
            if len(members) < 2:
                continue
            # Enumerate all UNORDERED (i<j) pairs. We shuffle then apply the
            # per-pocket cap so no small subset of ligands monopolises it.
            cand = []
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    cand.append((members[a_idx], members[b_idx]))
            rng_pockets.shuffle(cand)
            for i_row, j_row in cand:
                if per_pocket_pair_counts[pocket_idx] >= args.max_pairs_per_pocket:
                    per_pocket_capped.add(pocket_idx)
                    break
                tc = DataStructs.TanimotoSimilarity(fp_cache[i_row], fp_cache[j_row])
                if not (args.tc_min <= tc <= args.tc_max):
                    n_tc_reject += 1
                    continue
                tc_bins[round(tc, 1)] += 1
                per_pocket_pair_counts[pocket_idx] += 1
                # Bidirectional emission — each unordered pair yields TWO rows.
                # Row 1: i -> j (pose of j)
                src_smi_list.append(randomize_smi(canon_cache[i_row]))
                tgt_smi_list.append(canon_cache[j_row])
                res_idx_list.append(pocket_idx)
                pose_list.append(poses[j_row])
                tc_list.append(float(tc))
                class_list.append(wc)
                qsrc_list.append(-1.0)
                qtgt_list.append(-1.0)
                # Row 2: j -> i (pose of i)
                src_smi_list.append(randomize_smi(canon_cache[j_row]))
                tgt_smi_list.append(canon_cache[i_row])
                res_idx_list.append(pocket_idx)
                pose_list.append(poses[i_row])
                tc_list.append(float(tc))
                class_list.append(wc)
                qsrc_list.append(-1.0)
                qtgt_list.append(-1.0)
            if per_pocket_pair_counts[pocket_idx] >= args.max_pairs_per_pocket:
                break

    n_pairs = len(src_smi_list)  # already 2 * unordered_kept
    n_unordered = int(sum(per_pocket_pair_counts.values()))
    print(f"[Scheme B-max] formed {n_pairs} directed pairs "
           f"({n_unordered} unordered, x2 bidirectional). "
           f"Rejected by Tc filter: {n_tc_reject}", flush=True)

    # ---- Guardrails per supervisor prompt ----
    if n_pairs < 5000:
        raise SystemExit(
            f"[Scheme B-max] FAIL-FAST: only {n_pairs} pairs after filters. "
            f"Expected 30k-80k. Something is wrong (Tc floor too high? "
            f"pocket filter too aggressive?)."
        )
    if n_pairs > 200000:
        print(f"[Scheme B-max] WARN: {n_pairs} pairs exceeds 200k soft cap. "
               f"Downstream trainer may need per-epoch subsampling to stay "
               f"in the 5h wall budget.", flush=True)

    src_arr = np.array(src_smi_list, dtype=object)
    tgt_arr = np.array(tgt_smi_list, dtype=object)
    res_arr = np.array(res_idx_list, dtype=np.int32)
    pose_arr = np.stack(pose_list).astype(np.float32)
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
             residues_mask_idx=res_arr.copy(),
             pose=pose_arr,
             tc=tc_arr,
             warhead_class=class_arr,
             q_src=qsrc_arr,
             q_tgt=qtgt_arr,
             cache_path=str(cache_path),
             scheme="B-max")
    print(f"[Scheme B-max] wrote {out_path}", flush=True)

    per_pocket_summary = {
        str(pk): {"n_buckets": len(sizes), "sizes": sorted(sizes, reverse=True)}
        for pk, sizes in per_pocket_bucket_sizes.items()
    }
    raw_class_counts = Counter(row_raw_class.values())
    top10 = sorted(per_pocket_pair_counts.items(), key=lambda kv: -kv[1])[:10]

    stats = {
        "scheme": "B-max",
        "n_pairs_directed": int(n_pairs),
        "n_pairs_unordered": int(n_unordered),
        "n_source_rows": int(len(rows)),
        "n_rows_with_class": int(len(row_class)),
        "n_buckets": int(len(buckets)),
        "n_active_pockets": int(len(per_pocket_bucket_sizes)),
        "n_kept_pockets": int(len(kept_pockets)),
        "n_dropped_pockets_min_rows": int(dropped_pockets),
        "n_pockets_capped": int(len(per_pocket_capped)),
        "n_tc_rejected": int(n_tc_reject),
        "tc_bins": {str(k): int(v) for k, v in sorted(tc_bins.items())},
        "warhead_class_counts": {
            wc: int(sum(1 for x in class_list if x == wc))
            for wc in sorted(set(class_list))
        },
        "raw_warhead_class_counts_source_rows": {
            wc: int(n) for wc, n in raw_class_counts.most_common()
        },
        "per_pocket_bucket_sizes": per_pocket_summary,
        "per_pocket_unordered_pair_counts_top10": [
            {"pocket_idx": int(pk), "n_unordered_pairs": int(n)}
            for pk, n in top10
        ],
        "per_pocket_pair_counts": {
            str(pk): int(n)
            for pk, n in per_pocket_pair_counts.most_common()
        },
        "filter": {
            "tc_min": args.tc_min,
            "tc_max": args.tc_max,
            "max_pairs_per_pocket": args.max_pairs_per_pocket,
            "min_pocket_rows": args.min_pocket_rows,
            "warhead_match": args.warhead_match,
            "ranking": "NONE (Scheme B-max: no productivity rank)",
            "directionality": "bidirectional (each unordered pair emits 2 rows)",
        },
        "user_hypothesis": (
            "Scheme B's productivity rank (rank on z-scored pose[0]) leaked "
            "the pose channel into target selection. Scheme B-max drops the "
            "rank so pose_of_tgt is the sole disambiguator between (src, "
            "pose_A) and (src, pose_B), and widens Tc to 0.2 for a bigger, "
            "more diverse training set."
        ),
    }
    Path(args.out_stats).write_text(json.dumps(stats, indent=2))
    print(f"[Scheme B-max] wrote stats: {args.out_stats}", flush=True)
    if top10:
        print("[Scheme B-max] top 10 pockets by unordered pair count:", flush=True)
        for pk, n in top10:
            print(f"    pocket_idx={pk}  n_unordered_pairs={n}", flush=True)


if __name__ == "__main__":
    main()
