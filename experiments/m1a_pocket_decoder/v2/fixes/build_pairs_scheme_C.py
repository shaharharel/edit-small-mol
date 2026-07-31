"""Scheme C — Same-Struct Co-Crystal Pairs.

Offline pair builder. For every PDB struct_id that has EXACTLY 2 rows
(one per source: typically covindb_v2 + covbinder_inpdb), form BOTH
directed pairs (mol_a -> mol_b, mol_b -> mol_a). Both poses are
experimentally verified (crystal-derived) rather than Boltz-cofolded.

The point of Scheme C is to test whether high pose-quality on a SMALL
dataset (~1,400 pairs) teaches pose-control better than low-quality
cofold poses on the ~13k Scheme B / ~14k Scheme A datasets.

Emits `data/paper_pair_training/scheme_C/pairs_scheme_C.npz` with
fields (compatible with train_m1a_pairs.py):
  src_smi (object, (N,))               - non-canonical (randomized) src SMILES
  tgt_smi (object, (N,))               - canonical target SMILES
  residues_emb_idx (int32, (N,))       - ESM cache pocket index (tgt's pocket)
  residues_mask_idx (int32, (N,))      - same index
  pose (float32, (N, 3))               - z-scored TARGET pose (from cache)
  tc (float32, (N,))                   - src/tgt Tanimoto similarity (audit)
  source_a (object, (N,))              - source-tag of src row
  source_b (object, (N,))              - source-tag of tgt row
  struct_id (object, (N,))             - PDB struct id (audit)
  same_pocket (bool, (N,))             - whether src+tgt rows share pocket_idx

Plus `pairs_scheme_C_stats.json`.

Filters:
  * struct_ids with EXACTLY 2 rows (matches the 705 co-crystal pair set)
  * Both mols must parse with RDKit AND have MW >= 200 (drop cofactors /
    ions / water / crystallization additives). MW is on the canonical
    SMILES (no salt stripping — we assume the cache is already clean).

Directionality:
  For each 2-row struct_id with rows (i, j) we emit BOTH:
    (canonical(smi_i) -> canonical(smi_j))  with pose_j and pocket_j
    (canonical(smi_j) -> canonical(smi_i))  with pose_i and pocket_i
  Src is randomized via non-canonical walk so training sees varied
  input tokens for the same mol identity.

If a struct_id's two rows sit in different pocket_idx buckets (the ~89
cross-pocket case observed in the cache), we STILL emit the pair — the
model receives the target row's pocket_idx so the ESM tokens correspond
to where pose_tgt is measured. This is the honest interpretation
because "struct_id" identifies the PDB structure regardless of how the
downstream ESM cache split it.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

MW_MIN = 200.0
FP_RADIUS = 2
FP_NBITS = 2048


def mol_from_smi(smi: str):
    m = Chem.MolFromSmiles(smi)
    return m


def canonicalize(smi: str) -> str | None:
    m = mol_from_smi(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def randomize_smi(smi: str) -> str:
    m = mol_from_smi(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=False, doRandom=True)


def mol_mw(smi: str) -> float | None:
    m = mol_from_smi(smi)
    if m is None:
        return None
    return float(Descriptors.MolWt(m))


def mol_fp(smi: str):
    m = mol_from_smi(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--out_npz", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/scheme_C/pairs_scheme_C.npz"))
    ap.add_argument("--out_stats", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/scheme_C/pairs_scheme_C_stats.json"))
    ap.add_argument("--mw_min", type=float, default=MW_MIN,
                    help="Minimum molecular weight (Da). Drops cofactors, "
                         "ions, water, small crystallization additives.")
    ap.add_argument("--min_pairs", type=int, default=300,
                    help="Fail-fast if fewer than this many pairs after "
                         "filtering. Something is upstream-wrong if we can't "
                         "clear ~300 pairs from a 705 co-crystal set.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    cache_path = Path(args.cache)
    print(f"[Scheme C] Loading cache: {cache_path}", flush=True)
    d = np.load(cache_path, allow_pickle=True)
    sources = d["sources"]
    smiles = d["smiles"]
    struct_ids = d["struct_ids"]
    row_seq_idx = d["row_seq_idx"].astype(np.int32)
    poses = d["poses"].astype(np.float32)
    n_rows_cache = len(smiles)
    print(f"[Scheme C] cache rows: {n_rows_cache}", flush=True)

    # ---- Group rows by struct_id ----
    by_struct: dict[str, list[int]] = defaultdict(list)
    for i in range(n_rows_cache):
        sid = str(struct_ids[i])
        by_struct[sid].append(i)

    total_structs = len(by_struct)
    struct_size_hist = Counter(len(v) for v in by_struct.values())

    # ---- Keep EXACTLY-2 struct_ids ----
    struct_ids_2 = [sid for sid, rows in by_struct.items() if len(rows) == 2]
    print(f"[Scheme C] struct_ids with exactly 2 rows: {len(struct_ids_2)} "
          f"(of {total_structs} total)", flush=True)

    # ---- MW filter (both mols must have MW >= mw_min) ----
    kept_structs: list[tuple[str, int, int]] = []
    dropped_reasons = Counter()
    canon_of: dict[int, str] = {}
    mw_of: dict[int, float] = {}
    for sid in struct_ids_2:
        i, j = by_struct[sid]
        smi_i, smi_j = str(smiles[i]), str(smiles[j])
        ci = canonicalize(smi_i)
        cj = canonicalize(smi_j)
        if ci is None or cj is None:
            dropped_reasons["parse_fail"] += 1
            continue
        mi = mol_mw(ci)
        mj = mol_mw(cj)
        if mi is None or mj is None:
            dropped_reasons["mw_fail"] += 1
            continue
        if mi < args.mw_min or mj < args.mw_min:
            dropped_reasons["mw_below_thresh"] += 1
            continue
        # ADDITIONAL: drop identity edits (same canonical smi on both sides)
        if ci == cj:
            dropped_reasons["identity_edit"] += 1
            continue
        canon_of[i] = ci
        canon_of[j] = cj
        mw_of[i] = mi
        mw_of[j] = mj
        kept_structs.append((sid, i, j))
    print(f"[Scheme C] kept struct_ids after MW/parse filter: "
          f"{len(kept_structs)}   dropped: {dict(dropped_reasons)}",
          flush=True)

    # ---- Emit bidirectional pairs ----
    src_smi_list: list[str] = []
    tgt_smi_list: list[str] = []
    res_idx_list: list[int] = []
    pose_list: list[np.ndarray] = []
    tc_list: list[float] = []
    source_a_list: list[str] = []
    source_b_list: list[str] = []
    struct_id_list: list[str] = []
    same_pocket_list: list[bool] = []
    tc_bins = Counter()
    n_same_pocket = 0
    n_cross_pocket = 0

    for sid, i, j in kept_structs:
        ci, cj = canon_of[i], canon_of[j]
        fp_i = mol_fp(ci)
        fp_j = mol_fp(cj)
        if fp_i is None or fp_j is None:
            continue
        tc = float(DataStructs.TanimotoSimilarity(fp_i, fp_j))
        # Symmetric Tc — same value for both directions.
        pocket_i = int(row_seq_idx[i])
        pocket_j = int(row_seq_idx[j])
        same_pocket = pocket_i == pocket_j
        if same_pocket:
            n_same_pocket += 1
        else:
            n_cross_pocket += 1

        # Direction 1: i -> j (pose_j, pocket_j)
        src_smi_list.append(randomize_smi(ci))
        tgt_smi_list.append(cj)
        res_idx_list.append(pocket_j)
        pose_list.append(poses[j])
        tc_list.append(tc)
        source_a_list.append(str(sources[i]))
        source_b_list.append(str(sources[j]))
        struct_id_list.append(sid)
        same_pocket_list.append(same_pocket)
        tc_bins[round(tc, 1)] += 1

        # Direction 2: j -> i (pose_i, pocket_i)
        src_smi_list.append(randomize_smi(cj))
        tgt_smi_list.append(ci)
        res_idx_list.append(pocket_i)
        pose_list.append(poses[i])
        tc_list.append(tc)
        source_a_list.append(str(sources[j]))
        source_b_list.append(str(sources[i]))
        struct_id_list.append(sid)
        same_pocket_list.append(same_pocket)
        tc_bins[round(tc, 1)] += 1

    n_pairs = len(src_smi_list)
    print(f"[Scheme C] emitted {n_pairs} directed pairs "
          f"({n_same_pocket} same-pocket + {n_cross_pocket} cross-pocket "
          f"struct_ids, each x2 directions)", flush=True)

    # ---- Fail-fast ----
    if n_pairs < args.min_pairs:
        stats = {
            "scheme": "C",
            "status": "FAILED_MIN_PAIRS",
            "n_pairs": int(n_pairs),
            "min_pairs_required": int(args.min_pairs),
            "n_struct_ids_exact2": int(len(struct_ids_2)),
            "n_struct_ids_kept": int(len(kept_structs)),
            "dropped_reasons": dict(dropped_reasons),
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(stats, indent=2))
        raise SystemExit(f"[Scheme C] FAIL-FAST: only {n_pairs} pairs "
                         f"(need >= {args.min_pairs}). Stats written.")

    # ---- Save NPZ ----
    src_arr = np.array(src_smi_list, dtype=object)
    tgt_arr = np.array(tgt_smi_list, dtype=object)
    res_arr = np.array(res_idx_list, dtype=np.int32)
    pose_arr = np.stack(pose_list).astype(np.float32)
    tc_arr = np.array(tc_list, dtype=np.float32)
    src_source_arr = np.array(source_a_list, dtype=object)
    tgt_source_arr = np.array(source_b_list, dtype=object)
    struct_id_arr = np.array(struct_id_list, dtype=object)
    same_pocket_arr = np.array(same_pocket_list, dtype=bool)

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
             struct_id=struct_id_arr,
             same_pocket=same_pocket_arr,
             cache_path=str(cache_path),
             scheme="C")
    print(f"[Scheme C] wrote {out_path}", flush=True)

    # ---- Source-pair breakdown for audit ----
    source_pair_counts = Counter()
    for a, b in zip(source_a_list, source_b_list):
        source_pair_counts[f"{a} -> {b}"] += 1

    stats = {
        "scheme": "C",
        "status": "OK",
        "n_pairs": int(n_pairs),
        "n_struct_ids_total": int(total_structs),
        "n_struct_ids_exact2": int(len(struct_ids_2)),
        "n_struct_ids_kept_after_mw_filter": int(len(kept_structs)),
        "n_same_pocket_structs": int(n_same_pocket),
        "n_cross_pocket_structs": int(n_cross_pocket),
        "dropped_reasons": dict(dropped_reasons),
        "struct_size_histogram": {str(k): int(v)
                                   for k, v in sorted(struct_size_hist.items())},
        "tc_bins": {str(k): int(v) for k, v in sorted(tc_bins.items())},
        "source_pair_counts": dict(source_pair_counts),
        "filter": {
            "mw_min": args.mw_min,
            "min_pairs": args.min_pairs,
        },
    }
    Path(args.out_stats).write_text(json.dumps(stats, indent=2))
    print(f"[Scheme C] wrote stats: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
