"""Build v2-curriculum pairs — the geometry-forcing training set.

MOTIVATION
----------
Prior v2 training used self-reconstruction (src==tgt), so the [POSE] token
carried no information (target already fully specified by anchor). The
retrained diagnostic showed v2 IGNORES the pose channel.

FIX
---
Enumerate within-pocket pairs (A, B) where B has BETTER covalent geometry
than A by our validation criterion:
    |theta_B - 105|  < |theta_A - 105|  by >= dtheta_gap  (default 10 deg)
    |d_B - 3.5|      < |d_A - 3.5|      by >= dd_gap      (default 0.5 A)
    Tc(A, B)         >= tc_min                             (default 0.35)
    A != B canonical SMILES

Convention: theta target is 105 deg (the ideal S-C-C=O attack angle for
acrylamide covalent inhibitors; the pose value in the ESM cache is in
degrees pre-normalization). d target is 3.5 A (van-der-Waals sum ~3.4-3.6 A
for S..Cbeta pre-reaction), consistent with the pre-reaction 'ready-to-fire'
geometry we want to steer the model toward.

Rather than reading the raw poses back from the CSV, we operate on the ESM
cache directly:
    poses_unnorm  (N, 3)  = [d_b_nuc_A, bd_angle_deg, planar_dihedral_deg]
    poses         (N, 3)  = z-scored version (matches training convention)
    row_seq_idx   (N,)    = per-row pocket index (unique pocket = residue emb)

We form pairs *within pocket* (row_seq_idx[A] == row_seq_idx[B]) and
emit both directions asymmetrically: only (worse -> better).

For each pocket, we optionally cap the emitted pairs to avoid one very
rich pocket (ZAP70 has 1546 rows -> up to ~1M in-pocket pairs) from
dominating training. Default cap: 200 pairs per pocket.

Emits NPZ compatible with `train_m1a_pairs.py`:
  src_smi, tgt_smi, residues_emb_idx, residues_mask_idx, pose (=B's z-scored pose),
  tc, pocket_idx, dtheta, dd, source_a, source_b

Also emits a stats JSON + pair_stats.md summary.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

# Ideal geometry targets — empirically calibrated from crystal-verified rows
# (covindb_pdb, covbinder_inpdb, covindb_v2) in the ESM cache:
#   theta median ~110° (post-reaction S-C-C angle in covalent adduct)
#   d median ~1.75 Å (post-reaction S-C bond length)
# Boltz cofolds systematically over-linearize (median 146°) and elongate,
# so pairs of (worse -> crystal-like) are readily available.
THETA_IDEAL = 110.0
D_IDEAL = 1.75

FP_RADIUS = 2
FP_NBITS = 2048


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def randomize_smi(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=False, doRandom=True)


def mol_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/v2_curriculum"))
    ap.add_argument("--dtheta_gap", type=float, default=10.0,
                    help="Minimum absolute-error reduction in bd_angle to accept.")
    ap.add_argument("--dd_gap", type=float, default=0.25,
                    help="Minimum absolute-error reduction in d_b_nuc to accept.")
    ap.add_argument("--tc_min", type=float, default=0.30,
                    help="Minimum Tanimoto similarity (same chemotype).")
    ap.add_argument("--per_pocket_cap", type=int, default=1000,
                    help="Max curriculum pairs per pocket (avoid one pocket "
                         "dominating training).")
    ap.add_argument("--require_both_geoms", type=int, default=1,
                    help="If 1, require BOTH dtheta_gap AND dd_gap; if 0, "
                         "accept pair if EITHER improvement is met.")
    ap.add_argument("--val_pocket_frac", type=float, default=0.20,
                    help="Fraction of pockets held out for validation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include_scheme_c", action="store_true", default=True,
                    help="Include Scheme C (crystal-pair) pairs — high-quality "
                         "improve-to-crystal signal.")
    ap.add_argument("--scheme_c_npz", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/scheme_C/pairs_scheme_C.npz"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[curriculum] out_dir={out_dir}")

    # ---- Load ESM cache ----
    cache_path = Path(args.cache)
    d = np.load(cache_path, allow_pickle=True)
    smiles = d["smiles"]
    sources = d["sources"]
    struct_ids = d["struct_ids"]
    row_seq_idx = d["row_seq_idx"].astype(np.int64)
    poses = d["poses"].astype(np.float32)          # z-scored
    poses_unnorm = d["poses_unnorm"].astype(np.float32)  # (d, theta, phi) raw
    pose_mean = d["pose_mean"]
    pose_std = d["pose_std"]
    print(f"[curriculum] cache: N={len(smiles)} rows, "
          f"{row_seq_idx.max()+1} unique pockets")

    # ---- Group rows by pocket ----
    by_pocket: dict[int, list[int]] = defaultdict(list)
    for i in range(len(smiles)):
        by_pocket[int(row_seq_idx[i])].append(i)

    multi = {p: rows for p, rows in by_pocket.items() if len(rows) >= 2}
    print(f"[curriculum] pockets with >=2 rows: {len(multi)}")

    # ---- Pocket-based train/val split ----
    pockets_list = sorted(multi.keys())
    rng.shuffle(pockets_list)
    n_val = max(1, int(len(pockets_list) * args.val_pocket_frac))
    val_pockets = set(pockets_list[:n_val])
    train_pockets = set(pockets_list[n_val:])
    print(f"[curriculum] pocket split: train={len(train_pockets)} "
          f"val={len(val_pockets)}")

    # ---- Enumerate ordered pairs ----
    def emit_pairs_for_pocket(pocket_id: int, row_indices: list[int]):
        d_raw = poses_unnorm[row_indices, 0]
        theta_raw = poses_unnorm[row_indices, 1]
        # Canonical SMILES for uniqueness + Tc precompute
        canons = [canon(str(smiles[i])) for i in row_indices]
        fps = [mol_fp(c) if c else None for c in canons]

        eligible: list[dict] = []
        for a in range(len(row_indices)):
            if canons[a] is None or fps[a] is None:
                continue
            err_theta_a = abs(theta_raw[a] - THETA_IDEAL)
            err_d_a = abs(d_raw[a] - D_IDEAL)
            for b in range(len(row_indices)):
                if a == b: continue
                if canons[b] is None or fps[b] is None: continue
                if canons[a] == canons[b]: continue
                err_theta_b = abs(theta_raw[b] - THETA_IDEAL)
                err_d_b = abs(d_raw[b] - D_IDEAL)
                theta_ok = (err_theta_a - err_theta_b) >= args.dtheta_gap
                d_ok = (err_d_a - err_d_b) >= args.dd_gap
                if args.require_both_geoms:
                    if not (theta_ok and d_ok):
                        continue
                else:
                    # OR-mode: at least one geom must improve, and the OTHER
                    # must NOT get worse by more than half the gap (soft floor).
                    if not (theta_ok or d_ok):
                        continue
                    if (err_theta_b - err_theta_a) > 0.5 * args.dtheta_gap:
                        continue
                    if (err_d_b - err_d_a) > 0.5 * args.dd_gap:
                        continue
                tc = float(DataStructs.TanimotoSimilarity(fps[a], fps[b]))
                if tc < args.tc_min:
                    continue
                eligible.append(dict(
                    a=row_indices[a], b=row_indices[b],
                    dtheta=err_theta_a - err_theta_b,
                    dd=err_d_a - err_d_b,
                    tc=tc,
                ))
        # Cap per pocket: keep the pairs with the LARGEST combined improvement
        if len(eligible) > args.per_pocket_cap:
            eligible.sort(key=lambda x: (x["dtheta"] + 20.0 * x["dd"] + 5.0 * x["tc"]),
                          reverse=True)
            eligible = eligible[:args.per_pocket_cap]
        return eligible

    train_pairs: list[dict] = []
    val_pairs: list[dict] = []
    per_pocket_emit = Counter()

    for p_id, row_indices in multi.items():
        pairs = emit_pairs_for_pocket(p_id, row_indices)
        per_pocket_emit[p_id] = len(pairs)
        if p_id in val_pockets:
            val_pairs.extend((p_id, p) for p in pairs)
        else:
            train_pairs.extend((p_id, p) for p in pairs)

    print(f"[curriculum] curriculum pairs: train={len(train_pairs)} "
          f"val={len(val_pairs)}")
    top10 = per_pocket_emit.most_common(10)
    print(f"[curriculum] top-10 pockets by emitted pairs: {top10}")

    # ---- Optionally splice in Scheme C (crystal pairs) ----
    scheme_c_added_train = 0
    scheme_c_added_val = 0
    if args.include_scheme_c and Path(args.scheme_c_npz).exists():
        sc = np.load(args.scheme_c_npz, allow_pickle=True)
        n_sc = len(sc["src_smi"])
        sc_pocket_idx = sc["residues_emb_idx"].astype(np.int64)
        # Route scheme C pairs into train/val by their pocket
        for k in range(n_sc):
            p_id = int(sc_pocket_idx[k])
            # If pocket appears in val_pockets, add to val; else to train.
            # If pocket is unseen (not in either), route to train.
            record = {
                "_scheme_c": True,
                "src_smi": str(sc["src_smi"][k]),
                "tgt_smi": str(sc["tgt_smi"][k]),
                "pocket_id": p_id,
                "pose": sc["pose"][k].astype(np.float32),
                "tc": float(sc["tc"][k]),
                "source_a": str(sc["source_a"][k]),
                "source_b": str(sc["source_b"][k]),
            }
            if p_id in val_pockets:
                val_pairs.append((p_id, record))
                scheme_c_added_val += 1
            else:
                train_pairs.append((p_id, record))
                scheme_c_added_train += 1
    print(f"[curriculum] scheme_C spliced: train+={scheme_c_added_train} "
          f"val+={scheme_c_added_val}")

    # ---- Build output arrays ----
    def _serialize(pairs, tag):
        src_smi_list, tgt_smi_list, res_idx_list = [], [], []
        pose_list, tc_list, source_a_list, source_b_list = [], [], [], []
        dtheta_list, dd_list, pocket_list, is_crystal_list = [], [], [], []
        for pocket_id, pair in pairs:
            if pair.get("_scheme_c"):
                # Pre-baked crystal pair
                src_smi_list.append(pair["src_smi"])   # already randomized in scheme_C
                tgt_smi_list.append(pair["tgt_smi"])
                res_idx_list.append(pair["pocket_id"])
                pose_list.append(pair["pose"])
                tc_list.append(pair["tc"])
                source_a_list.append(pair["source_a"])
                source_b_list.append(pair["source_b"])
                dtheta_list.append(np.nan)  # crystal pair; not measured against pre-reaction ideal
                dd_list.append(np.nan)
                pocket_list.append(pair["pocket_id"])
                is_crystal_list.append(True)
            else:
                a = pair["a"]; b = pair["b"]
                src_smi_list.append(randomize_smi(str(smiles[a])))
                tgt_smi_list.append(str(smiles[b]))
                res_idx_list.append(int(row_seq_idx[b]))  # target's pocket (== A's pocket in-pocket)
                pose_list.append(poses[b])                 # z-scored TARGET pose
                tc_list.append(pair["tc"])
                source_a_list.append(str(sources[a]))
                source_b_list.append(str(sources[b]))
                dtheta_list.append(pair["dtheta"])
                dd_list.append(pair["dd"])
                pocket_list.append(int(row_seq_idx[b]))
                is_crystal_list.append(False)
        return dict(
            src_smi=np.array(src_smi_list, dtype=object),
            tgt_smi=np.array(tgt_smi_list, dtype=object),
            residues_emb_idx=np.array(res_idx_list, dtype=np.int32),
            residues_mask_idx=np.array(res_idx_list, dtype=np.int32),
            pose=np.stack(pose_list).astype(np.float32) if pose_list else np.zeros((0, 3), np.float32),
            tc=np.array(tc_list, dtype=np.float32),
            source_a=np.array(source_a_list, dtype=object),
            source_b=np.array(source_b_list, dtype=object),
            dtheta_reduction=np.array(dtheta_list, dtype=np.float32),
            dd_reduction=np.array(dd_list, dtype=np.float32),
            pocket_idx=np.array(pocket_list, dtype=np.int32),
            is_crystal=np.array(is_crystal_list, dtype=bool),
            scheme=f"v2_curriculum_{tag}",
            cache_path=str(cache_path),
        )

    train_arr = _serialize(train_pairs, "train")
    val_arr = _serialize(val_pairs, "val")

    train_path = out_dir / "pairs_train.npz"
    val_path = out_dir / "pairs_val.npz"
    np.savez(train_path, **train_arr)
    np.savez(val_path, **val_arr)
    print(f"[curriculum] wrote {train_path}  N={len(train_arr['src_smi'])}")
    print(f"[curriculum] wrote {val_path}    N={len(val_arr['src_smi'])}")

    # ---- Also write parquet (spec deliverable) ----
    try:
        import pandas as pd
        def to_df(arr):
            return pd.DataFrame({
                "src_smi": arr["src_smi"],
                "tgt_smi": arr["tgt_smi"],
                "pocket_idx": arr["pocket_idx"],
                "pose_d_z": arr["pose"][:, 0],
                "pose_theta_z": arr["pose"][:, 1],
                "pose_phi_z": arr["pose"][:, 2],
                "tc": arr["tc"],
                "dtheta_reduction": arr["dtheta_reduction"],
                "dd_reduction": arr["dd_reduction"],
                "source_a": arr["source_a"],
                "source_b": arr["source_b"],
                "is_crystal": arr["is_crystal"],
            })
        to_df(train_arr).to_parquet(out_dir / "pairs_train.parquet", index=False)
        to_df(val_arr).to_parquet(out_dir / "pairs_val.parquet", index=False)
        print("[curriculum] wrote parquet mirrors")
    except Exception as e:
        print(f"[curriculum] parquet write failed: {e} (NPZ is the source of truth)")

    # ---- Stats + pair_stats.md ----
    stats = {
        "cache_path": str(cache_path),
        "filters": {
            "dtheta_gap_deg": args.dtheta_gap,
            "dd_gap_A": args.dd_gap,
            "tc_min": args.tc_min,
            "per_pocket_cap": args.per_pocket_cap,
            "theta_ideal_deg": THETA_IDEAL,
            "d_ideal_A": D_IDEAL,
        },
        "pocket_split": {
            "n_train_pockets": len(train_pockets),
            "n_val_pockets": len(val_pockets),
            "val_pocket_frac": args.val_pocket_frac,
            "seed": args.seed,
        },
        "counts": {
            "n_train_pairs": len(train_arr["src_smi"]),
            "n_val_pairs": len(val_arr["src_smi"]),
            "n_train_curriculum": len(train_arr["src_smi"]) - scheme_c_added_train,
            "n_val_curriculum": len(val_arr["src_smi"]) - scheme_c_added_val,
            "n_train_crystal": scheme_c_added_train,
            "n_val_crystal": scheme_c_added_val,
        },
        "top10_pockets_by_pairs": [(int(k), int(v)) for k, v in top10],
        "pose_normalizer": {
            "mean": pose_mean.tolist(),
            "std": pose_std.tolist(),
        },
    }
    (out_dir / "pair_stats.json").write_text(json.dumps(stats, indent=2))
    md = [
        "# v2 Curriculum Pairs — Stats",
        "",
        f"Cache: `{cache_path}` (N={len(smiles)} rows, "
        f"{row_seq_idx.max()+1} pockets)",
        "",
        "## Filters",
        f"- Δθ >= {args.dtheta_gap}° (improvement in |bd_angle - {THETA_IDEAL}°|)",
        f"- Δd >= {args.dd_gap} Å (improvement in |d_SγCβ - {D_IDEAL} Å|)",
        f"- Tc(A, B) >= {args.tc_min} (same chemotype family)",
        f"- Per-pocket cap: {args.per_pocket_cap} pairs",
        "",
        "## Counts",
        f"- Train pockets: {len(train_pockets)}",
        f"- Val pockets: {len(val_pockets)}",
        f"- Train pairs (curriculum): {len(train_arr['src_smi']) - scheme_c_added_train}",
        f"- Train pairs (crystal spliced): {scheme_c_added_train}",
        f"- **Train total: {len(train_arr['src_smi'])}**",
        f"- Val pairs (curriculum): {len(val_arr['src_smi']) - scheme_c_added_val}",
        f"- Val pairs (crystal spliced): {scheme_c_added_val}",
        f"- **Val total: {len(val_arr['src_smi'])}**",
        "",
        "## Top 10 pockets by emitted pair count",
        *[f"- pocket_idx={k}: {v} pairs" for k, v in top10],
        "",
        "## Pose normalizer (persisted with checkpoint)",
        f"- mean = {list(pose_mean)}",
        f"- std  = {list(pose_std)}",
    ]
    (out_dir / "pair_stats.md").write_text("\n".join(md))
    print(f"[curriculum] wrote pair_stats.md")


if __name__ == "__main__":
    main()
