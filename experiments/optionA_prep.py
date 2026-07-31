"""OptionA — Data prep: build train/val/test parquets with rotation-invariant local-frame
    beta-carbon coordinates (bc_local_xyz), BD angle, and planar torsion phi.

Split is by target_name (leakage-free), roughly 80/10/10.

Outputs:
    data/optionA/train.parquet
    data/optionA/val.parquet
    data/optionA/test.parquet
    data/optionA/splits.json
    data/optionA/local_frame_unit_test.json  (SO(3) invariance check)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "m1a_triples_v2" / "covindb_v2_triples_smarts_fixed.parquet"
OUT = REPO / "data" / "optionA"
OUT.mkdir(parents=True, exist_ok=True)


def parse_json_or_list(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def build_local_frame(nuc_xyz: np.ndarray, ca_xyzs: np.ndarray):
    """Given nucleophile origin and >=3 pocket Cα coordinates (sorted nearest→farthest),
    return orthonormal rotation R (3x3) whose columns are e1, e2, e3.
    Frame: e1 = unit vector to nearest Cα; e2 = Gram-Schmidt on 2nd Cα; e3 = e1 x e2.
    """
    if ca_xyzs.shape[0] < 3:
        return None
    v1 = ca_xyzs[0] - nuc_xyz
    n1 = np.linalg.norm(v1)
    if n1 < 1e-6:
        return None
    e1 = v1 / n1

    v2 = ca_xyzs[1] - nuc_xyz
    v2p = v2 - np.dot(v2, e1) * e1
    n2 = np.linalg.norm(v2p)
    if n2 < 1e-6:
        # try third
        v2 = ca_xyzs[2] - nuc_xyz
        v2p = v2 - np.dot(v2, e1) * e1
        n2 = np.linalg.norm(v2p)
        if n2 < 1e-6:
            return None
    e2 = v2p / n2

    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=1)  # cols = e1, e2, e3
    return R


def process_row(row):
    """Return a dict of derived fields, or None on failure."""
    pocket = parse_json_or_list(row.get("pocket_residues"))
    nuc = parse_json_or_list(row.get("nucleophile_xyz"))
    wb = parse_json_or_list(row.get("warhead_b_xyz"))
    pose6d = parse_json_or_list(row.get("warhead_pose_6d"))
    if pocket is None or nuc is None or wb is None:
        return None
    if len(pocket) < 3:
        return None
    nuc = np.array(nuc, dtype=np.float64)
    wb = np.array(wb, dtype=np.float64)
    if nuc.shape != (3,) or wb.shape != (3,):
        return None

    # Sort pocket residues by d (nearest first)
    pocket = sorted(pocket, key=lambda r: float(r.get("d", 1e9)))
    ca_all = []
    aa_all = []
    for r in pocket:
        xyz = r.get("ca_xyz")
        if xyz is None:
            continue
        xyz = np.array(xyz, dtype=np.float64)
        if xyz.shape != (3,):
            continue
        ca_all.append(xyz)
        aa_all.append(r.get("aa", "X"))
    if len(ca_all) < 3:
        return None
    ca_all = np.stack(ca_all)  # (K, 3)

    # Build local frame from top 3 nearest CAs
    R = build_local_frame(nuc, ca_all[:3])
    if R is None:
        return None

    # bc_local = R.T @ (wb - nuc)
    bc_local = R.T @ (wb - nuc)  # (3,)

    bd_angle = float(row.get("bd_angle_deg")) if row.get("bd_angle_deg") is not None else None
    # phi from pose_6d[5]
    phi = None
    if pose6d is not None and len(pose6d) >= 6:
        try:
            phi = float(pose6d[5])
        except Exception:
            phi = None
    # Verify (informational)
    if pose6d is not None and len(pose6d) >= 5 and bd_angle is not None:
        p_bd = float(pose6d[4])
        # allow numerical drift
        # (we won't hard-fail; just note if drift is large)

    return {
        "struct_id": row.get("struct_id"),
        "pdb_id": row.get("pdb_id"),
        "canon_smi": row.get("canon_smi"),
        "target_name": row.get("target_name"),
        "nucleophile_resname": row.get("nucleophile_resname"),
        "pocket_ca_xyz": [list(map(float, x)) for x in ca_all.tolist()],
        "pocket_aa": aa_all,
        "nucleophile_xyz": [float(x) for x in nuc.tolist()],
        "warhead_b_xyz": [float(x) for x in wb.tolist()],
        "bc_local_xyz": [float(x) for x in bc_local.tolist()],
        "bd_angle": bd_angle,
        "phi_planar": phi,
        "d_b_nuc": float(row.get("d_b_nuc")) if row.get("d_b_nuc") is not None else None,
        "local_frame_R": R.flatten().tolist(),
    }


def local_frame_invariance_test(rows, n=5, seed=0):
    """Verify: apply random SO(3) rotation + translation to nuc, pocket, wb;
    recompute local frame; bc_local must be identical.
    """
    from scipy.spatial.transform import Rotation as ScRot
    rng = np.random.default_rng(seed)
    results = []
    for i in range(min(n, len(rows))):
        r = rows[i]
        nuc = np.array(r["nucleophile_xyz"])
        wb = np.array(r["warhead_b_xyz"])
        ca_all = np.array(r["pocket_ca_xyz"])
        # baseline bc_local
        R0 = build_local_frame(nuc, ca_all[:3])
        bc0 = R0.T @ (wb - nuc)
        # random rotation + translation
        Rrand = ScRot.random(random_state=rng.integers(1e6)).as_matrix()
        t = rng.normal(0, 10, size=3)
        nuc_r = Rrand @ nuc + t
        wb_r = Rrand @ wb + t
        ca_r = (Rrand @ ca_all.T).T + t
        R1 = build_local_frame(nuc_r, ca_r[:3])
        bc1 = R1.T @ (wb_r - nuc_r)
        err = float(np.linalg.norm(bc0 - bc1))
        results.append({"row_i": i, "struct_id": r["struct_id"], "bc_local_orig": bc0.tolist(), "bc_local_rot": bc1.tolist(), "err": err})
    return results


def main():
    print(f"[prep] loading {SRC}", flush=True)
    df = pd.read_parquet(SRC)
    print(f"[prep] rows={len(df)} targets={df['target_name'].nunique()}", flush=True)

    processed = []
    n_fail = 0
    for _, row in df.iterrows():
        out = process_row(row.to_dict())
        if out is None:
            n_fail += 1
            continue
        processed.append(out)
    print(f"[prep] processed={len(processed)} failed={n_fail}", flush=True)

    # SO(3) invariance test
    test_rows = local_frame_invariance_test(processed, n=5, seed=42)
    max_err = max(r["err"] for r in test_rows)
    print(f"[prep] SO(3) invariance test max_err={max_err:.3e}", flush=True)
    if max_err > 1e-4:
        print(f"[prep] FAIL: SO(3) invariance broken (max_err={max_err})", flush=True)
        # dump for debug
        (OUT / "local_frame_unit_test.json").write_text(json.dumps(test_rows, indent=2))
        sys.exit(1)
    (OUT / "local_frame_unit_test.json").write_text(json.dumps({"max_err": max_err, "cases": test_rows}, indent=2))

    # Split by target
    df_proc = pd.DataFrame(processed)
    targets = sorted(df_proc["target_name"].dropna().unique().tolist())
    rng = np.random.default_rng(1234)
    rng.shuffle(targets)
    n = len(targets)
    n_train = int(round(n * 0.80))
    n_val = int(round(n * 0.10))
    train_tgts = set(targets[:n_train])
    val_tgts = set(targets[n_train : n_train + n_val])
    test_tgts = set(targets[n_train + n_val :])
    assert not (train_tgts & val_tgts) and not (train_tgts & test_tgts) and not (val_tgts & test_tgts)
    print(f"[prep] targets: train={len(train_tgts)} val={len(val_tgts)} test={len(test_tgts)}", flush=True)

    tr = df_proc[df_proc["target_name"].isin(train_tgts)].reset_index(drop=True)
    va = df_proc[df_proc["target_name"].isin(val_tgts)].reset_index(drop=True)
    te = df_proc[df_proc["target_name"].isin(test_tgts)].reset_index(drop=True)
    print(f"[prep] rows: train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    tr.to_parquet(OUT / "train.parquet")
    va.to_parquet(OUT / "val.parquet")
    te.to_parquet(OUT / "test.parquet")
    (OUT / "splits.json").write_text(json.dumps({
        "train": sorted(train_tgts), "val": sorted(val_tgts), "test": sorted(test_tgts),
        "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)), "n_test_rows": int(len(te)),
        "n_train_targets": len(train_tgts), "n_val_targets": len(val_tgts), "n_test_targets": len(test_tgts),
    }, indent=2))
    print(f"[prep] wrote to {OUT}", flush=True)


if __name__ == "__main__":
    main()
