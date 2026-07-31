"""Phase 2: Free-sampling distribution shift under 4 pose conditions.

For each pose condition, sample 100 molecules with anchor=Mol1, pocket=ZAP70.
Compute per-condition property distributions and pairwise KS tests.

Pose conditions:
  d_low       : d=2.5, theta=105, phi=0 (typical BD)
  d_high      : d=5.0, theta=105, phi=0 (far, no reaction geometry)
  d_shuffled  : pose randomly permuted from batch of training poses
  d_zeroed    : all pose dims = 0

Interpretation:
  - If observables shift only for zeroed pose but NOT for d=2.5 vs d=5.0
    (or shuffled) -> decorative-token: model responds to HAVING vs LACKING
    pose signal, not to pose CONTENT.
  - If observables shift monotonically or dramatically between d=2.5 vs d=5.0
    -> real pose-conditional generation.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
REINVENT_ROOT = Path("/Users/shaharharel/Documents/github/REINVENT4")
FIXES_DIR = LOCAL_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"
for p in [REINVENT_ROOT, FIXES_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from m1a_v2_model import load_m1a_v2  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS = "C=CC(=O)N"

CKPT = LOCAL_ROOT / "models/v2_curriculum_clean/best.chkpt"
PRIOR = LOCAL_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"
CACHE = LOCAL_ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"
VAL_NPZ = LOCAL_ROOT / "data/paper_pair_training/v2_curriculum/pairs_val.npz"
OUT_DIR = LOCAL_ROOT / "data/paper_pair_training/pose_sweep_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def randomize_smi(smi: str) -> str:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def sample_cohort(model, anchor_smi, res_emb, res_mask, pose_norm,
                  n=100, batch_size=32, max_length=128, temperature=1.0,
                  seed=42):
    """Sample n molecules with fixed pose."""
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    out_rows = []
    n_done = 0
    t0 = time.time()
    while n_done < n:
        batch_n = min(batch_size, n - n_done)
        anchors = [randomize_smi(anchor_smi) for _ in range(batch_n)]
        seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64)
                 for s in anchors]
        L = max(len(s) for s in seqs)
        src = np.zeros((batch_n, L), dtype=np.int64)
        for j, s in enumerate(seqs):
            src[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        res_emb_b = torch.from_numpy(np.tile(res_emb[None], (batch_n, 1, 1))).to(device)
        res_mask_b = torch.from_numpy(np.tile(res_mask[None], (batch_n, 1))).to(device)
        pose_b = torch.from_numpy(np.tile(pose_norm[None], (batch_n, 1))).to(device)
        out_smiles, nlls = model.sample_multinomial(
            src_t, src_mask, res_emb_b, res_mask_b, pose_b,
            max_length=max_length, temperature=temperature)
        for s, nll, anc in zip(out_smiles, nlls, anchors):
            out_rows.append({"SMILES": s, "Input_SMILES": anc,
                              "NLL": float(nll)})
        n_done += batch_n
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-6)
        print(f"    {n_done}/{n}  ({rate:.1f} mol/s)", flush=True)
    return out_rows


def compute_props(smi):
    """Return dict of properties or None if invalid."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    RDLogger.DisableLog("rdApp.*")

    try:
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() == 0:
            return None
        acryl_pattern = Chem.MolFromSmarts(ACRYL_SMARTS)
        # Largest fragment
        try:
            frags = Chem.GetMolFrags(m, asMols=True)
            if len(frags) > 1:
                largest = max(frags, key=lambda f: f.GetNumAtoms())
            else:
                largest = m
        except Exception:
            largest = m
        has_acryl_largest = largest.HasSubstructMatch(acryl_pattern)
        # Scaffold
        try:
            scaffold = MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            scaffold = ""
        first_token = smi[0] if smi else ""
        return {
            "valid": True,
            "MW": float(Descriptors.MolWt(m)),
            "TPSA": float(Descriptors.TPSA(m)),
            "logP": float(Descriptors.MolLogP(m)),
            "HBA": int(Descriptors.NumHAcceptors(m)),
            "HBD": int(Descriptors.NumHDonors(m)),
            "RB": int(Descriptors.NumRotatableBonds(m)),
            "acryl_largest": bool(has_acryl_largest),
            "scaffold": scaffold,
            "first_token": first_token,
            "n_atoms_largest": int(largest.GetNumAtoms()),
        }
    except Exception:
        return None


def cohort_summary(df):
    valid = df[df["valid"] == True]
    return {
        "n": len(df),
        "n_valid": len(valid),
        "pct_valid": float(len(valid) / max(len(df), 1)),
        "MW_median": float(valid["MW"].median()) if len(valid) else None,
        "MW_mean": float(valid["MW"].mean()) if len(valid) else None,
        "TPSA_median": float(valid["TPSA"].median()) if len(valid) else None,
        "logP_median": float(valid["logP"].median()) if len(valid) else None,
        "HBA_mean": float(valid["HBA"].mean()) if len(valid) else None,
        "HBD_mean": float(valid["HBD"].mean()) if len(valid) else None,
        "pct_acryl_largest": float(valid["acryl_largest"].mean()) if len(valid) else None,
        "n_unique_scaffolds": int(valid["scaffold"].nunique()) if len(valid) else None,
        "top_first_tokens": dict(valid["first_token"].value_counts().head(5).to_dict()) if len(valid) else {},
    }


def ks_matrix(cohorts_valid, feature: str):
    """Pairwise KS-test between cohorts on a numeric feature."""
    from scipy.stats import ks_2samp
    names = list(cohorts_valid.keys())
    mat = {}
    for a in names:
        mat[a] = {}
        for b in names:
            if a == b:
                mat[a][b] = {"stat": 0.0, "p": 1.0}
                continue
            x = cohorts_valid[a][feature].dropna().values
            y = cohorts_valid[b][feature].dropna().values
            if len(x) < 5 or len(y) < 5:
                mat[a][b] = {"stat": None, "p": None}
                continue
            k = ks_2samp(x, y)
            mat[a][b] = {"stat": float(k.statistic), "p": float(k.pvalue)}
    return mat


def main():
    N = 100
    BATCH = 32
    TEMP = 1.0
    device = torch.device("cpu")

    print("[phase2] loading model", flush=True)
    model = load_m1a_v2(str(PRIOR), device, ckpt_path=str(CKPT))
    model.eval()
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()

    print("[phase2] loading ZAP70 pocket from cache", flush=True)
    cache = np.load(CACHE, allow_pickle=True)
    seq_idx = 5
    zap70_emb = cache["residues_emb"][seq_idx]
    zap70_mask = cache["residues_mask"][seq_idx]
    print(f"[phase2] pocket_idx={seq_idx}, residues={int(zap70_mask.sum())}",
          flush=True)

    val = np.load(VAL_NPZ, allow_pickle=True)
    val_poses_norm = val["pose"].astype(np.float32)
    val_poses_raw = val_poses_norm * pose_std + pose_mean
    median_pose_raw = np.median(val_poses_raw, axis=0)

    def norm(p_raw):
        return ((p_raw - pose_mean) / np.maximum(pose_std, 1e-6)).astype(np.float32)

    # 4 pose conditions (all with theta=105, phi=0 unless noted)
    pose_d_low = np.array([2.5, 105.0, 0.0], dtype=np.float32)
    pose_d_high = np.array([5.0, 105.0, 0.0], dtype=np.float32)
    # Shuffled random from val
    rng = np.random.RandomState(7)
    shuf_idx = rng.choice(len(val_poses_raw), 1)[0]
    pose_shuffled_raw = val_poses_raw[shuf_idx].astype(np.float32).copy()
    # Zeroed pose
    pose_zero_raw = np.zeros(3, dtype=np.float32)

    variants = {
        "d_2.5":     {"raw": pose_d_low,        "norm": norm(pose_d_low)},
        "d_5.0":     {"raw": pose_d_high,       "norm": norm(pose_d_high)},
        "shuffled":  {"raw": pose_shuffled_raw, "norm": norm(pose_shuffled_raw)},
        "zeroed":    {"raw": pose_zero_raw,     "norm": norm(pose_zero_raw)},
    }

    all_dfs = {}
    for name, cfg in variants.items():
        out_csv = OUT_DIR / f"phase2_samples_{name}.csv"
        print(f"\n[phase2] === Cohort {name} ===", flush=True)
        print(f"[phase2] pose_raw={cfg['raw']}  pose_norm={cfg['norm']}",
              flush=True)
        if out_csv.exists() and out_csv.stat().st_size > 500:
            print(f"[phase2] loading cached {out_csv}", flush=True)
            df = pd.read_csv(out_csv)
        else:
            rows = sample_cohort(model, MOL1_SMI, zap70_emb, zap70_mask,
                                  cfg["norm"], n=N, batch_size=BATCH,
                                  max_length=128, temperature=TEMP,
                                  seed=42)
            for r in rows:
                r["cohort"] = name
                r["pose_raw"] = json.dumps(cfg["raw"].tolist())
            df = pd.DataFrame(rows)
            # Compute properties
            props = df["SMILES"].apply(compute_props)
            for k in ["valid", "MW", "TPSA", "logP", "HBA", "HBD", "RB",
                       "acryl_largest", "scaffold", "first_token",
                       "n_atoms_largest"]:
                df[k] = props.apply(lambda p, key=k: (p or {}).get(key,
                                                                   False if key == "valid" or key == "acryl_largest" else None))
            df.to_csv(out_csv, index=False)
        all_dfs[name] = df

    # Summaries
    print("\n===== PHASE 2 COHORT SUMMARIES =====")
    summaries = {}
    for name, df in all_dfs.items():
        s = cohort_summary(df)
        summaries[name] = s
        print(f"\n{name}: pose_raw={variants[name]['raw'].tolist()}")
        for k, v in s.items():
            print(f"    {k}: {v}")

    # Valid-only subsets for KS
    cohorts_valid = {n: df[df["valid"] == True] for n, df in all_dfs.items()}
    features = ["MW", "TPSA", "logP", "HBA", "HBD", "RB"]
    ks_results = {}
    print("\n===== KS TESTS ON CONTINUOUS FEATURES =====")
    for feat in features:
        ks_results[feat] = ks_matrix(cohorts_valid, feat)
        print(f"\n{feat}:")
        names = list(cohorts_valid.keys())
        header = "        " + "  ".join(f"{n:>9}" for n in names)
        print(header)
        for a in names:
            row = f"  {a:>6}  "
            for b in names:
                p = ks_results[feat][a][b]["p"]
                stat = ks_results[feat][a][b]["stat"]
                if p is None:
                    cell = "    n/a "
                else:
                    marker = "*" if p < 0.01 else ""
                    cell = f"p={p:.3g}{marker:>2}"
                row += f"{cell:>10} "
            print(row)

    # Acryl retention
    print("\n===== % ACRYL (largest fragment) =====")
    acryl_pct = {}
    for name in cohorts_valid:
        d = cohorts_valid[name]
        pct = float(d["acryl_largest"].mean()) if len(d) else None
        acryl_pct[name] = pct
        print(f"  {name}: {pct:.3f} ({int(d['acryl_largest'].sum())}/{len(d)})")

    # First-token distribution
    print("\n===== FIRST-TOKEN DISTRIBUTION =====")
    tok_dist = {}
    for name in cohorts_valid:
        d = cohorts_valid[name]
        counts = d["first_token"].value_counts().to_dict()
        tok_dist[name] = counts
        print(f"  {name}: {counts}")

    out = {
        "config": {
            "N_per_cohort": N,
            "temperature": TEMP,
            "batch_size": BATCH,
            "pocket_idx": seq_idx,
            "median_pose_raw": [float(x) for x in median_pose_raw],
        },
        "pose_conditions": {n: {"raw": v["raw"].tolist(), "norm": v["norm"].tolist()} for n, v in variants.items()},
        "summaries": summaries,
        "ks_results": ks_results,
        "acryl_pct": acryl_pct,
        "first_token_dist": tok_dist,
    }
    out_path = OUT_DIR / "phase2_free_sampling.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[phase2] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
