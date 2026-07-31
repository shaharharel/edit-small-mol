"""Phase 3: Permutation null test.

Sample 100 mols with the GROUND-TRUTH training pose (paired with the
reference-target's val row) vs 100 mols with a randomly permuted pose
from another training row. Run KS-test on all sample-level observables.

If ground-truth pose vs permuted pose produce indistinguishable samples,
this confirms the decorative-token hypothesis (model doesn't respond to
pose CONTENT, only to pose PRESENCE).
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
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
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
        for s, nll in zip(out_smiles, nlls):
            out_rows.append({"SMILES": s, "NLL": float(nll)})
        n_done += batch_n
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-6)
        print(f"    {n_done}/{n}  ({rate:.1f} mol/s)", flush=True)
    return out_rows


def compute_props(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() == 0:
            return None
        acryl = Chem.MolFromSmarts(ACRYL_SMARTS)
        try:
            frags = Chem.GetMolFrags(m, asMols=True)
            largest = max(frags, key=lambda f: f.GetNumAtoms()) if len(frags) > 1 else m
        except Exception:
            largest = m
        try:
            scaffold = MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            scaffold = ""
        return {
            "valid": True,
            "MW": float(Descriptors.MolWt(m)),
            "TPSA": float(Descriptors.TPSA(m)),
            "logP": float(Descriptors.MolLogP(m)),
            "HBA": int(Descriptors.NumHAcceptors(m)),
            "HBD": int(Descriptors.NumHDonors(m)),
            "RB": int(Descriptors.NumRotatableBonds(m)),
            "acryl_largest": bool(largest.HasSubstructMatch(acryl)),
            "scaffold": scaffold,
            "first_token": smi[0] if smi else "",
        }
    except Exception:
        return None


def main():
    N = 100
    BATCH = 32
    TEMP = 1.0
    device = torch.device("cpu")

    print("[phase3] loading model", flush=True)
    model = load_m1a_v2(str(PRIOR), device, ckpt_path=str(CKPT))
    model.eval()
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()

    print("[phase3] loading ZAP70 pocket from cache", flush=True)
    cache = np.load(CACHE, allow_pickle=True)
    seq_idx = 5
    zap70_emb = cache["residues_emb"][seq_idx]
    zap70_mask = cache["residues_mask"][seq_idx]

    val = np.load(VAL_NPZ, allow_pickle=True)
    val_poses_norm = val["pose"].astype(np.float32)
    val_poses_raw = val_poses_norm * pose_std + pose_mean

    # Reference row (median pose) — same as phase 1
    target_pose = np.array([2.1, 114.0, -17.0])
    dist = np.linalg.norm((val_poses_raw - target_pose) / pose_std, axis=1)
    ref_i = int(np.argmin(dist))
    gt_pose_norm = val_poses_norm[ref_i]
    gt_pose_raw = val_poses_raw[ref_i]
    print(f"[phase3] ground-truth pose raw={gt_pose_raw}", flush=True)

    # Permuted pose: pick another random row that's far in pose space
    rng = np.random.RandomState(11)
    # Sort by distance descending; pick one from top 30%
    all_dist = np.linalg.norm((val_poses_raw - gt_pose_raw) / pose_std, axis=1)
    far_ix = np.argsort(-all_dist)[:int(0.3 * len(all_dist))]
    perm_i = int(rng.choice(far_ix))
    perm_pose_norm = val_poses_norm[perm_i]
    perm_pose_raw = val_poses_raw[perm_i]
    print(f"[phase3] permuted pose raw={perm_pose_raw}  (i={perm_i}, "
          f"dist={all_dist[perm_i]:.3f} from GT)", flush=True)

    variants = {
        "gt_pose":   {"raw": gt_pose_raw,   "norm": gt_pose_norm},
        "perm_pose": {"raw": perm_pose_raw, "norm": perm_pose_norm},
    }

    all_dfs = {}
    for name, cfg in variants.items():
        out_csv = OUT_DIR / f"phase3_samples_{name}.csv"
        print(f"\n[phase3] === Cohort {name} ===", flush=True)
        print(f"[phase3] pose_raw={cfg['raw']}  pose_norm={cfg['norm']}",
              flush=True)
        if out_csv.exists() and out_csv.stat().st_size > 500:
            print(f"[phase3] loading cached {out_csv}", flush=True)
            df = pd.read_csv(out_csv)
        else:
            rows = sample_cohort(model, MOL1_SMI, zap70_emb, zap70_mask,
                                  cfg["norm"].astype(np.float32),
                                  n=N, batch_size=BATCH,
                                  max_length=128, temperature=TEMP,
                                  seed=42)
            for r in rows:
                r["cohort"] = name
                r["pose_raw"] = json.dumps(cfg["raw"].tolist())
            df = pd.DataFrame(rows)
            props = df["SMILES"].apply(compute_props)
            for k in ["valid", "MW", "TPSA", "logP", "HBA", "HBD", "RB",
                       "acryl_largest", "scaffold", "first_token"]:
                df[k] = props.apply(lambda p, key=k: (p or {}).get(
                    key, False if key in ("valid", "acryl_largest") else None))
            df.to_csv(out_csv, index=False)
        all_dfs[name] = df

    # Summaries
    print("\n===== PHASE 3 COHORT SUMMARIES =====")
    summaries = {}
    for name, df in all_dfs.items():
        v = df[df["valid"] == True]
        summaries[name] = {
            "n": len(df),
            "n_valid": len(v),
            "pct_valid": float(len(v) / max(len(df), 1)),
            "MW_median": float(v["MW"].median()) if len(v) else None,
            "TPSA_median": float(v["TPSA"].median()) if len(v) else None,
            "logP_median": float(v["logP"].median()) if len(v) else None,
            "pct_acryl_largest": float(v["acryl_largest"].mean()) if len(v) else None,
            "n_unique_scaffolds": int(v["scaffold"].nunique()) if len(v) else None,
        }
        print(f"\n{name}:")
        for k, val in summaries[name].items():
            print(f"    {k}: {val}")

    # KS tests
    from scipy.stats import ks_2samp
    print("\n===== KS TESTS (GT vs PERM) =====")
    ks_res = {}
    gt_v = all_dfs["gt_pose"][all_dfs["gt_pose"]["valid"] == True]
    pv = all_dfs["perm_pose"][all_dfs["perm_pose"]["valid"] == True]
    for feat in ["MW", "TPSA", "logP", "HBA", "HBD", "RB"]:
        x = gt_v[feat].dropna().values
        y = pv[feat].dropna().values
        if len(x) < 5 or len(y) < 5:
            ks_res[feat] = {"stat": None, "p": None}
        else:
            k = ks_2samp(x, y)
            ks_res[feat] = {"stat": float(k.statistic), "p": float(k.pvalue)}
        print(f"  {feat}: stat={ks_res[feat]['stat']} p={ks_res[feat]['p']}")

    # Scaffold Jaccard
    gt_scafs = set(gt_v["scaffold"])
    perm_scafs = set(pv["scaffold"])
    inter = len(gt_scafs & perm_scafs)
    union = len(gt_scafs | perm_scafs)
    jaccard = inter / max(union, 1)
    print(f"\nScaffold overlap: GT={len(gt_scafs)} PERM={len(perm_scafs)} "
          f"inter={inter} Jaccard={jaccard:.3f}")

    # First-token overlap
    gt_toks = gt_v["first_token"].value_counts().to_dict()
    perm_toks = pv["first_token"].value_counts().to_dict()
    print(f"\nFirst-token distributions:")
    print(f"  GT: {gt_toks}")
    print(f"  PERM: {perm_toks}")

    # Acryl percent
    gt_acryl = float(gt_v["acryl_largest"].mean()) if len(gt_v) else None
    perm_acryl = float(pv["acryl_largest"].mean()) if len(pv) else None
    print(f"\n%acryl (largest): GT={gt_acryl} PERM={perm_acryl}")

    # Verdict
    n_sig = sum(1 for r in ks_res.values()
                 if r["p"] is not None and r["p"] < 0.01)
    n_total = sum(1 for r in ks_res.values() if r["p"] is not None)
    print(f"\n===== PHASE 3 VERDICT =====")
    print(f"KS tests significant at p<0.01: {n_sig}/{n_total}")
    if n_sig == 0:
        print("--> GT vs PERM samples INDISTINGUISHABLE on all features.")
        print("--> Consistent with DECORATIVE-TOKEN hypothesis: model does "
              "NOT respond to pose CONTENT.")
    else:
        print("--> Some features shift between GT and PERM poses; "
              "pose content IS influencing sample distribution.")

    out = {
        "gt_pose_raw": [float(x) for x in gt_pose_raw],
        "perm_pose_raw": [float(x) for x in perm_pose_raw],
        "gt_perm_pose_distance_zscore": float(all_dist[perm_i]),
        "summaries": summaries,
        "ks_tests": ks_res,
        "scaffold_jaccard": jaccard,
        "first_token_dist": {"gt": gt_toks, "perm": perm_toks},
        "acryl_pct": {"gt": gt_acryl, "perm": perm_acryl},
        "n_significant_ks_p01": n_sig,
        "n_total_ks": n_total,
    }
    out_path = OUT_DIR / "phase3_permutation_null.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[phase3] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
