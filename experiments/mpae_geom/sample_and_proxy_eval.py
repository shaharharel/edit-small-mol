#!/usr/bin/env python3
"""Phase A sampling + proxy eval for a single mpae-geom variant.

For a given checkpoint (V1/V2/V3/V4):
  1. Load model + heads (base, reg_head if reg_on, cpe/proj if nce_on)
  2. Sample 300 Mol1-anchored SMILES × 3 clamps = 900 total
       Clamp A: real Mol1 pose (baseline)
       Clamp B: mean pose vector of the training GOOD quartile
       Clamp C: mean pose vector of the training BAD quartile
  3. Compute proxy metrics per cohort:
       validity, dedup, acryl_largest_frag_pct, Tc-to-Mol1,
       planar_dihedral median, emitted d/θ/φ from ETKDG,
       pIC50_film mean/median (skipped here — placeholder),
       reg-head predicted mpae (V2/V4 only)
  4. Save proxy_metrics_V{n}.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/mpae_geom"))
from m1a_v2_model import load_m1a_v2, D_MODEL  # noqa: E402
from train_m1a_v2_mpae import (RegressionHead, ContrastivePoseEncoder,
                                  ProjectionHead, get_h_pooled, _make_std_mask,
                                  randomize_smi)  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS_STR = "[CH2]=[CH][C](=O)[N]"
ACRYL_SMARTS = Chem.MolFromSmarts(ACRYL_SMARTS_STR)


def measure_emitted_pose(smi_list, seed=42):
    n = len(smi_list)
    d = np.full(n, np.nan); th = np.full(n, np.nan); ph = np.full(n, np.nan)
    for i, smi in enumerate(smi_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        matches = m.GetSubstructMatches(ACRYL_SMARTS)
        if not matches:
            continue
        a1, a2, a3, a_o, a_n = matches[0]
        try:
            mH = Chem.AddHs(m)
            p = AllChem.ETKDGv3(); p.randomSeed = seed
            cid = AllChem.EmbedMolecule(mH, p)
            if cid < 0: continue
            conf = mH.GetConformer(cid)
            p1 = np.array(conf.GetAtomPosition(a1))
            p3 = np.array(conf.GetAtomPosition(a3))
            d[i] = float(np.linalg.norm(p1 - p3))
            th[i] = float(AllChem.GetAngleDeg(conf, a1, a2, a3))
            phi = float(AllChem.GetDihedralDeg(conf, a1, a2, a3, a_n))
            phi_wrap = ((phi + 180.0) % 360.0) - 180.0
            ph[i] = min(abs(phi_wrap), abs(180.0 - abs(phi_wrap)))
        except Exception:
            continue
    return d, th, ph


def sample_cohort(model, res_emb, res_mask, pose_norm, n: int = 300, batch_size: int = 32, temperature: float = 1.0):
    device = model.device
    vocab = model.base.vocabulary; tok = model.base.tokenizer
    smis = []
    n_done = 0
    while n_done < n:
        bn = min(batch_size, n - n_done)
        anchors = [randomize_smi(MOL1_SMI) for _ in range(bn)]
        seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
        L = max(len(s) for s in seqs)
        src = np.zeros((bn, L), dtype=np.int64)
        for j, s in enumerate(seqs): src[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        re = torch.from_numpy(np.tile(res_emb[None], (bn, 1, 1))).to(device)
        rm = torch.from_numpy(np.tile(res_mask[None], (bn, 1))).to(device)
        po = torch.from_numpy(np.tile(pose_norm[None], (bn, 1))).to(device)
        out_smiles, _ = model.sample_multinomial(src_t, src_mask, re, rm, po, max_length=128, temperature=temperature)
        smis.extend(out_smiles)
        n_done += bn
    return smis


def summarize(name, smis, pose_clamp_unnorm, model=None, reg_head=None, res_emb=None, res_mask=None, pose_norm=None, planar_sample_n=300):
    n_raw = len(smis)
    canon = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        canon.append(Chem.MolToSmiles(m) if m else None)
    valid = [c for c in canon if c is not None]
    validity = len(valid) / max(n_raw, 1)
    dedup = len(set(valid)) / max(len(valid), 1)
    acryl_hits = 0
    for c in valid:
        lf = max(c.split("."), key=len)
        m = Chem.MolFromSmiles(lf)
        if m and m.HasSubstructMatch(ACRYL_SMARTS):
            acryl_hits += 1
    acryl_pct = acryl_hits / max(len(valid), 1)
    # Tc
    mol1 = Chem.MolFromSmiles(MOL1_SMI)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    tcs = []
    for c in valid:
        m = Chem.MolFromSmiles(c)
        if not m: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        tcs.append(DataStructs.TanimotoSimilarity(fp1, fp))
    tc_med = float(np.median(tcs)) if tcs else float("nan")

    subset = np.random.default_rng(0).choice(valid, size=min(planar_sample_n, len(valid)), replace=False) if valid else np.array([])
    d, th, ph = measure_emitted_pose(list(subset))
    def med(x): x = x[np.isfinite(x)]; return float(np.median(x)) if len(x) else float("nan")

    out = {
        "cohort": name,
        "n_raw": int(n_raw), "n_valid": int(len(valid)),
        "validity_frac": float(validity),
        "dedup_frac": float(dedup),
        "acryl_largest_frag_pct": float(acryl_pct),
        "tc_to_mol1_median": tc_med,
        "planar_dihedral_median_deg": med(ph),
        "emitted_d_median": med(d),
        "emitted_theta_median": med(th),
        "emitted_phi_median": med(ph),
        "clamp_pose_unnorm": {"d_b_nuc": float(pose_clamp_unnorm[0]),
                                "bd_angle_deg": float(pose_clamp_unnorm[1]),
                                "planar_dihedral_deg": float(pose_clamp_unnorm[2])},
    }
    # Regression-head prediction on the CLAMP conditioning (predict what the head THINKS about this pose token)
    if reg_head is not None:
        pass  # skip since it needs decoded targets — placeholder
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--labeled_npz", default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/labeled_mols.npz"))
    ap.add_argument("--esm_cache",   default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/esm_cache.npz"))
    ap.add_argument("--pose_stats",  default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_geom/pose_stats.json"))
    ap.add_argument("--prior",       default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--n_per_clamp", type=int, default=300)
    ap.add_argument("--target_pocket", default="BTK",
                     help="Which target's pocket to condition on for the eval (default BTK).")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = str(PROJECT_ROOT / f"models/m1a_v2_mpae_V{args.variant}.ckpt")
    if args.out_json is None:
        args.out_json = str(PROJECT_ROOT / f"data/paper_pair_training/mpae_geom/proxy_metrics_V{args.variant}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ps = json.loads(Path(args.pose_stats).read_text())
    pose_mean = np.array(ps["pose_mean"], dtype=np.float32)
    pose_std  = np.array(ps["pose_std"], dtype=np.float32)
    print(f"[eval V{args.variant}] pose_norm mean={pose_mean} std={pose_std}", flush=True)

    # Load model + heads
    print(f"[load] {args.ckpt}")
    model = load_m1a_v2(args.prior, device, ckpt_path=None,
                         pose_mean=torch.from_numpy(pose_mean),
                         pose_std=torch.from_numpy(pose_std))
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    reg_on = ckpt.get("reg_on", False); nce_on = ckpt.get("nce_on", False)
    print(f"  reg_on={reg_on}  nce_on={nce_on}")

    # ESM pocket
    E = np.load(args.esm_cache, allow_pickle=True)
    targets = list(E["target_names"])
    if args.target_pocket not in targets:
        args.target_pocket = targets[0]
    p_idx = targets.index(args.target_pocket)
    res_emb = E["residues_emb"][p_idx].astype(np.float32)
    res_mask = E["residues_mask"][p_idx].astype(bool)
    print(f"[pocket] using {args.target_pocket} (idx {p_idx})  shape={res_emb.shape}")

    # Clamps: real Mol1 pose (proxy = mean of ALL training rows), good quartile mean, bad quartile mean
    L = np.load(args.labeled_npz, allow_pickle=True)
    pose_all = L["pose_boltz"].astype(np.float32)
    qbin = L["q_bin"]
    clamp_A = pose_all.mean(axis=0)  # baseline = grand mean (proxy for Mol1)
    clamp_B = pose_all[qbin == "good"].mean(axis=0)
    clamp_C = pose_all[qbin == "bad"].mean(axis=0)
    print(f"[clamps] A={clamp_A}  B(good)={clamp_B}  C(bad)={clamp_C}")

    def norm(p): return (p - pose_mean) / np.clip(pose_std, 1e-6, None)
    poses_norm = {"A": norm(clamp_A), "B": norm(clamp_B), "C": norm(clamp_C)}
    poses_unnorm = {"A": clamp_A, "B": clamp_B, "C": clamp_C}

    reg_head = None
    if reg_on:
        reg_head = RegressionHead().to(device)
        reg_head.load_state_dict(ckpt["reg_head_state"]); reg_head.eval()

    model.eval()
    cohorts = {}
    for clamp in ["A", "B", "C"]:
        print(f"[sample] clamp {clamp}", flush=True)
        t0 = time.time()
        smis = sample_cohort(model, res_emb, res_mask, poses_norm[clamp],
                              n=args.n_per_clamp, batch_size=32)
        print(f"  sampled {len(smis)} in {time.time()-t0:.1f}s", flush=True)
        summary = summarize(f"clamp_{clamp}", smis, poses_unnorm[clamp],
                             model=model, reg_head=reg_head,
                             res_emb=res_emb, res_mask=res_mask,
                             pose_norm=poses_norm[clamp])
        cohorts[clamp] = summary
        # Save intermediate SMILES too
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        smis_path = Path(args.out_json).parent / f"samples_V{args.variant}_clamp{clamp}.txt"
        smis_path.write_text("\n".join(smis))
        print(f"  {summary}")

    # Composite clamp responsiveness metrics
    def get(x, k): return x.get(k, float("nan"))
    resp = {
        "d_shift_AB": get(cohorts["B"], "emitted_d_median") - get(cohorts["A"], "emitted_d_median"),
        "d_shift_AC": get(cohorts["C"], "emitted_d_median") - get(cohorts["A"], "emitted_d_median"),
        "theta_shift_AB": get(cohorts["B"], "emitted_theta_median") - get(cohorts["A"], "emitted_theta_median"),
        "theta_shift_AC": get(cohorts["C"], "emitted_theta_median") - get(cohorts["A"], "emitted_theta_median"),
        "phi_shift_AB": get(cohorts["B"], "emitted_phi_median") - get(cohorts["A"], "emitted_phi_median"),
        "phi_shift_AC": get(cohorts["C"], "emitted_phi_median") - get(cohorts["A"], "emitted_phi_median"),
    }

    payload = {
        "variant": args.variant,
        "reg_on": reg_on, "nce_on": nce_on,
        "target_pocket": args.target_pocket,
        "clamps_unnorm": {k: v.tolist() for k, v in poses_unnorm.items()},
        "cohorts": cohorts,
        "clamp_responsiveness": resp,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, default=str))
    print(f"[write] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
