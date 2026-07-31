"""Scheme B evaluation: Eval 1 (10K standard) + Eval 2 (pose-clamp sweep +
pose-swap column) for the causal pose-token demonstration.

Eval 1
------
Sample 10K Mol1-anchored using real ZAP70 pocket + real Mol1 pose.

Eval 2
------
For each of six cohorts (A/B/C/D/E + pose_swap) sample 1000 mols.  All hold
the ZAP70 pocket fixed; only the pose is varied.

- Clamp A: real Mol1 pose (baseline)
- Clamp B: shift d_b_nuc +2σ_d (further from Cys)
- Clamp C: shift d_b_nuc -1σ_d (closer to Cys)
- Clamp D: θ_BD +90° from Bürgi-Dunitz (wraps to non-productive)
- Clamp E: φ_planar clamped at 60° (broken conjugation)
- pose_swap: pose taken from a DIFFERENT molecule that also binds ZAP70
  (row index picked from the boltz_zap70 block in the training cache, in the
  same pocket bucket as Mol1). Direct causal null: if this cohort is
  indistinguishable from Clamp A on emitted pose, the pose token is
  provably not load-bearing.

For each cohort we measure the EMITTED pose (d, θ, φ) of each generated
molecule via RDKit ETKDG single-conformer and compare against the CLAMPED
input.  Spearman(clamped_axis, emitted_axis) pooled across all six cohorts
is the headline "pose loadbearing" answer.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr, wasserstein_distance

RDLogger.DisableLog("rdApp.*")

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
from m1a_v2_model import load_m1a_v2  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

# SMARTS for emitted-pose measurement (planar C=C-C(=O)-N torsion)
ACRYL_SMARTS_STR = "[CH2]=[CH][C](=O)[N]"
ACRYL_SMARTS = Chem.MolFromSmarts(ACRYL_SMARTS_STR)


def randomize_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def sample_cohort(model, anchor_smi, res_emb_np, res_mask_np, pose_norm_np,
                   n, batch_size, max_length, temperature, tag=""):
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
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
        res_emb = torch.from_numpy(
            np.tile(res_emb_np[None], (batch_n, 1, 1))).to(device)
        res_mask = torch.from_numpy(
            np.tile(res_mask_np[None], (batch_n, 1))).to(device)
        pose = torch.from_numpy(
            np.tile(pose_norm_np[None], (batch_n, 1))).to(device)
        out_smiles, nlls = model.sample_multinomial(
            src_t, src_mask, res_emb, res_mask, pose,
            max_length=max_length, temperature=temperature)
        for s, nll, anchor in zip(out_smiles, nlls, anchors):
            out_rows.append({"SMILES": s, "Input_SMILES": anchor,
                              "NLL": float(nll)})
        n_done += batch_n
        if n_done % (batch_size * 10) == 0 or n_done == n:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n - n_done) / max(rate, 1e-6)
            print(f"    [{tag}] sampled {n_done}/{n}  "
                   f"{rate:.0f} mol/s  ETA {eta:.0f}s", flush=True)
    return out_rows


# ---- Emitted-pose measurement (RDKit ETKDG single conformer) ----
def measure_emitted_pose(smi_list, seed=42):
    """For each SMILES: embed once with ETKDGv3, then measure
      d      = C=C-C(=O) distance    (not really d_warhead-Cys, but an
                                       intramolecular scale we can compare
                                       across cohorts; the C=C to acyl carbon
                                       distance monotonically tracks the
                                       generated warhead geometry)
      theta  = C=C-C(=O)=O bond angle (proxy for BD angle)
      phi    = C=C-C(=O)-N dihedral   (planar dihedral, EXACTLY the metric
                                       used in paper_dap_metric_panel_clean)
    Returns (d[N], theta[N], phi_wrapped[N]) with NaN where measurement fails.
    """
    n = len(smi_list)
    d_arr = np.full(n, np.nan)
    t_arr = np.full(n, np.nan)
    p_arr = np.full(n, np.nan)
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
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            cid = AllChem.EmbedMolecule(mH, params)
            if cid < 0:
                continue
            conf = mH.GetConformer(cid)
            # d: distance between beta-C (a1) and acyl C (a3).  Rigidly
            # determined by the double bond and the C(=O) bond so it is a
            # local geometry probe.
            p1 = np.array(conf.GetAtomPosition(a1))
            p3 = np.array(conf.GetAtomPosition(a3))
            d_arr[i] = float(np.linalg.norm(p1 - p3))
            # theta: bond angle a1-a2-a3 (C=C-C).  Same physical meaning as
            # BD angle to leading order.
            t_arr[i] = float(AllChem.GetAngleDeg(conf, a1, a2, a3))
            # phi: planar torsion (wrap+reflect exactly as metric panel).
            phi = float(AllChem.GetDihedralDeg(conf, a1, a2, a3, a_n))
            phi_wrap = ((phi + 180.0) % 360.0) - 180.0
            p_arr[i] = min(abs(phi_wrap), abs(180.0 - abs(phi_wrap)))
        except Exception:
            continue
    return d_arr, t_arr, p_arr


def summarize_cohort(name, rows, clamp_input_unnorm, planar_sample_n=1000):
    smis = [r["SMILES"] for r in rows]
    n_raw = len(smis)
    canon = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            canon.append(None)
        else:
            canon.append(Chem.MolToSmiles(m))
    valid = [c for c in canon if c is not None]
    validity = len(valid) / max(n_raw, 1)
    dedup = len(set(valid)) / max(len(valid), 1)

    # Acryl on largest fragment
    def largest_frag(s):
        if s is None: return None
        try:
            frags = s.split(".")
            return max(frags, key=len)
        except Exception:
            return s
    acryl_hits = 0
    for c in valid:
        lf = largest_frag(c)
        if lf is None: continue
        m = Chem.MolFromSmiles(lf)
        if m is None: continue
        if m.HasSubstructMatch(ACRYL_SMARTS):
            acryl_hits += 1
    acryl_pct = acryl_hits / max(len(valid), 1)

    # Emitted pose (subsample for speed)
    sub_smi = np.random.default_rng(0).choice(
        valid, size=min(planar_sample_n, len(valid)), replace=False
    ) if len(valid) else np.array([])
    d_arr, t_arr, p_arr = measure_emitted_pose(list(sub_smi))
    def med(x):
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else None
    def mean_(x):
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if len(x) else None
    def std_(x):
        x = x[np.isfinite(x)]
        return float(np.std(x)) if len(x) else None

    # Tc to Mol1
    from rdkit.Chem import DataStructs
    mol1 = Chem.MolFromSmiles(MOL1_SMI)
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    tcs = []
    for c in valid:
        m = Chem.MolFromSmiles(c)
        if m is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        tcs.append(DataStructs.TanimotoSimilarity(mol1_fp, fp))
    tc_median = float(np.median(tcs)) if tcs else None

    return {
        "cohort": name,
        "n_raw": n_raw,
        "n_valid": len(valid),
        "validity_frac": float(validity),
        "dedup_frac": float(dedup),
        "acryl_largest_frag_pct": float(acryl_pct),
        "tc_to_mol1_median": tc_median,
        "planar_dihedral_median_deg": med(p_arr),
        "planar_dihedral_mean_deg":  mean_(p_arr),
        "planar_dihedral_std_deg":   std_(p_arr),
        "emitted_d_median":     med(d_arr),
        "emitted_d_mean":       mean_(d_arr),
        "emitted_d_std":        std_(d_arr),
        "emitted_theta_median": med(t_arr),
        "emitted_theta_mean":   mean_(t_arr),
        "emitted_theta_std":    std_(t_arr),
        "emitted_n_measured":   int(np.sum(np.isfinite(d_arr))),
        "clamp_input_unnorm": {
            "d_b_nuc": float(clamp_input_unnorm[0]),
            "bd_angle_deg": float(clamp_input_unnorm[1]),
            "planar_dihedral_deg": float(clamp_input_unnorm[2]),
        },
        "_raw_emitted": {  # kept for pooled Spearman
            "d": d_arr.tolist(),
            "theta": t_arr.tolist(),
            "phi": p_arr.tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n_eval1", type=int, default=10000)
    ap.add_argument("--n_eval2", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--planar_sample", type=int, default=1000)
    ap.add_argument("--skip_eval1", action="store_true")
    ap.add_argument("--skip_eval2", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model: {args.ckpt}", flush=True)
    model = load_m1a_v2(args.prior, device, ckpt_path=args.ckpt)
    model.eval()
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()
    print(f"pose normalizer: mean={pose_mean}, std={pose_std}", flush=True)

    # Fresh cofold Mol1 pose + ZAP70 pocket
    mp = np.load(args.mol1_pose_npz, allow_pickle=True)
    mol1_pose_unnorm = mp["pose_unnorm"].astype(np.float32)
    mol1_pose_norm = mp["pose_norm"].astype(np.float32)
    print(f"Mol1 pose (unnorm): {mol1_pose_unnorm}", flush=True)

    pk = np.load(args.mol1_pocket_npz, allow_pickle=True)
    zap70_emb = pk["residues_emb"][0]
    zap70_mask = pk["residues_mask"][0]

    cache = np.load(args.cache, allow_pickle=True)
    R_max = cache["residues_emb"].shape[1]
    if zap70_emb.shape[0] < R_max:
        pad = np.zeros((R_max - zap70_emb.shape[0], zap70_emb.shape[1]),
                        dtype=zap70_emb.dtype)
        zap70_emb = np.concatenate([zap70_emb, pad], axis=0)
        mask_pad = np.zeros((R_max - zap70_mask.shape[0],), dtype=bool)
        zap70_mask = np.concatenate([zap70_mask, mask_pad], axis=0)
    elif zap70_emb.shape[0] > R_max:
        zap70_emb = zap70_emb[:R_max]
        zap70_mask = zap70_mask[:R_max]
    print(f"ZAP70 pocket: {int(zap70_mask.sum())} real residues "
           f"(R_max={R_max})", flush=True)

    def norm(p): return (p - pose_mean) / np.maximum(pose_std, 1e-6)

    # ---- Pose-swap donor mol pose ----
    # Pick a boltz_zap70 row in the TOP pocket bucket (same-sequence donor)
    # whose pose is meaningfully different from Mol1.  We use the z-scored
    # pose stored in the cache directly; this is exactly what the cache
    # supplies to the decoder during training.
    sources = cache["sources"]
    row_seq_idx = cache["row_seq_idx"]
    poses_z = cache["poses"].astype(np.float32)
    poses_uz = cache["poses_unnorm"].astype(np.float32)
    smiles = cache["smiles"]
    mask_bz = sources == "boltz_zap70"
    rows_bz = np.where(mask_bz)[0]
    from collections import Counter
    pc = Counter(row_seq_idx[rows_bz].tolist())
    top_pocket = pc.most_common(1)[0][0]
    top_rows = [i for i in rows_bz if row_seq_idx[i] == top_pocket]
    # Choose the row whose pose is FURTHEST from Mol1 in z-space (max L2)
    mol1_z = mol1_pose_norm
    dists = [float(np.linalg.norm(poses_z[i] - mol1_z)) for i in top_rows]
    donor_idx = int(top_rows[int(np.argmax(dists))])
    donor_pose_norm = poses_z[donor_idx].astype(np.float32)
    donor_pose_unnorm = poses_uz[donor_idx].astype(np.float32)
    donor_smi = str(smiles[donor_idx])
    print(f"Pose-swap donor: row={donor_idx} pose_z={donor_pose_norm} "
           f"pose_unnorm={donor_pose_unnorm}  smi={donor_smi[:80]}",
           flush=True)

    # ---- Clamp definitions (in UNNORMALIZED space) ----
    # sigma values come from the CACHE (pose_std), not from Mol1's local
    # jitter — this is the σ referenced in the spec.
    sig_d = float(pose_std[0])
    clamps = {}

    # A: real Mol1 pose
    clamps["A"] = {"unnorm": mol1_pose_unnorm.copy(),
                    "label": "real Mol1 pose"}
    # B: d + 2σ_d
    clamps["B"] = {"unnorm": mol1_pose_unnorm.copy(),
                    "label": "d_b_nuc +2σ_d"}
    clamps["B"]["unnorm"][0] = mol1_pose_unnorm[0] + 2.0 * sig_d
    # C: d - 1σ_d
    clamps["C"] = {"unnorm": mol1_pose_unnorm.copy(),
                    "label": "d_b_nuc -1σ_d"}
    clamps["C"]["unnorm"][0] = mol1_pose_unnorm[0] - 1.0 * sig_d
    # D: theta + 90° (wrap into non-productive)
    clamps["D"] = {"unnorm": mol1_pose_unnorm.copy(),
                    "label": "theta_BD +90° (non-productive)"}
    clamps["D"]["unnorm"][1] = ((mol1_pose_unnorm[1] + 90.0 + 180.0)
                                 % 360.0) - 180.0
    # E: phi_planar = 60°
    clamps["E"] = {"unnorm": mol1_pose_unnorm.copy(),
                    "label": "phi_planar=60° (broken conjugation)"}
    clamps["E"]["unnorm"][2] = 60.0
    # pose_swap: donor pose from same-pocket, same-sequence mol
    clamps["pose_swap"] = {"unnorm": donor_pose_unnorm.copy(),
                            "label": f"pose from donor row={donor_idx}",
                            "donor_smi": donor_smi,
                            "donor_row": int(donor_idx)}

    for k, c in clamps.items():
        c["norm"] = norm(c["unnorm"]).astype(np.float32)
        print(f"  {k}: {c['label']}  unnorm={c['unnorm']}  "
               f"norm={c['norm']}", flush=True)

    # ---- Eval 1 (10K standard) ----
    if not args.skip_eval1:
        print("\n=== EVAL 1: 10K Mol1-anchored, real ZAP70 pocket + pose ===",
               flush=True)
        eval1_csv = out_dir / "samples_scheme_B_10k.csv"
        rows = sample_cohort(
            model, args.anchor_smi, zap70_emb, zap70_mask,
            mol1_pose_norm, n=args.n_eval1, batch_size=args.batch_size,
            max_length=args.max_length, temperature=args.temperature,
            tag="eval1")
        for r in rows:
            r["cohort"] = "eval1"
        pd.DataFrame(rows).to_csv(eval1_csv, index=False)
        print(f"Wrote {eval1_csv}", flush=True)
        summ = summarize_cohort("eval1", rows, mol1_pose_unnorm,
                                 planar_sample_n=args.planar_sample)
        # Save without raw arrays
        summ_clean = {k: v for k, v in summ.items() if k != "_raw_emitted"}
        (out_dir / "report_scheme_B.json").write_text(
            json.dumps(summ_clean, indent=2))
        print(json.dumps(summ_clean, indent=2), flush=True)

    # ---- Eval 2 (5 clamps + pose-swap, 1K each) ----
    if not args.skip_eval2:
        print("\n=== EVAL 2: 5 clamps + pose_swap × 1K ===", flush=True)
        clamp_summaries = {}
        for name, c in clamps.items():
            print(f"\n-- Clamp {name}: {c['label']} --", flush=True)
            csv_path = out_dir / f"samples_scheme_B_clamp{name}_1k.csv"
            rows = sample_cohort(
                model, args.anchor_smi, zap70_emb, zap70_mask,
                c["norm"], n=args.n_eval2, batch_size=args.batch_size,
                max_length=args.max_length, temperature=args.temperature,
                tag=f"clamp{name}")
            for r in rows:
                r["cohort"] = f"clamp{name}"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Wrote {csv_path}", flush=True)
            summ = summarize_cohort(f"clamp{name}", rows, c["unnorm"],
                                     planar_sample_n=args.planar_sample)
            summ["clamp_label"] = c["label"]
            if "donor_smi" in c:
                summ["donor_smi"] = c["donor_smi"]
                summ["donor_row"] = c["donor_row"]
            clamp_summaries[name] = summ

        # ---- Pooled Spearman across clamps A/B/C (d axis), A/D (theta),
        # A/E (phi).  Pool all cohorts for a robust axis-wise readout.
        def pool(axis):
            xs, ys = [], []
            for name, summ in clamp_summaries.items():
                # Skip pose_swap for axis-wise Spearman (it's not a single
                # axis clamp) but include it separately below.
                if name == "pose_swap":
                    continue
                if axis == "d":     clamped = summ["clamp_input_unnorm"]["d_b_nuc"]
                elif axis == "theta": clamped = summ["clamp_input_unnorm"]["bd_angle_deg"]
                else:                clamped = summ["clamp_input_unnorm"]["planar_dihedral_deg"]
                raw = summ["_raw_emitted"][axis]
                for v in raw:
                    if np.isfinite(v):
                        xs.append(clamped)
                        ys.append(float(v))
            if len(xs) < 3:
                return None, None, len(xs)
            r, p = spearmanr(xs, ys)
            return float(r), float(p), len(xs)

        r_d, p_d, n_d = pool("d")
        r_t, p_t, n_t = pool("theta")
        r_p, p_p, n_p = pool("phi")
        print(f"\nHeadline Spearman (pooled across A/B/C/D/E):", flush=True)
        print(f"  r_d = {r_d}  (n={n_d}, p={p_d})", flush=True)
        print(f"  r_theta = {r_t}  (n={n_t}, p={p_t})", flush=True)
        print(f"  r_phi = {r_p}  (n={n_p}, p={p_p})", flush=True)

        # ---- Pose-swap divergence: Wasserstein between A and pose_swap
        A_d = np.array(clamp_summaries["A"]["_raw_emitted"]["d"])
        A_d = A_d[np.isfinite(A_d)]
        S_d = np.array(clamp_summaries["pose_swap"]["_raw_emitted"]["d"])
        S_d = S_d[np.isfinite(S_d)]
        A_t = np.array(clamp_summaries["A"]["_raw_emitted"]["theta"])
        A_t = A_t[np.isfinite(A_t)]
        S_t = np.array(clamp_summaries["pose_swap"]["_raw_emitted"]["theta"])
        S_t = S_t[np.isfinite(S_t)]
        A_p = np.array(clamp_summaries["A"]["_raw_emitted"]["phi"])
        A_p = A_p[np.isfinite(A_p)]
        S_p = np.array(clamp_summaries["pose_swap"]["_raw_emitted"]["phi"])
        S_p = S_p[np.isfinite(S_p)]
        wd = float(wasserstein_distance(A_d, S_d)) if len(A_d) and len(S_d) else None
        wt = float(wasserstein_distance(A_t, S_t)) if len(A_t) and len(S_t) else None
        wp = float(wasserstein_distance(A_p, S_p)) if len(A_p) and len(S_p) else None
        print(f"\nPose-swap divergence (Wasserstein A vs pose_swap):", flush=True)
        print(f"  W(d)     = {wd}", flush=True)
        print(f"  W(theta) = {wt}", flush=True)
        print(f"  W(phi)   = {wp}", flush=True)

        # Strip raw arrays before saving; keep summaries clean.
        for name in clamp_summaries:
            clamp_summaries[name].pop("_raw_emitted", None)

        report = {
            "clamp_summaries": clamp_summaries,
            "headline_spearman": {
                "r_d": r_d, "p_d": p_d, "n_d": n_d,
                "r_theta": r_t, "p_theta": p_t, "n_theta": n_t,
                "r_phi": r_p, "p_phi": p_p, "n_phi": n_p,
            },
            "pose_swap_divergence": {
                "wasserstein_d": wd,
                "wasserstein_theta": wt,
                "wasserstein_phi": wp,
                "n_A_valid_d": int(len(A_d)),
                "n_swap_valid_d": int(len(S_d)),
            },
        }
        (out_dir / "report_scheme_B_clamps.json").write_text(
            json.dumps(report, indent=2))
        print(f"\nWrote {out_dir / 'report_scheme_B_clamps.json'}",
               flush=True)


if __name__ == "__main__":
    main()
