"""Scheme A evaluation: Eval 1 (10K standard) + Eval 2 (pose-clamp sweep +
pose-swap column) + Eval 3 (pocket-clamp sweep) for the causal pose+pocket
token loadbearingness demonstration.

Eval 1
------
Sample 10K Mol1-anchored using real ZAP70 pocket + real Mol1 pose.

Eval 2 (pose clamps + pose-swap)  — same as Scheme B
--------------------------------
- Clamp A: real Mol1 pose (baseline)
- Clamp B: d_b_nuc +2σ_d
- Clamp C: d_b_nuc -1σ_d
- Clamp D: θ_BD +90° (wrap to non-productive)
- Clamp E: φ_planar clamped at 60° (broken conjugation)
- pose_swap: pose taken from a DIFFERENT molecule that also binds ZAP70
  (row picked from the boltz_zap70 block in the training cache, in the same
  pocket bucket as Mol1). Direct causal null.

Eval 3 (pocket clamps)  — NEW for Scheme A
--------------------------------
- Pocket-clamp A: real ZAP70 pocket (baseline, same as pose-clamp A)
- Pocket-clamp B: permute ESM residue rows of the ZAP70 pocket
  (destroys residue identity while preserving mask/shape/statistics)
- Pocket-clamp C: swap in a different pocket from the corpus
  (a random kinase from covindb_v2 that is NOT the ZAP70 struct_id)

All held-fixed: Mol1 anchor, real Mol1 pose.  Only pocket varies.
If Pocket-clamp B/C are indistinguishable from A, the pocket token is inert.

Metrics
-------
For each cohort: n_valid, validity_frac, dedup_frac, acryl_largest_frag_pct,
tc_to_mol1_median, emitted (d, theta, phi) median/mean/std, planar_dihedral.

For Eval 2: pooled Spearman across clamps A/B/C (d axis), A/D (theta), A/E (phi).
For pose_swap: Wasserstein A vs pose_swap on d, theta, phi.
For Eval 3 (pocket): Wasserstein A vs B and A vs C on d, theta, phi.
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


def measure_emitted_pose(smi_list, seed=42):
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
            p1 = np.array(conf.GetAtomPosition(a1))
            p3 = np.array(conf.GetAtomPosition(a3))
            d_arr[i] = float(np.linalg.norm(p1 - p3))
            t_arr[i] = float(AllChem.GetAngleDeg(conf, a1, a2, a3))
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
        "_raw_emitted": {
            "d": d_arr.tolist(),
            "theta": t_arr.tolist(),
            "phi": p_arr.tolist(),
        },
    }


def wass(a, b):
    a = np.array(a); a = a[np.isfinite(a)]
    b = np.array(b); b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    return float(wasserstein_distance(a, b))


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
    ap.add_argument("--skip_eval3", action="store_true")
    ap.add_argument("--other_pocket_source",
                    default="covindb_v2",
                    help="Substring in `sources` to pick pocket-clamp C from.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print(f"Loading model: {args.ckpt}", flush=True)
    model = load_m1a_v2(args.prior, device, ckpt_path=args.ckpt)
    model.eval()
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()
    print(f"pose normalizer: mean={pose_mean}, std={pose_std}", flush=True)

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
    sources = cache["sources"]
    row_seq_idx = cache["row_seq_idx"]
    poses_z = cache["poses"].astype(np.float32)
    poses_uz = cache["poses_unnorm"].astype(np.float32)
    smiles_c = cache["smiles"]
    mask_bz = sources == "boltz_zap70"
    rows_bz = np.where(mask_bz)[0]
    from collections import Counter
    pc = Counter(row_seq_idx[rows_bz].tolist())
    top_pocket = pc.most_common(1)[0][0]
    top_rows = [i for i in rows_bz if row_seq_idx[i] == top_pocket]
    mol1_z = mol1_pose_norm
    dists = [float(np.linalg.norm(poses_z[i] - mol1_z)) for i in top_rows]
    donor_idx = int(top_rows[int(np.argmax(dists))])
    donor_pose_norm = poses_z[donor_idx].astype(np.float32)
    donor_pose_unnorm = poses_uz[donor_idx].astype(np.float32)
    donor_smi = str(smiles_c[donor_idx])
    print(f"Pose-swap donor: row={donor_idx} pose_z={donor_pose_norm} "
          f"pose_unnorm={donor_pose_unnorm}  smi={donor_smi[:80]}", flush=True)

    # ---- Clamp definitions (pose, unnormalized) ----
    sig_d = float(pose_std[0])
    clamps = {}
    clamps["A"] = {"unnorm": mol1_pose_unnorm.copy(),
                   "label": "real Mol1 pose"}
    clamps["B"] = {"unnorm": mol1_pose_unnorm.copy(),
                   "label": "d_b_nuc +2σ_d"}
    clamps["B"]["unnorm"][0] = mol1_pose_unnorm[0] + 2.0 * sig_d
    clamps["C"] = {"unnorm": mol1_pose_unnorm.copy(),
                   "label": "d_b_nuc -1σ_d"}
    clamps["C"]["unnorm"][0] = mol1_pose_unnorm[0] - 1.0 * sig_d
    clamps["D"] = {"unnorm": mol1_pose_unnorm.copy(),
                   "label": "theta_BD +90° (non-productive)"}
    clamps["D"]["unnorm"][1] = ((mol1_pose_unnorm[1] + 90.0 + 180.0)
                                % 360.0) - 180.0
    clamps["E"] = {"unnorm": mol1_pose_unnorm.copy(),
                   "label": "phi_planar=60° (broken conjugation)"}
    clamps["E"]["unnorm"][2] = 60.0
    clamps["pose_swap"] = {"unnorm": donor_pose_unnorm.copy(),
                           "label": f"pose from donor row={donor_idx}",
                           "donor_smi": donor_smi,
                           "donor_row": int(donor_idx)}
    for k, c in clamps.items():
        c["norm"] = norm(c["unnorm"]).astype(np.float32)
        print(f"  pose-clamp {k}: {c['label']}  unnorm={c['unnorm']}",
              flush=True)

    # ---- Pocket-clamp donor pockets ----
    # A: real ZAP70 pocket (same as clamp A)
    # B: permute rows within pocket
    # C: swap in a different pocket
    pocket_A_emb, pocket_A_mask = zap70_emb.copy(), zap70_mask.copy()
    pocket_B_emb = zap70_emb.copy()
    # Permute ONLY the "real residue" rows; keep padding zeros in place so
    # the padded region stays inert.
    real_idx = np.where(zap70_mask.astype(bool))[0]
    perm = rng.permutation(real_idx)
    pocket_B_emb[real_idx] = zap70_emb[perm]
    pocket_B_mask = zap70_mask.copy()

    # C: pick a different pocket from cache.  Prefer a covind_v2 kinase row.
    src_arr = np.array([str(s) for s in sources])
    mask_other = np.array([args.other_pocket_source in s for s in src_arr])
    mask_other &= ~mask_bz  # not ZAP70
    other_rows = np.where(mask_other)[0]
    if len(other_rows) == 0:
        print(f"  [warn] no source containing '{args.other_pocket_source}'; "
              f"falling back to any non-ZAP70 pocket.", flush=True)
        other_rows = np.where(~mask_bz)[0]
    # Choose one whose pocket bucket has full residues (max sum of mask)
    cache_res_mask = cache["residues_mask"]
    cache_res_emb = cache["residues_emb"]
    best_row = None; best_mass = -1
    n_scan = min(500, len(other_rows))
    rng2 = np.random.default_rng(args.seed + 1)
    scan = rng2.choice(other_rows, size=n_scan, replace=False)
    for i in scan:
        sidx = int(row_seq_idx[int(i)])
        mass = int(cache_res_mask[sidx].sum())
        if mass > best_mass:
            best_mass = mass; best_row = int(i)
    donor_pk_seq = int(row_seq_idx[best_row])
    pocket_C_emb = cache_res_emb[donor_pk_seq].astype(zap70_emb.dtype)
    pocket_C_mask = cache_res_mask[donor_pk_seq].astype(zap70_mask.dtype)
    donor_pk_src = str(sources[best_row])
    print(f"  pocket-clamp C donor: cache_row={best_row}  "
          f"src={donor_pk_src}  residues={best_mass}", flush=True)

    pocket_clamps = {
        "A": {"emb": pocket_A_emb, "mask": pocket_A_mask,
              "label": "real ZAP70 pocket"},
        "B": {"emb": pocket_B_emb, "mask": pocket_B_mask,
              "label": "ZAP70 with permuted residue rows"},
        "C": {"emb": pocket_C_emb, "mask": pocket_C_mask,
              "label": f"different pocket ({donor_pk_src})",
              "donor_row": int(best_row),
              "donor_src": donor_pk_src,
              "donor_seq_idx": int(donor_pk_seq)},
    }

    # =========================================================
    # Eval 1
    # =========================================================
    if not args.skip_eval1:
        print("\n=== EVAL 1: 10K Mol1-anchored, real ZAP70 pocket + pose ===",
              flush=True)
        eval1_csv = out_dir / "samples_scheme_A_10k.csv"
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
        summ_clean = {k: v for k, v in summ.items() if k != "_raw_emitted"}
        (out_dir / "report_scheme_A.json").write_text(
            json.dumps(summ_clean, indent=2))
        print(json.dumps(summ_clean, indent=2), flush=True)

    # =========================================================
    # Eval 2 (pose clamps + pose_swap)
    # =========================================================
    if not args.skip_eval2:
        print("\n=== EVAL 2: 5 pose clamps + pose_swap x 1K ===", flush=True)
        clamp_summaries = {}
        for name, c in clamps.items():
            print(f"\n-- Pose-clamp {name}: {c['label']} --", flush=True)
            csv_path = out_dir / f"samples_scheme_A_clamp{name}_1k.csv"
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

        def pool(axis):
            xs, ys = [], []
            for name, summ in clamp_summaries.items():
                if name == "pose_swap":
                    continue
                if axis == "d":
                    clamped = summ["clamp_input_unnorm"]["d_b_nuc"]
                elif axis == "theta":
                    clamped = summ["clamp_input_unnorm"]["bd_angle_deg"]
                else:
                    clamped = summ["clamp_input_unnorm"][
                        "planar_dihedral_deg"]
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
        print(f"\nHeadline pose-clamp Spearman (pooled A/B/C/D/E):",
              flush=True)
        print(f"  r_d     = {r_d}  (n={n_d}, p={p_d})", flush=True)
        print(f"  r_theta = {r_t}  (n={n_t}, p={p_t})", flush=True)
        print(f"  r_phi   = {r_p}  (n={n_p}, p={p_p})", flush=True)

        # Pose-swap Wasserstein A vs swap on d/theta/phi
        A_raw = clamp_summaries["A"]["_raw_emitted"]
        S_raw = clamp_summaries["pose_swap"]["_raw_emitted"]
        w_d = wass(A_raw["d"], S_raw["d"])
        w_t = wass(A_raw["theta"], S_raw["theta"])
        w_p = wass(A_raw["phi"], S_raw["phi"])
        print(f"\nPose-swap divergence (Wasserstein A vs pose_swap):",
              flush=True)
        print(f"  W(d)     = {w_d}", flush=True)
        print(f"  W(theta) = {w_t}", flush=True)
        print(f"  W(phi)   = {w_p}", flush=True)

        for name in clamp_summaries:
            clamp_summaries[name].pop("_raw_emitted", None)

        report = {
            "clamp_summaries": clamp_summaries,
            "headline_spearman": {
                "r_d": r_d, "p_d": p_d, "n_d": n_d,
                "r_theta": r_t, "p_theta": p_t, "n_theta": n_t,
                "r_phi": r_p, "p_phi": p_p, "n_phi": n_p,
            },
            "pose_swap": {
                "wasserstein_d": w_d,
                "wasserstein_theta": w_t,
                "wasserstein_phi": w_p,
            },
        }
        (out_dir / "report_scheme_A_pose_clamps.json").write_text(
            json.dumps(report, indent=2))
        print(f"\nWrote {out_dir / 'report_scheme_A_pose_clamps.json'}",
              flush=True)

    # =========================================================
    # Eval 3 (pocket clamps)
    # =========================================================
    if not args.skip_eval3:
        print("\n=== EVAL 3: 3 pocket clamps x 1K ===", flush=True)
        pocket_summaries = {}
        for name, c in pocket_clamps.items():
            print(f"\n-- Pocket-clamp {name}: {c['label']} --", flush=True)
            csv_path = out_dir / f"samples_scheme_A_pocket_clamp{name}_1k.csv"
            rows = sample_cohort(
                model, args.anchor_smi, c["emb"], c["mask"],
                mol1_pose_norm, n=args.n_eval2, batch_size=args.batch_size,
                max_length=args.max_length, temperature=args.temperature,
                tag=f"pkclamp{name}")
            for r in rows:
                r["cohort"] = f"pocket_clamp{name}"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Wrote {csv_path}", flush=True)
            summ = summarize_cohort(
                f"pocket_clamp{name}", rows, mol1_pose_unnorm,
                planar_sample_n=args.planar_sample)
            summ["pocket_label"] = c["label"]
            if "donor_src" in c:
                summ["donor_pocket_src"] = c["donor_src"]
                summ["donor_pocket_seq_idx"] = c["donor_seq_idx"]
            pocket_summaries[name] = summ

        # Pocket-clamp divergence: A vs B and A vs C on d,theta,phi
        A_raw = pocket_summaries["A"]["_raw_emitted"]
        B_raw = pocket_summaries["B"]["_raw_emitted"]
        C_raw = pocket_summaries["C"]["_raw_emitted"]
        div = {
            "AB": {
                "wasserstein_d":     wass(A_raw["d"],     B_raw["d"]),
                "wasserstein_theta": wass(A_raw["theta"], B_raw["theta"]),
                "wasserstein_phi":   wass(A_raw["phi"],   B_raw["phi"]),
            },
            "AC": {
                "wasserstein_d":     wass(A_raw["d"],     C_raw["d"]),
                "wasserstein_theta": wass(A_raw["theta"], C_raw["theta"]),
                "wasserstein_phi":   wass(A_raw["phi"],   C_raw["phi"]),
            },
        }
        # Also Tc-to-Mol1 divergence between pockets — chemistry shift under
        # a different pocket is the *chemistry* readout (parallel to pose
        # readout on emitted-pose).
        def tc_list(rows):
            from rdkit.Chem import DataStructs
            mol1 = Chem.MolFromSmiles(MOL1_SMI)
            mol1_fp = AllChem.GetMorganFingerprintAsBitVect(
                mol1, 2, nBits=2048)
            out = []
            for r in rows:
                m = Chem.MolFromSmiles(r["SMILES"])
                if m is None: continue
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    m, 2, nBits=2048)
                out.append(DataStructs.TanimotoSimilarity(mol1_fp, fp))
            return np.array(out)
        # We already recorded medians per summary; add pairwise Wasserstein
        # on the Tc distributions (loaded from CSVs quickly).
        tc_A = tc_list(pd.read_csv(
            out_dir / "samples_scheme_A_pocket_clampA_1k.csv"
        ).to_dict("records"))
        tc_B = tc_list(pd.read_csv(
            out_dir / "samples_scheme_A_pocket_clampB_1k.csv"
        ).to_dict("records"))
        tc_C = tc_list(pd.read_csv(
            out_dir / "samples_scheme_A_pocket_clampC_1k.csv"
        ).to_dict("records"))
        div["AB"]["wasserstein_tc_to_mol1"] = wass(tc_A, tc_B)
        div["AC"]["wasserstein_tc_to_mol1"] = wass(tc_A, tc_C)

        print(f"\nPocket-clamp divergence (Wasserstein):", flush=True)
        for pair in ["AB", "AC"]:
            print(f"  {pair}: d={div[pair]['wasserstein_d']}  "
                  f"theta={div[pair]['wasserstein_theta']}  "
                  f"phi={div[pair]['wasserstein_phi']}  "
                  f"tc_to_mol1={div[pair]['wasserstein_tc_to_mol1']}",
                  flush=True)

        for name in pocket_summaries:
            pocket_summaries[name].pop("_raw_emitted", None)

        rep = {
            "pocket_clamp_summaries": pocket_summaries,
            "pocket_clamp_divergence": div,
        }
        (out_dir / "report_scheme_A_pocket_clamps.json").write_text(
            json.dumps(rep, indent=2))
        print(f"\nWrote {out_dir / 'report_scheme_A_pocket_clamps.json'}",
              flush=True)


if __name__ == "__main__":
    main()
