"""Phase 3 evaluation: InfoNCE checkpoint, with CFG sampling.

Same eval panel as Scheme A/B (so results are directly comparable):
    Eval 1: 10K Mol1-anchored samples, real ZAP70 pocket + real Mol1 pose,
            w=1 (no CFG). Metrics: validity, acryl, planar median, Tc-to-Mol1.
    Eval 2: 5 clamps A/B/C/D/E × 1K each × 3 CFG scales (w=1, w=2, w=3).
            Headline: r_d, r_θ, r_φ pooled Spearman per CFG scale (we want
            |r| to GROW as w increases). Pose-swap Wasserstein W(d), W(θ), W(φ)
            at w=1 (direct causal test).
    Eval 3: CFG dose-response. For FIXED pose clamp (Clamp D: θ+90°),
            sample 1K at w=0.5, 1, 2, 3, 4. Plot planar-median vs w. If
            control installed → monotone response.

CFG (Classifier-Free Guidance) sampling here is applied at the LEVEL OF THE
POSE-CONDITIONED MEMORY. For each decoder step we compute:
    logits_cond   = decoder(memory_cond)
    logits_uncond = decoder(memory_uncond)      # pose = NULL embedding
    logits_final  = logits_uncond + w * (logits_cond - logits_uncond)
    prob          = softmax(logits_final / temperature)

At w=1 → equal to standard conditional sampling. At w=0 → unconditional.
w>1 sharpens the conditional. w<0 not used.

The NULL pose embedding comes from the InfoNCE ckpt (extra key
'null_pose_emb'). If the ckpt does not contain it, we default to zeros and
report that CFG behaves like a "hard-zero pose" ablation (still meaningful).
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
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr, wasserstein_distance

RDLogger.DisableLog("rdApp.*")

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
V100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else V100_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
from m1a_v2_model import load_m1a_v2, D_MODEL  # noqa: E402

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


# ============================================================================
# CFG sampling
# ============================================================================
@torch.no_grad()
def sample_with_cfg(model, src, src_mask, residue_emb, residue_mask,
                     pose_norm, null_pose_emb, cfg_scale: float,
                     max_length: int = 128, temperature: float = 1.0):
    """Autoregressive sampling with classifier-free guidance.

    Args:
        model: M1aV2ConditionedModel wrapper
        src, src_mask: encoder input
        residue_emb, residue_mask: pocket
        pose_norm: (B, POSE_DIM) z-scored pose
        null_pose_emb: (D_MODEL,) learned NULL pose embedding (or zeros)
        cfg_scale: guidance scale w. w=1 == standard conditional sampling.
                    w=0 == unconditional. w>1 sharpens conditional.
    """
    from torch.autograd import Variable
    from reinvent.models.transformer.core.network.module.subsequent_mask import (
        subsequent_mask)

    base = model.base
    device = model.device
    B = src.shape[0]

    # Encoded pose (conditional).
    pose_vec_cond = model.pose_enc(pose_norm)  # (B, D_MODEL)
    # Unconditional pose vector = broadcast NULL embedding.
    null_bt = null_pose_emb.unsqueeze(0).expand(B, -1).to(device)

    # Build conditional and unconditional memory (share the encoder pass).
    memory = base.network.encoder(base.network.src_embed(src), src_mask)
    pocket_vec = model.pocket_enc(residue_emb, residue_mask)

    def build_ext(pose_vec):
        cond = torch.stack([pocket_vec, pose_vec], dim=1)
        memory_ext = torch.cat([cond, memory], dim=1)
        if src_mask.dim() == 2:
            extra = torch.ones(B, 2, dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=1)
        else:
            extra = torch.ones(B, src_mask.shape[1], 2,
                                dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=2)
        return memory_ext, mask_ext

    mem_c, mask_c = build_ext(pose_vec_cond)
    mem_u, mask_u = build_ext(null_bt)

    ys = torch.ones(1, device=device).repeat(B, 1).long()
    break_cond = torch.zeros(B, dtype=torch.bool, device=device)
    nlls = torch.zeros(B, device=device)
    end_token = base.vocabulary["$"]

    for i in range(max_length - 1):
        sub_mask = Variable(subsequent_mask(ys.size(1)).type_as(src))
        # Conditional forward
        out_c = base.network.decode(mem_c, mask_c, Variable(ys), sub_mask)
        logits_c = base.network.generator.proj(out_c[:, -1])
        # Unconditional forward (uses null pose in memory)
        out_u = base.network.decode(mem_u, mask_u, Variable(ys), sub_mask)
        logits_u = base.network.generator.proj(out_u[:, -1])

        # CFG-guided logits
        logits_g = logits_u + cfg_scale * (logits_c - logits_u)

        # Convert to log-prob at the model temperature (matches the standard
        # sampler's post-softmax behaviour).
        log_prob = F.log_softmax(logits_g / base.temperature, dim=-1)
        prob = torch.exp(log_prob)
        mask_prop = base.mask_property_tokens(B)
        prob = prob.masked_fill(mask_prop, 0)
        # Guard against numerical zero-row (all mass zero → renormalize).
        prob = prob + 1e-12
        prob = prob / prob.sum(dim=-1, keepdim=True)
        next_word = torch.multinomial(prob, 1)
        break_t = torch.unsqueeze(break_cond, 1)
        next_word = next_word.masked_fill(break_t, 0)
        ys = torch.cat([ys, next_word], dim=1)
        nw = next_word.reshape(-1)
        nlls += base._nll_loss(log_prob, nw)
        break_cond = break_cond | (nw == end_token)
        if bool(break_cond.all()):
            break

    from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
    tok = SMILESTokenizer()
    out_smiles = [tok.untokenize(base.vocabulary.decode(seq))
                   for seq in ys.detach().cpu().numpy()]
    return out_smiles, nlls.detach().cpu().numpy()


def sample_cohort(model, anchor_smi, res_emb_np, res_mask_np, pose_norm_np,
                   null_pose_emb, n, batch_size, max_length, temperature,
                   cfg_scale, tag=""):
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
        out_smiles, nlls = sample_with_cfg(
            model, src_t, src_mask, res_emb, res_mask, pose,
            null_pose_emb=null_pose_emb, cfg_scale=cfg_scale,
            max_length=max_length, temperature=temperature)
        for s, nll, anchor in zip(out_smiles, nlls, anchors):
            out_rows.append({"SMILES": s, "Input_SMILES": anchor,
                              "NLL": float(nll)})
        n_done += batch_n
        if n_done % (batch_size * 10) == 0 or n_done == n:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n - n_done) / max(rate, 1e-6)
            print(f"    [{tag} w={cfg_scale}] sampled {n_done}/{n}  "
                   f"{rate:.0f} mol/s  ETA {eta:.0f}s", flush=True)
    return out_rows


# ---- Emitted-pose measurement (RDKit ETKDG single conformer) ----
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


# ============================================================================
# main
# ============================================================================
def load_infonce_ckpt(prior, device, ckpt_path):
    """Load the base wrapper from the InfoNCE ckpt.

    The InfoNCE checkpoint stores both:
      - "base_wrapper_state": the M1aV2ConditionedModel state_dict (what eval
        uses at inference)
      - "null_pose_emb":       numpy (D_MODEL,) — the learned NULL pose vector
      - "model_state":          full outer InfoNCEWrapper state dict

    We prefer base_wrapper_state (guaranteed strict load). If absent, fall
    back to the outer model_state and strip the "wrapper." prefix.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "base_wrapper_state" in ckpt:
        base_state = ckpt["base_wrapper_state"]
        strip_prefix = None
    elif "model_state" in ckpt:
        # Might be either a saved InfoNCEWrapper OR a M1aV2 wrapper.
        raw = ckpt["model_state"]
        # If keys have wrapper. prefix, strip.
        if any(k.startswith("wrapper.") for k in raw.keys()):
            base_state = {k[len("wrapper."):]: v for k, v in raw.items()
                           if k.startswith("wrapper.")}
            strip_prefix = "wrapper."
        else:
            base_state = raw
            strip_prefix = None
    else:
        base_state = ckpt
        strip_prefix = None

    # Recover pose_mean/pose_std from ckpt buffers, else fall back to cache.
    pose_mean = base_state.get("pose_mean", None)
    pose_std = base_state.get("pose_std", None)
    if pose_mean is None or pose_std is None:
        print("[eval] WARN: pose_mean/pose_std not in ckpt; will load from cache",
               flush=True)
    if isinstance(pose_mean, torch.Tensor):
        pose_mean = pose_mean.cpu().numpy()
    if isinstance(pose_std, torch.Tensor):
        pose_std = pose_std.cpu().numpy()

    model = load_m1a_v2(prior, device,
                         pose_mean=pose_mean, pose_std=pose_std)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"[eval] Loaded ckpt (strip={strip_prefix}). "
           f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"    (first 5 missing: {missing[:5]})", flush=True)
    if unexpected:
        print(f"    (first 5 unexpected: {unexpected[:5]})", flush=True)

    # NULL pose embedding.
    if "null_pose_emb" in ckpt:
        npe = ckpt["null_pose_emb"]
        if isinstance(npe, np.ndarray):
            npe_t = torch.from_numpy(npe).float()
        else:
            npe_t = torch.as_tensor(npe).float()
    else:
        print("[eval] WARN: null_pose_emb not in ckpt; defaulting to zeros. "
               "CFG unconditional branch will be hard-zero pose.", flush=True)
        npe_t = torch.zeros(D_MODEL)
    npe_t = npe_t.to(device)
    return model, npe_t


def pool_spearman(clamp_summaries, exclude_keys=("pose_swap",)):
    """Pooled Spearman across all included clamps for each axis."""
    def pool(axis):
        xs, ys = [], []
        for name, summ in clamp_summaries.items():
            if name in exclude_keys:
                continue
            if axis == "d":     clamped = summ["clamp_input_unnorm"]["d_b_nuc"]
            elif axis == "theta": clamped = summ["clamp_input_unnorm"]["bd_angle_deg"]
            else:                clamped = summ["clamp_input_unnorm"]["planar_dihedral_deg"]
            raw = summ["_raw_emitted"][axis]
            for v in raw:
                if np.isfinite(v):
                    xs.append(clamped); ys.append(float(v))
        if len(xs) < 3:
            return None, None, len(xs)
        r, p = spearmanr(xs, ys)
        return float(r), float(p), len(xs)
    r_d, p_d, n_d = pool("d")
    r_t, p_t, n_t = pool("theta")
    r_p, p_p, n_p = pool("phi")
    return {
        "r_d": r_d, "p_d": p_d, "n_d": n_d,
        "r_theta": r_t, "p_theta": p_t, "n_theta": n_t,
        "r_phi": r_p, "p_phi": p_p, "n_phi": n_p,
    }


def wasserstein_pair(summA, summB, axis):
    A = np.array(summA["_raw_emitted"][axis]);  A = A[np.isfinite(A)]
    B = np.array(summB["_raw_emitted"][axis]);  B = B[np.isfinite(B)]
    if len(A) == 0 or len(B) == 0:
        return None
    return float(wasserstein_distance(A, B))


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
    ap.add_argument("--n_eval3", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--planar_sample", type=int, default=1000)
    ap.add_argument("--cfg_scales_eval2", type=str, default="1,2,3",
                    help="CFG scales for Eval 2 pose clamps.")
    ap.add_argument("--cfg_scales_eval3", type=str, default="0.5,1,2,3,4",
                    help="CFG scales for Eval 3 dose-response.")
    ap.add_argument("--eval3_clamp", type=str, default="D",
                    help="Which clamp to sweep w over. A/B/C/D/E.")
    ap.add_argument("--skip_eval1", action="store_true")
    ap.add_argument("--skip_eval2", action="store_true")
    ap.add_argument("--skip_eval3", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model.
    print(f"[eval] Loading ckpt: {args.ckpt}", flush=True)
    model, null_pose_emb = load_infonce_ckpt(args.prior, device, args.ckpt)
    model.eval()
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()
    print(f"[eval] pose normalizer: mean={pose_mean} std={pose_std}", flush=True)
    print(f"[eval] null_pose_emb norm = {float(null_pose_emb.norm()):.4f}",
           flush=True)

    # Real Mol1 pose + ZAP70 pocket
    mp = np.load(args.mol1_pose_npz, allow_pickle=True)
    mol1_pose_unnorm = mp["pose_unnorm"].astype(np.float32)
    mol1_pose_norm = mp["pose_norm"].astype(np.float32)
    print(f"[eval] Mol1 pose (unnorm): {mol1_pose_unnorm}", flush=True)

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
    print(f"[eval] ZAP70 pocket: {int(zap70_mask.sum())} real residues "
           f"(R_max={R_max})", flush=True)

    def norm(p): return (p - pose_mean) / np.maximum(pose_std, 1e-6)

    # Pose-swap donor (same-pocket, largest z-dist from Mol1)
    sources = cache["sources"]
    row_seq_idx = cache["row_seq_idx"]
    poses_z = cache["poses"].astype(np.float32)
    poses_uz = cache["poses_unnorm"].astype(np.float32)
    smiles = cache["smiles"]
    mask_bz = sources == "boltz_zap70"
    rows_bz = np.where(mask_bz)[0]
    if len(rows_bz) > 0:
        from collections import Counter
        pc = Counter(row_seq_idx[rows_bz].tolist())
        top_pocket = pc.most_common(1)[0][0]
        top_rows = [i for i in rows_bz if row_seq_idx[i] == top_pocket]
        mol1_z = mol1_pose_norm
        dists = [float(np.linalg.norm(poses_z[i] - mol1_z)) for i in top_rows]
        donor_idx = int(top_rows[int(np.argmax(dists))])
        donor_pose_norm = poses_z[donor_idx].astype(np.float32)
        donor_pose_unnorm = poses_uz[donor_idx].astype(np.float32)
        donor_smi = str(smiles[donor_idx])
    else:
        print("[eval] WARN: no boltz_zap70 rows in cache; using max-L2 pose in "
               "the WHOLE corpus for pose_swap.", flush=True)
        dists = np.linalg.norm(poses_z - mol1_pose_norm[None], axis=1)
        donor_idx = int(np.argmax(dists))
        donor_pose_norm = poses_z[donor_idx].astype(np.float32)
        donor_pose_unnorm = poses_uz[donor_idx].astype(np.float32)
        donor_smi = str(smiles[donor_idx])
    print(f"[eval] pose_swap donor row={donor_idx} unnorm={donor_pose_unnorm}",
           flush=True)

    # Clamp definitions in UNNORMALIZED space, then normalized.
    sig_d = float(pose_std[0])
    clamps = {}
    clamps["A"] = {"unnorm": mol1_pose_unnorm.copy(), "label": "real Mol1 pose"}
    clamps["B"] = {"unnorm": mol1_pose_unnorm.copy(), "label": "d_b_nuc +2σ_d"}
    clamps["B"]["unnorm"][0] = mol1_pose_unnorm[0] + 2.0 * sig_d
    clamps["C"] = {"unnorm": mol1_pose_unnorm.copy(), "label": "d_b_nuc -1σ_d"}
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
        print(f"  {k}: {c['label']} unnorm={c['unnorm']} norm={c['norm']}",
               flush=True)

    cfg_scales_eval2 = [float(x) for x in args.cfg_scales_eval2.split(",")]
    cfg_scales_eval3 = [float(x) for x in args.cfg_scales_eval3.split(",")]

    # ---- Eval 1 (10K standard, w=1) ----
    if not args.skip_eval1:
        print("\n=== EVAL 1: 10K, real ZAP70 pocket + real Mol1 pose, w=1 ===",
               flush=True)
        rows = sample_cohort(
            model, args.anchor_smi, zap70_emb, zap70_mask, mol1_pose_norm,
            null_pose_emb=null_pose_emb, n=args.n_eval1,
            batch_size=args.batch_size, max_length=args.max_length,
            temperature=args.temperature, cfg_scale=1.0, tag="eval1")
        for r in rows: r["cohort"] = "eval1"; r["cfg_scale"] = 1.0
        eval1_csv = out_dir / "samples_infonce_10k.csv"
        pd.DataFrame(rows).to_csv(eval1_csv, index=False)
        print(f"Wrote {eval1_csv}", flush=True)
        summ = summarize_cohort("eval1", rows, mol1_pose_unnorm,
                                 planar_sample_n=args.planar_sample)
        summ_clean = {k: v for k, v in summ.items() if k != "_raw_emitted"}
        (out_dir / "report_infonce_eval1.json").write_text(
            json.dumps(summ_clean, indent=2))
        print(json.dumps(summ_clean, indent=2), flush=True)

    # ---- Eval 2 (5 clamps + pose_swap × 1K, × CFG scales) ----
    eval2_report = {"cfg_scale_headlines": {}}
    if not args.skip_eval2:
        print(f"\n=== EVAL 2: 5 clamps + pose_swap × 1K × CFG "
               f"{cfg_scales_eval2} ===", flush=True)
        for w in cfg_scales_eval2:
            print(f"\n--- CFG w={w} ---", flush=True)
            clamp_summaries = {}
            for name, c in clamps.items():
                print(f"-- Clamp {name}: {c['label']} @ w={w} --", flush=True)
                rows = sample_cohort(
                    model, args.anchor_smi, zap70_emb, zap70_mask, c["norm"],
                    null_pose_emb=null_pose_emb, n=args.n_eval2,
                    batch_size=args.batch_size, max_length=args.max_length,
                    temperature=args.temperature, cfg_scale=w,
                    tag=f"clamp{name}")
                for r in rows:
                    r["cohort"] = f"clamp{name}"; r["cfg_scale"] = w
                csv_path = (out_dir /
                             f"samples_infonce_clamp{name}_w{w}_1k.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                summ = summarize_cohort(
                    f"clamp{name}_w{w}", rows, c["unnorm"],
                    planar_sample_n=args.planar_sample)
                summ["clamp_label"] = c["label"]
                summ["cfg_scale"] = w
                if "donor_smi" in c:
                    summ["donor_smi"] = c["donor_smi"]
                    summ["donor_row"] = c["donor_row"]
                clamp_summaries[name] = summ

            headline = pool_spearman(clamp_summaries)
            print(f"[eval2 w={w}] Pooled Spearman (A/B/C/D/E):", flush=True)
            print(f"  r_d = {headline['r_d']}  (n={headline['n_d']}, "
                   f"p={headline['p_d']})", flush=True)
            print(f"  r_θ = {headline['r_theta']}  (n={headline['n_theta']}, "
                   f"p={headline['p_theta']})", flush=True)
            print(f"  r_φ = {headline['r_phi']}  (n={headline['n_phi']}, "
                   f"p={headline['p_phi']})", flush=True)

            wd = wasserstein_pair(clamp_summaries["A"],
                                     clamp_summaries["pose_swap"], "d")
            wt = wasserstein_pair(clamp_summaries["A"],
                                     clamp_summaries["pose_swap"], "theta")
            wp = wasserstein_pair(clamp_summaries["A"],
                                     clamp_summaries["pose_swap"], "phi")
            print(f"[eval2 w={w}] Pose-swap divergence W (A vs pose_swap):",
                   flush=True)
            print(f"  W(d)     = {wd}", flush=True)
            print(f"  W(theta) = {wt}", flush=True)
            print(f"  W(phi)   = {wp}", flush=True)

            # Strip raw arrays before recording; keep summaries clean.
            saved = {}
            for name in clamp_summaries:
                s = {k: v for k, v in clamp_summaries[name].items()
                      if k != "_raw_emitted"}
                saved[name] = s
            eval2_report["cfg_scale_headlines"][str(w)] = {
                "headline_spearman": headline,
                "pose_swap_wasserstein": {
                    "wasserstein_d": wd, "wasserstein_theta": wt,
                    "wasserstein_phi": wp,
                },
                "clamp_summaries": saved,
            }
        (out_dir / "report_infonce_eval2.json").write_text(
            json.dumps(eval2_report, indent=2))
        print(f"Wrote {out_dir / 'report_infonce_eval2.json'}", flush=True)

    # ---- Eval 3 (CFG dose-response for a fixed clamp) ----
    if not args.skip_eval3:
        clamp = clamps[args.eval3_clamp]
        print(f"\n=== EVAL 3: CFG dose-response on Clamp {args.eval3_clamp} "
               f"({clamp['label']}); w in {cfg_scales_eval3} ===", flush=True)
        eval3 = {"clamp": args.eval3_clamp, "label": clamp["label"],
                  "cfg_scales": cfg_scales_eval3,
                  "per_w": {}}
        for w in cfg_scales_eval3:
            print(f"-- Eval3 clamp{args.eval3_clamp} w={w} --", flush=True)
            rows = sample_cohort(
                model, args.anchor_smi, zap70_emb, zap70_mask,
                clamp["norm"], null_pose_emb=null_pose_emb,
                n=args.n_eval3, batch_size=args.batch_size,
                max_length=args.max_length, temperature=args.temperature,
                cfg_scale=w, tag=f"eval3_w{w}")
            for r in rows:
                r["cohort"] = f"eval3_clamp{args.eval3_clamp}"
                r["cfg_scale"] = w
            csv_path = (out_dir /
                         f"samples_infonce_eval3_clamp{args.eval3_clamp}"
                         f"_w{w}.csv")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            summ = summarize_cohort(
                f"eval3_clamp{args.eval3_clamp}_w{w}", rows, clamp["unnorm"],
                planar_sample_n=args.planar_sample)
            summ["clamp_label"] = clamp["label"]
            summ["cfg_scale"] = w
            summ_clean = {k: v for k, v in summ.items() if k != "_raw_emitted"}
            eval3["per_w"][str(w)] = summ_clean
        # dose-response tuples (planar median vs w)
        eval3["dose_response_planar_median"] = {
            str(w): eval3["per_w"][str(w)]["planar_dihedral_median_deg"]
            for w in cfg_scales_eval3
        }
        eval3["dose_response_d_median"] = {
            str(w): eval3["per_w"][str(w)]["emitted_d_median"]
            for w in cfg_scales_eval3
        }
        eval3["dose_response_theta_median"] = {
            str(w): eval3["per_w"][str(w)]["emitted_theta_median"]
            for w in cfg_scales_eval3
        }
        (out_dir / "report_infonce_eval3_cfg_dose.json").write_text(
            json.dumps(eval3, indent=2))
        print(f"Wrote {out_dir / 'report_infonce_eval3_cfg_dose.json'}",
               flush=True)
        print("\nDose-response (planar median vs w):", flush=True)
        for w in cfg_scales_eval3:
            print(f"  w={w}: planar_median = "
                   f"{eval3['dose_response_planar_median'][str(w)]}",
                   flush=True)

    print("\n=== Phase 3 InfoNCE evaluation complete ===", flush=True)


if __name__ == "__main__":
    main()
