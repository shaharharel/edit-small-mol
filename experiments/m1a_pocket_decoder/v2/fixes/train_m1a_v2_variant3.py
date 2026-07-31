"""Variant 3: Joint DAP + DPO fine-tune of M1a v2-cond on ZAP70 pocket + Mol1 pose.

Combines two loss terms:
  L(θ) = L_DPO(θ; s+, s-) + λ * L_DAP(θ; s, R(s, m0))

- DPO: pulls π_θ toward preferred (Boltz-scored) covalent-mol samples over less-preferred
       within-anchor (§2.5). β=0.1. Both s+ and s- come from the same anchor (v2 or covft).
- DAP: same as Variant 1 — anchored composite reward (pIC50 + acrylamide SMARTS + QED),
       geometric mean R^{0.5} * R_wh^{0.4} * R_QED^{0.1}, sigma-scaled likelihood target.

Preference pair construction (from data/m1a_v2_vs_covft/per_mol_scored.csv):
  q(s) = 0.4 * σ((iptm-0.78)/0.05) + 0.3 * σ((2.0-mPAE)/1.0) + 0.3 * exp(-(θ_BD-105)^2 / (2*8^2))
  Only include rows with cofold_attempted=True & scored=True & finite iptm/mPAE/bd_angle.
  Pair up within the same cohort (v2 or covft), keeping pairs with q(s+)-q(s-) >= 0.15.
  Anchor prompt for DPO likelihood: Mol1 SMILES (same as DAP anchor).

Freeze pocket_enc + pose_enc + base.encoder + src_embed; train decoder only.

Usage:
  python train_m1a_v2_variant3.py \
    --prior models/reinvent4_mol2mol_covalent_ft.prior \
    --base_ckpt models/m1a_v2.ckpt \
    --out_ckpt models/m1a_v2_variant3.ckpt \
    --preferences data/m1a_v2_vs_covft/per_mol_scored.csv \
    --max_steps 50 --batch_size 16 --dpo_batch_size 8 --sigma 128 --beta 0.1 --lam 0.5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, QED

RDLogger.DisableLog("rdApp.*")

from m1a_v2_model import load_m1a_v2  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP  # noqa: E402

# ---- reuse pieces from Variant 1 -----
from train_m1a_v2_dap import (  # noqa: E402
    MOL1_SMI,
    acryl_on_largest,
    qed_score,
    morgan_fp_np,
    load_film_predictor,
    score_pIC50_batch,
    sigmoid_transform,
    compose_reward,
    randomize_smi,
    sample_and_score,
    all_parameters,
    all_named_parameters,
    freeze_encoders,
)


# --------------------------------------------------------------------------------------
# Preference-dataset builder (§2.5 composite pose quality)
# --------------------------------------------------------------------------------------

def _sigmoid_np(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_pose_quality(iptm: float, mpae: float, bd_angle: float) -> float:
    """§2.5 composite pose-quality function q(s) in [0,1].

    q = 0.4*σ((iptm-0.78)/0.05)
      + 0.3*σ((2.0-mPAE)/1.0)
      + 0.3*exp(-(θ_BD-105)^2 / (2*8^2))
    """
    if not (np.isfinite(iptm) and np.isfinite(mpae) and np.isfinite(bd_angle)):
        return float("nan")
    iptm_term = _sigmoid_np((iptm - 0.78) / 0.05)
    mpae_term = _sigmoid_np((2.0 - mpae) / 1.0)
    bd_term = np.exp(-((bd_angle - 105.0) ** 2) / (2.0 * (8.0 ** 2)))
    return float(0.4 * iptm_term + 0.3 * mpae_term + 0.3 * bd_term)


def build_preference_pairs(csv_path: Path, margin: float = 0.15,
                            max_pairs_per_cohort: int = 5000,
                            seed: int = 0):
    """Build preference pairs (s+, s-) from per_mol_scored.csv.

    Only within-cohort pairs (v2-v2 or covft-covft) with q(s+) - q(s-) >= margin.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["scored"] == True) & (df["cofold_attempted"] == True)].copy()
    # Use mPAE_paper if finite, else mPAE_london (fallback)
    df["mpae_use"] = np.where(np.isfinite(df["mPAE_paper"]),
                              df["mPAE_paper"], df["mPAE_london"])
    df["bd_use"] = df["bd_angle_deg"].astype(float)
    df["iptm_use"] = df["iptm"].astype(float)
    df["q"] = df.apply(lambda r: compute_pose_quality(
        r["iptm_use"], r["mpae_use"], r["bd_use"]), axis=1)
    df = df[np.isfinite(df["q"])].copy()

    rng = np.random.default_rng(seed)
    pairs_all = []
    for cohort, sub in df.groupby("cohort"):
        sub = sub.sort_values("q").reset_index(drop=True)
        smis = sub["smiles"].values
        qs = sub["q"].values
        n = len(sub)
        # Vectorized: for each pair (i<j), diff = qs[j]-qs[i] >= margin
        # Enumerate high-q winners against random low-q losers to avoid O(n²)
        winners_mask = qs >= (qs.max() - 0.05)  # top slice
        winner_idx = np.where(winners_mask)[0]
        losers_mask = qs <= (qs.min() + 0.15)
        loser_idx = np.where(losers_mask)[0]
        # Also add middle-quality pairs
        # Full pairs approach: pick m random j's for each i where possible
        pair_cnt = 0
        # First: enumerate all valid pairs (limit)
        for i in range(n):
            if pair_cnt >= max_pairs_per_cohort:
                break
            # winners with q >= qs[i]+margin
            valid_j = np.where(qs >= qs[i] + margin)[0]
            if len(valid_j) == 0:
                continue
            # Sample up to 5 winners per loser to spread coverage
            k = min(5, len(valid_j))
            js = rng.choice(valid_j, size=k, replace=False)
            for j in js:
                pairs_all.append({
                    "chosen": smis[j],
                    "rejected": smis[i],
                    "q_chosen": float(qs[j]),
                    "q_rejected": float(qs[i]),
                    "q_gap": float(qs[j] - qs[i]),
                    "cohort": cohort,
                })
                pair_cnt += 1
                if pair_cnt >= max_pairs_per_cohort:
                    break
    pairs_df = pd.DataFrame(pairs_all)
    return pairs_df, df


# --------------------------------------------------------------------------------------
# DPO log-likelihood for a batch of (chosen, rejected) SMILES on the m1a wrapper
# --------------------------------------------------------------------------------------

def _tokenize_batch(smiles_list, vocab, tokenizer):
    """Return (trg_ids padded tensor, trg_mask 3D, kept_indices)."""
    from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask
    trg_seqs = []
    keep_idx = []
    for i, s in enumerate(smiles_list):
        try:
            toks = tokenizer.tokenize(s)
            ids = vocab.encode(toks)
            trg_seqs.append(np.asarray(ids, dtype=np.int64))
            keep_idx.append(i)
        except Exception:
            continue
    if not trg_seqs:
        return None, None, []
    Tm = max(len(s) for s in trg_seqs)
    trg = np.zeros((len(trg_seqs), Tm), dtype=np.int64)
    for j, s in enumerate(trg_seqs):
        trg[j, :len(s)] = s
    trg_t = torch.from_numpy(trg)
    trg_pad = (trg_t != 0).unsqueeze(-2)
    trg_mask_full = (trg_pad & subsequent_mask(trg_t.size(-1)).type_as(trg_pad))
    trg_mask_t = trg_mask_full[:, :-1, :-1].long()
    return trg_t, trg_mask_t, keep_idx


def _encode_anchor_batch(anchor_smi: str, batch_n: int, vocab, tokenizer,
                         randomize: bool = True):
    """Return (src, src_mask) tensors for anchor prompt repeated batch_n times."""
    anchors = [randomize_smi(anchor_smi) if randomize else anchor_smi
               for _ in range(batch_n)]
    seqs = [np.array(vocab.encode(tokenizer.tokenize(s)), dtype=np.int64)
            for s in anchors]
    L = max(len(s) for s in seqs)
    src = np.zeros((batch_n, L), dtype=np.int64)
    for j, s in enumerate(seqs):
        src[j, :len(s)] = s
    src_t = torch.from_numpy(src)
    src_mask_t = (src_t != 0).unsqueeze(-2).long()
    return src_t, src_mask_t


def dpo_loss(agent, prior, chosen_smis, rejected_smis, anchor_smi,
              res_emb_t, res_mask_t, pose_t, beta: float, device):
    """Compute DPO loss for a batch of (chosen, rejected) preference pairs.

    Loss = -log σ(β * [logπ_θ(chosen) - logπ_ref(chosen) - logπ_θ(rejected) + logπ_ref(rejected)])
    All conditioned on Mol1 anchor + ZAP70 pocket + Mol1 pose (matches DAP conditioning).
    """
    vocab = agent.base.vocabulary
    tok = agent.base.tokenizer

    # Tokenize both sides
    trg_c, trg_mask_c, keep_c = _tokenize_batch(chosen_smis, vocab, tok)
    trg_r, trg_mask_r, keep_r = _tokenize_batch(rejected_smis, vocab, tok)
    # Only keep pairs where BOTH sides tokenize
    both_keep = sorted(set(keep_c) & set(keep_r))
    if len(both_keep) < 2:
        return None, {"n_valid": len(both_keep)}
    # Filter down
    idx_c_map = {orig: j for j, orig in enumerate(keep_c)}
    idx_r_map = {orig: j for j, orig in enumerate(keep_r)}
    sel_c = [idx_c_map[i] for i in both_keep]
    sel_r = [idx_r_map[i] for i in both_keep]
    trg_c = trg_c[sel_c].to(device)
    trg_mask_c = trg_mask_c[sel_c].to(device)
    trg_r = trg_r[sel_r].to(device)
    trg_mask_r = trg_mask_r[sel_r].to(device)
    B = trg_c.size(0)

    # Encode anchor B times
    src, src_mask = _encode_anchor_batch(anchor_smi, B, vocab, tok, randomize=True)
    src = src.to(device); src_mask = src_mask.to(device)

    # Conditioning tensors
    res_emb = res_emb_t.unsqueeze(0).expand(B, -1, -1).contiguous()
    res_mask = res_mask_t.unsqueeze(0).expand(B, -1).contiguous()
    pose = pose_t.unsqueeze(0).expand(B, -1).contiguous()

    # NLL sums (positive; higher = worse likelihood)
    agent_nll_c = agent.likelihood(src, src_mask, trg_c, trg_mask_c,
                                    res_emb, res_mask, pose)
    agent_nll_r = agent.likelihood(src, src_mask, trg_r, trg_mask_r,
                                    res_emb, res_mask, pose)
    with torch.no_grad():
        prior_nll_c = prior.likelihood(src, src_mask, trg_c, trg_mask_c,
                                        res_emb, res_mask, pose)
        prior_nll_r = prior.likelihood(src, src_mask, trg_r, trg_mask_r,
                                        res_emb, res_mask, pose)

    # log π = -NLL. DPO logit:
    # z = β * [(logπ_θ(c) - logπ_ref(c)) - (logπ_θ(r) - logπ_ref(r))]
    #   = β * [(-agent_nll_c + prior_nll_c) - (-agent_nll_r + prior_nll_r)]
    #   = β * [prior_nll_c - agent_nll_c - prior_nll_r + agent_nll_r]
    z = beta * ((prior_nll_c - agent_nll_c) - (prior_nll_r - agent_nll_r))
    loss = -F.logsigmoid(z).mean()
    # Reward-gap (chosen - rejected in agent's implicit reward)
    reward_gap = (prior_nll_c - agent_nll_c - prior_nll_r + agent_nll_r).detach()
    info = {
        "n_valid": B,
        "reward_gap_mean": float(reward_gap.mean().cpu().item()),
        "reward_acc": float((reward_gap > 0).float().mean().cpu().item()),
        "z_mean": float(z.detach().mean().cpu().item()),
    }
    return loss, info


# --------------------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--base_ckpt", default=str(PROJECT_ROOT / "models/m1a_v2.ckpt"))
    ap.add_argument("--out_ckpt", default=str(PROJECT_ROOT / "models/m1a_v2_variant3.ckpt"))
    ap.add_argument("--film_cache", default=str(PROJECT_ROOT / "results/paper_evaluation/reinvent4_film_model.pt"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT / "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT / "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--preferences", default=str(PROJECT_ROOT / "data/m1a_v2_vs_covft/per_mol_scored.csv"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--batch_size", type=int, default=16)           # DAP batch
    ap.add_argument("--dpo_batch_size", type=int, default=8)        # DPO batch
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=128.0)
    ap.add_argument("--beta", type=float, default=0.1)              # DPO temperature
    ap.add_argument("--lam", type=float, default=0.5)               # DAP weight
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dpo_margin", type=float, default=0.15)
    ap.add_argument("--log_json", default=str(PROJECT_ROOT / "results/variant3/train_log.json"))
    ap.add_argument("--pairs_out", default=str(PROJECT_ROOT / "results/variant3/preference_pairs.csv"))
    ap.add_argument("--min_pairs", type=int, default=200)
    ap.add_argument("--acryl_halt_pct", type=float, default=0.50)
    ap.add_argument("--acryl_halt_consec", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v3] device={device}", flush=True)

    # ---- Build preference pairs BEFORE loading heavy models (QA gate) ----
    print(f"[v3] building preference pairs from {args.preferences}", flush=True)
    pairs_df, scored_df = build_preference_pairs(
        Path(args.preferences), margin=args.dpo_margin, seed=args.seed)
    n_pairs = len(pairs_df)
    print(f"[v3] preference pairs formed: n_total={n_pairs}", flush=True)
    print(f"[v3] pairs by cohort: {pairs_df['cohort'].value_counts().to_dict() if n_pairs > 0 else '{}'}",
          flush=True)
    if n_pairs < args.min_pairs:
        print(f"[v3][FATAL] only {n_pairs} pairs (need >= {args.min_pairs})", flush=True)
        sys.exit(2)
    Path(args.pairs_out).parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(args.pairs_out, index=False)
    print(f"[v3] wrote pairs to {args.pairs_out}", flush=True)
    print(f"[v3] q_chosen mean={pairs_df['q_chosen'].mean():.3f} "
          f"q_rejected mean={pairs_df['q_rejected'].mean():.3f} "
          f"gap mean={pairs_df['q_gap'].mean():.3f}", flush=True)

    # ---- Load agent + prior models ----
    print(f"[v3] loading agent from {args.base_ckpt}", flush=True)
    agent = load_m1a_v2(args.prior, device, ckpt_path=args.base_ckpt)
    agent.train()
    print(f"[v3] loading prior (frozen ref) from {args.base_ckpt}", flush=True)
    prior = load_m1a_v2(args.prior, device, ckpt_path=args.base_ckpt)
    prior.eval()
    for p in all_parameters(prior):
        p.requires_grad = False

    trainable_names, frozen_names = freeze_encoders(agent)
    all_agent = all_parameters(agent)
    n_trainable = sum(p.numel() for p in all_agent if p.requires_grad)
    n_frozen = sum(p.numel() for p in all_agent if not p.requires_grad)
    print(f"[v3] trainable params: {n_trainable/1e6:.2f}M   frozen: {n_frozen/1e6:.2f}M",
          flush=True)

    opt = torch.optim.AdamW(
        [p for p in all_agent if p.requires_grad],
        lr=args.lr, weight_decay=1e-2)

    # ---- FiLM predictor for DAP reward ----
    print(f"[v3] loading FiLM predictor: {args.film_cache}", flush=True)
    film_model, scaler, anchor_embs, anchor_pIC50 = load_film_predictor(Path(args.film_cache))
    print(f"[v3] FiLM anchors: {len(anchor_pIC50)}", flush=True)

    # ---- Load Mol1 pose + ZAP70 pocket (same conditioning as DAP) ----
    pose_d = np.load(args.mol1_pose_npz, allow_pickle=True)
    mol1_pose_norm = pose_d["pose_norm"].astype(np.float32)
    pocket_d = np.load(args.mol1_pocket_npz, allow_pickle=True)
    zap70_emb = pocket_d["residues_emb"][0]
    zap70_mask = pocket_d["residues_mask"][0]
    cache = np.load(args.cache, allow_pickle=True)
    R_max = cache["residues_emb"].shape[1]
    if zap70_emb.shape[0] < R_max:
        pad = np.zeros((R_max - zap70_emb.shape[0], zap70_emb.shape[1]), dtype=zap70_emb.dtype)
        zap70_emb = np.concatenate([zap70_emb, pad], axis=0)
        mask_pad = np.zeros((R_max - zap70_mask.shape[0],), dtype=bool)
        zap70_mask = np.concatenate([zap70_mask, mask_pad], axis=0)
    elif zap70_emb.shape[0] > R_max:
        zap70_emb = zap70_emb[:R_max]
        zap70_mask = zap70_mask[:R_max]
    print(f"[v3] ZAP70 pocket residues: {int(zap70_mask.sum())}", flush=True)
    res_emb_t = torch.from_numpy(zap70_emb).to(device)
    res_mask_t = torch.from_numpy(zap70_mask).to(device)
    pose_t = torch.from_numpy(mol1_pose_norm).to(device)

    # ---- Pre-training QA: DAP sample once, verify validity ----
    print("[v3][QA] pre-training smoke: sample 4 SMILES from current agent", flush=True)
    with torch.no_grad():
        qa_batch = sample_and_score(
            agent, args.anchor_smi, res_emb_t, res_mask_t, pose_t,
            batch_size=4, max_length=args.max_length,
            temperature=args.temperature, film_bundle=None)
    if qa_batch is None or len(qa_batch["smiles"]) == 0:
        print("[v3][FATAL] pre-training sampling produced 0 valid SMILES", flush=True)
        sys.exit(3)
    valid_smi = [s for s in qa_batch["smiles"] if Chem.MolFromSmiles(s) is not None]
    print(f"[v3][QA] sampled {len(qa_batch['smiles'])} tokens, {len(valid_smi)} parseable:",
          flush=True)
    for s in qa_batch["smiles"][:4]:
        print(f"        {s}", flush=True)

    # ---- Joint DAP+DPO loop ----
    rng = np.random.default_rng(args.seed)
    logs = []
    t_start = time.time()
    seen_smiles = set()
    lam = float(args.lam)
    acryl_halt_streak = 0

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        # ==== DAP: sample + reward ====
        batch = sample_and_score(
            agent, args.anchor_smi, res_emb_t, res_mask_t, pose_t,
            batch_size=args.batch_size, max_length=args.max_length,
            temperature=args.temperature, film_bundle=None)
        if batch is None or len(batch["smiles"]) == 0:
            print(f"[v3] step {step}: no valid DAP samples, skipping", flush=True)
            continue
        pIC50 = score_pIC50_batch(batch["smiles"], film_model, scaler,
                                   anchor_embs, anchor_pIC50)
        rewards, comps = compose_reward(pIC50, batch["smiles"])
        rewards_t = torch.from_numpy(rewards.astype(np.float32)).to(device)

        agent_nll = agent.likelihood(
            batch["src"], batch["src_mask"], batch["trg"], batch["trg_mask"],
            batch["res_emb"], batch["res_mask"], batch["pose"])
        with torch.no_grad():
            prior_nll = prior.likelihood(
                batch["src"], batch["src_mask"], batch["trg"], batch["trg_mask"],
                batch["res_emb"], batch["res_mask"], batch["pose"])
        # DAP: loss = mean((-agent_nll - (-prior_nll + sigma * R))**2)
        aug_logp = -prior_nll + args.sigma * rewards_t
        L_dap = ((-agent_nll) - aug_logp).pow(2).mean()

        # ==== DPO: sample pair batch from static dataset ====
        pair_idx = rng.choice(len(pairs_df), size=args.dpo_batch_size, replace=False)
        pair_rows = pairs_df.iloc[pair_idx]
        chosen_smis = pair_rows["chosen"].tolist()
        rejected_smis = pair_rows["rejected"].tolist()
        L_dpo_out = dpo_loss(agent, prior, chosen_smis, rejected_smis, args.anchor_smi,
                              res_emb_t, res_mask_t, pose_t, args.beta, device)
        if L_dpo_out[0] is None:
            print(f"[v3] step {step}: DPO batch had 0 valid pairs; running DAP only", flush=True)
            L_dpo = torch.tensor(0.0, device=device)
            dpo_info = {"n_valid": 0, "reward_gap_mean": 0.0, "reward_acc": 0.0, "z_mean": 0.0}
        else:
            L_dpo, dpo_info = L_dpo_out

        # ==== Combined ====
        loss = L_dpo + lam * L_dap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in all_agent if p.requires_grad], 1.0)
        opt.step()

        n_new = sum(1 for s in batch["smiles"] if s not in seen_smiles)
        for s in batch["smiles"]:
            seen_smiles.add(s)
        # Batch acryl retention
        acryl_pct = float(np.mean([1.0 if acryl_on_largest(s) else 0.0
                                    for s in batch["smiles"]]))
        dt = time.time() - t0
        step_log = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            "L_dap": float(L_dap.detach().cpu().item()),
            "L_dpo": float(L_dpo.detach().cpu().item()) if L_dpo_out[0] is not None else 0.0,
            "lam": lam,
            "agent_nll_mean": float(agent_nll.detach().mean().cpu().item()),
            "prior_nll_mean": float(prior_nll.detach().mean().cpu().item()),
            "R_composite_mean": comps["R_composite_mean"],
            "R_composite_max": comps["R_composite_max"],
            "R_pIC50_mean": comps["R_pIC50_mean"],
            "R_wh_mean": comps["R_wh_mean"],
            "R_QED_mean": comps["R_QED_mean"],
            "pIC50_mean": comps["pIC50_mean"],
            "acryl_pct_batch": acryl_pct,
            "dpo_n_valid": dpo_info["n_valid"],
            "dpo_reward_gap_mean": dpo_info["reward_gap_mean"],
            "dpo_reward_acc": dpo_info["reward_acc"],
            "dpo_z_mean": dpo_info["z_mean"],
            "n_valid_in_batch": len(batch["smiles"]),
            "n_new_unique": n_new,
            "step_time_s": dt,
            "cum_unique": len(seen_smiles),
        }
        logs.append(step_log)
        print(f"[v3] step {step:02d}  loss={step_log['loss']:.3f}  "
              f"L_dpo={step_log['L_dpo']:.3f}  L_dap={step_log['L_dap']:.3f}  "
              f"R={comps['R_composite_mean']:.3f}  pIC50={comps['pIC50_mean']:.3f}  "
              f"acryl={acryl_pct:.2f}  dpo_acc={dpo_info['reward_acc']:.2f}  "
              f"dt={dt:.1f}s", flush=True)

        # ---- QA: acryl retention floor ----
        if acryl_pct < args.acryl_halt_pct:
            acryl_halt_streak += 1
        else:
            acryl_halt_streak = 0
        if acryl_halt_streak >= args.acryl_halt_consec:
            print(f"[v3][HALT] acryl retention < {args.acryl_halt_pct} for "
                  f"{acryl_halt_streak} consecutive steps — DPO destabilized the model",
                  flush=True)
            # Save checkpoint anyway for post-mortem
            payload = {
                "model_state": agent.state_dict(),
                "base_network_state": agent.base.network.state_dict(),
                "config": vars(args),
                "steps_completed": len(logs),
                "halted_reason": "acryl_retention_collapse",
            }
            torch.save(payload, args.out_ckpt)
            Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.log_json, "w") as f:
                json.dump({"config": vars(args), "steps": logs,
                            "elapsed_s": time.time() - t_start,
                            "halted": "acryl_retention_collapse"}, f, indent=2)
            print(f"[v3] wrote halted checkpoint to {args.out_ckpt}", flush=True)
            sys.exit(4)

        # Intermediate log
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f:
            json.dump({"config": vars(args), "steps": logs,
                        "elapsed_s": time.time() - t_start}, f, indent=2)

    # ---- Save checkpoint ----
    print(f"[v3] saving checkpoint to {args.out_ckpt}", flush=True)
    payload = {
        "model_state": agent.state_dict(),
        "base_network_state": agent.base.network.state_dict(),
        "config": vars(args),
        "steps_completed": len(logs),
        "final_reward_mean": logs[-1]["R_composite_mean"] if logs else None,
    }
    torch.save(payload, args.out_ckpt)

    with open(args.log_json, "w") as f:
        json.dump({"config": vars(args), "steps": logs,
                    "elapsed_s": time.time() - t_start}, f, indent=2)
    total = time.time() - t_start
    print(f"[v3] DONE. Steps={len(logs)}  total={total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
