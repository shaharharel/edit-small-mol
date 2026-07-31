"""DAP RL fine-tune of M1a v2-cond on ZAP70 pocket + Mol1 pose.

Reward:
  R = R_pIC50^0.50 * R_wh^0.40 * R_QED^0.10
  R_pIC50 = sigmoid((pIC50_pred - 6.5) / 0.5) → [0,1] (matches TOML: low=5.5, high=7.5, k=0.5)
  R_wh    = 1.0 if acrylamide SMARTS present on largest fragment else 0.5 (soft floor per §2.3)
  R_QED   = RDKit QED

Loss: DAP augmented log-likelihood
  L = ( -NLL_agent(s) - (-NLL_prior(s) + sigma * R(s)) )^2   averaged over batch

Only decoder cross-attn + output weights update; pocket_enc + pose_enc frozen.
Sampling condition: Mol1 anchor + Mol1 cofold pose + ZAP70 pocket (variant A).

Usage:
  python train_m1a_v2_dap.py \
    --prior /home/shaharh_quris_ai/edit-small-mol/models/reinvent4_mol2mol_covalent_ft.prior \
    --base_ckpt /home/shaharh_quris_ai/edit-small-mol/models/m1a_v2.ckpt \
    --out_ckpt /home/shaharh_quris_ai/edit-small-mol/models/m1a_v2_dap.ckpt \
    --max_steps 50 --batch_size 16 --sigma 128 --lr 1e-4
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, QED

RDLogger.DisableLog("rdApp.*")

from m1a_v2_model import load_m1a_v2, save_m1a_v2  # noqa: E402

# FiLM predictor: reuse the trained + cached model.pt from the scorer.
from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS_PAT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def largest_fragment_smi(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return None
    largest = max(frags, key=lambda mm: mm.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest)


def acryl_on_largest(smi: str) -> bool:
    lf = largest_fragment_smi(smi)
    if lf is None:
        return False
    m = Chem.MolFromSmiles(lf)
    if m is None:
        return False
    return bool(m.GetSubstructMatches(ACRYL_SMARTS_PAT))


def qed_score(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0.0
    try:
        return float(QED.qed(m))
    except Exception:
        return 0.0


def morgan_fp_np(smi: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_film_predictor(cache_path: Path):
    """Load the cached FiLM predictor (reinvent4_film_model.pt)."""
    ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    anchor_embs = ckpt["anchor_embs"]  # already scaled torch tensor
    anchor_pIC50 = ckpt["anchor_pIC50"]
    return model, scaler, anchor_embs, anchor_pIC50


def score_pIC50_batch(smiles_list, film_model, scaler, anchor_embs, anchor_pIC50) -> np.ndarray:
    """Anchor-based FiLMDelta pIC50 predictions. NaN for invalid SMILES."""
    n = len(smiles_list)
    out = np.full(n, np.nan, dtype=np.float64)
    valid_fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        arr = morgan_fp_np(smi)
        if arr is None:
            continue
        valid_fps.append(arr)
        valid_idx.append(i)
    if not valid_fps:
        return out
    fps_arr = np.asarray(valid_fps, dtype=np.float32)
    batch_embs = torch.FloatTensor(scaler.transform(fps_arr))
    n_anch = len(anchor_pIC50)
    with torch.no_grad():
        for k, orig_i in enumerate(valid_idx):
            tgt = batch_embs[k:k+1].expand(n_anch, -1)
            deltas = film_model(anchor_embs, tgt).numpy().flatten()
            out[orig_i] = float(np.mean(anchor_pIC50 + deltas))
    return out


def sigmoid_transform(x: float, low: float = 5.5, high: float = 7.5, k: float = 0.5) -> float:
    """REINVENT4-style sigmoid: x=low→~0, x=high→~1, k controls slope.

    Matches the double-sigmoid formulation used in REINVENT: score = 1/(1 + 10^(k*(mid-x)*10/(high-low)))
    For simplicity here we use standard sigmoid on (x - mid) / half-width.
    """
    if not np.isfinite(x):
        return 0.0
    mid = 0.5 * (low + high)
    hw = 0.5 * (high - low)
    z = (x - mid) / max(hw * k, 1e-6)
    return float(1.0 / (1.0 + np.exp(-z)))


def compose_reward(pIC50_arr: np.ndarray, smiles_list: list[str]) -> tuple[np.ndarray, dict]:
    """Composite reward with per-component tracking."""
    n = len(smiles_list)
    r_pIC = np.zeros(n)
    r_wh = np.zeros(n)
    r_qed = np.zeros(n)
    for i, (smi, p) in enumerate(zip(smiles_list, pIC50_arr)):
        r_pIC[i] = sigmoid_transform(p) if np.isfinite(p) else 0.0
        r_wh[i] = 1.0 if acryl_on_largest(smi) else 0.5
        r_qed[i] = qed_score(smi)
    # geometric-mean-style composite (weight exponents sum to 1.0)
    r = np.power(np.clip(r_pIC, 1e-6, 1), 0.5) * \
        np.power(np.clip(r_wh, 1e-6, 1), 0.4) * \
        np.power(np.clip(r_qed, 1e-6, 1), 0.1)
    comps = {
        "R_pIC50_mean": float(np.mean(r_pIC)),
        "R_wh_mean": float(np.mean(r_wh)),
        "R_QED_mean": float(np.mean(r_qed)),
        "R_composite_mean": float(np.mean(r)),
        "R_composite_max": float(np.max(r)),
        "pIC50_mean": float(np.nanmean(pIC50_arr)) if np.any(np.isfinite(pIC50_arr)) else float("nan"),
    }
    return r, comps


def randomize_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def sample_and_score(model, anchor_smi: str, res_emb_t, res_mask_t, pose_t,
                     batch_size: int, max_length: int, temperature: float,
                     film_bundle):
    """Sample K SMILES from current model, score them.

    Returns:
      out_smiles: list[str] (length batch_size)
      src_t, src_mask_t: encoder inputs (needed for teacher-forced likelihood)
      nlls_sampled: torch tensor (batch_size,) — NLL of the sampled sequences UNDER THE MODEL AT SAMPLING TIME
      trg_padded, trg_mask: decoder target tensors for later likelihood recomputation
    """
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer

    # Encode anchor(s) with randomization for exploration diversity
    anchors = [randomize_smi(anchor_smi) for _ in range(batch_size)]
    seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
    L = max(len(s) for s in seqs)
    src = np.zeros((batch_size, L), dtype=np.int64)
    for j, s in enumerate(seqs):
        src[j, :len(s)] = s
    src_t = torch.from_numpy(src).to(device)
    src_mask_t = (src_t != 0).unsqueeze(-2).long()

    res_emb = res_emb_t.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    res_mask = res_mask_t.unsqueeze(0).expand(batch_size, -1).contiguous()
    pose = pose_t.unsqueeze(0).expand(batch_size, -1).contiguous()

    # sample_multinomial has @torch.no_grad; result gives us out_smiles + NLLs
    out_smiles, sampled_nlls = model.sample_multinomial(
        src_t, src_mask_t, res_emb, res_mask, pose,
        max_length=max_length, temperature=temperature)

    # Now we need the token-level target sequences to run a differentiable likelihood.
    # Re-tokenize the SAMPLED SMILES ourselves — the reinvent decoder starts with token 1 ("^")
    # and ends with "$". If tokenization fails, skip that row.
    trg_seqs = []
    kept_mask = []
    for s in out_smiles:
        try:
            toks = tok.tokenize(s)  # yields ["^", ..., "$"]
            ids = vocab.encode(toks)
            trg_seqs.append(np.asarray(ids, dtype=np.int64))
            kept_mask.append(True)
        except Exception:
            kept_mask.append(False)

    if not trg_seqs:
        return None
    Tm = max(len(s) for s in trg_seqs)
    trg = np.zeros((len(trg_seqs), Tm), dtype=np.int64)
    for j, s in enumerate(trg_seqs):
        trg[j, :len(s)] = s
    trg_t = torch.from_numpy(trg).to(device)
    # trg_mask matches REINVENT4's paired_dataset conventions:
    #   trg_pad = (trg != 0).unsqueeze(-2)   # (B, 1, T)
    #   trg_mask = trg_pad & subsequent_mask(T)   # (B, T, T)
    #   trg_mask = trg_mask[:, :-1, :-1]   # match trg[:, :-1] used in decoder
    from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask
    trg_pad = (trg_t != 0).unsqueeze(-2)  # (B, 1, T)
    trg_mask_full = (trg_pad & subsequent_mask(trg_t.size(-1)).type_as(trg_pad))
    trg_mask_t = trg_mask_full[:, :-1, :-1].long()

    keep_idx = [i for i, k in enumerate(kept_mask) if k]
    src_kept = src_t[keep_idx]
    src_mask_kept = src_mask_t[keep_idx]
    res_emb_kept = res_emb[keep_idx]
    res_mask_kept = res_mask[keep_idx]
    pose_kept = pose[keep_idx]
    out_smiles_kept = [out_smiles[i] for i in keep_idx]

    return {
        "smiles": out_smiles_kept,
        "src": src_kept,
        "src_mask": src_mask_kept,
        "trg": trg_t,
        "trg_mask": trg_mask_t,
        "res_emb": res_emb_kept,
        "res_mask": res_mask_kept,
        "pose": pose_kept,
    }


def all_parameters(model):
    """Return all leaf parameters across the wrapper AND base.network.

    NOTE: Mol2MolModel is NOT an nn.Module — the wrapper only registers pocket_enc /
    pose_enc / buffers. base.network is an nn.Module carrying the transformer weights
    (encoder, decoder, src_embed, tgt_embed, generator). We need to iterate both.
    """
    seen = set()
    out = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p)); out.append(p)
    for p in model.base.network.parameters():
        if id(p) not in seen:
            seen.add(id(p)); out.append(p)
    return out


def all_named_parameters(model):
    seen = set()
    out = []
    for n, p in model.named_parameters():
        if id(p) not in seen:
            seen.add(id(p)); out.append((n, p))
    for n, p in model.base.network.named_parameters(prefix="base.network"):
        if id(p) not in seen:
            seen.add(id(p)); out.append((n, p))
    return out


def freeze_encoders(model):
    """Freeze pocket_enc + pose_enc + base.network.encoder + src_embed. Decoder +
    tgt_embed + generator stay trainable.

    Rationale: freeze the geometry conditioning (pocket_enc, pose_enc) and the base
    encoder + src_embed so anchor interpretation is preserved. Only decoder learns.
    """
    # Enable all
    for p in all_parameters(model):
        p.requires_grad = True
    # Freeze geometry
    for p in model.pocket_enc.parameters():
        p.requires_grad = False
    for p in model.pose_enc.parameters():
        p.requires_grad = False
    # Freeze base encoder + src_embed
    for p in model.base.network.encoder.parameters():
        p.requires_grad = False
    for p in model.base.network.src_embed.parameters():
        p.requires_grad = False
    trainable = [n for n, p in all_named_parameters(model) if p.requires_grad]
    frozen = [n for n, p in all_named_parameters(model) if not p.requires_grad]
    return trainable, frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--base_ckpt", default=str(PROJECT_ROOT / "models/m1a_v2.ckpt"))
    ap.add_argument("--out_ckpt", default=str(PROJECT_ROOT / "models/m1a_v2_dap.ckpt"))
    ap.add_argument("--film_cache", default=str(PROJECT_ROOT / "results/paper_evaluation/reinvent4_film_model.pt"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT / "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT / "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--min_steps", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=128.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--early_stop_reward", type=float, default=0.7,
                    help="Stop when rolling-max composite reward exceeds this.")
    ap.add_argument("--rolling_window", type=int, default=3)
    ap.add_argument("--log_json", default=str(PROJECT_ROOT / "results/m1a_v2_dap_train_log.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dap] device={device}", flush=True)

    # ---- Load agent model (trainable, initialized from base_ckpt) ----
    print(f"[dap] loading agent from base_ckpt: {args.base_ckpt}", flush=True)
    agent = load_m1a_v2(args.prior, device, ckpt_path=args.base_ckpt)
    agent.train()

    # ---- Load prior model (frozen; identical init) ----
    print(f"[dap] loading prior (frozen copy) from base_ckpt", flush=True)
    prior = load_m1a_v2(args.prior, device, ckpt_path=args.base_ckpt)
    prior.eval()
    for p in all_parameters(prior):
        p.requires_grad = False

    # ---- Freeze pocket_enc + pose_enc + encoder + src_embed in AGENT ----
    trainable_names, frozen_names = freeze_encoders(agent)
    all_agent = all_parameters(agent)
    n_trainable = sum(p.numel() for p in all_agent if p.requires_grad)
    n_frozen = sum(p.numel() for p in all_agent if not p.requires_grad)
    print(f"[dap] trainable params: {n_trainable/1e6:.2f}M   frozen: {n_frozen/1e6:.2f}M", flush=True)
    print(f"[dap] trainable names (first 5): {trainable_names[:5]}", flush=True)

    # ---- Optimizer over trainable params only ----
    opt = torch.optim.Adam(
        [p for p in all_agent if p.requires_grad],
        lr=args.lr)

    # ---- Load FiLM predictor ----
    print(f"[dap] loading FiLM predictor: {args.film_cache}", flush=True)
    film_model, scaler, anchor_embs, anchor_pIC50 = load_film_predictor(Path(args.film_cache))
    print(f"[dap] FiLM anchors: {len(anchor_pIC50)}", flush=True)

    # ---- Load Mol1 pose + ZAP70 pocket (variant A conditioning) ----
    pose_d = np.load(args.mol1_pose_npz, allow_pickle=True)
    mol1_pose_norm = pose_d["pose_norm"].astype(np.float32)
    pocket_d = np.load(args.mol1_pocket_npz, allow_pickle=True)
    zap70_emb = pocket_d["residues_emb"][0]
    zap70_mask = pocket_d["residues_mask"][0]
    # Align R_max
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
    print(f"[dap] ZAP70 pocket residues: {int(zap70_mask.sum())}", flush=True)

    res_emb_t = torch.from_numpy(zap70_emb).to(device)
    res_mask_t = torch.from_numpy(zap70_mask).to(device)
    pose_t = torch.from_numpy(mol1_pose_norm).to(device)

    # ---- DAP loop ----
    logs = []
    rolling_max = []
    t_start = time.time()
    seen_smiles = set()

    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        batch = sample_and_score(
            agent, args.anchor_smi, res_emb_t, res_mask_t, pose_t,
            batch_size=args.batch_size, max_length=args.max_length,
            temperature=args.temperature, film_bundle=None)
        if batch is None or len(batch["smiles"]) == 0:
            print(f"[dap] step {step}: no valid samples, skipping", flush=True)
            continue

        # ---- Score reward ----
        pIC50 = score_pIC50_batch(batch["smiles"], film_model, scaler,
                                  anchor_embs, anchor_pIC50)
        rewards, comps = compose_reward(pIC50, batch["smiles"])
        rewards_t = torch.from_numpy(rewards.astype(np.float32)).to(device)

        # ---- Compute agent NLL (grad-enabled) and prior NLL (no grad) ----
        # NB: model.likelihood already returns SUM(NLL) per row (positive), i.e., -log_pi.
        agent_nll = agent.likelihood(
            batch["src"], batch["src_mask"], batch["trg"], batch["trg_mask"],
            batch["res_emb"], batch["res_mask"], batch["pose"])
        with torch.no_grad():
            prior_nll = prior.likelihood(
                batch["src"], batch["src_mask"], batch["trg"], batch["trg_mask"],
                batch["res_emb"], batch["res_mask"], batch["pose"])

        # DAP augmented log-likelihood:
        #   log pi_aug(s) = log pi_prior(s) + sigma * R(s)
        #   loss = MSE( log pi_agent(s) , log pi_aug(s) )
        # Using NLL (= -log pi):
        #   log pi = -NLL
        #   loss = mean( (-agent_nll - (-prior_nll + sigma * R))**2 )
        aug_logp = -prior_nll + args.sigma * rewards_t
        loss = ((-agent_nll) - aug_logp).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in all_agent if p.requires_grad], 1.0)
        opt.step()

        # ---- Log ----
        n_new = sum(1 for s in batch["smiles"] if s not in seen_smiles)
        for s in batch["smiles"]:
            seen_smiles.add(s)
        dt = time.time() - t0
        step_log = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            "agent_nll_mean": float(agent_nll.detach().mean().cpu().item()),
            "prior_nll_mean": float(prior_nll.detach().mean().cpu().item()),
            "R_composite_mean": comps["R_composite_mean"],
            "R_composite_max": comps["R_composite_max"],
            "R_pIC50_mean": comps["R_pIC50_mean"],
            "R_wh_mean": comps["R_wh_mean"],
            "R_QED_mean": comps["R_QED_mean"],
            "pIC50_mean": comps["pIC50_mean"],
            "n_valid_in_batch": len(batch["smiles"]),
            "n_new_unique": n_new,
            "step_time_s": dt,
            "cum_unique": len(seen_smiles),
        }
        logs.append(step_log)
        print(f"[dap] step {step:02d}  loss={step_log['loss']:.3f}  "
              f"R={comps['R_composite_mean']:.3f} (max {comps['R_composite_max']:.3f})  "
              f"pIC50={comps['pIC50_mean']:.3f}  "
              f"R_pIC={comps['R_pIC50_mean']:.3f}  R_wh={comps['R_wh_mean']:.3f}  "
              f"R_qed={comps['R_QED_mean']:.3f}  n_valid={len(batch['smiles'])}/{args.batch_size}  "
              f"dt={dt:.1f}s", flush=True)

        # Early stop on rolling MEAN (not max) — max is dominated by sampling variance,
        # mean is what we actually want to lift.
        rolling_max.append(comps["R_composite_mean"])
        if step >= args.min_steps and len(rolling_max) >= args.rolling_window:
            recent_mean = float(np.mean(rolling_max[-args.rolling_window:]))
            if recent_mean >= args.early_stop_reward:
                print(f"[dap] early stop at step {step}: rolling mean R={recent_mean:.3f} >= {args.early_stop_reward}",
                      flush=True)
                break

        # Write intermediate log every step
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f:
            json.dump({
                "config": vars(args),
                "steps": logs,
                "elapsed_s": time.time() - t_start,
            }, f, indent=2)

    # ---- Save checkpoint ----
    # NB: wrapper.state_dict() only contains pocket_enc / pose_enc / buffers because
    # base is not an nn.Module. But DAP updates base.network.decoder / tgt_embed /
    # generator. We save both — the wrapper state under "model_state" (as v2 does)
    # AND the base.network state under "base_network_state". A companion loader is
    # provided (load_m1a_v2_with_base) in the sample script.
    print(f"[dap] saving fine-tuned checkpoint to {args.out_ckpt}", flush=True)
    payload = {
        "model_state": agent.state_dict(),
        "base_network_state": agent.base.network.state_dict(),
        "dap_config": vars(args),
        "dap_steps_completed": len(logs),
        "dap_final_reward_mean": logs[-1]["R_composite_mean"] if logs else None,
    }
    torch.save(payload, args.out_ckpt)

    # ---- Final log ----
    with open(args.log_json, "w") as f:
        json.dump({
            "config": vars(args),
            "steps": logs,
            "elapsed_s": time.time() - t_start,
        }, f, indent=2)
    total = time.time() - t_start
    print(f"[dap] DONE. Steps={len(logs)} total={total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
