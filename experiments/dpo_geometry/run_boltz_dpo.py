#!/usr/bin/env python3
"""Boltz DPO Campaign — train C2/C3/C4 variants on the covFT prior.

Cohorts:
  C2 boltz_geom_dpo_only         : pure DPO on geom pairs
  C3 boltz_geom_dpo_plus_dap     : DPO + occasional pIC50 (DAP) reward term
  C4 boltz_geom_dpo_regularized  : C3 + guardrail penalty (acryl_lf + Tc>=0.35)

Reuses experiments.run_dpo_generation infra:
  - DPOConfig, ReinventMol2MolBackend, PreferencePair, setup_logging

Note: uses SAME grad-preserving log-prob patch as run_stream_c_dpo.py.
"""
from __future__ import annotations
import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
_USER_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "REINVENT4"))

from experiments.run_dpo_generation import (
    DPOConfig,
    ReinventMol2MolBackend,
    PreferencePair,
    setup_logging,
)

os.environ["CUDA_VISIBLE_DEVICES"] = _USER_CUDA

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# grad-preserving log-prob patch (copied from run_stream_c_dpo)
# ---------------------------------------------------------------------------
def _grad_compute_log_probs(self, source_smiles, target_smiles, batch_size=32):
    from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
    from torch.utils.data import DataLoader

    dataset = PairedDataset(
        source_smiles, target_smiles,
        vocabulary=self.model.vocabulary,
        tokenizer=self.model.tokenizer,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=PairedDataset.collate_fn, drop_last=False,
    )
    nlls = []
    for batch in loader:
        nll = self.model.likelihood(
            batch.input, batch.input_mask, batch.output, batch.output_mask
        )
        nlls.append(nll)  # NO .detach() — preserve grad
    return -torch.cat(nlls, dim=0)


ReinventMol2MolBackend.compute_log_probs = _grad_compute_log_probs


ANCHOR_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS = "[CH2]=[CH][C](=O)[N]"


# ---------------------------------------------------------------------------
# Reward helpers (used only by C3/C4)
# ---------------------------------------------------------------------------

def largest_fragment_smi(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    frags = sorted(frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True)
    return Chem.MolToSmiles(frags[0], canonical=True)


def has_acryl_on_lf(smi: str, patt) -> bool:
    lf = largest_fragment_smi(smi)
    if lf is None:
        return False
    m = Chem.MolFromSmiles(lf)
    if m is None:
        return False
    return m.HasSubstructMatch(patt)


def morgan_fp(smi: str, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)


# ---------------------------------------------------------------------------
# Optional pIC50 predictor (Morgan+MLP) for DAP reward
# ---------------------------------------------------------------------------

class SimplePIC50Predictor:
    """Lightweight ZAP70 pIC50 predictor: Morgan FP → 2-layer MLP.

    Trained on ZAP70 all-mols (280) at init. Provides pIC50 estimates on the fly.
    Used by C3/C4 for DAP-style reward.
    """

    def __init__(self, device: torch.device):
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)
        self.scaler = StandardScaler()
        self.mean = 6.0
        self.std = 1.0
        self.trained = False

    def train_from_zap70(self, logger: logging.Logger):
        from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints
        smiles_df, _ = load_zap70_molecules()
        smis = smiles_df["smiles"].tolist()
        pic = smiles_df["pIC50"].values.astype(np.float64)
        logger.info(f"[dap] training pIC50 predictor on {len(smis)} ZAP70 mols")
        fps = compute_fingerprints(smis, "morgan", radius=2, n_bits=2048)
        X = torch.tensor(fps, dtype=torch.float32, device=self.device)
        y = torch.tensor(pic, dtype=torch.float32, device=self.device)
        self.mean = float(y.mean().item())
        self.std = float(y.std().item()) or 1.0
        y_norm = (y - self.mean) / self.std

        opt = torch.optim.AdamW(self.mlp.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(200):
            self.mlp.train()
            opt.zero_grad()
            pred = self.mlp(X).squeeze(-1)
            loss = loss_fn(pred, y_norm)
            loss.backward()
            opt.step()
        logger.info(f"[dap] pIC50 predictor final MSE={loss.item():.4f} (normalized units)")
        self.trained = True

    @torch.no_grad()
    def score(self, smiles_list: list[str]) -> np.ndarray:
        """Return predicted pIC50 for each SMILES (0.0 if invalid)."""
        if not self.trained:
            return np.zeros(len(smiles_list))
        self.mlp.eval()
        fps = np.zeros((len(smiles_list), 2048), dtype=np.float32)
        valid = np.zeros(len(smiles_list), dtype=bool)
        for i, s in enumerate(smiles_list):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            DataStructs.ConvertToNumpyArray(fp, fps[i])
            valid[i] = True
        X = torch.tensor(fps, dtype=torch.float32, device=self.device)
        y_norm = self.mlp(X).squeeze(-1).cpu().numpy()
        y = y_norm * self.std + self.mean
        y[~valid] = 0.0
        return y


# ---------------------------------------------------------------------------
# Pair loader
# ---------------------------------------------------------------------------

def load_pairs(path: Path, split: str = "train") -> list[PreferencePair]:
    df = pd.read_parquet(path)
    df = df[df["split"] == split].copy()
    return [
        PreferencePair(
            source_smi=row["source_smiles"],
            win_smi=row["chosen_smiles"],
            lose_smi=row["rejected_smiles"],
            win_score=float(row["q_chosen"]),
            lose_score=float(row["q_rejected"]),
        )
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Guardrail loss (C4 only) — penalize policy for putting probability on
# targets that fail acryl_lf OR fall below Tc>=0.35 to Mol1.
# ---------------------------------------------------------------------------

def guardrail_penalty(win_smiles: list[str], patt, mol1_fp, tc_min: float) -> np.ndarray:
    """Return per-item penalty in [0,1]: 1 = fully off-policy (penalize)."""
    out = np.zeros(len(win_smiles), dtype=np.float32)
    for i, s in enumerate(win_smiles):
        acryl_ok = has_acryl_on_lf(s, patt)
        fp = morgan_fp(s)
        tc = DataStructs.TanimotoSimilarity(fp, mol1_fp) if fp else 0.0
        pen = 0.0
        if not acryl_ok:
            pen += 0.5
        if tc < tc_min:
            pen += 0.5
        out[i] = min(1.0, pen)
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cohort(
    cfg: DPOConfig,
    pi_theta: ReinventMol2MolBackend,
    pi_ref: ReinventMol2MolBackend,
    train_pairs: list[PreferencePair],
    val_pairs: list[PreferencePair],
    logger: logging.Logger,
    checkpoint_path: Path,
    checkpoint_every: int,
    warmup_steps: int,
    dap_predictor: SimplePIC50Predictor | None,
    dap_weight: float,
    dap_every_n: int,
    guardrail_weight: float,
    tc_min_guardrail: float,
) -> dict:
    """Unified training loop for C2/C3/C4."""
    logger.info(
        f"cohort: train={len(train_pairs)} val={len(val_pairs)} epochs={cfg.n_epochs} "
        f"batch={cfg.train_batch_size} beta={cfg.beta_dpo} lr={cfg.lr} "
        f"dap_weight={dap_weight} dap_every_n={dap_every_n} guardrail={guardrail_weight}"
    )

    pi_theta.set_mode("training")
    pi_ref.set_mode("inference")

    optimizer = torch.optim.AdamW(
        pi_theta.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    total_steps = max(1, (len(train_pairs) // cfg.train_batch_size) * cfg.n_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "step": [], "epoch": [], "train_loss": [], "dpo_loss": [], "dap_loss": [],
        "guardrail_loss": [], "train_acc": [], "lr": [], "reward_gap": [],
    }
    val_history = {
        "epoch": [], "val_loss": [], "val_acc": [], "val_reward_gap": [], "kl_div": [],
    }

    patt = Chem.MolFromSmarts(ACRYL_SMARTS)
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ANCHOR_SMI), 2, 2048)

    global_step = 0
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        rng = np.random.default_rng(cfg.seed + epoch)
        idx = rng.permutation(len(train_pairs))
        pi_theta.set_mode("training")

        for i in range(0, len(idx), cfg.train_batch_size):
            batch_idx = idx[i : i + cfg.train_batch_size]
            batch = [train_pairs[j] for j in batch_idx]
            if not batch:
                continue

            src_list = [p.source_smi for p in batch]
            win_list = [p.win_smi for p in batch]
            lose_list = [p.lose_smi for p in batch]

            lp_theta_win = pi_theta.compute_log_probs(src_list, win_list, batch_size=len(batch))
            lp_theta_lose = pi_theta.compute_log_probs(src_list, lose_list, batch_size=len(batch))

            with torch.no_grad():
                lp_ref_win = pi_ref.compute_log_probs(src_list, win_list, batch_size=len(batch))
                lp_ref_lose = pi_ref.compute_log_probs(src_list, lose_list, batch_size=len(batch))

            log_ratio_win = lp_theta_win - lp_ref_win.to(lp_theta_win.device)
            log_ratio_lose = lp_theta_lose - lp_ref_lose.to(lp_theta_lose.device)

            logits = cfg.beta_dpo * (log_ratio_win - log_ratio_lose)
            dpo_loss = -F.logsigmoid(logits).mean()
            reward_gap = logits.detach().mean().item()
            acc = (logits.detach() > 0).float().mean().item()

            dap_loss = torch.tensor(0.0, device=dpo_loss.device)
            if dap_predictor is not None and dap_weight > 0 and (global_step % dap_every_n == 0):
                # DAP-style reward: encourage log prob of winners weighted by pred pIC50.
                # Score winners with pIC50 predictor (numpy), then weight the log-prob of
                # winners (theta) so that HIGHER pIC50 → higher weight.
                pic_win = dap_predictor.score(win_list)
                # Normalize to 0..1 via sigmoid transform around 6.5 (typical decision threshold)
                dap_w = torch.tensor(
                    1.0 / (1.0 + np.exp(-(pic_win - 6.5))),
                    dtype=torch.float32, device=lp_theta_win.device,
                )
                # DAP loss: -mean(dap_w * lp_theta_win)  (maximize weighted log-prob)
                dap_loss = -(dap_w * lp_theta_win).mean()

            guardrail_loss = torch.tensor(0.0, device=dpo_loss.device)
            if guardrail_weight > 0:
                pen = guardrail_penalty(win_list, patt, mol1_fp, tc_min_guardrail)
                pen_t = torch.tensor(pen, dtype=torch.float32, device=lp_theta_win.device)
                # If chosen violates guardrail, we DON'T want the policy to increase
                # its prob → penalize by adding lp_theta_win weighted by pen.
                guardrail_loss = (pen_t * lp_theta_win).mean()

            total_loss = dpo_loss + dap_weight * dap_loss + guardrail_weight * guardrail_loss

            optimizer.zero_grad()
            total_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(pi_theta.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            history["step"].append(global_step)
            history["epoch"].append(epoch)
            history["train_loss"].append(total_loss.item())
            history["dpo_loss"].append(dpo_loss.item())
            history["dap_loss"].append(float(dap_loss.item()))
            history["guardrail_loss"].append(float(guardrail_loss.item()))
            history["train_acc"].append(acc)
            history["lr"].append(scheduler.get_last_lr()[0])
            history["reward_gap"].append(reward_gap)

            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  ep{epoch} step{global_step}/{total_steps} loss={total_loss.item():.4f} "
                    f"dpo={dpo_loss.item():.4f} dap={dap_loss.item():.4f} "
                    f"grd={guardrail_loss.item():.4f} acc={acc:.3f} gap={reward_gap:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.0f}s"
                )

            if global_step % checkpoint_every == 0:
                pi_theta.save(str(checkpoint_path))
                logger.info(f"  [ckpt] saved → {checkpoint_path} @ step {global_step}")
                _write_history(history, val_history, checkpoint_path.parent)

        val_loss, val_acc, val_gap, kl = _validate(pi_theta, pi_ref, val_pairs, cfg)
        val_history["epoch"].append(epoch)
        val_history["val_loss"].append(val_loss)
        val_history["val_acc"].append(val_acc)
        val_history["val_reward_gap"].append(val_gap)
        val_history["kl_div"].append(kl)
        logger.info(
            f"=== Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"val_gap={val_gap:.3f} KL={kl:.4f} ==="
        )
        pi_theta.save(str(checkpoint_path))
        logger.info(f"  [ckpt] end-of-epoch → {checkpoint_path}")
        _write_history(history, val_history, checkpoint_path.parent)

    return {"train": history, "val": val_history}


def _validate(pi_theta, pi_ref, val_pairs, cfg):
    pi_theta.set_mode("inference")
    losses, gaps, kl_acc = [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(val_pairs), cfg.train_batch_size):
            batch = val_pairs[i : i + cfg.train_batch_size]
            if not batch:
                continue
            src_list = [p.source_smi for p in batch]
            win_list = [p.win_smi for p in batch]
            lose_list = [p.lose_smi for p in batch]
            lp_theta_win = pi_theta.compute_log_probs(src_list, win_list, batch_size=len(batch))
            lp_theta_lose = pi_theta.compute_log_probs(src_list, lose_list, batch_size=len(batch))
            lp_ref_win = pi_ref.compute_log_probs(src_list, win_list, batch_size=len(batch))
            lp_ref_lose = pi_ref.compute_log_probs(src_list, lose_list, batch_size=len(batch))
            log_ratio_win = lp_theta_win - lp_ref_win.to(lp_theta_win.device)
            log_ratio_lose = lp_theta_lose - lp_ref_lose.to(lp_theta_lose.device)
            logits = cfg.beta_dpo * (log_ratio_win - log_ratio_lose)
            loss = -F.logsigmoid(logits).mean()
            losses.append(loss.item())
            gaps.append(logits.mean().item())
            correct += (logits > 0).sum().item()
            total += len(batch)
            kl_acc.append((lp_theta_win - lp_ref_win.to(lp_theta_win.device)).mean().item())
    return (
        float(np.mean(losses)) if losses else float("nan"),
        correct / max(total, 1),
        float(np.mean(gaps)) if gaps else float("nan"),
        float(np.mean(kl_acc)) if kl_acc else float("nan"),
    )


def _write_history(history, val_history, out_dir):
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(val_history).to_csv(out_dir / "validation_history.csv", index=False)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_cohort(
    backend: ReinventMol2MolBackend,
    source_smi: str,
    n_samples: int,
    batch_size: int,
    out_csv: Path,
    logger: logging.Logger,
    temperature: float = 1.0,
) -> None:
    backend.set_mode("inference")
    logger.info(f"sampling {n_samples} from anchor={source_smi}")
    all_samples = []
    n_remaining = n_samples
    t0 = time.time()
    while n_remaining > 0:
        bsz = min(n_remaining, batch_size)
        gens = backend.sample(
            [source_smi], n_samples_per_source=bsz,
            temperature=temperature, batch_size=bsz,
        )
        all_samples.extend(gens[0])
        n_remaining -= bsz
        if (n_samples - n_remaining) % 500 == 0 or n_remaining <= 0:
            logger.info(
                f"  sampled {n_samples - n_remaining}/{n_samples} "
                f"({(n_samples - n_remaining) / max(time.time()-t0, 1):.1f}/s)"
            )
    canon_set = {}
    n_valid = 0
    for raw in all_samples:
        if not raw:
            continue
        m = Chem.MolFromSmiles(raw)
        if m is None:
            continue
        n_valid += 1
        c = Chem.MolToSmiles(m, canonical=True)
        if c not in canon_set:
            canon_set[c] = raw
    rows = [
        {"SMILES": c, "SMILES_state": 1, "Input_SMILES": source_smi, "raw_smiles": r}
        for c, r in canon_set.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info(f"sampled: raw={len(all_samples)} valid={n_valid} unique={len(df)} → {out_csv}")


def filter_to_400(input_csv: Path, output_csv: Path, logger: logging.Logger) -> int:
    """Filter samples down to 400 that pass: valid, unique, acryl_lf, Tc>=0.35 to Mol1."""
    df = pd.read_csv(input_csv)
    logger.info(f"[filter] input: {len(df)}")
    patt = Chem.MolFromSmarts(ACRYL_SMARTS)
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ANCHOR_SMI), 2, 2048)

    kept = []
    for smi in df["SMILES"].tolist():
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, canonical=True)
        # dedupe within kept
        if canon in [k["SMILES"] for k in kept[-200:]]:
            continue  # cheap partial dedupe on tail
        if not has_acryl_on_lf(canon, patt):
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        tc = DataStructs.TanimotoSimilarity(fp, mol1_fp)
        if tc < 0.35:
            continue
        kept.append({"SMILES": canon, "tc_to_mol1": tc})
    # dedupe globally
    seen = set()
    final = []
    for r in kept:
        if r["SMILES"] in seen:
            continue
        seen.add(r["SMILES"])
        final.append(r)
        if len(final) >= 400:
            break
    logger.info(f"[filter] kept: {len(final)}")
    pd.DataFrame(final).to_csv(output_csv, index=False)
    return len(final)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COHORTS = {
    "C2": {
        "name": "boltz_geom_dpo_only",
        "dap": False,
        "guardrail": False,
        "epochs": 5,
        "batch": 16,
        "beta": 0.1,
        "lr": 5e-6,
    },
    "C3": {
        "name": "boltz_geom_dpo_plus_dap",
        "dap": True,
        "dap_weight": 0.3,
        "dap_every_n": 5,
        "guardrail": False,
        "epochs": 4,
        "batch": 16,
        "beta": 0.1,
        "lr": 5e-6,
    },
    "C4": {
        "name": "boltz_geom_dpo_regularized",
        "dap": True,
        "dap_weight": 0.3,
        "dap_every_n": 5,
        "guardrail": True,
        "guardrail_weight": 0.5,
        "tc_min_guardrail": 0.35,
        "epochs": 4,
        "batch": 16,
        "beta": 0.1,
        "lr": 5e-6,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True, choices=list(COHORTS.keys()))
    p.add_argument("--action", required=True, choices=["train", "sample", "filter", "both"])
    p.add_argument("--pairs", default=str(
        PROJECT_ROOT / "data" / "paper_pair_training" / "boltz_dpo_campaign" / "pairs_geom.parquet"
    ))
    p.add_argument("--base-ckpt", default=str(
        PROJECT_ROOT / "models" / "reinvent4_mol2mol_covalent_ft.prior"
    ))
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--sample-batch", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-val-pairs", type=int, default=400)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cohort_spec = COHORTS[args.cohort]
    cohort_name = cohort_spec["name"]

    results_dir = PROJECT_ROOT / "results" / "paper_evaluation" / "boltz_dpo_campaign" / cohort_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = PROJECT_ROOT / "models" / "dpo_checkpoints" / cohort_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trained_ckpt = ckpt_dir / "dpo_latest.chkpt"
    cohort_dir = PROJECT_ROOT / "data" / "paper_pair_training" / "boltz_dpo_campaign"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    samples_raw = cohort_dir / f"samples_{cohort_name}_raw.csv"
    samples_400 = cohort_dir / f"samples_{cohort_name}_400.csv"

    logger = setup_logging(results_dir)
    logger.info(f"=== Boltz DPO Campaign: cohort={args.cohort} ({cohort_name}) action={args.action} ===")
    logger.info(f"ckpt_dir={ckpt_dir}")
    logger.info(f"trained_ckpt={trained_ckpt}")
    logger.info(f"pairs={args.pairs}")
    logger.info(f"base_ckpt={args.base_ckpt}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    cfg = DPOConfig(
        prior_path=args.base_ckpt,
        results_dir=str(results_dir),
        model_backend="reinvent_mol2mol",
        device=str(device),
        beta_dpo=cohort_spec["beta"],
        lr=cohort_spec["lr"],
        weight_decay=args.wd,
        n_epochs=cohort_spec["epochs"],
        train_batch_size=cohort_spec["batch"],
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        seed=args.seed,
    )

    if args.action in ("train", "both"):
        train_pairs = load_pairs(Path(args.pairs), split="train")
        val_pairs = load_pairs(Path(args.pairs), split="val")
        logger.info(f"loaded pairs: train={len(train_pairs)} val={len(val_pairs)}")
        if args.max_val_pairs and args.max_val_pairs < len(val_pairs):
            rng = np.random.default_rng(args.seed)
            keep = rng.choice(len(val_pairs), size=args.max_val_pairs, replace=False)
            val_pairs = [val_pairs[i] for i in keep]
            logger.info(f"subsampled val to {len(val_pairs)}")

        # Resume from latest chkpt if exists
        prior_for_theta = args.base_ckpt
        if trained_ckpt.exists():
            logger.info(f"[resume] existing ckpt found → resuming from {trained_ckpt}")
            prior_for_theta = str(trained_ckpt)

        logger.info(f"loading pi_theta from {prior_for_theta}")
        pi_theta = ReinventMol2MolBackend(
            prior_path=prior_for_theta, device=device, mode="training"
        )
        logger.info(f"loading pi_ref from {args.base_ckpt}")
        pi_ref = ReinventMol2MolBackend(
            prior_path=args.base_ckpt, device=device, mode="inference"
        )
        for p in pi_ref.model.get_network_parameters():
            p.requires_grad_(False)

        # DAP predictor (C3/C4)
        dap_predictor = None
        dap_weight = 0.0
        dap_every_n = 10**9
        if cohort_spec.get("dap"):
            dap_predictor = SimplePIC50Predictor(device=device)
            dap_predictor.train_from_zap70(logger)
            dap_weight = float(cohort_spec.get("dap_weight", 0.3))
            dap_every_n = int(cohort_spec.get("dap_every_n", 5))

        guardrail_weight = float(cohort_spec.get("guardrail_weight", 0.0))
        tc_min_guardrail = float(cohort_spec.get("tc_min_guardrail", 0.35))

        history = train_cohort(
            cfg, pi_theta, pi_ref, train_pairs, val_pairs, logger,
            checkpoint_path=trained_ckpt,
            checkpoint_every=args.checkpoint_every,
            warmup_steps=args.warmup_steps,
            dap_predictor=dap_predictor,
            dap_weight=dap_weight,
            dap_every_n=dap_every_n,
            guardrail_weight=guardrail_weight,
            tc_min_guardrail=tc_min_guardrail,
        )

        pi_theta.save(str(trained_ckpt))
        logger.info(f"FINAL ckpt → {trained_ckpt}")

        (results_dir / "training_summary.json").write_text(json.dumps({
            "cohort": args.cohort,
            "cohort_name": cohort_name,
            "config": asdict(cfg),
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "final_train_loss": history["train"]["train_loss"][-1] if history["train"]["train_loss"] else None,
            "final_val_loss": history["val"]["val_loss"][-1] if history["val"]["val_loss"] else None,
            "final_val_acc": history["val"]["val_acc"][-1] if history["val"]["val_acc"] else None,
            "final_kl": history["val"]["kl_div"][-1] if history["val"]["kl_div"] else None,
            "dap_enabled": bool(cohort_spec.get("dap")),
            "guardrail_enabled": bool(cohort_spec.get("guardrail")),
        }, indent=2))

        # Free ref before sampling
        del pi_ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.action == "both":
            sample_cohort(pi_theta, ANCHOR_SMI, args.n_samples, args.sample_batch,
                          samples_raw, logger, temperature=args.temperature)
            n_kept = filter_to_400(samples_raw, samples_400, logger)
            logger.info(f"[done] {n_kept} filtered samples in {samples_400}")
        return

    if args.action == "sample":
        logger.info(f"loading trained policy from {trained_ckpt}")
        backend = ReinventMol2MolBackend(
            prior_path=str(trained_ckpt), device=device, mode="inference"
        )
        sample_cohort(backend, ANCHOR_SMI, args.n_samples, args.sample_batch,
                      samples_raw, logger, temperature=args.temperature)
        n_kept = filter_to_400(samples_raw, samples_400, logger)
        logger.info(f"[done] {n_kept} filtered samples in {samples_400}")
        return

    if args.action == "filter":
        n_kept = filter_to_400(samples_raw, samples_400, logger)
        logger.info(f"[done] {n_kept} filtered samples in {samples_400}")


if __name__ == "__main__":
    main()
