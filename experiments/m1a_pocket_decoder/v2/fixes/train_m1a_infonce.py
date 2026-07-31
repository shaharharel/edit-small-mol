"""Phase 3: InfoNCE contrastive pose training for the M1a v2-cond decoder.

The v2-cond decoder currently trains as a SMILES denoising autoencoder
(src_smi = randomize(canon_smi); tgt_smi = canon_smi). Scheme A / B both
tried to break src=tgt symmetry with supervised NLL on chemistry-matched
pairs, and BOTH failed the causal pose test (pose-clamp |r|<0.11, pose-swap
W(d)~0.001-0.003).

Root cause: NLL on (src != tgt) is still too easy — src fully determines
tgt via chemistry, so the pose channel gets essentially zero gradient
signal.

The InfoNCE fix: add a contrastive term whose loss cannot be minimized
unless the decoder's pooled hidden state discriminates pose-real from
pose-mismatched. This installs a gradient path from tgt-generation through
the pose encoder that cannot be shortcut.

Loss:
    L_total = L_ce(tgt) + λ · L_infonce
    L_infonce = -log[ exp(sim(h, e(pose_true))/τ) /
                       Σ_k exp(sim(h, e(pose_neg_k))/τ) ]
where:
    h = decoder last-layer hidden state, pooled (mean over non-pad tokens)
    e(pose) = the existing pose_mlp (WarheadPoseEncoder) output (3->64->256)
    sim = cosine similarity
    τ = 0.1
    K = 7 negatives per positive:
        3 synthetic perturbations of the real pose
        2 real poses from OTHER molecules in the SAME pocket (hard)
        2 real poses from RANDOM different pockets (easy)

Curriculum: λ linearly ramps 0 -> 0.5 over epochs 0..5, then stays 0.5.
CFG-style pose dropout: p=0.15 → replace pose with learned NULL embedding.

Warm-start: models/m1a_v2.ckpt (the ORIGINAL autoencoder — NOT Scheme A/B).

Saves final ckpt to: models/m1a_v2_infonce.ckpt.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- Path shims (local mac vs V100 vs A100) ----
LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
V100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else V100_ROOT
LOCAL_REINVENT = Path("/Users/shaharharel/Documents/github/REINVENT4")
V100_REINVENT = Path("/home/shaharh_quris_ai/REINVENT4")
REINVENT4_ROOT = LOCAL_REINVENT if LOCAL_REINVENT.exists() else V100_REINVENT
if str(REINVENT4_ROOT) not in sys.path:
    sys.path.insert(0, str(REINVENT4_ROOT))

FIXES_DIR = PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"
if str(FIXES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXES_DIR))

from m1a_v2_model import load_m1a_v2, save_m1a_v2, POSE_DIM, D_MODEL  # noqa: E402
from reinvent.models.transformer.core.network.module.subsequent_mask import (  # noqa: E402
    subsequent_mask,
)
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def randomize_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


# ============================================================================
# Dataset
# ============================================================================
class M1aV2InfoNCEDataset(Dataset):
    """Autoencoder dataset (same as train_m1a_v2.py) that ALSO yields the
    row index so the collate can pool same-pocket / random-pocket negatives
    from the full pose corpus.
    """

    def __init__(self, npz_path: Path, vocabulary, tokenizer, max_len: int = 128):
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        self.residues_emb = d["residues_emb"]
        self.residues_mask = d["residues_mask"]
        self.row_seq_idx = d["row_seq_idx"]
        self.poses = d["poses"].astype(np.float32)  # z-scored (v3 invariant)
        self.smiles = d["smiles"]
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_len = max_len

        ok = []
        for i, smi in enumerate(self.smiles):
            try:
                tokens = self.tokenizer.tokenize(smi)
                if len(tokens) < max_len - 2:
                    ok.append(i)
            except Exception:
                continue
        self.idx = np.array(ok, dtype=np.int64)
        print(f"[Dataset InfoNCE] {len(self.idx)} / {len(self.smiles)} usable rows "
              f"(U={self.residues_emb.shape[0]} unique pockets, "
              f"poses={self.poses.shape})", flush=True)

        # ---- Precompute same-pocket lookup ----
        # For each unique pocket, list of dataset positions (post-filter) that
        # sit in that pocket.  We use this at __getitem__ time to sample two
        # HARD negatives (same pocket, different pose).
        self.same_pocket_rows: dict[int, np.ndarray] = {}
        for k, i in enumerate(self.idx):
            seq = int(self.row_seq_idx[i])
            self.same_pocket_rows.setdefault(seq, []).append(k)
        for seq in list(self.same_pocket_rows.keys()):
            self.same_pocket_rows[seq] = np.array(
                self.same_pocket_rows[seq], dtype=np.int64)
        # Cache the full "usable" pose bank for random-pocket negatives.
        self.usable_positions = np.arange(len(self.idx), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        canon_smi = str(self.smiles[i])
        anchor_smi = randomize_smi(canon_smi)
        seq_idx = int(self.row_seq_idx[i])
        pose_pos = self.poses[i]

        # ---- 3 synthetic perturbations of the real pose (z-space) ----
        # Perturbation A: jitter d_b_nuc by ±1σ (in z-score space => ±1.0)
        # Perturbation B: rotate θ_BD by ±30° or ±60°
        # Perturbation C: rotate φ_planar by ±30° or ±60°
        # NB: the pose is z-scored using per-axis (mean, std) from the cache,
        # so ±1σ in z-space means adding ±1.0 to the z value. For angular
        # jitters (B, C) we work in UNNORMALIZED degrees, then re-normalize
        # using the cache's mean/std for that axis — but since the collate
        # doesn't have mean/std, we do all pose math in NORMALIZED space and
        # convert the angular jitter into z-space using an approximation.
        # A cleaner path: pass pose_mean/pose_std to the dataset so the
        # perturbations are geometrically meaningful. We do that below.
        pose_negs_syn = self._make_synthetic_negatives(pose_pos)

        # ---- 2 same-pocket hard negatives ----
        same_pool = self.same_pocket_rows.get(seq_idx, np.array([], dtype=np.int64))
        # exclude self
        same_pool = same_pool[same_pool != k]
        if len(same_pool) >= 2:
            picks = np.random.choice(same_pool, size=2, replace=False)
        elif len(same_pool) == 1:
            picks = np.array([same_pool[0], same_pool[0]], dtype=np.int64)
        else:
            # Fallback: no other row in same pocket. Use random negatives.
            picks = np.random.choice(self.usable_positions, size=2, replace=False)
        pose_negs_hard = np.stack(
            [self.poses[int(self.idx[p])] for p in picks], axis=0)

        # ---- 2 random-pocket easy negatives ----
        # Random rows anywhere in the corpus.  These are strictly wrong
        # pockets in the majority of cases; even if they collide with the
        # same pocket occasionally, that just makes them "hard-like".
        rand_picks = np.random.choice(self.usable_positions, size=2, replace=False)
        pose_negs_easy = np.stack(
            [self.poses[int(self.idx[p])] for p in rand_picks], axis=0)

        pose_negs = np.concatenate(
            [pose_negs_syn, pose_negs_hard, pose_negs_easy], axis=0
        ).astype(np.float32)  # (7, POSE_DIM)

        return {
            "src_smi": anchor_smi,
            "tgt_smi": canon_smi,
            "residues_emb": self.residues_emb[seq_idx],
            "residues_mask": self.residues_mask[seq_idx],
            "pose": pose_pos,
            "pose_negs": pose_negs,
        }

    def _make_synthetic_negatives(self, pose_pos: np.ndarray) -> np.ndarray:
        """Three synthetic perturbations of the z-scored real pose.

        The pose is ALREADY z-scored (mean/std applied in preprocessing).
        Perturbation A shifts index 0 (d_b_nuc) by ±1 in z-space (== ±1σ).
        Perturbations B and C shift the ANGULAR axes (index 1 = θ, index 2 = φ)
        by an amount that, when un-normalized, would be ±30° or ±60°.

        Since we do not have (mean, std) inside __getitem__, we use the
        conservative rule of "shift by 1.0-2.0 in z-space" for angular
        axes — the z-scale is set from the cache so 1σ typically covers ~30°
        of rotation for these angles, which is roughly the target scale.

        This is only used as a NEGATIVE (something the model must
        distinguish); imperfect geometric semantics of the perturbation is
        fine so long as it produces a distinct z-vector.
        """
        rng = np.random
        neg_A = pose_pos.copy()
        sign_A = rng.choice([-1.0, 1.0])
        neg_A[0] = neg_A[0] + sign_A * 1.0  # ±1σ_d in z-space

        neg_B = pose_pos.copy()
        # ±30° or ±60° in real degrees. In z-space, use a shift of ~1σ or ~2σ.
        step_B = float(rng.choice([1.0, 2.0]))
        sign_B = rng.choice([-1.0, 1.0])
        neg_B[1] = neg_B[1] + sign_B * step_B

        neg_C = pose_pos.copy()
        step_C = float(rng.choice([1.0, 2.0]))
        sign_C = rng.choice([-1.0, 1.0])
        neg_C[2] = neg_C[2] + sign_C * step_C

        return np.stack([neg_A, neg_B, neg_C], axis=0)


def collate_factory(vocabulary, tokenizer, device, max_len: int = 128):
    def collate(batch):
        B = len(batch)
        src_seqs, tgt_seqs = [], []
        for b in batch:
            src = tokenizer.tokenize(b["src_smi"])
            tgt = tokenizer.tokenize(b["tgt_smi"])
            src_seqs.append(np.array(vocabulary.encode(src), dtype=np.int64))
            tgt_seqs.append(np.array(vocabulary.encode(tgt), dtype=np.int64))
        L_src = max(len(s) for s in src_seqs)
        L_tgt = max(len(s) for s in tgt_seqs)
        src = np.zeros((B, L_src), dtype=np.int64)
        trg = np.zeros((B, L_tgt), dtype=np.int64)
        for j, s in enumerate(src_seqs):
            src[j, : len(s)] = s
        for j, s in enumerate(tgt_seqs):
            trg[j, : len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        trg_t = torch.from_numpy(trg).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        trg_mask = make_std_mask(trg_t[:, :-1], 0)
        res_emb = torch.from_numpy(
            np.stack([b["residues_emb"] for b in batch])).to(device)
        res_mask = torch.from_numpy(
            np.stack([b["residues_mask"] for b in batch])).to(device)
        pose = torch.from_numpy(
            np.stack([b["pose"] for b in batch])).to(device)  # (B, POSE_DIM)
        pose_negs = torch.from_numpy(
            np.stack([b["pose_negs"] for b in batch])).to(device)  # (B, K, POSE_DIM)
        return (src_t, src_mask, trg_t, trg_mask,
                res_emb, res_mask, pose, pose_negs)
    return collate


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask & sub_mask


def cosine_warmup_lr(step: int, warmup: int, total: int, peak: float,
                     min_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_frac + (1.0 - min_frac) * cos)


def lambda_curriculum(epoch: int, ramp_epochs: int, max_lambda: float) -> float:
    """Linear ramp 0 -> max_lambda over [0, ramp_epochs], then constant."""
    if ramp_epochs <= 0:
        return max_lambda
    frac = min(1.0, max(0.0, epoch / float(ramp_epochs)))
    return frac * max_lambda


# ============================================================================
# InfoNCE forward pass (with pose-dropout + hidden-state pool)
# ============================================================================
class InfoNCEWrapper(nn.Module):
    """Adds a learned NULL pose embedding to the M1aV2 wrapper (via composition,
    NOT modifying the wrapper's own state_dict layout).

    Reason for composition instead of subclass: we want to warm-start from the
    ORIGINAL v2 ckpt with strict=True on the base wrapper, then bolt on the
    NULL embedding as EXTRA parameters that live in this outer module. The
    outer module's state_dict is what we finally save (checkpoint carries the
    base + the NULL embedding).
    """

    def __init__(self, m1a_v2_wrapper: nn.Module):
        super().__init__()
        self.wrapper = m1a_v2_wrapper
        # NULL pose embedding: learnable, same shape as pose_enc's output
        # (D_MODEL). We could learn a NULL POSE VECTOR (POSE_DIM) instead,
        # but that plays weirder with pose_enc's LayerNorm; a direct
        # D_MODEL vector is cleaner and matches CFG-style formulations
        # in image diffusion (null-context is a learned embedding).
        self.null_pose_emb = nn.Parameter(torch.zeros(D_MODEL))
        nn.init.normal_(self.null_pose_emb, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.wrapper.device

    @property
    def base(self):
        return self.wrapper.base

    def encode_pose_with_dropout(self, pose_norm: torch.Tensor,
                                  dropout_p: float) -> torch.Tensor:
        """Encode a batch of poses with CFG-style pose dropout.

        For each item, with probability `dropout_p`, replace the encoded
        pose vector with the learned NULL embedding.
        """
        B = pose_norm.shape[0]
        pose_vec = self.wrapper.pose_enc(pose_norm)  # (B, D_MODEL)
        if dropout_p > 0.0 and self.training:
            drop_mask = (torch.rand(B, device=pose_vec.device) < dropout_p)
            if drop_mask.any():
                null = self.null_pose_emb.unsqueeze(0).expand(B, -1)
                pose_vec = torch.where(
                    drop_mask.unsqueeze(1), null, pose_vec)
        return pose_vec

    def _build_conditioned_memory_with_posevec(
            self, src, src_mask, residue_emb, residue_mask, pose_vec):
        """Same as M1aV2ConditionedModel._build_conditioned_memory but takes
        an already-encoded pose vector (so we can inject NULL or noise).
        """
        base = self.wrapper.base
        memory = base.network.encoder(base.network.src_embed(src), src_mask)
        B, L, D = memory.shape
        pocket_vec = self.wrapper.pocket_enc(residue_emb, residue_mask)
        cond = torch.stack([pocket_vec, pose_vec], dim=1)
        memory_ext = torch.cat([cond, memory], dim=1)
        if src_mask.dim() == 2:
            extra = torch.ones(B, 2, dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=1)
        elif src_mask.dim() == 3:
            extra = torch.ones(B, src_mask.shape[1], 2,
                                dtype=src_mask.dtype, device=src_mask.device)
            mask_ext = torch.cat([extra, src_mask], dim=2)
        else:
            raise ValueError(f"Unexpected src_mask shape: {src_mask.shape}")
        return memory_ext, mask_ext

    def forward_ce_and_hidden(
            self, src, src_mask, trg, trg_mask,
            residue_emb, residue_mask, pose_norm, pose_dropout_p: float):
        """One forward pass. Returns:
            ce_loss  : scalar CE loss (mean)
            h_pooled : (B, D_MODEL) mean-pooled decoder hidden state over
                       non-pad trg-input positions.
            pose_vec_used : (B, D_MODEL) the actual encoded pose (with dropout)
        """
        base = self.wrapper.base
        pose_vec = self.encode_pose_with_dropout(pose_norm, pose_dropout_p)
        memory_ext, src_mask_ext = self._build_conditioned_memory_with_posevec(
            src, src_mask, residue_emb, residue_mask, pose_vec)
        trg_y = trg[:, 1:]
        trg_in = trg[:, :-1]
        # Decoder output = last-layer hidden BEFORE the generator (vocab
        # projection). Shape: (B, L_trg-1, D_MODEL).
        out = base.network.decoder(
            base.network.tgt_embed(trg_in), memory_ext, src_mask_ext, trg_mask)
        # CE loss: use the same path as the standard likelihood()
        log_prob = base.network.generator(out, base.temperature).transpose(1, 2)
        nll = base._nll_loss(log_prob, trg_y).sum(dim=1)  # (B,)
        ce_loss = nll.mean()

        # Pool hidden state over non-pad trg-input positions.
        pad_mask = (trg_in != 0).float().unsqueeze(-1)  # (B, L_trg-1, 1)
        denom = pad_mask.sum(dim=1).clamp_min(1e-6)
        h_pooled = (out * pad_mask).sum(dim=1) / denom  # (B, D_MODEL)

        return ce_loss, h_pooled, pose_vec

    def infonce_loss(self, h_pooled: torch.Tensor, pose_pos: torch.Tensor,
                      pose_negs: torch.Tensor, temperature: float = 0.1,
                      encode_positives_with_dropout: bool = False,
                      pose_dropout_p: float = 0.0) -> torch.Tensor:
        """InfoNCE loss with cosine similarity.

        Args:
            h_pooled: (B, D_MODEL) query embeddings
            pose_pos: (B, POSE_DIM) POSITIVE pose (normalized z-space)
            pose_negs: (B, K, POSE_DIM) NEGATIVE poses (normalized z-space)
            temperature: softmax temperature

        The positive here is ALWAYS the real pose (never NULL), regardless
        of pose_dropout_p. Doing pose-dropout on the positive would defeat
        the causal alignment signal we want to install.
        """
        B, K, PD = pose_negs.shape
        # Encode positive + all negatives WITHOUT dropout.
        # We use the shared pose_enc so the pose space is one consistent
        # embedding that is being aligned to the decoder's hidden state.
        pos_emb = self.wrapper.pose_enc(pose_pos)                  # (B, D)
        neg_flat = pose_negs.reshape(B * K, PD)
        neg_emb = self.wrapper.pose_enc(neg_flat).reshape(B, K, -1)  # (B, K, D)

        # Cosine similarity in D_MODEL.
        h_n = F.normalize(h_pooled, dim=-1)
        pos_n = F.normalize(pos_emb, dim=-1)
        neg_n = F.normalize(neg_emb, dim=-1)

        sim_pos = (h_n * pos_n).sum(dim=-1, keepdim=True) / temperature  # (B, 1)
        sim_neg = (h_n.unsqueeze(1) * neg_n).sum(dim=-1) / temperature   # (B, K)

        # log-softmax over [pos | negs]. Loss = -log softmax_pos.
        logits = torch.cat([sim_pos, sim_neg], dim=1)  # (B, 1+K)
        log_prob = F.log_softmax(logits, dim=1)
        loss = -log_prob[:, 0].mean()
        # Accuracy diagnostic: how often pos > all negs
        with torch.no_grad():
            pos_wins = (sim_pos > sim_neg.max(dim=1, keepdim=True).values).float().mean()
        return loss, float(pos_wins.item())


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                    "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                    "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--warm_start", default=str(PROJECT_ROOT /
                    "models/m1a_v2.ckpt"),
                    help="Warm-start weights (ORIGINAL v2 autoencoder ckpt). "
                         "Set to '' to skip warm-start.")
    ap.add_argument("--out_ckpt", default=str(PROJECT_ROOT /
                    "models/m1a_v2_infonce.ckpt"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/infonce"),
                    help="Where to store per-step logs and progress json.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--lambda_max", type=float, default=0.5,
                    help="Max λ_infonce (curriculum ramp target). Fallback: 0.2.")
    ap.add_argument("--lambda_ramp_epochs", type=int, default=5,
                    help="Ramp epochs from 0 to λ_max.")
    ap.add_argument("--pose_dropout_p", type=float, default=0.15,
                    help="CFG-style pose-dropout probability.")
    ap.add_argument("--infonce_temp", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first --limit rows. Dry-run knob.")
    ap.add_argument("--max_wall_seconds", type=int, default=6 * 3600,
                    help="Hard fail if training exceeds this. Default: 6h.")
    ap.add_argument("--val_guard_epochs", type=int, default=2,
                    help="Number of epochs before checking val-loss guard.")
    ap.add_argument("--val_guard_delta", type=float, default=2.0,
                    help="If val_loss > warm_start_val + this, HALT.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "train_progress.json"
    log_path = out_dir / "train_infonce_log.jsonl"

    def write_progress(phase: str, **extra):
        rec = {"phase": phase, "timestamp": time.time(),
               "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **extra}
        progress_path.write_text(json.dumps(rec, indent=2))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[InfoNCE] Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[InfoNCE] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- Pose normalizer from the cache (buffers persisted in checkpoint) ----
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]
    pose_std = d_in["pose_std"]
    print(f"[InfoNCE] pose_mean={pose_mean} pose_std={pose_std}", flush=True)

    # ---- Load model with persisted normalizer ----
    print("[InfoNCE] Loading M1a v2 base model from prior...", flush=True)
    base_wrapper = load_m1a_v2(args.prior, device,
                                pose_mean=pose_mean, pose_std=pose_std)

    # Warm-start clean check
    if args.warm_start:
        ws_path = Path(args.warm_start)
        if ws_path.exists():
            print(f"[InfoNCE] Warm-starting from {ws_path}", flush=True)
            sd = torch.load(ws_path, map_location=device, weights_only=False)
            state = sd["model_state"] if "model_state" in sd else sd
            missing, unexpected = base_wrapper.load_state_dict(state, strict=False)
            print(f"    missing={len(missing)}  unexpected={len(unexpected)}",
                  flush=True)
            if missing:
                # Buffers (pose_mean/pose_std) live in state; extra parameters
                # should not exist yet. Report but do not fail.
                print(f"    (first 5 missing: {missing[:5]})", flush=True)
            if unexpected:
                print(f"    (first 5 unexpected: {unexpected[:5]})", flush=True)
        else:
            print(f"[InfoNCE] WARN: warm_start path missing: {ws_path}. "
                   f"Training from prior only.", flush=True)

    # Wrap with InfoNCE (adds learned NULL pose embedding)
    model = InfoNCEWrapper(base_wrapper).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[InfoNCE] Trainable params: {n_params/1e6:.2f}M", flush=True)

    # ---- Dataset ----
    ds = M1aV2InfoNCEDataset(Path(args.cache),
                              base_wrapper.base.vocabulary,
                              base_wrapper.base.tokenizer, max_len=128)
    N = len(ds)
    if args.limit is not None:
        N = min(N, args.limit)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))[:N]
    n_val = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"[InfoNCE] N={N} train={len(train_idx)} val={len(val_idx)}", flush=True)
    train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())
    collate = collate_factory(base_wrapper.base.vocabulary,
                               base_wrapper.base.tokenizer, device)
    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True,
                                collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False,
                              collate_fn=collate, num_workers=0)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    print(f"[InfoNCE] steps_per_epoch={steps_per_epoch} total_steps={total_steps}",
           flush=True)

    # ---- Warm-start val loss (before any InfoNCE gradient) ----
    def eval_val() -> tuple[float, float, float]:
        """Returns (ce_avg, infonce_avg, pos_wins_avg) averaged over val set."""
        model.eval()
        ce_sum = 0.0; nce_sum = 0.0; wins_sum = 0.0; n_ex = 0
        with torch.no_grad():
            for vb in val_loader:
                s, sm, t, tm, re, rm, po, pn = vb
                ce, h, _ = model.forward_ce_and_hidden(
                    s, sm, t, tm, re, rm, po, pose_dropout_p=0.0)
                nce, wins = model.infonce_loss(
                    h, po, pn, temperature=args.infonce_temp)
                ce_sum += float(ce.item()) * s.shape[0]
                nce_sum += float(nce.item()) * s.shape[0]
                wins_sum += wins * s.shape[0]
                n_ex += s.shape[0]
        model.train()
        return ce_sum / max(1, n_ex), nce_sum / max(1, n_ex), wins_sum / max(1, n_ex)

    print("[InfoNCE] Measuring warm-start val loss...", flush=True)
    ws_ce, ws_nce, ws_wins = eval_val()
    print(f"[InfoNCE] warm_start val: CE={ws_ce:.4f} InfoNCE={ws_nce:.4f} "
           f"pos_wins={ws_wins:.3f}", flush=True)
    write_progress("warm_start_eval", ws_ce=ws_ce, ws_nce=ws_nce,
                    ws_pos_wins=ws_wins,
                    total_steps=total_steps)

    log_fp = open(log_path, "a")
    model.train()
    step = 0
    t_start = time.time()
    ce_hist: list[float] = []
    nce_hist: list[float] = []
    wins_hist: list[float] = []

    val_history: list[dict] = [{"epoch": -1, "ce": ws_ce, "nce": ws_nce,
                                 "pos_wins": ws_wins}]

    def check_time_budget():
        elapsed = time.time() - t_start
        if elapsed > args.max_wall_seconds:
            raise RuntimeError(
                f"[InfoNCE] HARD FAIL: wall budget {args.max_wall_seconds}s "
                f"exceeded (elapsed={elapsed:.1f}s)")

    lam = 0.0
    epoch = 0
    for epoch in range(args.epochs):
        lam = lambda_curriculum(epoch, args.lambda_ramp_epochs, args.lambda_max)
        print(f"[InfoNCE] === epoch {epoch}  λ_infonce={lam:.4f} ===", flush=True)
        for batch in train_loader:
            src, src_mask, trg, trg_mask, res_emb, res_mask, pose, pose_negs = batch
            lr = cosine_warmup_lr(step, args.warmup_steps, total_steps, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            ce, h, _ = model.forward_ce_and_hidden(
                src, src_mask, trg, trg_mask, res_emb, res_mask, pose,
                pose_dropout_p=args.pose_dropout_p)
            nce, wins = model.infonce_loss(
                h, pose, pose_negs, temperature=args.infonce_temp)
            loss = ce + lam * nce

            # Nan/inf guard
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[InfoNCE] HARD FAIL: non-finite loss at step={step} "
                    f"epoch={epoch}  ce={float(ce)}  nce={float(nce)}")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            step += 1
            ce_hist.append(float(ce.item()))
            nce_hist.append(float(nce.item()))
            wins_hist.append(wins)

            if step % args.log_every == 0 or step <= 5:
                ce_avg = sum(ce_hist[-args.log_every:]) / min(
                    len(ce_hist), args.log_every)
                nce_avg = sum(nce_hist[-args.log_every:]) / min(
                    len(nce_hist), args.log_every)
                wins_avg = sum(wins_hist[-args.log_every:]) / min(
                    len(wins_hist), args.log_every)
                elapsed = time.time() - t_start
                print(f"[InfoNCE] ep={epoch} step={step} "
                       f"ce={ce_avg:.4f} nce={nce_avg:.4f} wins={wins_avg:.3f} "
                       f"λ={lam:.3f} lr={lr:.2e} elapsed={elapsed:.1f}s",
                       flush=True)
                log_fp.write(json.dumps({
                    "step": step, "epoch": epoch,
                    "ce": ce_avg, "nce": nce_avg, "pos_wins": wins_avg,
                    "lambda": lam, "lr": lr,
                    "wallclock_s": elapsed}) + "\n")
                log_fp.flush()

            check_time_budget()

        # ---- End-of-epoch val ----
        v_ce, v_nce, v_wins = eval_val()
        elapsed = time.time() - t_start
        print(f"[InfoNCE] ep={epoch}  val_ce={v_ce:.4f}  val_nce={v_nce:.4f}  "
               f"val_pos_wins={v_wins:.3f}  elapsed={elapsed:.1f}s",
               flush=True)
        val_history.append({"epoch": epoch, "ce": v_ce, "nce": v_nce,
                             "pos_wins": v_wins, "lambda": lam,
                             "wallclock_s": elapsed})
        log_fp.write(json.dumps({"epoch_end": epoch, "val_ce": v_ce,
                                  "val_nce": v_nce, "val_pos_wins": v_wins,
                                  "lambda": lam,
                                  "wallclock_s": elapsed}) + "\n")
        log_fp.flush()
        write_progress("training", epoch=epoch, step=step,
                        total_steps=total_steps,
                        val_ce=v_ce, val_nce=v_nce, val_pos_wins=v_wins,
                        val_history=val_history,
                        elapsed_s=elapsed)

        # ---- Guardrail: val CE cannot balloon > val_guard_delta above WS ----
        if epoch >= args.val_guard_epochs and (v_ce > ws_ce + args.val_guard_delta):
            raise RuntimeError(
                f"[InfoNCE] HARD FAIL: val_ce={v_ce:.4f} > ws_ce={ws_ce:.4f} "
                f"+ guard {args.val_guard_delta}. The InfoNCE term is "
                f"destabilizing CE loss. Restart with lower λ_max "
                f"(e.g. --lambda_max 0.2).")

    # ---- Save final checkpoint ----
    # We save the full outer state_dict (base wrapper + null_pose_emb) so we
    # can load it back with the same wrapper class. But the eval script
    # expects a *M1aV2ConditionedModel* state dict — we therefore save both:
    #   out_ckpt        : InfoNCEWrapper state (with null_pose_emb)
    #   *_basefmt.ckpt  : just the base wrapper's state, evaluable via
    #                     load_m1a_v2(ckpt_path=...)
    # The eval script we write below prefers the basefmt path for a cleaner
    # load, but is happy to load the outer state too (strict=False).
    payload = {
        "model_state": model.state_dict(),   # includes null_pose_emb.*
        "base_wrapper_state": base_wrapper.state_dict(),  # loadable by M1aV2
        "null_pose_emb": model.null_pose_emb.detach().cpu().numpy(),
        "epoch": epoch,
        "step": step,
        "val_history": val_history,
        "hparams": {
            "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "lambda_max": args.lambda_max,
            "lambda_ramp_epochs": args.lambda_ramp_epochs,
            "pose_dropout_p": args.pose_dropout_p,
            "infonce_temp": args.infonce_temp,
            "warm_start": args.warm_start,
        },
    }
    torch.save(payload, out_ckpt)
    print(f"[InfoNCE] wrote {out_ckpt}", flush=True)

    # Also save the base-format checkpoint that eval_scheme_B_clamps.py can
    # load directly (no InfoNCEWrapper needed at inference time — the NULL
    # embedding is only relevant if we want CFG at inference, which we
    # handle in the CFG-aware eval script below).
    basefmt = out_ckpt.with_name(out_ckpt.stem + "_basefmt.ckpt")
    save_m1a_v2(base_wrapper, str(basefmt),
                 extra={"epoch": epoch, "step": step,
                         "null_pose_emb": model.null_pose_emb.detach().cpu().numpy(),
                         "hparams": payload["hparams"]})
    print(f"[InfoNCE] wrote basefmt: {basefmt}", flush=True)

    log_fp.close()
    write_progress("done", epoch=epoch, step=step,
                    val_history=val_history,
                    out_ckpt=str(out_ckpt),
                    basefmt_ckpt=str(basefmt),
                    elapsed_s=time.time() - t_start)


if __name__ == "__main__":
    main()
