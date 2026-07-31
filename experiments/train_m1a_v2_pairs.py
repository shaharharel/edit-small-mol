"""Exp-P5: train v2_cond with TRUE (src, tgt) PAIRS instead of randomized-autoencoder.

Same architecture as train_m1a_v2.py but PairedDataset reads pair indices from a pkl.
For each pair: src_smi = cache.smiles[src_idx], tgt_smi = cache.smiles[tgt_idx].
Pose + pocket come from tgt (target ligand's crystal pose in target's pocket).

Usage:
  python train_m1a_v2_pairs.py --pairs_pkl data/exp_P5/pairs_warhead_swap.pkl \\
      --cache data/m1a_triples_v2/esm2_cache_posefix_v3.npz \\
      --out_dir data/exp_P5/models/P5-4 --epochs 30 --lr 1e-4 --batch_size 16 --seed 42 \\
      --ckpt_interval 2000 --progress_path data/exp_P5/logs/P5-4_progress.json
"""
from __future__ import annotations
import argparse, json, pickle, sys, time, os
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Reuse everything from the paper's train script
from train_m1a_v2 import (
    randomize_smi, collate_factory, make_std_mask, cosine_warmup_lr
)
from m1a_v2_model import load_m1a_v2  # noqa: E402


class PairedM1aV2Dataset(Dataset):
    """Dataset backed by a pair pkl + ESM cache.

    Each item: (src_smi from pair.src_idx, tgt_smi from pair.tgt_idx,
                pocket from tgt_idx, pose from tgt_idx).
    """
    def __init__(self, pairs_pkl: Path, cache_npz: Path, vocabulary, tokenizer, max_len: int = 128):
        super().__init__()
        with open(pairs_pkl, 'rb') as f:
            payload = pickle.load(f)
        self.pairs = payload['pairs']
        self.scheme = payload.get('scheme', 'unknown')
        d = np.load(cache_npz, allow_pickle=True)
        self.residues_emb = d["residues_emb"]
        self.residues_mask = d["residues_mask"]
        self.row_seq_idx = d["row_seq_idx"]
        self.poses = d["poses"]
        self.smiles = d["smiles"]
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Filter: keep pairs where both src and tgt are tokenizable within max_len
        ok = []
        for i, p in enumerate(self.pairs):
            src_smi = str(self.smiles[p['src_idx']])
            tgt_smi = str(self.smiles[p['tgt_idx']])
            try:
                if (len(tokenizer.tokenize(src_smi)) < max_len - 2 and
                    len(tokenizer.tokenize(tgt_smi)) < max_len - 2):
                    ok.append(i)
            except Exception:
                continue
        self.idx = np.array(ok)
        print(f"[PairedDataset] scheme={self.scheme}  {len(self.idx)}/{len(self.pairs)} pairs usable "
               f"(U={self.residues_emb.shape[0]} unique pockets)", flush=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k: int):
        p = self.pairs[self.idx[k]]
        src_idx, tgt_idx = p['src_idx'], p['tgt_idx']
        src_smi = str(self.smiles[src_idx])
        tgt_smi = str(self.smiles[tgt_idx])
        seq_idx = self.row_seq_idx[tgt_idx]  # pocket from tgt
        return {
            "src_smi": src_smi,
            "tgt_smi": tgt_smi,
            "residues_emb": self.residues_emb[seq_idx],
            "residues_mask": self.residues_mask[seq_idx],
            "pose": self.poses[tgt_idx],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--base_ckpt", default=None, help="Optional starting ckpt to fine-tune from")
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--pairs_pkl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_interval", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--progress_path", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load pose normalizer from cache
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]

    # Load prior + optionally base ckpt
    if args.base_ckpt:
        print(f"Loading base ckpt {args.base_ckpt}", flush=True)
        model = load_m1a_v2(args.prior, device, pose_mean=pose_mean, pose_std=pose_std)
        sd = torch.load(args.base_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd["model_state"])
    else:
        print(f"Loading prior {args.prior}", flush=True)
        model = load_m1a_v2(args.prior, device, pose_mean=pose_mean, pose_std=pose_std)
    vocabulary = model.base.vocabulary
    tokenizer = model.base.tokenizer
    print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    ds = PairedM1aV2Dataset(args.pairs_pkl, args.cache, vocabulary, tokenizer, max_len=args.max_len)
    collate = collate_factory(vocabulary, tokenizer, device, max_len=args.max_len)
    train_size = int(0.9 * len(ds)); val_size = len(ds) - train_size
    from torch.utils.data import random_split
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate, num_workers=0)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    print(f"Train={train_size} Val={val_size}", flush=True)
    print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    step = 0; t0 = time.time()
    ckpt_paths = []

    for epoch in range(args.epochs):
        model.train()
        for batch in train_dl:
            src, src_mask, trg, trg_mask, res_emb, res_mask, pose = batch
            lr = cosine_warmup_lr(step, args.warmup_steps, total_steps, args.lr)
            for g_ in optim.param_groups:
                g_["lr"] = lr
            nll = model.likelihood(src, src_mask, trg, trg_mask, res_emb, res_mask, pose)
            loss = nll.mean()
            optim.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 50 == 0:
                dt = time.time() - t0
                print(f"step={step}  epoch={epoch}  loss={loss.item():.4f}  lr={lr:.2e}  steps/s={step/dt:.2f}", flush=True)
            if step % args.ckpt_interval == 0:
                cp = out_dir / f"m1a_v2_step{step:06d}.ckpt"
                torch.save({'model_state': model.state_dict(), 'vocab': vocabulary,
                            'tokenizer_state': None, 'args': vars(args)}, cp)
                ckpt_paths.append(str(cp))
                print(f"[ckpt] {cp}", flush=True)
                if args.progress_path:
                    Path(args.progress_path).write_text(json.dumps({
                        'phase': 'training', 'epoch': epoch, 'step': step,
                        'total_steps': total_steps, 'timestamp': time.time()
                    }, indent=2))
        # end epoch — save mid-epoch checkpoint via ckpt_interval logic

    final = out_dir / "m1a_v2_final.ckpt"
    torch.save({'model_state': model.state_dict(), 'vocab': vocabulary,
                'tokenizer_state': None, 'args': vars(args)}, final)
    print(f"FINAL: {final}  (step={step})", flush=True)
    # Also symlink for downstream compatibility
    spec = out_dir.parent / "m1a_v2.ckpt"
    try:
        if spec.exists(): spec.unlink()
        spec.symlink_to(final.resolve())
    except Exception: pass
    print(f"SPEC-NAMED: {spec}", flush=True)


if __name__ == '__main__':
    main()
