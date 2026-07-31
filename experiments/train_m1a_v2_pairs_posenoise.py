"""Exp-P6-POSENOISE: same as train_m1a_v2_pairs.py but with Gaussian pose noise.

Motivation: the paper's v2_cond model conditions on tgt-pose XYZ; pair training on
same-pocket peer molecules may allow the model to memorize exact XYZ features and
short-circuit generation. Adding small XYZ noise (σ=0.3Å) forces pose-invariance
and improves generalization to novel anchor poses at inference time.

Usage identical to train_m1a_v2_pairs.py plus --pose_noise_std.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

from torch.utils.data import Dataset, DataLoader, random_split
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from train_m1a_v2 import (
    randomize_smi, collate_factory, make_std_mask, cosine_warmup_lr
)
from m1a_v2_model import load_m1a_v2  # noqa: E402
from train_m1a_v2_pairs import PairedM1aV2Dataset


class NoisyPairedM1aV2Dataset(PairedM1aV2Dataset):
    """Same as parent but adds Gaussian XYZ noise to tgt pose each __getitem__."""

    def __init__(self, *args, pose_noise_std: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_noise_std = pose_noise_std
        print(f"[NoisyPaired] pose_noise_std={pose_noise_std}Å applied at each fetch",
              flush=True)

    def __getitem__(self, k: int):
        item = super().__getitem__(k)
        if self.pose_noise_std > 0:
            noise = np.random.normal(0.0, self.pose_noise_std, item["pose"].shape).astype(np.float32)
            item["pose"] = item["pose"] + noise
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--base_ckpt", default=None)
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
    ap.add_argument("--pose_noise_std", type=float, default=0.3)
    ap.add_argument("--progress_path", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]

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

    ds = NoisyPairedM1aV2Dataset(args.pairs_pkl, args.cache, vocabulary, tokenizer,
                                  max_len=args.max_len,
                                  pose_noise_std=args.pose_noise_std)
    collate = collate_factory(vocabulary, tokenizer, device, max_len=args.max_len)
    train_size = int(0.9 * len(ds)); val_size = len(ds) - train_size
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
                print(f"step={step}  epoch={epoch}  loss={loss.item():.4f}  lr={lr:.2e}  steps/s={step/dt:.2f}",
                       flush=True)
            if step % args.ckpt_interval == 0:
                cp = out_dir / f"m1a_v2_step{step:06d}.ckpt"
                torch.save({'model_state': model.state_dict(), 'vocab': vocabulary,
                            'tokenizer_state': None, 'args': vars(args)}, cp)
                print(f"[ckpt] {cp}", flush=True)
                if args.progress_path:
                    Path(args.progress_path).write_text(json.dumps({
                        'phase': 'training', 'epoch': epoch, 'step': step,
                        'total_steps': total_steps, 'timestamp': time.time()
                    }, indent=2))

    final = out_dir / "m1a_v2_final.ckpt"
    torch.save({'model_state': model.state_dict(), 'vocab': vocabulary,
                'tokenizer_state': None, 'args': vars(args)}, final)
    print(f"FINAL: {final}  (step={step})", flush=True)


if __name__ == '__main__':
    main()
