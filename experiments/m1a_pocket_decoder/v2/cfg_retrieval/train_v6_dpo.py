"""v6 DPO fine-tune of covFT+v2 on Vina cov score preference pairs.

Loss: standard DPO (Rafailov et al. 2023):
  L = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                       - log π_θ(y_l|x) + log π_ref(y_l|x)))]

where x = anchor Mol1 SMILES + ZAP70 pocket/pose context.

Reference model: frozen copy of covFT+v2 (`m1a_v2.ckpt`).
Policy model: trainable copy with new modules at LR=5e-6, base at LR=1e-6
(both an order of magnitude lower than v3/v4 since DPO signal is fragile).

2-3 epochs over the pair set.  Each pair -> two forward passes each on
policy and ref (winner + loser), so ~4× the compute per step vs SFT.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/cfg_retrieval"))

from m1a_v2_film_model import load_film_model, save_film_model  # noqa
from reinvent.models.transformer.core.network.module.subsequent_mask \
    import subsequent_mask  # noqa
from rdkit import Chem, RDLogger  # noqa
RDLogger.DisableLog("rdApp.*")

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def randomize_smi(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def _load_zap70(cache_npz, esm_npz, pose_npz):
    """Return (residues_emb, residues_mask, pose_norm) for ZAP70/Mol1."""
    d = np.load(cache_npz, allow_pickle=True)
    r_max = d["residues_emb"].shape[1]
    p = np.load(esm_npz, allow_pickle=True)
    emb = p["residues_emb"][0]; mask = p["residues_mask"][0]
    if emb.shape[0] < r_max:
        pad = np.zeros((r_max - emb.shape[0], emb.shape[1]), dtype=emb.dtype)
        emb = np.concatenate([emb, pad], axis=0)
        mp = np.zeros((r_max - mask.shape[0],), dtype=bool)
        mask = np.concatenate([mask, mp], axis=0)
    elif emb.shape[0] > r_max:
        emb = emb[:r_max]; mask = mask[:r_max]
    pose_d = np.load(pose_npz, allow_pickle=True)
    return emb, mask, pose_d["pose_norm"].astype(np.float32)


class DPOPairDataset(Dataset):
    """Each item = (winner_smi, loser_smi, anchor_smi, pocket_emb, pocket_mask, pose)."""
    def __init__(self, pairs_csv, tokenizer, vocabulary,
                  esm_emb, esm_mask, pose_norm,
                  anchor_smi=MOL1_SMI, max_len=128):
        self.df = pd.read_csv(pairs_csv)
        # Filter: tokens must fit.
        keep = []
        for i, row in self.df.iterrows():
            try:
                w_ok = len(tokenizer.tokenize(row["winner_smi"])) < max_len - 2
                l_ok = len(tokenizer.tokenize(row["loser_smi"])) < max_len - 2
                if w_ok and l_ok:
                    keep.append(i)
            except Exception:
                continue
        self.df = self.df.iloc[keep].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.anchor_smi = anchor_smi
        self.esm_emb = esm_emb; self.esm_mask = esm_mask
        self.pose_norm = pose_norm
        self.max_len = max_len
        print(f"[DPODataset] {len(self.df)} pairs after token-length filter",
               flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, k):
        row = self.df.iloc[k]
        return {
            "winner_smi": str(row["winner_smi"]),
            "loser_smi": str(row["loser_smi"]),
            "anchor_smi": randomize_smi(self.anchor_smi),
            "esm_emb": self.esm_emb,
            "esm_mask": self.esm_mask,
            "pose": self.pose_norm,
        }


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    return tgt_mask & sub_mask


def _encode_batch(smis, vocabulary, tokenizer, device):
    seqs = [np.array(vocabulary.encode(tokenizer.tokenize(s)), dtype=np.int64)
             for s in smis]
    L = max(len(s) for s in seqs)
    arr = np.zeros((len(seqs), L), dtype=np.int64)
    for j, s in enumerate(seqs):
        arr[j, :len(s)] = s
    t = torch.from_numpy(arr).to(device)
    mask = (t != 0).unsqueeze(-2).long()
    return t, mask


def collate_dpo_factory(vocabulary, tokenizer, device):
    def collate(batch):
        B = len(batch)
        anchors = [b["anchor_smi"] for b in batch]
        winners = [b["winner_smi"] for b in batch]
        losers = [b["loser_smi"] for b in batch]
        src, src_mask = _encode_batch(anchors, vocabulary, tokenizer, device)
        w_trg, _ = _encode_batch(winners, vocabulary, tokenizer, device)
        l_trg, _ = _encode_batch(losers, vocabulary, tokenizer, device)
        w_mask = make_std_mask(w_trg[:, :-1], 0)
        l_mask = make_std_mask(l_trg[:, :-1], 0)
        re = torch.from_numpy(np.stack([b["esm_emb"] for b in batch])).to(device)
        rm = torch.from_numpy(np.stack([b["esm_mask"] for b in batch])).to(device)
        po = torch.from_numpy(np.stack([b["pose"] for b in batch])).to(device)
        return {"src": src, "src_mask": src_mask,
                 "w_trg": w_trg, "w_trg_mask": w_mask,
                 "l_trg": l_trg, "l_trg_mask": l_mask,
                 "res_emb": re, "res_mask": rm, "pose": po}
    return collate


def logprob_sequence(model, src, src_mask, trg, trg_mask,
                       res_emb, res_mask, pose):
    """Return the sum of log-probs of trg's teacher-forced tokens (excluding
    the leading BOS)."""
    return -model.likelihood(src, src_mask, trg, trg_mask,
                                res_emb, res_mask, pose)


def cosine_warmup_lr(step, warmup, total, peak, min_frac=0.1):
    if step < warmup:
        return peak * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return peak * (min_frac + (1.0 - min_frac) * cos)


def append_csv(path, row, header_ref):
    write_header = not header_ref[0] and not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    header_ref[0] = True


def write_progress(p, **fields):
    rec = {"timestamp": time.time(),
             "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    Path(p).write_text(json.dumps(rec, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v6_dpo/dpo_pairs.csv"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--v2_ckpt", default=str(PROJECT_ROOT /
                     "models/m1a_v2.ckpt"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "models/cfg_retrieval_v6_dpo"))
    ap.add_argument("--log_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v6_dpo"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Small: DPO does 4x forward per step (policy_w, "
                          "policy_l, ref_w, ref_l).")
    ap.add_argument("--lr_new", type=float, default=5e-6)
    ap.add_argument("--lr_base", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--ckpt_interval", type=int, default=100)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"args = {vars(args)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = log_dir / "dpo_progress.json"
    train_csv = log_dir / "train_log.csv"
    header_ref = [False]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]

    print(f"Loading POLICY model (trainable) from {args.v2_ckpt}", flush=True)
    policy = load_film_model(args.prior, device,
                                pose_mean=pose_mean, pose_std=pose_std,
                                init_from_v2_ckpt=args.v2_ckpt)
    policy.film_enabled = False   # DPO on baseline v2 pathway
    print(f"Loading REF model (frozen)  from {args.v2_ckpt}", flush=True)
    ref = load_film_model(args.prior, device,
                             pose_mean=pose_mean, pose_std=pose_std,
                             init_from_v2_ckpt=args.v2_ckpt)
    ref.film_enabled = False
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    # ---- Optim: two-LR (base + new) ----
    base_ids = set(id(p) for p in policy.base.network.parameters())
    new_params = [p for p in policy.parameters()
                    if id(p) not in base_ids and p.requires_grad]
    base_params = [p for p in policy.base.network.parameters() if p.requires_grad]
    print(f"Trainable params: new={sum(p.numel() for p in new_params)/1e6:.2f}M "
           f"base={sum(p.numel() for p in base_params)/1e6:.2f}M", flush=True)
    optim = torch.optim.AdamW(
        [{"params": new_params, "lr": args.lr_new, "weight_decay": args.weight_decay},
         {"params": base_params, "lr": args.lr_base, "weight_decay": args.weight_decay}])

    # ---- Dataset ----
    emb, mask, pose = _load_zap70(args.cache, args.mol1_pocket_npz, args.mol1_pose_npz)
    global pd  # for DPOPairDataset
    import pandas as _pd
    globals()["pd"] = _pd
    ds = DPOPairDataset(args.pairs_csv, policy.base.tokenizer,
                          policy.base.vocabulary, emb, mask, pose)
    N = len(ds)
    perm = np.random.default_rng(args.seed).permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"Train={len(train_idx)} Val={len(val_idx)}", flush=True)
    train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())
    collate = collate_dpo_factory(policy.base.vocabulary, policy.base.tokenizer, device)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    def dpo_step(batch):
        # policy log-probs (with grad).
        lp_w = logprob_sequence(policy, batch["src"], batch["src_mask"],
                                    batch["w_trg"], batch["w_trg_mask"],
                                    batch["res_emb"], batch["res_mask"],
                                    batch["pose"])
        lp_l = logprob_sequence(policy, batch["src"], batch["src_mask"],
                                    batch["l_trg"], batch["l_trg_mask"],
                                    batch["res_emb"], batch["res_mask"],
                                    batch["pose"])
        # ref log-probs (no grad).
        with torch.no_grad():
            lr_w = logprob_sequence(ref, batch["src"], batch["src_mask"],
                                        batch["w_trg"], batch["w_trg_mask"],
                                        batch["res_emb"], batch["res_mask"],
                                        batch["pose"])
            lr_l = logprob_sequence(ref, batch["src"], batch["src_mask"],
                                        batch["l_trg"], batch["l_trg_mask"],
                                        batch["res_emb"], batch["res_mask"],
                                        batch["pose"])
        # DPO logits.
        logits = args.beta * ((lp_w - lr_w) - (lp_l - lr_l))
        loss = -F.logsigmoid(logits).mean()
        # Diagnostics: accuracy (fraction of pairs where policy prefers winner).
        acc = (logits > 0).float().mean()
        margin = logits.mean()
        return loss, acc, margin, (lp_w - lp_l).mean(), (lr_w - lr_l).mean()

    def evaluate_val():
        policy.eval()
        with torch.no_grad():
            n = 0; s_loss = 0.0; s_acc = 0.0; s_margin = 0.0
            for vb in val_loader:
                loss, acc, margin, _, _ = dpo_step(vb)
                B = vb["src"].shape[0]
                s_loss += float(loss.item()) * B
                s_acc += float(acc.item()) * B
                s_margin += float(margin.item()) * B
                n += B
        policy.train()
        return {"val_dpo_loss": s_loss / max(1, n),
                 "val_dpo_acc": s_acc / max(1, n),
                 "val_dpo_margin": s_margin / max(1, n)}

    policy.train()
    step = 0; epoch = 0; t_start = time.time()
    accum_loss = 0.0; accum_n = 0
    write_progress(progress_path, phase="dpo_training", step=step,
                     total_steps=total_steps)

    # Baseline eval.
    val0 = evaluate_val()
    print(f"[baseline step 0] {val0}", flush=True)
    append_csv(train_csv, {"step": 0, "epoch": 0, "loss": float("nan"),
                             "acc": float("nan"), "margin": float("nan"),
                             **val0, "lr_new": 0.0, "lr_base": 0.0,
                             "wallclock_s": 0.0,
                             "timestamp_iso":
                                 time.strftime("%Y-%m-%dT%H:%M:%S")},
                 header_ref)

    for epoch in range(args.epochs):
        for batch in train_loader:
            lr_new = cosine_warmup_lr(step + 1, args.warmup_steps, total_steps, args.lr_new)
            lr_base = cosine_warmup_lr(step + 1, args.warmup_steps, total_steps, args.lr_base)
            optim.param_groups[0]["lr"] = lr_new
            optim.param_groups[1]["lr"] = lr_base
            loss, acc, margin, delta_policy, delta_ref = dpo_step(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            optim.step()
            B = batch["src"].shape[0]
            accum_loss += float(loss.item()) * B
            accum_n += B
            step += 1
            if step % 10 == 0:
                avg = accum_loss / accum_n
                elapsed = time.time() - t_start
                ips = step / max(elapsed, 1e-6)
                print(f"step={step} epoch={epoch} loss={avg:.4f} "
                       f"acc={float(acc.item()):.3f} "
                       f"margin={float(margin.item()):+.3f} "
                       f"Δπ={float(delta_policy.item()):+.2f} "
                       f"Δref={float(delta_ref.item()):+.2f} "
                       f"lr_new={lr_new:.2e} lr_base={lr_base:.2e} "
                       f"steps/s={ips:.2f}", flush=True)
                accum_loss = 0.0; accum_n = 0
            if step % args.ckpt_interval == 0 or step == total_steps:
                val = evaluate_val()
                ck = out_dir / f"dpo_step{step:06d}.ckpt"
                # QA #8 fix: also save the base transformer state dict
                # (Mol2MolModel isn't nn.Module so wrapper.state_dict() drops it).
                torch.save({"model_state": policy.state_dict(),
                             "base_network_state": policy.base.network.state_dict(),
                             "optim_state": optim.state_dict(),
                             "step": step, "epoch": epoch,
                             "val": val}, ck)
                cks = sorted(out_dir.glob("dpo_step*.ckpt"))
                for old in cks[:-2]:
                    if int(old.stem.split("step")[-1]) % 500 != 0:
                        old.unlink(missing_ok=True)
                append_csv(train_csv, {"step": step, "epoch": epoch,
                                         "loss": float(loss.item()),
                                         "acc": float(acc.item()),
                                         "margin": float(margin.item()),
                                         **val,
                                         "lr_new": lr_new, "lr_base": lr_base,
                                         "wallclock_s": time.time() - t_start,
                                         "timestamp_iso":
                                             time.strftime("%Y-%m-%dT%H:%M:%S")},
                             header_ref)
                write_progress(progress_path, phase="dpo_training_ckpt",
                                 step=step, epoch=epoch,
                                 val_dpo_loss=val["val_dpo_loss"],
                                 val_dpo_acc=val["val_dpo_acc"],
                                 val_dpo_margin=val["val_dpo_margin"],
                                 total_steps=total_steps,
                                 wallclock_s=time.time() - t_start)
                print(f"[ckpt] step={step} {val} -> {ck}", flush=True)

    final = out_dir / "dpo_final.ckpt"
    save_film_model(policy, str(final),
                      extra={"step": step, "epoch": args.epochs})
    print(f"FINAL: {final}", flush=True)
    write_progress(progress_path, phase="dpo_training_done",
                     final_step=step, final_epoch=args.epochs)


if __name__ == "__main__":
    main()
