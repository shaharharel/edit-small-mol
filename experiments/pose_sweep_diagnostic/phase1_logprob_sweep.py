"""Phase 1: Pose-sweep on log-prob (teacher-forced NLL).

Fix pocket=ZAP70-like (pocket_idx=5 from cache), fix anchor=Mol1, fix a
reference TARGET SMILES from val set. Sweep pose (d, theta, phi) one-at-a-time
and record log-prob of the reference target SMILES.

If log-prob is flat vs pose, [POSE] is IGNORED.
If log-prob has a clear preference / shifts smoothly, [POSE] is used at
scoring.

Outputs:
  data/paper_pair_training/pose_sweep_diagnostic/phase1_logprob.json
  data/paper_pair_training/pose_sweep_diagnostic/phase1_logprob.png  (plots)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
REINVENT_ROOT = Path("/Users/shaharharel/Documents/github/REINVENT4")
FIXES_DIR = LOCAL_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"
for p in [REINVENT_ROOT, FIXES_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from m1a_v2_model import load_m1a_v2  # noqa: E402
from train_m1a_pairs import make_std_mask  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

CKPT = LOCAL_ROOT / "models/v2_curriculum_clean/best.chkpt"
PRIOR = LOCAL_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"
CACHE = LOCAL_ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"
VAL_NPZ = LOCAL_ROOT / "data/paper_pair_training/v2_curriculum/pairs_val.npz"
OUT_DIR = LOCAL_ROOT / "data/paper_pair_training/pose_sweep_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_smi(smi, vocab, tok):
    return np.array(vocab.encode(tok.tokenize(smi)), dtype=np.int64)


def teacher_forced_nll(model, anchor_smi: str, target_smi: str,
                       res_emb, res_mask, pose_norm_batch):
    """Compute NLL of target_smi given anchor+pocket+pose (a batch of poses).

    pose_norm_batch: (B, 3) NORMALIZED poses.
    Returns: (B,) NLL per pose.
    """
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
    src_ids = encode_smi(anchor_smi, vocab, tok)
    tgt_ids = encode_smi(target_smi, vocab, tok)

    B = pose_norm_batch.shape[0]
    src = np.tile(src_ids[None], (B, 1))
    trg = np.tile(tgt_ids[None], (B, 1))
    src_t = torch.from_numpy(src).to(device)
    trg_t = torch.from_numpy(trg).to(device)
    src_mask = (src_t != 0).unsqueeze(-2).long()
    trg_mask = make_std_mask(trg_t[:, :-1], 0)

    res_emb_b = torch.from_numpy(np.tile(res_emb[None], (B, 1, 1))).to(device)
    res_mask_b = torch.from_numpy(np.tile(res_mask[None], (B, 1))).to(device)
    pose_t = torch.from_numpy(pose_norm_batch.astype(np.float32)).to(device)

    with torch.no_grad():
        nll = model.likelihood(src_t, src_mask, trg_t, trg_mask,
                                res_emb_b, res_mask_b, pose_t)
    return nll.detach().cpu().numpy()


def main():
    device = torch.device("cpu")
    print("[phase1] loading model", flush=True)
    model = load_m1a_v2(str(PRIOR), device, ckpt_path=str(CKPT))
    model.eval()
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()
    print(f"[phase1] pose_mean={pose_mean} pose_std={pose_std}", flush=True)

    print("[phase1] loading ZAP70 pocket from cache", flush=True)
    cache = np.load(CACHE, allow_pickle=True)
    row_seq_idx = cache["row_seq_idx"]
    # Use pocket_idx=5 (ZAP70 largest)
    seq_idx = 5
    zap70_emb = cache["residues_emb"][seq_idx]
    zap70_mask = cache["residues_mask"][seq_idx]
    print(f"[phase1] pocket_idx={seq_idx}, residues={int(zap70_mask.sum())}",
          flush=True)

    print("[phase1] picking reference target from val set (median pose)",
          flush=True)
    val = np.load(VAL_NPZ, allow_pickle=True)
    val_poses_norm = val["pose"]
    val_poses_raw = val_poses_norm * pose_std + pose_mean
    target_pose = np.array([2.1, 114.0, -17.0])
    dist = np.linalg.norm((val_poses_raw - target_pose) / pose_std, axis=1)
    ref_i = int(np.argmin(dist))
    ref_tgt_smi = str(val["tgt_smi"][ref_i])
    ref_pose_raw = val_poses_raw[ref_i]
    print(f"[phase1] reference i={ref_i} tgt='{ref_tgt_smi[:80]}'", flush=True)
    print(f"[phase1] reference raw pose={ref_pose_raw}", flush=True)

    # Median pose from training distribution (median of raw val poses)
    median_pose_raw = np.median(val_poses_raw, axis=0)
    print(f"[phase1] median training pose (raw): {median_pose_raw}",
          flush=True)

    def norm(p_raw):
        return (p_raw - pose_mean) / np.maximum(pose_std, 1e-6)

    # Sweep values
    d_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    theta_values = [80.0, 95.0, 105.0, 115.0, 130.0, 150.0]
    phi_values = [-40.0, -20.0, 0.0, 20.0, 40.0]

    def sweep_dim(dim: int, values, base_raw):
        poses = np.tile(base_raw, (len(values), 1))
        poses[:, dim] = values
        poses_norm = np.stack([norm(p) for p in poses])
        nlls = teacher_forced_nll(model, MOL1_SMI, ref_tgt_smi,
                                    zap70_emb, zap70_mask, poses_norm)
        return {"values": list(map(float, values)),
                "poses_raw": poses.tolist(),
                "poses_norm": poses_norm.tolist(),
                "nlls": [float(x) for x in nlls],
                "log_probs": [float(-x) for x in nlls]}

    print("\n[phase1] Sweep 1: d (Burgi-Dunitz distance)", flush=True)
    sweep_d = sweep_dim(0, d_values, median_pose_raw)
    for v, nll in zip(sweep_d["values"], sweep_d["nlls"]):
        print(f"    d={v:>4.1f}  NLL={nll:.4f}", flush=True)

    print("\n[phase1] Sweep 2: theta (Burgi-Dunitz angle)", flush=True)
    sweep_theta = sweep_dim(1, theta_values, median_pose_raw)
    for v, nll in zip(sweep_theta["values"], sweep_theta["nlls"]):
        print(f"    theta={v:>5.1f}  NLL={nll:.4f}", flush=True)

    print("\n[phase1] Sweep 3: phi (planar dihedral)", flush=True)
    sweep_phi = sweep_dim(2, phi_values, median_pose_raw)
    for v, nll in zip(sweep_phi["values"], sweep_phi["nlls"]):
        print(f"    phi={v:>5.1f}  NLL={nll:.4f}", flush=True)

    # Also probe with pose=null (all zeros raw -> normalized)
    zero_raw = np.zeros(3, dtype=np.float32)
    zero_norm = norm(zero_raw)[None]
    nll_zero = float(teacher_forced_nll(model, MOL1_SMI, ref_tgt_smi,
                                         zap70_emb, zap70_mask, zero_norm)[0])
    print(f"\n[phase1] NLL @ null pose (raw=0): {nll_zero:.4f}", flush=True)

    # Reference pose itself
    ref_pose_norm = norm(ref_pose_raw)[None]
    nll_ref = float(teacher_forced_nll(model, MOL1_SMI, ref_tgt_smi,
                                        zap70_emb, zap70_mask, ref_pose_norm)[0])
    print(f"[phase1] NLL @ reference pose (raw={ref_pose_raw}): {nll_ref:.4f}",
          flush=True)

    # Random poses (shuffled from val batch)
    rng = np.random.RandomState(42)
    rand_ix = rng.choice(len(val_poses_norm), 20, replace=False)
    rand_norm = val_poses_norm[rand_ix].astype(np.float32)
    nlls_rand = teacher_forced_nll(model, MOL1_SMI, ref_tgt_smi,
                                     zap70_emb, zap70_mask, rand_norm)
    print(f"[phase1] Random val-pose NLL: mean={nlls_rand.mean():.4f} "
          f"std={nlls_rand.std():.4f} min={nlls_rand.min():.4f} "
          f"max={nlls_rand.max():.4f}", flush=True)

    # Save
    out = {
        "reference": {
            "anchor_smi": MOL1_SMI,
            "target_smi": ref_tgt_smi,
            "target_i_in_val": ref_i,
            "pose_raw": [float(x) for x in ref_pose_raw],
            "pose_norm": [float(x) for x in norm(ref_pose_raw)],
        },
        "median_pose_raw": [float(x) for x in median_pose_raw],
        "pose_mean": [float(x) for x in pose_mean],
        "pose_std": [float(x) for x in pose_std],
        "pocket_idx": seq_idx,
        "sweep_d": sweep_d,
        "sweep_theta": sweep_theta,
        "sweep_phi": sweep_phi,
        "nll_null_pose": nll_zero,
        "nll_reference_pose": nll_ref,
        "nll_random_val_poses": {
            "values": [float(x) for x in nlls_rand],
            "mean": float(nlls_rand.mean()),
            "std": float(nlls_rand.std()),
            "min": float(nlls_rand.min()),
            "max": float(nlls_rand.max()),
        },
    }
    out_path = OUT_DIR / "phase1_logprob.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[phase1] saved {out_path}", flush=True)

    # Simple ASCII plot
    def ascii_bar(vals, title, labels):
        vmin = min(vals); vmax = max(vals)
        rng = max(vmax - vmin, 1e-6)
        print(f"\n{title}  (min={vmin:.3f} max={vmax:.3f} range={rng:.3f})")
        for v, lbl in zip(vals, labels):
            frac = (v - vmin) / rng
            bar = "#" * int(frac * 40)
            print(f"  {lbl:>8}  {v:8.4f}  {bar}")

    print("\n===== ASCII PLOTS: NLL by pose sweep =====")
    ascii_bar(sweep_d["nlls"], "NLL vs d",
              [f"{v:.1f}" for v in sweep_d["values"]])
    ascii_bar(sweep_theta["nlls"], "NLL vs theta",
              [f"{v:.0f}" for v in sweep_theta["values"]])
    ascii_bar(sweep_phi["nlls"], "NLL vs phi",
              [f"{v:.0f}" for v in sweep_phi["values"]])

    # Interpretation
    print("\n===== INTERPRETATION =====")
    for name, sw in [("d", sweep_d), ("theta", sweep_theta), ("phi", sweep_phi)]:
        n = np.array(sw["nlls"])
        rng = float(n.max() - n.min())
        std = float(n.std())
        print(f"{name}: NLL range={rng:.4f} std={std:.4f}")
    print(f"Null pose NLL - median pose NLL: "
          f"{nll_zero - sweep_d['nlls'][d_values.index(2.0) if 2.0 in d_values else 1]:.4f}")


if __name__ == "__main__":
    main()
