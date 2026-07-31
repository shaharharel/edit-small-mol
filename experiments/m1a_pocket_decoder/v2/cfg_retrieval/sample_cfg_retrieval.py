"""Sample the CFG × retrieval ablation matrix for ZAP70 Mol1.

For each combination in {s ∈ {1.0, 2.0, 3.0, 5.0}} × {retrieval ∈ {on, off}}:
  - Generate `--n` samples (default 200) using CFG scale s and, if
    retrieval='on', an encoder prefix built from the top-K retrieved SMILES.
  - Write incrementally to
    data/paper_pair_training/cfg_retrieval/samples_cfg_s{S}_retrieval_{X}.csv

s=1.0 recovers baseline (no CFG boost); s=1.0 with retrieval='off' is the
covFT+v2 baseline for reference.

Progress is written to progress.json every 25 samples.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/cfg_retrieval"))

from m1a_v2_cfg_model import load_cfg_model  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def randomize_smi(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=False, doRandom=True)
    except Exception:
        return smi


def write_progress(progress_path: Path, **fields):
    rec = {"timestamp": time.time(),
           "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    progress_path.write_text(json.dumps(rec, indent=2))


def sample_cell(model, anchor_smi, res_emb_np, res_mask_np, pose_norm_np,
                  cfg_scale, prefix_embs, n, batch_size, max_length,
                  temperature, out_csv, progress_path, cell_label):
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer

    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    n_done = 0
    if not write_header:
        # Resume: count rows already present.
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            for _ in reader:
                n_done += 1
        print(f"    resume: {n_done} rows already present in {out_csv}",
              flush=True)

    fp = open(out_csv, "a", newline="")
    fieldnames = ["cell", "cfg_scale", "retrieval", "sample_idx",
                    "SMILES", "Input_SMILES", "NLL"]
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

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
        res_emb = torch.from_numpy(np.tile(res_emb_np[None],
                                             (batch_n, 1, 1))).to(device)
        res_mask = torch.from_numpy(np.tile(res_mask_np[None],
                                              (batch_n, 1))).to(device)
        pose = torch.from_numpy(np.tile(pose_norm_np[None],
                                          (batch_n, 1))).to(device)
        out_smiles, nlls = model.sample_multinomial_cfg(
            src_t, src_mask, res_emb, res_mask, pose,
            cfg_scale=cfg_scale,
            retrieval_prefix_embs=prefix_embs,
            max_length=max_length, temperature=temperature)
        retrieval_on = int(prefix_embs is not None and prefix_embs.numel() > 0)
        for i, (s, nll, anchor) in enumerate(zip(out_smiles, nlls, anchors)):
            writer.writerow({
                "cell": cell_label,
                "cfg_scale": cfg_scale,
                "retrieval": retrieval_on,
                "sample_idx": n_done + i,
                "SMILES": s,
                "Input_SMILES": anchor,
                "NLL": float(nll),
            })
        fp.flush()
        n_done += batch_n
        elapsed = time.time() - t0
        rate = max(1, n_done) / max(elapsed, 1e-6)
        eta = (n - n_done) / max(rate, 1e-6)
        write_progress(progress_path, phase="sampling",
                         cell=cell_label, n_done=n_done, n_total=n,
                         rate_per_sec=round(rate, 2), eta_sec=round(eta, 1))
        if n_done % max(1, batch_size * 2) == 0 or n_done == n:
            print(f"    [{cell_label}] {n_done}/{n}  {rate:.1f} mol/s  "
                   f"ETA {eta:.0f}s", flush=True)
    fp.close()


def load_zap70_pocket_and_pose(mol1_esm_npz, mol1_pose_npz, r_max):
    pocket_d = np.load(mol1_esm_npz, allow_pickle=True)
    emb = pocket_d["residues_emb"][0]
    mask = pocket_d["residues_mask"][0]
    if emb.shape[0] < r_max:
        pad = np.zeros((r_max - emb.shape[0], emb.shape[1]), dtype=emb.dtype)
        emb = np.concatenate([emb, pad], axis=0)
        mask_pad = np.zeros((r_max - mask.shape[0],), dtype=bool)
        mask = np.concatenate([mask, mask_pad], axis=0)
    elif emb.shape[0] > r_max:
        emb = emb[:r_max]
        mask = mask[:r_max]
    pose_d = np.load(mol1_pose_npz, allow_pickle=True)
    pose_norm = pose_d["pose_norm"].astype(np.float32)
    return emb, mask, pose_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT /
                     "models/cfg_retrieval/cfg_final.ckpt"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--retrieval_json", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/retrieval_top5.json"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--cfg_scales", default="1.0,2.0,3.0,5.0")
    ap.add_argument("--retrieval_modes", default="off,on")
    ap.add_argument("--include_baseline_covft", action="store_true",
                    help="Also run baseline covFT via v2 checkpoint at s=1.0, "
                          "retrieval=off using the ORIGINAL v2 ckpt (no CFG "
                          "null token effect since s=1.0 = pure cond branch).")
    ap.add_argument("--tag", default="",
                    help="Optional tag appended to output CSV filenames, e.g. "
                          "`_unfrozen` so multiple checkpoints don't collide.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading CFG model: {args.ckpt}", flush=True)
    model = load_cfg_model(args.prior, device, ckpt_path=args.ckpt)
    model.eval()
    pose_mean = model.pose_mean.cpu().numpy()
    pose_std = model.pose_std.cpu().numpy()
    print(f"Model pose normalizer mean={pose_mean} std={pose_std}", flush=True)

    # ZAP70 pocket + pose.
    cache = np.load(args.cache, allow_pickle=True)
    r_max = cache["residues_emb"].shape[1]
    zap70_emb, zap70_mask, mol1_pose_norm = load_zap70_pocket_and_pose(
        args.mol1_pocket_npz, args.mol1_pose_npz, r_max)
    print(f"ZAP70 pocket: {int(zap70_mask.sum())} real residues; "
          f"pose_norm={mol1_pose_norm}", flush=True)

    # Retrieval prefix.
    retrieval_smiles = []
    if Path(args.retrieval_json).exists():
        rj = json.loads(Path(args.retrieval_json).read_text())
        retrieval_smiles = [p["smiles"] for p in rj.get("top_k", [])]
        print(f"Retrieval prefix ({len(retrieval_smiles)} SMILES):", flush=True)
        for i, s in enumerate(retrieval_smiles):
            print(f"  [{i+1}] {s}", flush=True)
    else:
        print(f"[warn] retrieval json not found at {args.retrieval_json}",
               flush=True)

    with torch.no_grad():
        prefix_embs = None
        if retrieval_smiles:
            prefix_embs = model.encode_smiles_to_prefix_embs(retrieval_smiles)
            print(f"Encoded retrieval prefix: shape={tuple(prefix_embs.shape)}",
                   flush=True)

    scales = [float(s) for s in args.cfg_scales.split(",") if s.strip()]
    modes = [m.strip().lower() for m in args.retrieval_modes.split(",")
                if m.strip()]

    cells = []
    for r in modes:
        for s in scales:
            cells.append((s, r))

    print(f"\nWill sample {len(cells)} cells x n={args.n}", flush=True)
    for (s, r) in cells:
        s_tag = f"{s:g}".replace(".", "p")
        cell_label = f"cfg_s{s_tag}_retrieval_{r}"
        if args.tag:
            cell_label = f"{cell_label}_{args.tag}"
        out_csv = out_dir / f"samples_{cell_label}.csv"
        print(f"\n=== Cell {cell_label}  (s={s}, retrieval={r}) ===", flush=True)
        write_progress(progress_path, phase="starting_cell", cell=cell_label)
        prefix_to_use = prefix_embs if r == "on" else None
        sample_cell(model, args.anchor_smi, zap70_emb, zap70_mask,
                     mol1_pose_norm, s, prefix_to_use, args.n,
                     args.batch_size, args.max_length, args.temperature,
                     out_csv, progress_path, cell_label)
        print(f"  wrote {out_csv}", flush=True)

    write_progress(progress_path, phase="sampling_done", cells=len(cells))
    print("\nALL CELLS DONE.", flush=True)


if __name__ == "__main__":
    main()
