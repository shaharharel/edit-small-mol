"""Sample the v5 retrieval-K sweep on the covFT+v2 baseline.

Cells: K ∈ {0 (baseline, no retrieval), 1, 5, 10, 20} — 5 cells × N=200.
For K=0, no retrieval prefix is prepended (identical to v3 baseline).

Model: covFT+v2 (`m1a_v2.ckpt`) loaded into the FiLM wrapper with
`film_enabled=False` so the FiLM path is inert (zero-init parameters +
disabled flag).  This gives the EXACT v2 sampling pathway with retrieval
prefix support.

Outputs one CSV per cell: samples_K{K}.csv (cell name `retrievalK_K{K}`).
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

from m1a_v2_film_model import load_film_model  # noqa
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


def write_progress(p, **fields):
    rec = {"timestamp": time.time(),
             "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    Path(p).write_text(json.dumps(rec, indent=2))


def sample_cell(model, anchor_smi, emb_np, mask_np, pose_np,
                  prefix_embs, K,
                  n, batch_size, max_length, temperature,
                  out_csv, progress_path, cell_label):
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    n_done = 0
    if not write_header:
        with open(out_csv) as f:
            for _ in csv.DictReader(f):
                n_done += 1
        print(f"  resume: {n_done} rows in {out_csv}", flush=True)
    fp = open(out_csv, "a", newline="")
    fieldnames = ["cell", "K", "sample_idx", "SMILES", "Input_SMILES", "NLL"]
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
        re = torch.from_numpy(np.tile(emb_np[None], (batch_n, 1, 1))).to(device)
        rm = torch.from_numpy(np.tile(mask_np[None], (batch_n, 1))).to(device)
        po = torch.from_numpy(np.tile(pose_np[None], (batch_n, 1))).to(device)
        # film_enabled=False for the whole v5 sweep -> pure covFT+v2 pathway.
        out_smiles, nlls = model.sample_multinomial_film_toggle(
            src_t, src_mask, re, rm, po, film_enabled=False,
            retrieval_prefix_embs=prefix_embs,
            max_length=max_length, temperature=temperature)
        for i, (s, nll, anc) in enumerate(zip(out_smiles, nlls, anchors)):
            writer.writerow({"cell": cell_label, "K": K,
                              "sample_idx": n_done + i,
                              "SMILES": s, "Input_SMILES": anc,
                              "NLL": float(nll)})
        fp.flush()
        n_done += batch_n
        el = time.time() - t0
        rate = max(1, n_done) / max(el, 1e-6)
        eta = (n - n_done) / max(rate, 1e-6)
        write_progress(progress_path, phase="sampling", cell=cell_label,
                         n_done=n_done, n_total=n,
                         rate=round(rate, 2), eta_sec=round(eta, 1))
        if n_done % max(1, batch_size * 2) == 0 or n_done == n:
            print(f"  [{cell_label}] {n_done}/{n} {rate:.1f} mol/s ETA {eta:.0f}s",
                   flush=True)
    fp.close()


def load_zap70_pocket_and_pose(mol1_esm_npz, mol1_pose_npz, r_max):
    p = np.load(mol1_esm_npz, allow_pickle=True)
    emb = p["residues_emb"][0]; mask = p["residues_mask"][0]
    if emb.shape[0] < r_max:
        pad = np.zeros((r_max - emb.shape[0], emb.shape[1]), dtype=emb.dtype)
        emb = np.concatenate([emb, pad], axis=0)
        mp = np.zeros((r_max - mask.shape[0],), dtype=bool)
        mask = np.concatenate([mask, mp], axis=0)
    elif emb.shape[0] > r_max:
        emb = emb[:r_max]; mask = mask[:r_max]
    pd = np.load(mol1_pose_npz, allow_pickle=True)
    return emb, mask, pd["pose_norm"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    # For v5 we need the covFT+v2 baseline weights.  Load into FiLM wrapper
    # (FiLM params zero-init and film_enabled=False -> identity path).
    ap.add_argument("--v2_ckpt", default=str(PROJECT_ROOT /
                     "models/m1a_v2.ckpt"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--retrieval_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v5_retrievalK"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v5_retrievalK"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--Ks", default="0,1,5,10,20")
    args = ap.parse_args()
    print(f"args = {vars(args)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "sampling_progress.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]
    print(f"Pose normalizer: mean={pose_mean} std={pose_std}", flush=True)

    print(f"Loading v2 weights into FiLM wrapper (film_enabled=False)"
           f" from {args.v2_ckpt}", flush=True)
    model = load_film_model(args.prior, device,
                              pose_mean=pose_mean, pose_std=pose_std,
                              init_from_v2_ckpt=args.v2_ckpt)
    model.film_enabled = False
    model.eval()

    cache = np.load(args.cache, allow_pickle=True)
    r_max = cache["residues_emb"].shape[1]
    emb, mask, pose = load_zap70_pocket_and_pose(
        args.mol1_pocket_npz, args.mol1_pose_npz, r_max)
    print(f"ZAP70 pocket: {int(mask.sum())} residues; pose_norm={pose}",
           flush=True)

    Ks = [int(k) for k in args.Ks.split(",")]
    print(f"K sweep: {Ks}", flush=True)

    for K in Ks:
        cell_label = f"retrievalK_K{K}"
        out_csv = out_dir / f"samples_{cell_label}.csv"
        prefix_embs = None
        retrieval_smiles = []
        if K > 0:
            rj_path = Path(args.retrieval_dir) / f"retrieval_top{K}.json"
            if not rj_path.exists():
                print(f"[skip K={K}] no {rj_path}", flush=True)
                continue
            rj = json.loads(rj_path.read_text())
            retrieval_smiles = [p["smiles"] for p in rj.get("top_k", [])][:K]
            with torch.no_grad():
                prefix_embs = model.encode_smiles_to_prefix_embs(retrieval_smiles)
        print(f"\n=== Cell {cell_label} (K={K}, {len(retrieval_smiles)} prefix SMILES) ===",
               flush=True)
        sample_cell(model, args.anchor_smi, emb, mask, pose,
                       prefix_embs=prefix_embs, K=K,
                       n=args.n, batch_size=args.batch_size,
                       max_length=args.max_length,
                       temperature=args.temperature,
                       out_csv=out_csv, progress_path=progress_path,
                       cell_label=cell_label)

    write_progress(progress_path, phase="sampling_done", cells=len(Ks))
    print("\nALL K CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
