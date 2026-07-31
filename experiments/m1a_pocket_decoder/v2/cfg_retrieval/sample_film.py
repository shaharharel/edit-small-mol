"""Sample the v4 FiLM 2x2 ablation matrix: {FiLM on/off} x {retrieval on/off}.

For the FiLM=off baseline: we use the SAME model checkpoint but toggle
`film_enabled` off at sampling time.  Since FiLM was zero-init and only
adds to the base decoder's output, FiLM=off recovers the pure v2 forward
path (with retrained base weights).

If `--use_v3_baseline_for_film_off`, we instead reuse the v3 samples for
the s=1.0 cell (identical mathematical baseline) — saves compute AND makes
the comparison a strict apples-to-apples with the v3 numbers.

Outputs to `data/paper_pair_training/cfg_retrieval_v4_film/samples_*.csv`
Same schema as v3 sample CSVs.
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


def write_progress(progress_path, **fields):
    rec = {"timestamp": time.time(),
           "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    Path(progress_path).write_text(json.dumps(rec, indent=2))


def sample_cell(model, anchor_smi, emb_np, mask_np, pose_np,
                  film_enabled, prefix_embs,
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
    fieldnames = ["cell", "film_enabled", "retrieval", "sample_idx",
                    "SMILES", "Input_SMILES", "NLL"]
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    t0 = time.time()
    while n_done < n:
        batch_n = min(batch_size, n - n_done)
        anchors = [randomize_smi(anchor_smi) for _ in range(batch_n)]
        seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
        L = max(len(s) for s in seqs)
        src = np.zeros((batch_n, L), dtype=np.int64)
        for j, s in enumerate(seqs):
            src[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        re = torch.from_numpy(np.tile(emb_np[None], (batch_n, 1, 1))).to(device)
        rm = torch.from_numpy(np.tile(mask_np[None], (batch_n, 1))).to(device)
        po = torch.from_numpy(np.tile(pose_np[None], (batch_n, 1))).to(device)
        out_smiles, nlls = model.sample_multinomial_film_toggle(
            src_t, src_mask, re, rm, po, film_enabled=film_enabled,
            retrieval_prefix_embs=prefix_embs,
            max_length=max_length, temperature=temperature)
        retr_on = int(prefix_embs is not None and prefix_embs.numel() > 0)
        film_int = int(film_enabled)
        for i, (s, nll, anc) in enumerate(zip(out_smiles, nlls, anchors)):
            writer.writerow({"cell": cell_label,
                              "film_enabled": film_int, "retrieval": retr_on,
                              "sample_idx": n_done + i,
                              "SMILES": s, "Input_SMILES": anc,
                              "NLL": float(nll)})
        fp.flush()
        n_done += batch_n
        el = time.time() - t0
        rate = max(1, n_done) / max(el, 1e-6)
        eta = (n - n_done) / max(rate, 1e-6)
        write_progress(progress_path, phase="sampling",
                         cell=cell_label, n_done=n_done, n_total=n,
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
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT /
                     "models/cfg_retrieval_v4_film/film_final.ckpt"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--mol1_pose_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz"))
    ap.add_argument("--mol1_pocket_npz", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz"))
    ap.add_argument("--retrieval_json", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/retrieval_top5.json"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v4_film"))
    ap.add_argument("--anchor_smi", default=MOL1_SMI)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()
    print(f"args = {vars(args)}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "sampling_progress.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading FiLM model: {args.ckpt}", flush=True)
    model = load_film_model(args.prior, device, ckpt_path=args.ckpt)
    model.eval()

    cache = np.load(args.cache, allow_pickle=True)
    r_max = cache["residues_emb"].shape[1]
    emb, mask, pose = load_zap70_pocket_and_pose(
        args.mol1_pocket_npz, args.mol1_pose_npz, r_max)
    print(f"ZAP70 pocket: {int(mask.sum())} residues; pose_norm={pose}",
           flush=True)

    retrieval_smiles = []
    if Path(args.retrieval_json).exists():
        rj = json.loads(Path(args.retrieval_json).read_text())
        retrieval_smiles = [p["smiles"] for p in rj.get("top_k", [])]
        print(f"Retrieval prefix ({len(retrieval_smiles)} SMILES)", flush=True)

    with torch.no_grad():
        prefix_embs = None
        if retrieval_smiles:
            prefix_embs = model.encode_smiles_to_prefix_embs(retrieval_smiles)
            print(f"Retrieval prefix shape: {tuple(prefix_embs.shape)}",
                   flush=True)

    # 4 cells: (film ∈ {off, on}) × (retrieval ∈ {off, on}).
    cells = []
    for film in [False, True]:
        for retr in [False, True]:
            f_tag = "on" if film else "off"
            r_tag = "on" if retr else "off"
            cell = f"film_{f_tag}_retrieval_{r_tag}"
            cells.append((cell, film, retr))

    print(f"Will sample {len(cells)} cells x n={args.n}", flush=True)
    for cell, film, retr in cells:
        print(f"\n=== Cell {cell} ===", flush=True)
        out_csv = out_dir / f"samples_{cell}.csv"
        pfx = prefix_embs if retr else None
        sample_cell(model, args.anchor_smi, emb, mask, pose,
                       film_enabled=film, prefix_embs=pfx,
                       n=args.n, batch_size=args.batch_size,
                       max_length=args.max_length, temperature=args.temperature,
                       out_csv=out_csv, progress_path=progress_path,
                       cell_label=cell)
    write_progress(progress_path, phase="sampling_done", cells=len(cells))
    print("\nALL CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
