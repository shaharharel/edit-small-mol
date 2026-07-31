"""v9 per-target sampler: uses a specified target's pocket ESM + anchor SMILES.

For a given target (BTK|EGFR|JAK3|KRAS_G12C), the target's pocket ESM
embedding is loaded from the v2 cache by struct_id. Pose vector is
zero-normalized (no target-specific pose available for non-ZAP70 targets).
Anchor SMILES is the target's canonical covalent inhibitor.

Outputs samples with per-target cell tags.
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

TARGETS = {
    "BTK": {
        "struct_id_pat": "5P9J_A_",
        "anchor_smi": "C=CC(=O)N1CCC[C@@H]1COc1cc(-c2nn(C)c3ncnc(N)c23)ccc1Nc1ccccc1",  # ibrutinib-like
        "anchor_note": "ibrutinib acrylamide (canonical BTK inhibitor)",
    },
    "EGFR": {
        "struct_id_pat": "5GTY_A_",
        "anchor_smi": "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",   # afatinib
        "anchor_note": "afatinib (EGFR)",
    },
    "JAK3": {
        "struct_id_pat": "5TOZ_A_",
        "anchor_smi": "C=CC(=O)N1CCC[C@@H]1n1cc(-c2ncnc3[nH]ccc23)cn1",   # PF-06651600-like
        "anchor_note": "PF-06651600-like (JAK3)",
    },
    "KRAS_G12C": {
        "struct_id_pat": "6OIM_A_",
        "anchor_smi": "C=CC(=O)N1CCN(C(=O)c2c(C)cc(-c3ccncc3)nc2Nc2cc(OC)nc(C)n2)CC1",   # sotorasib-like
        "anchor_note": "sotorasib (KRAS-G12C)",
    },
}


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


def sample_target_cell(model, anchor_smi, emb_np, mask_np, pose_np,
                          prefix_embs,
                          n, batch_size, max_length, temperature,
                          out_csv, progress_path, cell_label,
                          target, dpo, retrieval):
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    n_done = 0
    if not write_header:
        with open(out_csv) as f:
            for _ in csv.DictReader(f):
                n_done += 1
    fp = open(out_csv, "a", newline="")
    fieldnames = ["cell", "target", "dpo", "retrieval", "sample_idx",
                    "SMILES", "Input_SMILES", "NLL"]
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    if write_header: writer.writeheader()
    t0 = time.time()
    while n_done < n:
        batch_n = min(batch_size, n - n_done)
        anchors = [randomize_smi(anchor_smi) for _ in range(batch_n)]
        seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
        L = max(len(s) for s in seqs)
        src = np.zeros((batch_n, L), dtype=np.int64)
        for j, s in enumerate(seqs): src[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        re = torch.from_numpy(np.tile(emb_np[None], (batch_n, 1, 1))).to(device)
        rm = torch.from_numpy(np.tile(mask_np[None], (batch_n, 1))).to(device)
        po = torch.from_numpy(np.tile(pose_np[None], (batch_n, 1))).to(device)
        out_smiles, nlls = model.sample_multinomial_film_toggle(
            src_t, src_mask, re, rm, po, film_enabled=False,
            retrieval_prefix_embs=prefix_embs,
            max_length=max_length, temperature=temperature)
        for i, (s, nll, anc) in enumerate(zip(out_smiles, nlls, anchors)):
            writer.writerow({
                "cell": cell_label, "target": target,
                "dpo": int(dpo), "retrieval": int(retrieval),
                "sample_idx": n_done + i,
                "SMILES": s, "Input_SMILES": anc, "NLL": float(nll)})
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


def load_target_pocket(cache_npz, target_pat, r_max):
    """Return (emb, mask) for the first struct_id matching target_pat.
    Emb is R_max-padded/truncated to match training corpus."""
    d = np.load(cache_npz, allow_pickle=True)
    sids = [str(s) for s in d["struct_ids"]]
    row_seq_idx = d["row_seq_idx"]
    res_emb = d["residues_emb"]; res_mask = d["residues_mask"]
    for i, s in enumerate(sids):
        if target_pat.upper() in s.upper() or target_pat.lower() in s.lower():
            seq_idx = int(row_seq_idx[i])
            return res_emb[seq_idx], res_mask[seq_idx], s
    raise RuntimeError(f"No struct_id matching {target_pat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS.keys()))
    ap.add_argument("--prior", default=str(PROJECT_ROOT /
                     "models/reinvent4_mol2mol_covalent_ft.prior"))
    ap.add_argument("--dpo_ckpt", default=None,
                    help="If provided, sample with this DPO-trained ckpt.")
    ap.add_argument("--v2_ckpt", default=str(PROJECT_ROOT /
                     "models/m1a_v2.ckpt"))
    ap.add_argument("--cache", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--retrieval_json", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/retrieval_top5.json"))
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--progress_path", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--retrieval", action="store_true",
                    help="Include retrieval prefix.")
    args = ap.parse_args()
    print(f"args = {vars(args)}", flush=True)

    tinfo = TARGETS[args.target]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_in = np.load(args.cache, allow_pickle=True)
    pose_mean = d_in["pose_mean"]; pose_std = d_in["pose_std"]

    # Load target pocket.
    cache = np.load(args.cache, allow_pickle=True)
    r_max = cache["residues_emb"].shape[1]
    emb, mask, sid_used = load_target_pocket(args.cache,
                                                  tinfo["struct_id_pat"], r_max)
    print(f"Target {args.target}: struct_id={sid_used} residues={int(mask.sum())}",
           flush=True)

    # Neutral pose (target has no ZAP70-specific pose; use pose_mean = 0 after norm).
    # In normalised space, that's zeros.
    pose_norm = np.zeros(3, dtype=np.float32)

    # Model
    if args.dpo_ckpt is not None and Path(args.dpo_ckpt).exists():
        print(f"Loading DPO ckpt {args.dpo_ckpt}", flush=True)
        model = load_film_model(args.prior, device,
                                   pose_mean=pose_mean, pose_std=pose_std,
                                   ckpt_path=args.dpo_ckpt, strict=True)
        dpo_flag = True
    else:
        print(f"Loading covFT+v2 baseline {args.v2_ckpt}", flush=True)
        model = load_film_model(args.prior, device,
                                   pose_mean=pose_mean, pose_std=pose_std,
                                   init_from_v2_ckpt=args.v2_ckpt)
        dpo_flag = False
    model.film_enabled = False
    model.eval()

    prefix_embs = None
    if args.retrieval and Path(args.retrieval_json).exists():
        rj = json.loads(Path(args.retrieval_json).read_text())
        retrieval_smiles = [p["smiles"] for p in rj.get("top_k", [])]
        with torch.no_grad():
            prefix_embs = model.encode_smiles_to_prefix_embs(retrieval_smiles)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_path) if args.progress_path \
        else out_csv.parent / f"{out_csv.stem}_progress.json"

    cell_label = (f"{args.target}_dpo_{'on' if dpo_flag else 'off'}"
                    f"_retrieval_{'on' if args.retrieval else 'off'}")
    print(f"\n=== {cell_label} anchor={tinfo['anchor_note']} ===", flush=True)
    sample_target_cell(model, tinfo["anchor_smi"], emb, mask, pose_norm,
                          prefix_embs=prefix_embs,
                          n=args.n, batch_size=args.batch_size,
                          max_length=args.max_length,
                          temperature=args.temperature,
                          out_csv=out_csv,
                          progress_path=progress_path,
                          cell_label=cell_label,
                          target=args.target, dpo=dpo_flag,
                          retrieval=args.retrieval)


if __name__ == "__main__":
    main()
