"""Sample Mol1-anchored from a v1-style mol2mol checkpoint (loaded from ~/edit-small-mol/data/v1_ft_on_v2/).
Writes N=200 SMILES for planar-dihedral analysis, matching paper's variant-A protocol but WITHOUT pose/pocket.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
from optionA_mol2mol import Mol2MolTransformer, subsequent_mask, tokenize_smiles, detokenize

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    vocab = ck["vocab"]
    # rebuild model matching prior architecture
    from optionA_mol2mol import Mol2MolTransformer
    model = Mol2MolTransformer(vocab_size=len(vocab["tokens"]), N=6, d=256, h=8, d_ff=2048, dropout=0.1, max_len=5000)
    model.load_state_dict(ck["state_dict"], strict=False)
    return model.to(device).eval(), vocab


def sample(model, vocab, anchor, n, max_len, batch_size, device, temperature=1.0):
    tokens = vocab["tokens"]
    bos, eos, pad = vocab["bos_token"], vocab["eos_token"], vocab["pad_token"]
    inv = {v: k for k, v in tokens.items()}
    ids = tokenize_smiles(anchor, tokens, bos, eos, pad, max_len)
    src_single = torch.tensor([ids], dtype=torch.long, device=device)
    out_smis = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b = min(batch_size, n - i)
            src = src_single.repeat(b, 1)
            src_mask = (src != pad).unsqueeze(1)
            memory = model.encode(src, src_mask)
            ys = torch.full((b, 1), bos, dtype=torch.long, device=device)
            done = torch.zeros(b, dtype=torch.bool, device=device)
            for _ in range(max_len - 1):
                tm = subsequent_mask(ys.size(1)).to(device)
                h = model.decode(memory, src_mask, ys, tm)
                logp = model.generator(h[:, -1]) / temperature
                probs = logp.exp()
                nxt = torch.multinomial(probs, 1)
                nxt = torch.where(done.unsqueeze(-1), torch.full_like(nxt, eos), nxt)
                ys = torch.cat([ys, nxt], dim=1)
                done = done | (nxt.squeeze(-1) == eos)
                if done.all(): break
            for r in range(b):
                out_smis.append(detokenize(ys[r].tolist(), inv, bos, eos, pad))
    return out_smis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "data" / "v1_ft_on_v2" / "v1_ft_on_v2_pairs.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=str(REPO / "data" / "v1_ft_on_v2" / "samples_mol1_anchored.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[sample_v1_ft] device={device}, ckpt={args.ckpt}", flush=True)
    model, vocab = load_ckpt(args.ckpt, device)
    t0 = time.time()
    smis = sample(model, vocab, MOL1_SMI, args.n, args.max_len, args.batch_size, device, args.temperature)
    print(f"[sample_v1_ft] sampled {len(smis)} in {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame({"SMILES": smis, "anchor": MOL1_SMI}).to_csv(args.out, index=False)
    print(f"[sample_v1_ft] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
