"""Sample a Mol1-anchored cohort from a DPO-trained Mol2Mol checkpoint.

Multinomial sampling with randomized input SMILES per spec.

Output CSV columns: SMILES, Input_SMILES, NLL
"""
from __future__ import annotations
import argparse
import csv
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

try:
    from reinvent.runmodes.create_adapter import create_adapter
    from reinvent.models.transformer.core.dataset.dataset import Dataset
    from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
    from reinvent.chemistry import conversions
except Exception as e:
    sys.stderr.write(f"REINVENT4 import failed: {e}\n")
    raise

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sample_dpo_cohort")

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def randomize(s: str) -> str:
    try:
        std = conversions.convert_to_standardized_smiles(s)
    except Exception:
        std = s
    try:
        m = conversions.smile_to_mol(std)
        if m is not None:
            return conversions.mol_to_random_smiles(m, isomericSmiles=True)
    except Exception:
        pass
    return std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--anchor", default=MOL1)
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--randomize", action="store_true", default=True)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Loading {args.ckpt} on {device}")
    agent, _, mt = create_adapter(args.ckpt, "inference", device)
    assert mt == "Mol2Mol"
    if hasattr(agent, "set_temperature"):
        try:
            agent.set_temperature(args.temperature)
        except Exception:
            pass

    out_path = Path(args.out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w", newline=""); w = csv.writer(fh)
    w.writerow(["SMILES", "Input_SMILES", "NLL"])

    tokenizer = SMILESTokenizer()
    vocab = agent.get_vocabulary()

    n_done = 0
    t0 = time.time()
    agent.set_mode("inference")
    while n_done < args.n_samples:
        bsz = min(args.batch_size, args.n_samples - n_done)
        # generate fresh randomized inputs each batch for diversity
        seeds = [randomize(args.anchor) for _ in range(bsz)]
        ds = Dataset(seeds, vocab, tokenizer)
        loader = tud.DataLoader(ds, batch_size=bsz, shuffle=False, collate_fn=Dataset.collate_fn)
        for src, src_mask in loader:
            src = src.to(device); src_mask = src_mask.to(device)
            with torch.no_grad():
                sb = agent.sample(src, src_mask, "multinomial")
            outs = list(sb.output)
            ins = list(sb.input)
            nlls = sb.nlls
            if isinstance(nlls, torch.Tensor):
                nlls = nlls.detach().cpu().tolist()
            for inp, out, nll in zip(ins, outs, nlls):
                w.writerow([out or "", inp or "", f"{float(nll):.4f}"])
            n_done += len(outs)
        if n_done % (max(1, args.batch_size) * 50) == 0 or n_done >= args.n_samples:
            logger.info(f"n_done={n_done}/{args.n_samples}  elapsed={time.time()-t0:.1f}s")
        fh.flush()
    fh.close()
    logger.info(f"Done. wrote {out_path}  ({n_done} rows in {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
