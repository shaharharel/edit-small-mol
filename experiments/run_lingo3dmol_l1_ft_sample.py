"""L1-FT sampling driver — load a fine-tuned checkpoint and run vanilla
(L0-style) sampling at a given temperature.

Wraps `run_lingo3dmol_l0_smoke.run(args)` by:

  1. Loading the FT checkpoint (expected format: `{"model": state_dict, ...}`).
  2. Extracting the inner state_dict and saving it as a flat .pkl at a
     temp path that mimics what `gen_mol.pkl` looks like.
  3. Invoking the L0 smoke pipeline with `--caption_path` pointed at the
     temp path.

This is the simplest possible path that keeps `run_lingo3dmol_l0_smoke.py`
untouched.
"""
from __future__ import annotations
import os, sys, argparse, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG_CWD = Path(os.getcwd()).resolve()
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    cand = (ORIG_CWD / p).resolve()
    if cand.exists():
        return str(cand)
    cand2 = (ROOT / p).resolve()
    if cand2.exists():
        return str(cand2)
    return str(cand)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft_checkpoint", required=True,
                   help="Path to FT checkpoint (dict with 'model' key).")
    p.add_argument("--pocket_pdb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--contact_path", default="checkpoint/contact.pkl")
    p.add_argument("--gennums", type=int, default=30)
    p.add_argument("--min_acceptable", type=int, default=5)
    p.add_argument("--gen_frag_set", type=int, default=20)
    p.add_argument("--prod_time", type=int, default=1)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nci_thrs", type=float, default=0.7)
    p.add_argument("--coc_dis", type=float, default=2.5)
    p.add_argument("--frag_len_add", type=int, default=15)
    p.add_argument("--tempture", type=float, default=1.0)
    p.add_argument("--max_run_seconds", type=int, default=1800)
    args = p.parse_args()

    args.ft_checkpoint = _abs(args.ft_checkpoint)
    args.pocket_pdb = _abs(args.pocket_pdb)
    args.output = _abs(args.output)

    # Need to chdir into the Lingo3DMol root for relative paths to work
    os.chdir(str(ROOT / "external" / "Lingo3DMol"))

    import lingo3dmol_cpu_shim  # noqa: F401
    import torch

    # Extract inner state dict from FT checkpoint.
    print(f"[L1-FT sample] loading FT ckpt: {args.ft_checkpoint}")
    ck = torch.load(args.ft_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
        print(f"[L1-FT sample] extracted state_dict from FT wrapper "
              f"({len(sd)} keys)")
    else:
        sd = ck
        print(f"[L1-FT sample] using raw state_dict ({len(sd)} keys)")

    # Save flat state-dict to a temp path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tmp.close()
    torch.save(sd, tmp.name)
    print(f"[L1-FT sample] wrote flat state_dict: {tmp.name}")

    # Now invoke the L0 sampling pipeline with our caption_path.
    # Monkey-patch torch.nn.Module.load_state_dict to default to strict=False
    # — the model has been extended with extra heads (chassis) that aren't in
    # the FT'd state_dict, but the inference path doesn't use them.
    import torch.nn as _nn
    _orig_lsd = _nn.Module.load_state_dict
    def _lsd_nonstrict(self, sd, strict=True, assign=False):
        return _orig_lsd(self, sd, strict=False)
    _nn.Module.load_state_dict = _lsd_nonstrict

    from run_lingo3dmol_l0_smoke import run, Args

    l0_args = Args(
        pocket_pdb=args.pocket_pdb,
        output=args.output,
        contact_path=args.contact_path,
        caption_path=tmp.name,
        gennums=args.gennums,
        min_acceptable=args.min_acceptable,
        gen_frag_set=args.gen_frag_set,
        prod_time=args.prod_time,
        topk=args.topk,
        nci_thrs=args.nci_thrs,
        coc_dis=args.coc_dis,
        frag_len_add=args.frag_len_add,
        tempture=args.tempture,
        max_run_seconds=args.max_run_seconds,
        isTrain=False,
        USE_THRESHOLD=True,
        isMultiSample=True,
        isGuideSample=True,
        OnceMolGen=False,
    )

    summary = run(l0_args)
    print(f"[L1-FT sample] done: {summary}")

    # Clean up temp ckpt
    try:
        os.unlink(tmp.name)
    except Exception:
        pass


if __name__ == "__main__":
    main()
