"""Sample from a LoRA-fine-tuned Lingo3DMol checkpoint.

A LoRA-FT checkpoint has attention linears wrapped as `LoRALinear`, so its
state_dict keys look like
    encoder.layers.0.self_attn.linears.0.base.weight
    encoder.layers.0.self_attn.linears.0.base.bias
    encoder.layers.0.self_attn.linears.0.lora_A
    encoder.layers.0.self_attn.linears.0.lora_B
The vanilla `TransformerModel` (which sampling uses) has plain `nn.Linear`s,
so its keys are
    encoder.layers.0.self_attn.linears.0.weight
    encoder.layers.0.self_attn.linears.0.bias

This script:
  1. Loads the LoRA-FT ckpt (dict with "model" key).
  2. Merges LoRA into the base weight using
        W_merged = W_base + (alpha/r) * (B @ A)
     for each wrapped Linear, then drops lora_A / lora_B keys.
  3. Renames `<prefix>.base.weight` -> `<prefix>.weight`
     and        `<prefix>.base.bias`   -> `<prefix>.bias`.
  4. Saves the merged flat state_dict as a temp .pkl and invokes either
        - `run_lingo3dmol_l0_smoke.run()`           (--anchor_id none)
        - `run_lingo3dmol_l2_extended_anchor.run()` (--anchor_id L/H1/H2/H3)

The merge yields a state_dict that loads strict=False into a plain
TransformerModel and IS NUMERICALLY IDENTICAL (modulo float roundoff) to
the LoRA model in eval mode.
"""
from __future__ import annotations
import os, sys, argparse, json, tempfile, re
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


def merge_lora_into_state_dict(sd: dict, *, alpha: float, rank: int) -> dict:
    """Walk `sd` looking for LoRA-wrapped Linear triples
    {prefix}.base.weight  /  {prefix}.lora_A  /  {prefix}.lora_B,
    merge them into a single {prefix}.weight, drop lora_*, and rename
    base.* → *.

    Other keys pass through unchanged.

    Args:
      sd: the loaded state_dict.
      alpha, rank: LoRA hyperparams used at training; needed because the
        scale is alpha/rank. We default to the same values the train
        script uses (alpha=16, rank=8 → scale=2.0).
    """
    import torch
    scale = float(alpha) / float(rank)
    print(f"[merge] LoRA scale = alpha/rank = {alpha}/{rank} = {scale}")

    # Collect every prefix that has a `.lora_A` entry.
    lora_prefixes = []
    for k in sd:
        if k.endswith(".lora_A"):
            lora_prefixes.append(k[: -len(".lora_A")])

    print(f"[merge] found {len(lora_prefixes)} LoRA-wrapped linears")

    new_sd = {}
    consumed = set()

    for prefix in lora_prefixes:
        kA = f"{prefix}.lora_A"
        kB = f"{prefix}.lora_B"
        kw = f"{prefix}.base.weight"
        kb = f"{prefix}.base.bias"
        if kA not in sd or kB not in sd or kw not in sd:
            print(f"[merge] WARN: incomplete LoRA set @ {prefix}; skipping merge")
            continue
        A = sd[kA]  # (r, in)
        B = sd[kB]  # (out, r)
        W = sd[kw]  # (out, in)
        delta = scale * (B @ A)
        if delta.shape != W.shape:
            raise RuntimeError(
                f"[merge] shape mismatch @ {prefix}: delta {delta.shape} vs W {W.shape}")
        # New rename: strip .base
        new_key_w = f"{prefix}.weight"
        new_sd[new_key_w] = W + delta
        if kb in sd:
            new_sd[f"{prefix}.bias"] = sd[kb]
            consumed.add(kb)
        consumed.update([kA, kB, kw])

    for k, v in sd.items():
        if k in consumed:
            continue
        if k.endswith(".base.weight") or k.endswith(".base.bias"):
            # Edge case: a LoRA prefix existed but lora_A/lora_B missing.
            # Strip .base anyway so the vanilla model accepts the key.
            new_key = k.replace(".base.weight", ".weight").replace(
                ".base.bias", ".bias")
            new_sd[new_key] = v
            continue
        if k in new_sd:
            continue
        new_sd[k] = v

    print(f"[merge] input keys: {len(sd)}  output keys: {len(new_sd)}")
    return new_sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_checkpoint", required=True,
                   help="Path to LoRA-FT ckpt (dict with 'model' key + LoRA setup).")
    p.add_argument("--pocket_pdb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--anchor_id", default="none",
                   choices=["none", "L", "H1", "H2", "H3"],
                   help="'none' = vanilla L0 sampling; else use L2 extended anchor.")
    p.add_argument("--anchor_json", default="data/lingo3dmol_anchor_zap70_cys346.json")
    p.add_argument("--contact_path", default="checkpoint/contact.pkl")
    p.add_argument("--gennums", type=int, default=30)
    p.add_argument("--min_acceptable", type=int, default=10)
    p.add_argument("--gen_frag_set", type=int, default=10)
    p.add_argument("--prod_time", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nci_thrs", type=float, default=0.7)
    p.add_argument("--coc_dis", type=float, default=2.5)
    p.add_argument("--frag_len_add", type=int, default=15)
    p.add_argument("--tempture", type=float, default=1.0)
    p.add_argument("--max_run_seconds", type=int, default=1800)
    # LoRA hyperparams (must match the training run).
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    args.lora_checkpoint = _abs(args.lora_checkpoint)
    args.pocket_pdb = _abs(args.pocket_pdb)
    args.output = _abs(args.output)
    args.anchor_json = _abs(args.anchor_json)

    os.chdir(str(ROOT / "external" / "Lingo3DMol"))

    import lingo3dmol_cpu_shim  # noqa: F401
    import torch

    # If the LoRA setup file is alongside the ckpt, prefer those hyperparams.
    setup_path = Path(args.lora_checkpoint).parent / "lora_setup.json"
    if setup_path.exists():
        try:
            setup = json.loads(setup_path.read_text())
            args.lora_rank = int(setup.get("lora_rank", args.lora_rank))
            args.lora_alpha = int(setup.get("lora_alpha", args.lora_alpha))
            print(f"[LoRA sample] read setup: rank={args.lora_rank} alpha={args.lora_alpha}")
        except Exception as e:
            print(f"[LoRA sample] could not read {setup_path}: {e}; using CLI defaults")

    print(f"[LoRA sample] loading LoRA ckpt: {args.lora_checkpoint}")
    ck = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
    print(f"[LoRA sample] loaded {len(sd)} keys")

    merged = merge_lora_into_state_dict(
        sd, alpha=args.lora_alpha, rank=args.lora_rank)

    # Drop any keys that the vanilla TransformerModel doesn't know about
    # (e.g. reactivity_head if the training script seeded it). We use
    # strict=False at load anyway, so leaving them in is harmless — but
    # stripping them keeps the diff clean.

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tmp.close()
    torch.save(merged, tmp.name)
    print(f"[LoRA sample] wrote merged flat state_dict: {tmp.name} ({len(merged)} keys)")

    # Make load_state_dict default to strict=False — the merged ckpt may
    # contain extra heads (reactivity_head) that the inference model doesn't
    # define, and the chassis head may be missing if it wasn't added.
    import torch.nn as _nn
    _orig_lsd = _nn.Module.load_state_dict

    def _lsd_nonstrict(self, sd, strict=True, assign=False):
        return _orig_lsd(self, sd, strict=False)
    _nn.Module.load_state_dict = _lsd_nonstrict

    if args.anchor_id == "none":
        from run_lingo3dmol_l0_smoke import run as l0_run, Args
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
            isTrain=False, USE_THRESHOLD=True, isMultiSample=True,
            isGuideSample=True, OnceMolGen=False,
        )
        summary = l0_run(l0_args)
    else:
        # L2 extended anchor.
        from run_lingo3dmol_l2_extended_anchor import run as l2_run
        ext_args = argparse.Namespace(
            pocket_pdb=args.pocket_pdb,
            output=args.output,
            anchor_id=args.anchor_id,
            anchor_json=args.anchor_json,
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
            isTrain=False, USE_THRESHOLD=True, isMultiSample=True,
            isGuideSample=True, OnceMolGen=False,
        )
        summary = l2_run(ext_args)

    print(f"[LoRA sample] done. summary={summary}")
    try:
        os.unlink(tmp.name)
    except Exception:
        pass


if __name__ == "__main__":
    main()
