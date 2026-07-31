"""QA: validate a seed .smi file against REINVENT4's actual input parser.

Uses REINVENT4's `convert_to_standardized_smiles` and `validate_tokens` to
catch the EXACT failure modes that block RL launches. This is the QA we should
have run BEFORE first launch — instead we discovered each issue at runtime.

Run locally on the V100 (where REINVENT4 is installed), or via SSH.

Usage:
  python experiments/qa_reinvent_seed.py <seed.smi> <prior.prior>
"""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: qa_reinvent_seed.py <seed.smi> <prior_or_chkpt>")
        sys.exit(2)
    seed_path = sys.argv[1]
    prior_path = sys.argv[2]

    from reinvent.utils.config_parse import read_smiles_csv_file
    from reinvent.models.model_factory.sample_model_factory import SampleModelFactory
    from reinvent.runmodes.generator.adapters import sampling_adapter

    print(f"=== QA seed file: {seed_path}")
    print(f"=== Against prior: {prior_path}\n")

    # Load model to get its token vocab
    try:
        from reinvent.models.model_factory.transformer_adapter import TransformerAdapter
        model = TransformerAdapter.load_from_file(prior_path)
        allowed_tokens = model.get_vocabulary().tokens()
    except Exception as e:
        # Fallback: try the more general route
        import torch
        ckpt = torch.load(prior_path, map_location="cpu", weights_only=False)
        # Extract vocab tokens — try common locations
        if "vocabulary" in ckpt:
            allowed_tokens = set(ckpt["vocabulary"].tokens() if hasattr(ckpt["vocabulary"], "tokens") else ckpt["vocabulary"])
        elif isinstance(ckpt, dict) and "model" in ckpt:
            allowed_tokens = set(ckpt["model"].vocabulary.tokens())
        else:
            print(f"Could not extract vocab: {e}")
            sys.exit(1)
    print(f"Model vocab: {len(allowed_tokens)} tokens")

    # Use REINVENT4's parser
    try:
        smilies = read_smiles_csv_file(seed_path, 0, allowed_tokens)
        print(f"[PASS] All {len(smilies):,} SMILES validate against the prior's token vocab.")
    except ValueError as e:
        msg = str(e)
        # Show first offending SMILES
        print(f"[FAIL] {msg[:500]}")
        sys.exit(1)

    # Also test: each SMILES roundtrips through convert_to_standardized_smiles
    from reinvent.chemistry import conversions
    n_fail = 0
    failures = []
    for s in smilies[:1000]:  # Sample first 1000 for speed
        try:
            conversions.convert_to_standardized_smiles(s)
        except Exception as e:
            n_fail += 1
            if len(failures) < 5:
                failures.append((s, str(e)[:100]))
    if n_fail:
        print(f"[FAIL] {n_fail}/1000 SMILES failed convert_to_standardized_smiles")
        for s, err in failures:
            print(f"   {s[:80]}  ->  {err}")
        sys.exit(1)
    print(f"[PASS] First 1000 SMILES roundtrip through convert_to_standardized_smiles.")
    print(f"\nSeed file is REINVENT-compatible.")


if __name__ == "__main__":
    main()
