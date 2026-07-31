#!/usr/bin/env python3
"""
REINVENT4 De Novo generation with a fixed warhead-anchor token prefix.

We patch the prior model's autoregressive sampling loop so the first N tokens
are fixed to encode the acrylamide warhead ``C=CC(=O)N``. The RNN consumes
those tokens (priming its hidden state) and then samples continuations
left-to-right. Every output therefore starts with the warhead.

This is a *prefix*, not a *filter*: each sequence is constructed with the
warhead tokens already committed at the start, then continued by sampling
from the prior's conditional distribution given that prefix.

Fallback path (if the prior somehow rejects the warhead context): also
generate a *filtered* baseline (sample → keep only ``C=CC(=O)N``-prefixed
SMILES) for comparison and reporting.

Outputs
-------
* ``data/reinvent4_warhead_prefix_samples/samples.smi`` -- 500 unique SMILES
  whose canonical form starts with ``C=CC(=O)N``.
* ``results/paper_evaluation/seq_method_experiments/warhead_prefix.json`` --
  experiment metadata, acceptance rates, diversity stats, fallback flag.
* ``/tmp/seq_exp4_warhead_prefix.md`` -- short human-readable report.

QA: all 500 outputs match SMARTS ``C=CC(=O)N`` and >100 unique canonicals.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
PRIOR_PATH = REINVENT4_ROOT / "priors" / "reinvent.prior"

DATA_OUT_DIR = PROJECT_ROOT / "data" / "reinvent4_warhead_prefix_samples"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "seq_method_experiments"
REPORT_PATH = Path("/tmp/seq_exp4_warhead_prefix.md")

WARHEAD_SMILES = "C=CC(=O)N"
WARHEAD_SMARTS = "C=CC(=O)N"

TARGET_N = 500
BATCH_SIZE = 256
MAX_BATCHES = 80  # 80 * 256 = 20480 attempts, cap on prefix path
DEFAULT_SEED = 42

# Add REINVENT4 to import path
sys.path.insert(0, str(REINVENT4_ROOT))


def encode_prefix_tokens(model, smiles: str) -> list[int]:
    """Tokenize SMILES (with start marker) and encode to vocab ids.

    Returns a list of integer token ids beginning with the start token
    ``^`` and ending with the last warhead character (no stop token).
    """
    tokens = model.tokenizer.tokenize(smiles, with_begin_and_end=False)
    start_id = int(model.vocabulary[model.tokenizer.tokenize("", with_begin_and_end=True)[0]])
    # Build [START_TOKEN] + warhead tokens
    prefix_tokens = ["^"] + tokens
    ids = [int(model.vocabulary[t]) for t in prefix_tokens]
    assert ids[0] == start_id, "Start token id mismatch"
    return ids


@torch.no_grad()
def sample_with_prefix(model, prefix_ids: list[int], batch_size: int):
    """Sample sequences from the prior conditioned on a fixed token prefix.

    Mirrors ``Model._sample`` but feeds the prefix through the RNN first
    so the hidden state is properly primed before free sampling begins.

    Returns (sequences, smiles, nlls).
    """
    device = model.device
    prefix_len = len(prefix_ids)
    # Build prefix tensor of shape (batch, prefix_len)
    prefix_tensor = torch.tensor(
        prefix_ids, dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch_size, -1).contiguous()

    # Prime the hidden state by feeding the full prefix at once.
    # The RNN consumes tokens 0..prefix_len-1 and emits logits at every
    # step; we keep only the final hidden state for continuation.
    logits, hidden_state = model.network(prefix_tensor)
    # The last column of logits gives the distribution for the *next* token
    # after the prefix.
    next_logits = logits[:, -1, :]
    log_probs_next = next_logits.log_softmax(dim=1)
    probs_next = next_logits.softmax(dim=1)
    next_input = torch.multinomial(probs_next, num_samples=1).view(-1)

    nlls = torch.zeros(batch_size, device=device)
    # NLL contribution of the *transition* into next_input. We deliberately
    # do not score the deterministic prefix tokens.
    nlls += model._nll_loss(log_probs_next, next_input)

    # Collected sequences: prefix tokens followed by sampled token, then loop.
    sequences = [prefix_tensor, next_input.view(-1, 1)]
    input_vector = next_input

    remaining = model.max_sequence_length - prefix_len - 1
    for _ in range(remaining):
        logits, hidden_state = model.network(
            input_vector.unsqueeze(1), hidden_state
        )
        logits = logits.squeeze(1)
        log_probs = logits.log_softmax(dim=1)
        probs = logits.softmax(dim=1)
        input_vector = torch.multinomial(probs, num_samples=1).view(-1)
        sequences.append(input_vector.view(-1, 1))
        nlls += model._nll_loss(log_probs, input_vector)
        if input_vector.sum().item() == 0:
            break

    concat = torch.cat(sequences, dim=1)
    smiles = [
        model.tokenizer.untokenize(model.vocabulary.decode(seq))
        for seq in concat.cpu().numpy()
    ]
    return concat, smiles, nlls


def validate_warhead(smiles: str, patt) -> tuple[bool, str | None]:
    """Parse SMILES, return (warhead_intact, canonical_smiles or None).

    warhead_intact means the canonical SMILES *also* contains the
    acrylamide substructure. We do not require it to be at the literal
    start of the canonical string (canonicalization may reorder atoms).
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    canon = Chem.MolToSmiles(mol)
    has_warhead = mol.HasSubstructMatch(patt)
    return has_warhead, canon


def diversity_stats(smiles_list: list[str]) -> dict:
    """Compute simple diversity statistics."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    valid = []
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid.append(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    n = len(fps)
    if n < 2:
        return {
            "n_valid": n,
            "n_unique_canonical": len(set(valid)),
            "mean_pairwise_tanimoto": None,
        }

    # Sub-sample if very large to keep runtime sane
    rng = np.random.default_rng(0)
    if n > 200:
        idx = rng.choice(n, size=200, replace=False)
        sub_fps = [fps[i] for i in idx]
    else:
        sub_fps = fps

    sims = []
    for i in range(len(sub_fps)):
        for j in range(i + 1, len(sub_fps)):
            sims.append(DataStructs.TanimotoSimilarity(sub_fps[i], sub_fps[j]))

    return {
        "n_valid": n,
        "n_unique_canonical": len(set(valid)),
        "mean_pairwise_tanimoto": float(np.mean(sims)) if sims else None,
        "median_pairwise_tanimoto": float(np.median(sims)) if sims else None,
    }


def write_smi(path: Path, smiles: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in smiles:
            f.write(s + "\n")


def main():
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)

    from reinvent.models.reinvent.models.model import Model
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    warhead_patt = Chem.MolFromSmarts(WARHEAD_SMARTS)

    print(f"Loading REINVENT4 prior: {PRIOR_PATH}")
    save_dict = torch.load(PRIOR_PATH, map_location="cpu", weights_only=False)
    model = Model.create_from_dict(
        save_dict, mode="inference", device=torch.device("cpu")
    )
    print(f"Prior loaded. vocab={len(model.vocabulary)}, "
          f"max_seq_len={model.max_sequence_length}")

    prefix_ids = encode_prefix_tokens(model, WARHEAD_SMILES)
    prefix_tokens = [model.vocabulary[i] for i in prefix_ids]
    print(f"Warhead prefix tokens: {prefix_tokens}")
    print(f"Warhead prefix ids:    {prefix_ids}")

    # ---- PRIMARY PATH: prefix injection -----------------------------------
    print("\n[PRIMARY] Running prefix-conditioned sampling...")
    accepted: list[str] = []  # canonical SMILES that contain warhead
    accepted_raw: list[str] = []  # raw decoded SMILES (warhead-first)
    n_attempts = 0
    n_valid = 0
    n_warhead_match = 0
    seen_canon: set[str] = set()
    t0 = time.time()

    for batch_idx in range(MAX_BATCHES):
        if len(accepted) >= TARGET_N:
            break
        seqs, smiles, nlls = sample_with_prefix(
            model, prefix_ids, batch_size=BATCH_SIZE
        )
        for raw in smiles:
            n_attempts += 1
            if not raw.startswith(WARHEAD_SMILES):
                # By construction this should not happen, but be defensive.
                continue
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                continue
            n_valid += 1
            if not mol.HasSubstructMatch(warhead_patt):
                continue
            n_warhead_match += 1
            canon = Chem.MolToSmiles(mol)
            if canon in seen_canon:
                continue
            seen_canon.add(canon)
            accepted.append(canon)
            accepted_raw.append(raw)
        print(f"  batch {batch_idx + 1}/{MAX_BATCHES}: attempts={n_attempts} "
              f"valid={n_valid} warhead_ok={n_warhead_match} "
              f"unique_kept={len(accepted)}")

    elapsed_primary = time.time() - t0

    primary_acceptance_rate = (
        n_warhead_match / n_attempts if n_attempts else 0.0
    )
    primary_validity_rate = n_valid / n_attempts if n_attempts else 0.0

    used_fallback = False
    fallback_info: dict | None = None

    if len(accepted) < TARGET_N:
        # ---- FALLBACK PATH: unconditioned sample + filter ----------------
        print("\n[FALLBACK] Prefix path under-produced; falling back to "
              "unconditioned sample + filter.")
        used_fallback = True
        fb_attempts = 0
        fb_valid = 0
        fb_warhead_first = 0
        fb_t0 = time.time()
        for fb_batch in range(MAX_BATCHES * 2):
            if len(accepted) >= TARGET_N:
                break
            _, smiles, _ = model.sample(batch_size=BATCH_SIZE)
            for raw in smiles:
                fb_attempts += 1
                mol = Chem.MolFromSmiles(raw)
                if mol is None:
                    continue
                fb_valid += 1
                canon = Chem.MolToSmiles(mol)
                # Hard filter: canonical SMILES must START with the warhead
                if not canon.startswith(WARHEAD_SMILES):
                    continue
                fb_warhead_first += 1
                if canon in seen_canon:
                    continue
                seen_canon.add(canon)
                accepted.append(canon)
                accepted_raw.append(raw)
            print(f"  fallback batch {fb_batch + 1}: "
                  f"attempts={fb_attempts} valid={fb_valid} "
                  f"warhead_first={fb_warhead_first} "
                  f"unique_kept={len(accepted)}")

        fallback_info = {
            "attempts": fb_attempts,
            "valid": fb_valid,
            "warhead_first_canonical": fb_warhead_first,
            "warhead_first_rate": (
                fb_warhead_first / fb_attempts if fb_attempts else 0.0
            ),
            "elapsed_sec": time.time() - fb_t0,
        }

    # Trim to exactly TARGET_N if overshot
    accepted = accepted[:TARGET_N]
    accepted_raw = accepted_raw[:TARGET_N]

    # ---- QA --------------------------------------------------------------
    qa_smarts_match = 0
    qa_canon_starts_with_warhead = 0
    canon_list: list[str] = []
    for s in accepted:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        if mol.HasSubstructMatch(warhead_patt):
            qa_smarts_match += 1
        canon = Chem.MolToSmiles(mol)
        canon_list.append(canon)
        if canon.startswith(WARHEAD_SMILES):
            qa_canon_starts_with_warhead += 1

    div_stats = diversity_stats(accepted)

    # ---- Write outputs ---------------------------------------------------
    smi_out = DATA_OUT_DIR / "samples.smi"
    write_smi(smi_out, accepted)
    print(f"\nWrote {len(accepted)} SMILES to {smi_out}")

    # Also keep raw (pre-canonicalization) decoded sequences for forensics.
    write_smi(DATA_OUT_DIR / "samples_raw.smi", accepted_raw)

    payload = {
        "experiment": "seq_exp4_warhead_prefix",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warhead_smiles": WARHEAD_SMILES,
        "warhead_smarts": WARHEAD_SMARTS,
        "warhead_token_prefix": prefix_tokens,
        "warhead_token_ids": prefix_ids,
        "prior": str(PRIOR_PATH),
        "target_n": TARGET_N,
        "n_returned": len(accepted),
        "n_unique_canonical": len(set(accepted)),
        "qa": {
            "all_smarts_match": qa_smarts_match == len(accepted),
            "smarts_match_count": qa_smarts_match,
            "canon_starts_with_warhead": qa_canon_starts_with_warhead,
            "diversity_unique_gt_100": div_stats["n_unique_canonical"] > 100,
        },
        "diversity": div_stats,
        "primary_path": {
            "method": "prefix_injection",
            "attempts": n_attempts,
            "valid": n_valid,
            "warhead_match": n_warhead_match,
            "acceptance_rate": primary_acceptance_rate,
            "validity_rate": primary_validity_rate,
            "elapsed_sec": elapsed_primary,
            "batches": batch_idx + 1,
            "batch_size": BATCH_SIZE,
        },
        "used_fallback": used_fallback,
        "fallback": fallback_info,
        "outputs": {
            "smi_file": str(smi_out),
            "smi_file_raw": str(DATA_OUT_DIR / "samples_raw.smi"),
            "json_file": str(RESULTS_DIR / "warhead_prefix.json"),
            "report_file": str(REPORT_PATH),
        },
    }

    json_path = RESULTS_DIR / "warhead_prefix.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {json_path}")

    # ---- Human report ---------------------------------------------------
    lines = []
    lines.append("# Seq-Method Experiment 4: Warhead-Anchor Token Prefix")
    lines.append("")
    lines.append(f"- **Timestamp:** {payload['timestamp']}")
    lines.append(f"- **Prior:** `{PRIOR_PATH}`")
    lines.append(f"- **Warhead SMILES:** `{WARHEAD_SMILES}`")
    lines.append(
        f"- **Warhead token prefix:** `{prefix_tokens}` "
        f"(ids `{prefix_ids}`)"
    )
    lines.append(f"- **Target / returned:** {TARGET_N} / {len(accepted)}")
    lines.append(
        f"- **Used fallback:** {'YES' if used_fallback else 'NO'}"
    )
    lines.append("")
    lines.append("## Primary path (prefix injection)")
    lines.append("")
    lines.append(
        f"- attempts: {n_attempts}, valid: {n_valid}, "
        f"warhead-match: {n_warhead_match}"
    )
    lines.append(
        f"- validity rate: {primary_validity_rate:.3f}, "
        f"warhead acceptance: {primary_acceptance_rate:.3f}"
    )
    lines.append(f"- elapsed: {elapsed_primary:.1f}s")
    if used_fallback and fallback_info is not None:
        lines.append("")
        lines.append("## Fallback path (sample + filter)")
        lines.append("")
        lines.append(
            f"- attempts: {fallback_info['attempts']}, "
            f"valid: {fallback_info['valid']}, "
            f"warhead-first canonical: "
            f"{fallback_info['warhead_first_canonical']}"
        )
        lines.append(
            f"- warhead-first rate (canonical): "
            f"{fallback_info['warhead_first_rate']:.4f}"
        )
        lines.append(
            f"- elapsed: {fallback_info['elapsed_sec']:.1f}s"
        )
    lines.append("")
    lines.append("## QA")
    lines.append("")
    lines.append(
        f"- All outputs match warhead SMARTS: "
        f"**{payload['qa']['all_smarts_match']}** "
        f"({qa_smarts_match}/{len(accepted)})"
    )
    lines.append(
        f"- Canonical SMILES still starts with `C=CC(=O)N`: "
        f"{qa_canon_starts_with_warhead}/{len(accepted)}"
    )
    lines.append(
        f"- Unique canonical structures: "
        f"{div_stats['n_unique_canonical']} "
        f"(diversity > 100 requirement: "
        f"**{payload['qa']['diversity_unique_gt_100']}**)"
    )
    if div_stats.get("mean_pairwise_tanimoto") is not None:
        lines.append(
            f"- Mean pairwise Tanimoto "
            f"(Morgan2, 2048 bits, ≤200 sample): "
            f"{div_stats['mean_pairwise_tanimoto']:.3f}"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `{smi_out}`")
    lines.append(f"- `{DATA_OUT_DIR / 'samples_raw.smi'}`")
    lines.append(f"- `{json_path}`")
    lines.append(f"- `{REPORT_PATH}`")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")

    # ---- Final summary --------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Returned: {len(accepted)} / {TARGET_N}")
    print(f"All warhead-match: {payload['qa']['all_smarts_match']}")
    print(f"Unique canonical: {div_stats['n_unique_canonical']} "
          f"(>100 req: {payload['qa']['diversity_unique_gt_100']})")
    print(f"Used fallback: {used_fallback}")
    if not (payload['qa']['all_smarts_match']
            and payload['qa']['diversity_unique_gt_100']):
        print("WARNING: QA criteria not fully met. See JSON for details.")


if __name__ == "__main__":
    main()
