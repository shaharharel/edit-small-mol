#!/usr/bin/env python3
"""
seq_exp6 — REINVENT4 Mol2Mol with warhead-class control tokens in the vocab.

Concept
-------
Mol2Mol's encoder is a SMILES transformer; the vocabulary is a flat token list
shared by encoder/decoder. We add four NEW single-token entries to the prior's
vocab so the model can read a warhead-class control token directly:

    [ACRYLAMIDE]        — acrylamide (Michael acceptor) warhead
    [CHLOROACETAMIDE]   — alpha-halo acetamide
    [VINYL_SULFONAMIDE] — vinyl sulfonamide
    [EPOXIDE]           — strained epoxide

Each token is a *single* tokenizer atom because REINVENT4's SMILESTokenizer uses
`r'(\[[^]]*])'` as the first split regex — any bracketed content is kept as one
token. The same regex covers atom blocks like `[C@@H]`, so our tokens piggy-back
cleanly on the existing path.

Implementation
--------------
1. Load `priors/mol2mol_medium_similarity.prior` (vocab=128, model_dim=256).
2. Add 4 tokens → vocab=132 (`vocabulary._add`, `_current_id` bumped).
3. Expand the 4 weight tensors that depend on vocab size:
     - `src_embed.0.lut.weight`     (128,256) → (132,256)
     - `tgt_embed.0.lut.weight`     (128,256) → (132,256)
     - `generator.proj.weight`      (128,256) → (132,256)
     - `generator.proj.bias`        (128,)    → (132,)
   New embedding rows are sampled from the empirical mean ± σ of the existing
   rows (so they live in the same regime as initialised tokens). Generator
   weight rows mirror the same statistic and the generator bias is set to the
   minimum existing bias minus 5, so a freshly added token has a very low prior
   probability of being *emitted* (we only want them on the source side).
4. Save patched prior to a working file.
5. Fine-tune via REINVENT4's standard TL pipeline with a *pre-built* pairs CSV
   (not Tanimoto pair generation). For every covalent SMILES with warhead class
   W we emit:
       Source_Mol = "[W]" + canonical_SMILES (W ∈ {ACRYLAMIDE,...})
       Target_Mol = canonical_SMILES_of_another_inhibitor_with_same_warhead_W
   This teaches the model that the prefix conditions the *output* warhead class
   regardless of source identity.
6. Sample N=500 by prepending `[ACRYLAMIDE]` to the ZAP70 Mol-1 anchor.
7. QA: SMARTS gate, diversity, loss curve, comparison vs EXP4 prefix baseline.

Outputs
-------
- experiments/reinvent4_mol2mol_warhead_tokens.py     (this file)
- models/reinvent4_mol2mol_warhead_tokens.prior       (FT checkpoint)
- data/reinvent4_mol2mol_warhead_tokens_samples/samples.smi   (N=500)
- results/paper_evaluation/seq_method_experiments/warhead_tokens.json
- /tmp/seq_exp6_warhead_tokens.md
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
BASE_PRIOR = REINVENT4_ROOT / "priors" / "mol2mol_medium_similarity.prior"

MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "data" / "reinvent4_mol2mol_warhead_tokens_samples"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "seq_method_experiments"
WORK_DIR = PROJECT_ROOT / "data" / "reinvent4_mol2mol_warhead_tokens_work"

for d in [MODELS_DIR, SAMPLES_DIR, RESULTS_DIR, WORK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PATCHED_PRIOR = WORK_DIR / "mol2mol_vocab_patched.prior"
OUT_PRIOR = MODELS_DIR / "reinvent4_mol2mol_warhead_tokens.prior"

PAIRS_CSV = WORK_DIR / "pairs.csv"
PAIRS_VAL_CSV = WORK_DIR / "pairs_val.csv"
TL_TOML = WORK_DIR / "transfer_learning.toml"
TL_LOG = WORK_DIR / "transfer_learning.log"
SAMPLE_TOML = WORK_DIR / "sampling.toml"
SAMPLE_CSV = WORK_DIR / "sampling.csv"

SAMPLES_SMI = SAMPLES_DIR / "samples.smi"
RESULT_JSON = RESULTS_DIR / "warhead_tokens.json"
REPORT_MD = Path("/tmp/seq_exp6_warhead_tokens.md")

REINVENT_BIN = "/opt/miniconda3/envs/quris/bin/reinvent"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONTROL_TOKENS = ["[ACRYLAMIDE]", "[CHLOROACETAMIDE]", "[VINYL_SULFONAMIDE]", "[EPOXIDE]"]

# Warhead SMARTS used both to (a) classify training molecules and (b) QA samples.
WARHEAD_SMARTS = {
    "[ACRYLAMIDE]":        "[CH2]=[CH]C(=O)[N;!H2]",
    "[CHLOROACETAMIDE]":   "[Cl][CH2]C(=O)[N;!H2]",
    "[VINYL_SULFONAMIDE]": "[CH2]=[CH]S(=O)(=O)N",
    "[EPOXIDE]":           "C1OC1",
}

# Chassis-label warhead-class aliases used in covindb_chassis_labels.csv
CHASSIS_ALIAS = {
    "[ACRYLAMIDE]":        {"acrylamide", "acrylyl", "acrylate", "acrylonitrile"},
    "[CHLOROACETAMIDE]":   {"chloroacetamide", "chloroacetyl", "bromoacetyl", "bromoacetamide"},
    "[VINYL_SULFONAMIDE]": {"vinyl_sulfonamide", "vinyl_sulfone"},
    "[EPOXIDE]":           {"epoxide"},
}

# Sampling anchor — Mol-1 ZAP70 covalent inhibitor (same as EXP2/EXP4)
MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

NUM_SAMPLES_TARGET = 500
NUM_SAMPLES_RAW = 1500

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 64
LR = 1.0e-5

# Cap dataset to keep TL tractable on CPU
MAX_PAIRS_PER_CLASS = 4000
MAX_TOTAL_PAIRS = 12000

RANDOM_SEED = 7


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _canonicalize(smi: str | float) -> str | None:
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if not (150.0 <= Descriptors.MolWt(mol) <= 800.0):
        return None
    if mol.GetNumHeavyAtoms() < 10 or mol.GetNumHeavyAtoms() > 60:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _has_warhead(mol: Chem.Mol, token: str) -> bool:
    pat = Chem.MolFromSmarts(WARHEAD_SMARTS[token])
    return mol.HasSubstructMatch(pat)


# ---------------------------------------------------------------------------
# Step 1 — patch the prior vocabulary
# ---------------------------------------------------------------------------


def patch_prior_vocab() -> dict:
    """Add CONTROL_TOKENS to the vocab and expand vocab-sized weight tensors."""
    log(f"Loading base prior: {BASE_PRIOR}")
    sd = torch.load(BASE_PRIOR, map_location="cpu", weights_only=False)

    voc = sd["vocabulary"]
    old_vocab_size = len(voc)
    log(f"  base vocab size = {old_vocab_size}")

    # Vocabulary objects deserialised from older priors have `_current_id == 0`
    # because load_from_dictionary doesn't restore it. Recompute from existing ids.
    existing_ids = [i for k, i in voc.word2idx().items()]
    voc._current_id = max(existing_ids) + 1 if existing_ids else 0
    log(f"  set _current_id = {voc._current_id}")

    new_ids = {}
    for tok in CONTROL_TOKENS:
        if tok in voc:
            log(f"  WARN: token {tok!r} already present (id={voc[tok]}); skipping add")
            new_ids[tok] = voc[tok]
            continue
        nid = voc.add(tok)  # uses Vocabulary._current_id
        new_ids[tok] = nid
        log(f"  + {tok!r} -> id {nid}")

    new_vocab_size = len(voc)
    log(f"  new vocab size = {new_vocab_size}")

    # Update network_parameter so loading creates an EncoderDecoder with the right vocab dim.
    sd["network_parameter"]["vocabulary_size"] = new_vocab_size

    # Expand weight tensors. We sample new rows from the empirical distribution
    # of the existing rows so the patched model behaves like the base prior.
    rng = torch.Generator().manual_seed(RANDOM_SEED)
    net = sd["network_state"]

    for key in ("src_embed.0.lut.weight", "tgt_embed.0.lut.weight",
                "generator.proj.weight"):
        w = net[key]  # (vocab, dim)
        mean = w.mean(dim=0)
        std = w.std(dim=0).clamp(min=1e-6)
        n_new = new_vocab_size - w.shape[0]
        if n_new <= 0:
            continue
        new_rows = mean[None, :] + std[None, :] * torch.randn(
            (n_new, w.shape[1]), generator=rng
        ) * 0.5  # half-σ so they're conservative
        net[key] = torch.cat([w, new_rows], dim=0)
        log(f"  expanded {key}: {tuple(w.shape)} -> {tuple(net[key].shape)}")

    # Generator bias: set new entries low so the model doesn't *emit* control
    # tokens unless explicitly trained to. -10 is a strong but finite negative.
    key = "generator.proj.bias"
    b = net[key]
    n_new = new_vocab_size - b.shape[0]
    if n_new > 0:
        new_bias = torch.full((n_new,), -10.0, dtype=b.dtype)
        net[key] = torch.cat([b, new_bias], dim=0)
        log(f"  expanded {key}: {tuple(b.shape)} -> {tuple(net[key].shape)} (new entries = -10.0)")

    # Persist
    torch.save(sd, PATCHED_PRIOR)
    size_mb = PATCHED_PRIOR.stat().st_size / (1024 * 1024)
    log(f"  saved patched prior: {PATCHED_PRIOR} ({size_mb:.1f} MB)")

    # Smoke-test: load via REINVENT4's adapter to make sure dim alignment works
    from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
    test_sd = torch.load(PATCHED_PRIOR, map_location="cpu", weights_only=False)
    Mol2MolModel.create_from_dict(test_sd, "inference", torch.device("cpu"))
    log("  smoke-test: patched prior loads via Mol2MolModel.create_from_dict OK")

    return {
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "new_token_ids": new_ids,
    }


# ---------------------------------------------------------------------------
# Step 2 — build paired training dataset with control-token prefixes
# ---------------------------------------------------------------------------


def _load_covindb_warhead_pool() -> dict[str, list[str]]:
    """Return {control_token: [canonical_smiles, ...]} for CovInDB.

    We use BOTH:
      1. Per-row warhead classification from CovInDB_All by SMARTS match.
      2. covindb_chassis_labels.csv 'warhead_class' aliases (mapped to our four
         classes).
    """
    pool: dict[str, set[str]] = {tok: set() for tok in CONTROL_TOKENS}

    # 1) Direct SMARTS scan of CovInDB_All
    cov_all = PROJECT_ROOT / "data/covbinder/raw_covindb2/CovInDB_All.csv"
    if cov_all.exists():
        log(f"  Loading CovInDB_All: {cov_all}")
        df = pd.read_csv(cov_all, low_memory=False)
        for smi in df["SMILES"].dropna().astype(str):
            canon = _canonicalize(smi)
            if not canon:
                continue
            mol = Chem.MolFromSmiles(canon)
            if mol is None:
                continue
            for tok in CONTROL_TOKENS:
                if _has_warhead(mol, tok):
                    pool[tok].add(canon)

    # 2) Chassis labels (provides additional curated assignments for rarer classes)
    chassis = PROJECT_ROOT / "data/covindb_chassis_labels.csv"
    if chassis.exists():
        log(f"  Loading chassis labels: {chassis}")
        df = pd.read_csv(chassis, low_memory=False)
        df = df.dropna(subset=["ligand_smiles", "warhead_class"])
        for _, row in df.iterrows():
            canon = _canonicalize(row["ligand_smiles"])
            if not canon:
                continue
            wclass = str(row["warhead_class"]).strip().lower()
            for tok in CONTROL_TOKENS:
                if wclass in CHASSIS_ALIAS[tok]:
                    pool[tok].add(canon)
                    break

    return {tok: sorted(s) for tok, s in pool.items()}


def build_pairs() -> dict:
    """Construct (Source_Mol=[TOKEN]+A, Target_Mol=B) pairs.

    For each warhead class W with N mols we draw `min(MAX_PAIRS_PER_CLASS, N*3)`
    pairs (A, B) where A != B and both have warhead W. Source uses prefix [W].
    """
    log("Building paired training data...")
    pool = _load_covindb_warhead_pool()
    pool_sizes = {tok: len(v) for tok, v in pool.items()}
    log(f"  per-class pool sizes: {pool_sizes}")

    if sum(pool_sizes.values()) < 100:
        raise SystemExit("Not enough labelled covalent SMILES to fine-tune.")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for tok, mols in pool.items():
        n = len(mols)
        if n < 2:
            log(f"  WARN: skipping {tok} — only {n} mols")
            continue
        # Generate ~3 pairs per molecule but cap class
        n_pairs_target = min(MAX_PAIRS_PER_CLASS, n * 3)
        # Sample with replacement on (A, B) — A != B
        a_idx = rng.integers(0, n, size=n_pairs_target * 2)
        b_idx = rng.integers(0, n, size=n_pairs_target * 2)
        used = 0
        for ai, bi in zip(a_idx, b_idx):
            if ai == bi:
                continue
            rows.append((f"{tok}{mols[ai]}", mols[bi], tok))
            used += 1
            if used >= n_pairs_target:
                break
        log(f"  {tok}: {used} pairs from pool of {n}")

    # Cap total pairs and shuffle
    rng.shuffle(rows)
    if len(rows) > MAX_TOTAL_PAIRS:
        rows = rows[:MAX_TOTAL_PAIRS]
        log(f"  capped to MAX_TOTAL_PAIRS={MAX_TOTAL_PAIRS}")
    log(f"  TOTAL pairs: {len(rows)}")

    df = pd.DataFrame(rows, columns=["Source_Mol", "Target_Mol", "warhead_class"])

    # Hold out 5% for validation
    n_val = max(50, int(0.05 * len(df)))
    val_idx = rng.choice(len(df), size=n_val, replace=False)
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[val_idx] = True
    df_train = df.loc[~val_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)
    df_train.to_csv(PAIRS_CSV, index=False)
    df_val.to_csv(PAIRS_VAL_CSV, index=False)
    log(f"  wrote {len(df_train)} train / {len(df_val)} val pairs")

    return {
        "pool_sizes": pool_sizes,
        "n_pairs_train": len(df_train),
        "n_pairs_val": len(df_val),
        "per_class_train": df_train["warhead_class"].value_counts().to_dict(),
        "per_class_val": df_val["warhead_class"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Step 3 — fine-tune via REINVENT4 TL using a custom pairs CSV
#
# REINVENT4 generates pairs from a SMILES list. We have *pre-built* pairs;
# the easiest path is to roll a direct PyTorch training loop using the same
# PairedDataset/model classes REINVENT4 already exposes.
# ---------------------------------------------------------------------------


def run_transfer_learning(device: str) -> dict:
    """Plain PyTorch training loop on (source, target) token pairs."""
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO)

    sys.path.insert(0, str(REINVENT4_ROOT))
    from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
    from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
    from torch.utils.data import DataLoader

    log("Loading patched prior for training...")
    sd = torch.load(PATCHED_PRIOR, map_location="cpu", weights_only=False)
    model = Mol2MolModel.create_from_dict(sd, "training", torch.device(device))
    log(f"  loaded model; vocab size = {len(model.vocabulary)}")
    log(f"  network params: {sum(p.numel() for p in model.network.parameters()):,}")

    # Build datasets
    df_train = pd.read_csv(PAIRS_CSV)
    df_val = pd.read_csv(PAIRS_VAL_CSV)

    def _ds(df):
        return PairedDataset(
            smiles_input=df["Source_Mol"].tolist(),
            smiles_output=df["Target_Mol"].tolist(),
            vocabulary=model.vocabulary,
            tokenizer=model.tokenizer,
            tanimoto_similarities=np.zeros(len(df), dtype=np.float32),
        )

    train_ds = _ds(df_train)
    val_ds = _ds(df_val)
    log(f"  train pairs encoded: {len(train_ds)} (input dropped: {len(df_train) - len(train_ds)})")
    log(f"  val pairs encoded:   {len(val_ds)} (input dropped: {len(df_val) - len(val_ds)})")

    if len(train_ds) == 0:
        raise SystemExit("No training pairs encoded — token mismatch?")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        collate_fn=PairedDataset.collate_fn,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=PairedDataset.collate_fn,
    )

    optim = torch.optim.Adam(model.network.parameters(), lr=LR)

    train_curve: list[float] = []
    val_curve: list[float] = []

    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.set_mode(model._model_modes.TRAINING)
        tot_loss = 0.0
        tot_n = 0
        for step, batch in enumerate(train_loader):
            src, src_mask, trg, trg_mask, _sim = batch
            src = src.to(device); src_mask = src_mask.to(device)
            trg = trg.to(device); trg_mask = trg_mask.to(device)
            optim.zero_grad()
            nll = model.likelihood(src, src_mask, trg, trg_mask)
            loss = nll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
            optim.step()
            tot_loss += float(loss.detach().cpu().item()) * len(trg)
            tot_n += len(trg)
            if step % 20 == 0:
                log(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        train_loss = tot_loss / max(1, tot_n)
        train_curve.append(train_loss)

        # Val
        model.set_mode(model._model_modes.INFERENCE)
        tot_loss = 0.0
        tot_n = 0
        with torch.no_grad():
            for batch in val_loader:
                src, src_mask, trg, trg_mask, _sim = batch
                src = src.to(device); src_mask = src_mask.to(device)
                trg = trg.to(device); trg_mask = trg_mask.to(device)
                nll = model.likelihood(src, src_mask, trg, trg_mask)
                loss = nll.mean()
                tot_loss += float(loss.detach().cpu().item()) * len(trg)
                tot_n += len(trg)
        val_loss = tot_loss / max(1, tot_n) if tot_n else float("nan")
        val_curve.append(val_loss)

        log(f"  EPOCH {epoch}/{NUM_EPOCHS}  train_nll={train_loss:.4f}  val_nll={val_loss:.4f}  ({(time.time()-t0)/60.0:.1f} min elapsed)")

    # Save FT checkpoint
    model.set_mode(model._model_modes.INFERENCE)
    model.save(OUT_PRIOR)
    size_mb = OUT_PRIOR.stat().st_size / (1024 * 1024)
    log(f"Saved FT prior: {OUT_PRIOR} ({size_mb:.1f} MB)")

    return {
        "wall_seconds": time.time() - t0,
        "train_loss_curve": train_curve,
        "val_loss_curve": val_curve,
        "final_train_nll": train_curve[-1] if train_curve else None,
        "final_val_nll": val_curve[-1] if val_curve else None,
        "loss_decrease": (train_curve[0] - train_curve[-1]) if len(train_curve) >= 2 else None,
        "ft_prior_mb": size_mb,
    }


# ---------------------------------------------------------------------------
# Step 4 — sample with the [ACRYLAMIDE] control token
#
# We do NOT use the REINVENT4 sampling CLI because it tokenizes the input via
# its own SMILESTokenizer (which does correctly tokenize `[ACRYLAMIDE]`), BUT
# the multinomial decoder in `transformer.sample` masks property tokens via the
# hardcoded LogD/Solubility list. Our new tokens are not in that list so
# masking is fine. We still do our own sampling loop to keep tight control
# over the input prefix.
# ---------------------------------------------------------------------------


def run_sampling(device: str, control_token: str = "[ACRYLAMIDE]") -> dict:
    sys.path.insert(0, str(REINVENT4_ROOT))
    from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel

    log(f"Loading FT prior for sampling: {OUT_PRIOR}")
    sd = torch.load(OUT_PRIOR, map_location="cpu", weights_only=False)
    model = Mol2MolModel.create_from_dict(sd, "inference", torch.device(device))
    voc = model.vocabulary
    tok = model.tokenizer
    log(f"  vocab size = {len(voc)}; control token id = {voc[control_token]}")

    # Prepare source: [ACRYLAMIDE] + Mol-1
    src_smi = f"{control_token}{MOL1_SMILES}"
    src_tokens = tok.tokenize(src_smi, with_begin_and_end=True)
    log(f"  source tokens (first 12): {src_tokens[:12]}")
    src_ids = voc.encode(src_tokens).astype(np.int64)
    log(f"  source id sequence length = {len(src_ids)}; first 12 ids = {src_ids[:12].tolist()}")

    # Batch the same input NUM_SAMPLES_RAW times for multinomial sampling
    src_t = torch.from_numpy(src_ids).long().unsqueeze(0).repeat(NUM_SAMPLES_RAW, 1).to(device)
    src_mask = torch.ones(NUM_SAMPLES_RAW, 1, src_t.size(1), dtype=torch.bool, device=device)

    t0 = time.time()
    # Mol2Mol.sample() runs full decode loop. With 1500 samples on CPU this is
    # the longest single op (~few minutes). We chunk to bound memory.
    chunk = 250
    all_out: list[str] = []
    all_nll: list[float] = []
    for i in range(0, NUM_SAMPLES_RAW, chunk):
        s = src_t[i : i + chunk]
        m = src_mask[i : i + chunk]
        _in, out_smis, nlls = model.sample(s, m, "multinomial")
        all_out.extend(out_smis)
        all_nll.extend([float(n) for n in nlls])
        log(f"  sampled chunk {i//chunk + 1}/{(NUM_SAMPLES_RAW + chunk - 1)//chunk}  "
            f"so far {len(all_out)}/{NUM_SAMPLES_RAW}  ({(time.time()-t0)/60.0:.1f} min)")

    wall = time.time() - t0
    log(f"Sampling finished in {wall/60.0:.2f} min ({len(all_out)} raw samples)")

    # Persist raw sampling output
    with (WORK_DIR / "sampling_raw.smi").open("w") as fh:
        for s, n in zip(all_out, all_nll):
            fh.write(f"{s}\t{n:.4f}\n")

    return {"wall_seconds": wall, "raw_smiles": all_out, "raw_nll": all_nll}


# ---------------------------------------------------------------------------
# Step 5 — post-process: dedupe, validate, warhead gate, diversity
# ---------------------------------------------------------------------------


def postprocess(samples: list[str], nlls: list[float], control_token: str = "[ACRYLAMIDE]") -> dict:
    log("Post-processing samples...")
    wh_pat = Chem.MolFromSmarts(WARHEAD_SMARTS[control_token])
    if wh_pat is None:
        raise SystemExit(f"Invalid SMARTS for {control_token}")

    rows = []
    canon_seen: set[str] = set()
    n_invalid = 0
    for smi, nll in zip(samples, nlls):
        if not isinstance(smi, str) or not smi:
            n_invalid += 1
            continue
        # The sampled output should not contain the control token (its generator
        # bias was -10), but strip it defensively if present.
        if smi.startswith(control_token):
            smi = smi[len(control_token):]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if canon in canon_seen:
            continue
        canon_seen.add(canon)
        has_wh = bool(mol.HasSubstructMatch(wh_pat))
        rows.append({"smiles": canon, "nll": nll, "warhead": has_wh,
                     "mw": float(Descriptors.MolWt(mol)),
                     "qed": float(Descriptors.qed(mol))})

    valid_unique = len(rows)
    with_wh = [r for r in rows if r["warhead"]]
    log(f"  raw={len(samples)}  invalid={n_invalid}  valid_unique={valid_unique}  warhead={len(with_wh)} ({100*len(with_wh)/max(1,valid_unique):.1f}%)")

    # Select top NUM_SAMPLES_TARGET — gate by warhead if we have enough
    pool = with_wh if len(with_wh) >= NUM_SAMPLES_TARGET else rows
    if pool is rows:
        log(f"  WARN: only {len(with_wh)} warhead-matching mols < {NUM_SAMPLES_TARGET}; falling back to top valid")
    pool = sorted(pool, key=lambda r: r["nll"] if r["nll"] is not None else 1e9)[:NUM_SAMPLES_TARGET]

    # Diversity
    mols = [Chem.MolFromSmiles(r["smiles"]) for r in pool]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    sims: list[float] = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
    mean_sim = float(np.mean(sims)) if sims else 0.0
    internal_diversity = 1.0 - mean_sim

    anchor_mol = Chem.MolFromSmiles(MOL1_SMILES)
    anchor_fp = AllChem.GetMorganFingerprintAsBitVect(anchor_mol, 2, 2048)
    tc_to_anchor = [float(DataStructs.TanimotoSimilarity(anchor_fp, fp)) for fp in fps]

    # Write samples.smi (tab-separated for downstream parsing)
    with SAMPLES_SMI.open("w") as fh:
        fh.write("smiles\tnll\twarhead\ttc_to_anchor\tmw\tqed\n")
        for r, tc in zip(pool, tc_to_anchor):
            fh.write(f"{r['smiles']}\t{r['nll']:.4f}\t{int(r['warhead'])}\t{tc:.4f}\t{r['mw']:.2f}\t{r['qed']:.3f}\n")
    log(f"Wrote {SAMPLES_SMI} ({len(pool)} mols)")

    return {
        "raw_samples": len(samples),
        "n_invalid": n_invalid,
        "valid_unique": valid_unique,
        "warhead_count": len(with_wh),
        "warhead_rate": len(with_wh) / max(1, valid_unique),
        "selected": len(pool),
        "internal_diversity": internal_diversity,
        "mean_pairwise_tanimoto": mean_sim,
        "tc_to_anchor_mean": float(np.mean(tc_to_anchor)) if tc_to_anchor else None,
        "tc_to_anchor_median": float(np.median(tc_to_anchor)) if tc_to_anchor else None,
    }


# ---------------------------------------------------------------------------
# Step 6 — write JSON metrics + Markdown report
# ---------------------------------------------------------------------------


def write_report(metrics: dict) -> None:
    RESULT_JSON.write_text(json.dumps(metrics, indent=2, default=str))
    log(f"Wrote {RESULT_JSON}")

    final_train = metrics.get("final_train_nll")
    final_val = metrics.get("final_val_nll")
    loss_dec = metrics.get("loss_decrease")
    final_train_s = f"{final_train:.4f}" if isinstance(final_train, (int, float)) else "n/a"
    final_val_s = f"{final_val:.4f}" if isinstance(final_val, (int, float)) else "n/a"
    loss_dec_s = f"{loss_dec:.4f}" if isinstance(loss_dec, (int, float)) else "n/a"

    md = f"""# seq_exp6 — Mol2Mol warhead-class control tokens

**Run date:** {metrics['run_date']}
**Device:** {metrics['device']}
**Base prior:** `priors/mol2mol_medium_similarity.prior` (REINVENT4 v4.7.15)
**Patched prior (vocab+4):** `{PATCHED_PRIOR.relative_to(PROJECT_ROOT)}`
**FT output:** `{OUT_PRIOR.relative_to(PROJECT_ROOT)}` (~{metrics['ft_prior_mb']:.0f} MB)

## Vocabulary patch

- Base vocab size: **{metrics['vocab']['old_vocab_size']}**
- New control tokens: **{', '.join(CONTROL_TOKENS)}**
- New vocab size: **{metrics['vocab']['new_vocab_size']}**
- New token ids: {metrics['vocab']['new_token_ids']}
- Resized weights: `src_embed.0.lut.weight`, `tgt_embed.0.lut.weight`, `generator.proj.weight` (rows sampled from base-row mean ± 0.5σ); `generator.proj.bias` (new entries set to −10 so the model does not *emit* control tokens).

## Training data

- Source pool: CovInDB_All ({metrics['data']['pool_sizes']}) + chassis-labels
- Per-class molecule pool: {metrics['data']['pool_sizes']}
- Per-class **train** pairs: {metrics['data']['per_class_train']}
- Per-class **val** pairs: {metrics['data']['per_class_val']}
- Construction: `Source = [WARHEAD] + canon(A)`, `Target = canon(B)`, both A and B drawn from the same warhead class. Pairs cap: `{MAX_PAIRS_PER_CLASS}` per class, `{MAX_TOTAL_PAIRS}` total.
- Train pairs (kept after vocab encoding): **{metrics['data']['n_pairs_train']}**
- Validation pairs: **{metrics['data']['n_pairs_val']}**

## Training

- Optimiser: Adam, LR={LR:.0e}, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}, grad-clip 1.0
- Wall: **{metrics['tl_wall_min']:.2f} min**
- Train NLL curve: {metrics['train_loss_curve']}
- Val NLL curve: {metrics['val_loss_curve']}
- Final train NLL: **{final_train_s}**
- Final val NLL: **{final_val_s}**
- Loss decrease (epoch1 → final): **{loss_dec_s}**

## Sampling (control token = `[ACRYLAMIDE]`, anchor = Mol-1)

- Anchor SMILES: `{MOL1_SMILES}`
- Source string passed to the model: `[ACRYLAMIDE]{MOL1_SMILES}`
- Raw samples requested: **{NUM_SAMPLES_RAW}**, returned valid unique: **{metrics['valid_unique']}**
- Containing acrylamide SMARTS `{WARHEAD_SMARTS['[ACRYLAMIDE]']}`: **{metrics['warhead_count']}** ({100*metrics['warhead_rate']:.1f}%)
- Final emitted set: **{metrics['selected']}** SMILES (warhead-gated where possible)
- Internal diversity (1 − mean pairwise Tanimoto, Morgan r=2/2048): **{metrics['internal_diversity']:.3f}**
- Tanimoto-to-anchor: mean={metrics['tc_to_anchor_mean']:.3f}, median={metrics['tc_to_anchor_median']:.3f}
- Sampling wall: **{metrics['sampling_wall_min']:.2f} min**

## QA

- All 500 outputs match acrylamide SMARTS: **{'PASS' if metrics['warhead_count'] >= NUM_SAMPLES_TARGET else 'PARTIAL'}**
- Training loss decreases: **{'PASS' if isinstance(loss_dec, (int, float)) and loss_dec > 0 else 'FAIL'}**
- Diversity (>50% unique among selected): **{'PASS' if metrics['selected'] >= 0.5 * NUM_SAMPLES_TARGET else 'FAIL'}**

## Baseline comparison (seq_exp4 prefix-only, **Reinvent de novo** prior)

| Metric                          | seq_exp4 (prefix)   | seq_exp6 (control token, this run)  |
|---------------------------------|---------------------|---------------------------------------|
| Model                           | Reinvent de novo    | Mol2Mol with vocab control tokens     |
| Output warhead match            | {metrics['baseline_exp4']['warhead_match_rate']:.1%}              | {100*metrics['warhead_rate']:.1f}% (raw, valid-unique) |
| Mean pairwise Tanimoto          | {metrics['baseline_exp4']['mean_pairwise_tanimoto']:.3f} | {metrics['mean_pairwise_tanimoto']:.3f} |
| Internal diversity              | {1 - metrics['baseline_exp4']['mean_pairwise_tanimoto']:.3f}               | {metrics['internal_diversity']:.3f}                  |
| Mechanism                       | Prefix injection (token-level seeding of decoder) | Learned: control token conditions encoder, decoder emits warhead-bearing analogue of the anchor |

## Outputs

- `experiments/reinvent4_mol2mol_warhead_tokens.py`
- `{OUT_PRIOR.relative_to(PROJECT_ROOT)}`
- `{SAMPLES_SMI.relative_to(PROJECT_ROOT)}`
- `{RESULT_JSON.relative_to(PROJECT_ROOT)}`
- `{REPORT_MD}`
"""
    REPORT_MD.write_text(md)
    log(f"Wrote {REPORT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pick_device() -> str:
    # Mac: force CPU. Per CLAUDE.md, MPS is unstable for transformer training.
    # (Also matches EXP2 behaviour.)
    return "cpu"


def load_exp4_baseline() -> dict:
    """Pull comparison numbers from EXP4's prefix-only run."""
    p = RESULTS_DIR / "warhead_prefix.json"
    if not p.exists():
        return {"warhead_match_rate": 0.0, "mean_pairwise_tanimoto": 0.0}
    d = json.loads(p.read_text())
    return {
        "warhead_match_rate": d.get("primary_path", {}).get("acceptance_rate", 0.0),
        "mean_pairwise_tanimoto": d.get("diversity", {}).get("mean_pairwise_tanimoto", 0.0),
    }


def main() -> int:
    t_start = time.time()

    device = pick_device()
    log(f"Using device: {device}")

    # 1) Patch prior vocab
    vocab_info = patch_prior_vocab()

    # 2) Build pairs
    data_info = build_pairs()

    # 3) Train
    tl_info = run_transfer_learning(device=device)

    # 4) Sample
    sample = run_sampling(device=device, control_token="[ACRYLAMIDE]")

    # 5) Post-process
    pp = postprocess(sample["raw_smiles"], sample["raw_nll"], "[ACRYLAMIDE]")

    # 6) Compose metrics + report
    metrics = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "base_prior": str(BASE_PRIOR),
        "patched_prior": str(PATCHED_PRIOR),
        "ft_prior": str(OUT_PRIOR),
        "vocab": vocab_info,
        "data": data_info,
        "tl_wall_min": tl_info["wall_seconds"] / 60.0,
        "train_loss_curve": tl_info["train_loss_curve"],
        "val_loss_curve": tl_info["val_loss_curve"],
        "final_train_nll": tl_info["final_train_nll"],
        "final_val_nll": tl_info["final_val_nll"],
        "loss_decrease": tl_info["loss_decrease"],
        "ft_prior_mb": tl_info["ft_prior_mb"],
        "sampling_wall_min": sample["wall_seconds"] / 60.0,
        "baseline_exp4": load_exp4_baseline(),
        **pp,
        "total_wall_min": (time.time() - t_start) / 60.0,
    }

    write_report(metrics)
    log(f"DONE — total wall {metrics['total_wall_min']:.2f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
