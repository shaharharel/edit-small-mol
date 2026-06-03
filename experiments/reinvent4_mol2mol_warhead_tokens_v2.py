#!/usr/bin/env python
"""EXP6_v2 — Mol2Mol warhead-class control tokens, with:

  Fixes over EXP6_v1 (failed: 0% non-acrylamide retention):
  1. Tanimoto-filtered pairs (Tc >= TC_MIN_PAIR, default 0.5) — preserves
     Mol2Mol's high-similarity prior while letting the prefix shift warhead class.
  2. Class-balanced sampling — each epoch sees N_PAIRS_PER_CLASS pairs from EACH
     class, so minority classes (epoxide n=425, vinyl-sulf n=159) get equal weight.
  3. 6 epochs (vs 3).
  4. Generator bias for new tokens initialised to -3 (not -10) — allows training
     signal to lift the new-token probabilities without blowing up emission.
  5. Sampling test at the end across ALL 4 control tokens to verify conditioning.

Outputs:
  models/reinvent4_mol2mol_warhead_tokens_v2.prior
  data/reinvent4_mol2mol_warhead_tokens_v2_work/{pairs.csv, pairs_val.csv}
  data/reinvent4_mol2mol_warhead_tokens_v2_samples/per_class_50.csv
"""
from __future__ import annotations
import argparse, gc, math, os, random, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/home/shaharh_quris_ai/edit-small-mol") if Path("/home/shaharh_quris_ai/edit-small-mol").exists() \
        else Path("/Users/shaharharel/Documents/github/edit-small-mol")
REINVENT4 = Path("/home/shaharh_quris_ai/REINVENT4") if Path("/home/shaharh_quris_ai/REINVENT4").exists() \
          else Path("/Users/shaharharel/Documents/github/REINVENT4")

BASE_PRIOR = REINVENT4 / "priors/mol2mol_medium_similarity.prior"
PATCHED_PRIOR = PROJECT / "data/reinvent4_mol2mol_warhead_tokens_v2_work/mol2mol_vocab_patched_v2.prior"
FT_PRIOR = PROJECT / "models/reinvent4_mol2mol_warhead_tokens_v2.prior"
WORK_DIR = PROJECT / "data/reinvent4_mol2mol_warhead_tokens_v2_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_CSV = WORK_DIR / "pairs.csv"
PAIRS_VAL_CSV = WORK_DIR / "pairs_val.csv"
SAMPLES_DIR = PROJECT / "data/reinvent4_mol2mol_warhead_tokens_v2_samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_TOKENS = ["[ACRYLAMIDE]", "[CHLOROACETAMIDE]", "[VINYL_SULFONAMIDE]", "[EPOXIDE]"]
WARHEAD_SMARTS = {
    "[ACRYLAMIDE]":        "[CH2]=[CH]C(=O)N",
    "[CHLOROACETAMIDE]":   "Cl[CH2]C(=O)N",
    "[VINYL_SULFONAMIDE]": "[CH2]=[CH]S(=O)(=O)N",
    "[EPOXIDE]":           "[#6]1O[#6]1",
}
TC_MIN_PAIR = 0.5           # Tanimoto floor for source-target pairs
MAX_PAIRS_PER_CLASS = 5000  # cap per class (no cap if pool is smaller)
N_PAIRS_PER_CLASS = MAX_PAIRS_PER_CLASS  # alias used in build_tanimoto_filtered_pairs
PAIRS_PER_ANCHOR = 3        # top-Tc neighbors per anchor mol
N_VAL_PER_CLASS = 50
NUM_EPOCHS = 6
BATCH_SIZE = 32
LR = 1e-4
NEW_TOKEN_BIAS_INIT = -3.0  # softer than v1's -10.0
RANDOM_SEED = 42

# CovInDB 2.0 sources — pool ALL unique covalent SMILES found
COVINDB_SOURCES = [
    PROJECT / "data/covbinder/raw_covindb2/CovInDB_All.csv",
    PROJECT / "data/covbinder/raw_covindb2/Covalent_Complex_Records.csv",
    PROJECT / "data/covbinder/exp2_multi_warhead.csv",
    PROJECT / "data/covbinder/exp4_michael_acceptor.csv",
    PROJECT / "data/covbinder/covind_training_set.csv",
    PROJECT / "data/covindb_chassis_labels.csv",
]


def log(*a):
    print("[exp6_v2]", *a, flush=True)


def canonicalize(s):
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) if m else None


def morgan_fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def has_warhead(mol, tok):
    pat = Chem.MolFromSmarts(WARHEAD_SMARTS[tok])
    return mol.HasSubstructMatch(pat)


def load_covindb_warhead_pool():
    pool = {tok: set() for tok in CONTROL_TOKENS}
    cov_all = PROJECT / "data/covbinder/raw_covindb2/CovInDB_All.csv"
    if not cov_all.exists():
        raise SystemExit(f"missing {cov_all}")
    df = pd.read_csv(cov_all, low_memory=False)
    for smi in df["SMILES"].dropna().astype(str):
        canon = canonicalize(smi)
        if not canon: continue
        m = Chem.MolFromSmiles(canon)
        if m is None: continue
        for tok in CONTROL_TOKENS:
            if has_warhead(m, tok):
                pool[tok].add(canon)
    return {tok: sorted(v) for tok, v in pool.items()}


def build_tanimoto_filtered_pairs():
    log(f"Building Tc>={TC_MIN_PAIR} pairs per class (target {N_PAIRS_PER_CLASS}/class)...")
    pool = load_covindb_warhead_pool()
    for tok, mols in pool.items():
        log(f"  {tok}: pool={len(mols)}")
    rng = np.random.default_rng(RANDOM_SEED)
    train_rows, val_rows = [], []
    for tok, mols in pool.items():
        if len(mols) < 4:
            log(f"  WARN: {tok} has only {len(mols)} mols — skipping")
            continue
        fps = []
        for s in mols:
            fp = morgan_fp(s)
            if fp is not None: fps.append((s, fp))
        if len(fps) < 4: continue
        # For each anchor mol, find Tc>=threshold neighbors. Sample (anchor, neighbor) as pair.
        pairs_for_class = []
        anchor_indices = list(range(len(fps)))
        rng.shuffle(anchor_indices)
        for ai in anchor_indices:
            anchor_smi, anchor_fp = fps[ai]
            others = [(j, others_fp) for j, (_, others_fp) in enumerate(fps) if j != ai]
            if not others: continue
            sims = DataStructs.BulkTanimotoSimilarity(anchor_fp, [f for _, f in others])
            eligible = [(others[k][0], sims[k]) for k in range(len(others)) if sims[k] >= TC_MIN_PAIR]
            if not eligible: continue
            # Pick up to 3 neighbors per anchor (top-Tc) to keep training pairs information-dense
            eligible.sort(key=lambda x: -x[1])
            for nj, tc in eligible[:3]:
                target_smi = fps[nj][0]
                pairs_for_class.append((f"{tok}{anchor_smi}", target_smi, tok, float(tc)))
                if len(pairs_for_class) >= N_PAIRS_PER_CLASS + N_VAL_PER_CLASS:
                    break
            if len(pairs_for_class) >= N_PAIRS_PER_CLASS + N_VAL_PER_CLASS:
                break
        rng.shuffle(pairs_for_class)
        val = pairs_for_class[:N_VAL_PER_CLASS]
        train = pairs_for_class[N_VAL_PER_CLASS:N_VAL_PER_CLASS + N_PAIRS_PER_CLASS]
        log(f"  {tok}: built {len(pairs_for_class)} pairs (Tc range)  → train={len(train)} val={len(val)}")
        train_rows.extend(train)
        val_rows.extend(val)
    train_df = pd.DataFrame(train_rows, columns=["Source_Mol", "Target_Mol", "warhead_class", "Tc"])
    val_df   = pd.DataFrame(val_rows,   columns=["Source_Mol", "Target_Mol", "warhead_class", "Tc"])
    # Class-balanced shuffle (interleave by class)
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_df.to_csv(PAIRS_CSV, index=False)
    val_df.to_csv(PAIRS_VAL_CSV, index=False)
    log(f"Train pair counts per class:\n{train_df['warhead_class'].value_counts().to_string()}")
    log(f"Tc stats (train): mean={train_df.Tc.mean():.3f}  min={train_df.Tc.min():.3f}  max={train_df.Tc.max():.3f}")
    return train_df, val_df


def patch_prior_vocab():
    log(f"Patching prior {BASE_PRIOR.name}...")
    sd = torch.load(BASE_PRIOR, map_location="cpu", weights_only=False)
    voc = sd["vocabulary"]
    old_size = len(voc)
    existing_ids = set(voc.tokens.values()) if hasattr(voc, "tokens") else set(voc._tokens.values())
    next_id = max(existing_ids) + 1
    if hasattr(voc, "_current_id"):
        voc._current_id = next_id
    new_ids = {}
    for tok in CONTROL_TOKENS:
        if tok in voc.tokens if hasattr(voc, "tokens") else tok in voc._tokens:
            new_ids[tok] = voc.tokens[tok] if hasattr(voc, "tokens") else voc._tokens[tok]
            continue
        nid = voc.add(tok) if hasattr(voc, "add") else voc._add(tok)
        new_ids[tok] = nid
    new_size = len(voc)
    log(f"  vocab {old_size} → {new_size}")
    sd["network_parameter"]["vocabulary_size"] = new_size
    rng = torch.Generator().manual_seed(RANDOM_SEED)
    net = sd["network_state"]
    for key in ("src_embed.0.lut.weight", "tgt_embed.0.lut.weight", "generator.proj.weight"):
        w = net[key]
        n_new = new_size - w.shape[0]
        if n_new <= 0: continue
        mean = w.mean(dim=0); std = w.std(dim=0).clamp(min=1e-6)
        new_rows = mean[None, :] + std[None, :] * torch.randn((n_new, w.shape[1]), generator=rng) * 0.5
        net[key] = torch.cat([w, new_rows], dim=0)
    key = "generator.proj.bias"
    b = net[key]
    n_new = new_size - b.shape[0]
    if n_new > 0:
        new_bias = torch.full((n_new,), NEW_TOKEN_BIAS_INIT, dtype=b.dtype)
        net[key] = torch.cat([b, new_bias], dim=0)
    PATCHED_PRIOR.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, PATCHED_PRIOR)
    log(f"  saved {PATCHED_PRIOR.name}")
    return new_ids


def run_transfer_learning(device="cuda"):
    sys.path.insert(0, str(REINVENT4))
    from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
    from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
    from torch.utils.data import DataLoader

    log(f"Loading patched prior to {device}...")
    sd = torch.load(PATCHED_PRIOR, map_location="cpu", weights_only=False)
    model = Mol2MolModel.create_from_dict(sd, "training", torch.device(device))
    log(f"  vocab={len(model.vocabulary)}  params={sum(p.numel() for p in model.network.parameters()):,}")

    df_train = pd.read_csv(PAIRS_CSV)
    df_val = pd.read_csv(PAIRS_VAL_CSV)

    def _ds(df):
        return PairedDataset(
            smiles_input=df["Source_Mol"].tolist(),
            smiles_output=df["Target_Mol"].tolist(),
            vocabulary=model.vocabulary,
            tokenizer=model.tokenizer,
            tanimoto_similarities=df["Tc"].values.astype(np.float32),
        )
    train_ds = _ds(df_train); val_ds = _ds(df_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_ds.collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_ds.collate_fn, num_workers=0)
    log(f"  train batches={len(train_dl)}  val batches={len(val_dl)}")

    opt = torch.optim.Adam(model.network.parameters(), lr=LR)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.network.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for step, batch in enumerate(train_dl):
            opt.zero_grad()
            loss = model.likelihood(batch).mean()
            loss.backward()
            opt.step()
            tot += float(loss.item()); n += 1
            if step % 20 == 0:
                log(f"  epoch {epoch} step {step}/{len(train_dl)} loss={loss.item():.4f}")
        train_nll = tot / max(1, n)
        # Val
        model.network.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for batch in val_dl:
                v = model.likelihood(batch).mean()
                vtot += float(v.item()); vn += 1
            val_nll = vtot / max(1, vn)
        log(f"  EPOCH {epoch}/{NUM_EPOCHS}  train_nll={train_nll:.4f}  val_nll={val_nll:.4f}  ({(time.time()-t0)/60.0:.1f} min)")
    FT_PRIOR.parent.mkdir(parents=True, exist_ok=True)
    sd_out = model.get_save_dict()
    torch.save(sd_out, FT_PRIOR)
    log(f"Saved FT prior: {FT_PRIOR} ({FT_PRIOR.stat().st_size / 1024 / 1024:.1f} MB)")


def per_class_sampling_test(device="cuda", n_per_class=50):
    sys.path.insert(0, str(REINVENT4))
    from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
    log("Per-class sampling test...")
    sd = torch.load(FT_PRIOR, map_location="cpu", weights_only=False)
    model = Mol2MolModel.create_from_dict(sd, "inference", torch.device(device))
    voc, tok = model.vocabulary, model.tokenizer
    MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1"
    out_rows = []
    for ctok, smarts_str in WARHEAD_SMARTS.items():
        src_smi = f"{ctok}{MOL1}"
        toks = tok.tokenize(src_smi, with_begin_and_end=True)
        ids = voc.encode(toks).astype(np.int64)
        src_t = torch.from_numpy(ids).long().unsqueeze(0).repeat(n_per_class, 1).to(device)
        src_mask = torch.ones(n_per_class, 1, src_t.size(1), dtype=torch.bool, device=device)
        _in, outs, _ = model.sample(src_t, src_mask, "multinomial")
        pat = Chem.MolFromSmarts(smarts_str)
        n_valid = 0; n_hit = 0
        for s in outs:
            if s.startswith(ctok): s = s[len(ctok):]
            m = Chem.MolFromSmiles(s) if s else None
            if m is None: continue
            n_valid += 1
            if m.HasSubstructMatch(pat): n_hit += 1
            out_rows.append({"prompt_class": ctok, "smiles": s,
                             "matches_prompt_warhead": int(m.HasSubstructMatch(pat))})
        pct = 100.0 * n_hit / n_per_class
        log(f"  {ctok}: valid={n_valid}/{n_per_class}  warhead_match={n_hit} ({pct:.1f}%)")
    pd.DataFrame(out_rows).to_csv(SAMPLES_DIR / "per_class_50.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_pairs", action="store_true")
    ap.add_argument("--skip_patch", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    if not args.skip_pairs:
        build_tanimoto_filtered_pairs()
    if not args.skip_patch:
        patch_prior_vocab()
    if not args.skip_train:
        run_transfer_learning(args.device)
    per_class_sampling_test(args.device)
