#!/usr/bin/env python
"""EXP6 FT vs Base Mol2Mol ablation — generate from both models on the SAME inputs.

Compares the EXP6-FT'd prior to the un-FT'd Mol2Mol base prior. Same input seeds
(20 ZAP70 acrylamide actives), 5 samples per seed per model → 100 mols per model.
Then AD-CovDock score (via meeko tether + Vina score_only) on all 200 mols.

Output: results/paper_evaluation/cohort_eval/ft_vs_base_ablation.csv
"""
from __future__ import annotations
import sys, time, csv, json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
REINVENT4 = Path("/Users/shaharharel/Documents/github/REINVENT4")
BASE_PRIOR = REINVENT4 / "priors/mol2mol_medium_similarity.prior"
FT_PRIOR = PROJECT / "models/reinvent4_mol2mol_warhead_tokens.prior"

N_SAMPLES_PER_SEED = 10
N_SEEDS = 20  # → 200 mols per model, ~400 total dockings

sys.path.insert(0, str(REINVENT4))
sys.path.insert(0, str(PROJECT))

# Pull 20 ZAP70 actives with acrylamide as seeds
from experiments.run_zap70_v3 import load_zap70_molecules
smiles_df, _ = load_zap70_molecules()
acryl_pat = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
seeds = []
for s in smiles_df["smiles"]:
    m = Chem.MolFromSmiles(s) if s else None
    if m is None: continue
    if m.HasSubstructMatch(acryl_pat):
        seeds.append(s)
    if len(seeds) >= N_SEEDS:
        break
print(f"Using {len(seeds)} ZAP70 acrylamide seeds")

from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel

def gen_from_model(prior_path, prefix=""):
    print(f"\n=== Loading {prior_path.name} (prefix={prefix!r}) ===")
    sd = torch.load(prior_path, map_location="cpu", weights_only=False)
    model = Mol2MolModel.create_from_dict(sd, "inference", torch.device("cpu"))
    voc, tok = model.vocabulary, model.tokenizer
    out_smis = []
    for i, seed in enumerate(seeds):
        src_smi = f"{prefix}{seed}"
        toks = tok.tokenize(src_smi, with_begin_and_end=True)
        ids = voc.encode(toks).astype(np.int64)
        src_t = torch.from_numpy(ids).long().unsqueeze(0).repeat(N_SAMPLES_PER_SEED, 1)
        src_mask = torch.ones(N_SAMPLES_PER_SEED, 1, src_t.size(1), dtype=torch.bool)
        _in, outs, _ = model.sample(src_t, src_mask, "multinomial")
        for s in outs:
            if prefix and s.startswith(prefix): s = s[len(prefix):]
            out_smis.append((i, seed, s))
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{len(seeds)} seeds → {len(out_smis)} mols")
    return out_smis

base_out = gen_from_model(BASE_PRIOR, prefix="")
ft_out = gen_from_model(FT_PRIOR, prefix="[ACRYLAMIDE]")

# Save raw outputs
out_csv = PROJECT / "results/paper_evaluation/cohort_eval/ft_vs_base_raw.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","seed_idx","seed_smi","output_smi","valid","has_acryl"])
    for tag, lst in [("base", base_out), ("ft", ft_out)]:
        for si, seed, s in lst:
            m = Chem.MolFromSmiles(s) if s else None
            valid = int(m is not None)
            has = int(m is not None and m.HasSubstructMatch(acryl_pat))
            w.writerow([tag, si, seed, s, valid, has])
print(f"\nWrote raw: {out_csv}")

# Quick summary
import collections
for tag, lst in [("base", base_out), ("ft", ft_out)]:
    valid, acryl = 0, 0
    for _, _, s in lst:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None: continue
        valid += 1
        if m.HasSubstructMatch(acryl_pat): acryl += 1
    print(f"{tag}: n={len(lst)}  valid={valid}  acryl_match={acryl}  acryl%={100*acryl/max(1,len(lst)):.1f}")
