#!/usr/bin/env python
"""Quick test: sample 50 mols from EXP6 prior with each of the 4 warhead control tokens.
Verifies whether EXP6 learned to generate non-acrylamide warheads (chloroacet, vinyl-sulf, epoxide).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
REINVENT4_ROOT = Path("/Users/shaharharel/Documents/github/REINVENT4")
FT_PRIOR = PROJECT / "models/reinvent4_mol2mol_warhead_tokens.prior"
MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1"

WARHEAD_SMARTS = {
    "[ACRYLAMIDE]":         "[CH2]=[CH]C(=O)N",
    "[CHLOROACETAMIDE]":    "Cl[CH2]C(=O)N",
    "[VINYL_SULFONAMIDE]":  "[CH2]=[CH]S(=O)(=O)N",
    "[EPOXIDE]":            "[#6]1O[#6]1",
}

N_PER_CLASS = 50

sys.path.insert(0, str(REINVENT4_ROOT))
from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel

print(f"Loading {FT_PRIOR.name}...")
sd = torch.load(FT_PRIOR, map_location="cpu", weights_only=False)
model = Mol2MolModel.create_from_dict(sd, "inference", torch.device("cpu"))
voc = model.vocabulary
tok = model.tokenizer
print(f"vocab size = {len(voc)}")

results = {}
for ctok, smarts_str in WARHEAD_SMARTS.items():
    print(f"\n--- {ctok} (smarts={smarts_str}) ---")
    src_smi = f"{ctok}{MOL1}"
    src_tokens = tok.tokenize(src_smi, with_begin_and_end=True)
    src_ids = voc.encode(src_tokens).astype(np.int64)
    src_t = torch.from_numpy(src_ids).long().unsqueeze(0).repeat(N_PER_CLASS, 1)
    src_mask = torch.ones(N_PER_CLASS, 1, src_t.size(1), dtype=torch.bool)
    t0 = time.time()
    _in, out_smis, nlls = model.sample(src_t, src_mask, "multinomial")
    wall = time.time() - t0
    pat = Chem.MolFromSmarts(smarts_str)
    n_valid, n_unique, n_hit = 0, set(), 0
    sample_hits = []
    for s in out_smis:
        if s.startswith(ctok):
            s = s[len(ctok):]
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            continue
        n_valid += 1
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        n_unique.add(can)
        if mol.HasSubstructMatch(pat):
            n_hit += 1
            if len(sample_hits) < 3:
                sample_hits.append(can)
    pct = 100.0 * n_hit / N_PER_CLASS
    print(f"  sampled={N_PER_CLASS}  valid={n_valid}  unique={len(n_unique)}  warhead_match={n_hit} ({pct:.1f}%)  wall={wall:.1f}s")
    for s in sample_hits:
        print(f"    sample: {s}")
    results[ctok] = {"n_valid": n_valid, "n_unique": len(n_unique), "n_hit": n_hit, "pct": pct}

print("\n=== Summary ===")
print(f"{'class':<22}  sampled  valid  unique  match  retention%")
for c, r in results.items():
    print(f"{c:<22}  {N_PER_CLASS:>7}  {r['n_valid']:>5}  {r['n_unique']:>6}  {r['n_hit']:>5}  {r['pct']:>9.1f}")
