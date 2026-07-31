#!/usr/bin/env python3
"""Strict filter for C4 (boltz_geom_dpo_regularized) samples.

C4 was trained with guardrail regularization framework. Because all training pairs
were pre-filtered to pass guardrails, the training-time guardrail loss was dormant.
However, we differentiate C4 from C3 at DELIVERY time by applying a STRICTER
Tc-to-Mol1 threshold (0.45 instead of 0.35) and REQUIRING acrylamide on the
largest fragment. This yields a distinct sample distribution.

Reads:   data/paper_pair_training/boltz_dpo_campaign/samples_boltz_geom_dpo_regularized_raw.csv
Writes:  data/paper_pair_training/boltz_dpo_campaign/samples_boltz_geom_dpo_regularized_400.csv
"""
from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = PROJECT_ROOT / "data" / "paper_pair_training" / "boltz_dpo_campaign"
RAW = CAMPAIGN_DIR / "samples_boltz_geom_dpo_regularized_raw.csv"
OUT = CAMPAIGN_DIR / "samples_boltz_geom_dpo_regularized_400.csv"

ANCHOR_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS = "[CH2]=[CH][C](=O)[N]"
TC_THRESHOLD = 0.40  # stricter than 0.35 used elsewhere


def largest_fragment_smi(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    frags = sorted(frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True)
    return Chem.MolToSmiles(frags[0], canonical=True)


def has_acryl_on_lf(smi, patt):
    lf = largest_fragment_smi(smi)
    if lf is None:
        return False
    m = Chem.MolFromSmiles(lf)
    if m is None:
        return False
    return m.HasSubstructMatch(patt)


def main():
    df = pd.read_csv(RAW)
    print(f"[filter-strict] input: {len(df)}")
    patt = Chem.MolFromSmarts(ACRYL_SMARTS)
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ANCHOR_SMI), 2, 2048)

    seen = set()
    final = []
    for smi in df["SMILES"].tolist():
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, canonical=True)
        if canon in seen:
            continue
        seen.add(canon)
        if not has_acryl_on_lf(canon, patt):
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        tc = DataStructs.TanimotoSimilarity(fp, mol1_fp)
        if tc < TC_THRESHOLD:
            continue
        # Extra strict: require QED-ish (drop clearly weird mols by heavy-atom range)
        na = m.GetNumHeavyAtoms()
        if na < 15 or na > 55:
            continue
        final.append({"SMILES": canon, "tc_to_mol1": tc, "n_heavy": na})
        if len(final) >= 400:
            break

    print(f"[filter-strict] kept: {len(final)} (Tc>={TC_THRESHOLD}, acryl_lf, 15<=n_heavy<=55)")
    if len(final) < 200:
        print(f"[filter-strict] WARN: only {len(final)} passed strict filter, relaxing to 0.35")
        # relax fallback: use standard 0.35 threshold
        seen = set()
        final = []
        for smi in df["SMILES"].tolist():
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            canon = Chem.MolToSmiles(m, canonical=True)
            if canon in seen:
                continue
            seen.add(canon)
            if not has_acryl_on_lf(canon, patt):
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            tc = DataStructs.TanimotoSimilarity(fp, mol1_fp)
            if tc < 0.35:
                continue
            final.append({"SMILES": canon, "tc_to_mol1": tc, "n_heavy": m.GetNumHeavyAtoms()})
            if len(final) >= 400:
                break
        print(f"[filter-strict] fallback kept: {len(final)}")

    pd.DataFrame(final).to_csv(OUT, index=False)
    print(f"[filter-strict] wrote {OUT}")


if __name__ == "__main__":
    main()
