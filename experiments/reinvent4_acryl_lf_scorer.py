#!/usr/bin/env python3
"""REINVENT4 ExternalProcess scorer: acrylamide retention on the LARGEST FRAGMENT.

Fixes the "anywhere-SMARTS" bug where the standard `MatchingSubstructure`
component matches [CH2]=[CH]C(=O)N ANYWHERE in the molecule — inflating
warhead retention to ~1.0 in every sample regardless of chemistry.

Contract
--------
Input  (stdin): newline-separated SMILES.
Output (stdout JSON): {"version": 1, "payload": {"acryl_lf": [0/1 per SMILES]}}

Rules
-----
For each SMILES:
  1. Parse; if invalid → 0.0.
  2. Extract largest fragment by heavy atom count.
  3. Return 1.0 if largest fragment contains the acryl SMARTS pattern,
     else 0.0.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RDK_DEPRECATION_WARNING"] = "off"
logging.disable(logging.CRITICAL)

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH][C](=O)[N]")


def score_one(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0.0
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return 0.0
    lf = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return 1.0 if lf.HasSubstructMatch(ACRYL_SMARTS) else 0.0


def main():
    smiles = [line.strip() for line in sys.stdin if line.strip()]
    scores = [score_one(s) for s in smiles]
    sys.stdout.write(json.dumps({"version": 1, "payload": {"acryl_lf": scores}}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
