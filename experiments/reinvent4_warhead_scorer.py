#!/usr/bin/env python3
"""
REINVENT4 external scoring component: warhead-intact gate (0/1).

Reads SMILES from stdin (one per line), outputs JSON
  {"version": 1, "payload": {"warhead": [0.0, 1.0, ...]}}

Score:
  1.0 if SMILES contains the acrylamide warhead [CH2]=[CH]-C(=O)-[N;!H2]
  0.0 otherwise (or invalid SMILES).

Used as a multiplicative gate in REINVENT4 RL: candidates that lose the
warhead get score 0 → are penalised.

Usage in TOML:
  [[stage.scoring.component]]
  [stage.scoring.component.ExternalProcess]
  [[stage.scoring.component.ExternalProcess.endpoint]]
  name = "warhead"
  weight = 1.0
  params.executable = "conda"
  params.args = ["run", "--no-capture-output", "-n", "quris", "python",
                 "/abs/path/to/experiments/reinvent4_warhead_scorer.py"]
  [stage.scoring.component.ExternalProcess.endpoint.transform]
  type = "value_mapping"
  mapping = { 0.0 = 0.0, 1.0 = 1.0 }
"""

import sys
import json
import os
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
logging.disable(logging.CRITICAL)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

WARHEAD_SMARTS = "[CH2]=[CH]-C(=O)-[N;!H2]"
PATTERN = Chem.MolFromSmarts(WARHEAD_SMARTS)


def score(smi: str) -> float:
    if not smi:
        return 0.0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    return 1.0 if mol.HasSubstructMatch(PATTERN) else 0.0


def main():
    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    scores = [score(s) for s in smiles_list]
    print(json.dumps({"version": 1, "payload": {"warhead": scores}}))
    print(f"[warhead_scorer] {sum(scores):.0f}/{len(scores)} intact",
          file=sys.stderr)


if __name__ == "__main__":
    main()
