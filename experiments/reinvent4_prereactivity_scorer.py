#!/usr/bin/env python3
"""REINVENT4 ExternalProcess scorer: cheap 2D pre-reactivity geometry score.

For each input SMILES:
  1. Find acrylamide warhead via SMARTS: [CH2]=[CH]C(=O)N
  2. If absent -> score = 0.0
  3. ETKDG embed + brief MMFF optimization
  4. Measure C(beta)=C(alpha)-C(=O)-N dihedral
  5. planar_dev = min(|d|, |180-|d||)  (i.e. distance to 0 or 180)
  6. Score = max(0, 1 - planar_dev / 60.0). Score=1 when planar (within ~0 deg
     of cis/trans), linearly decays to 0 at 60 deg deviation.

Output JSON: {"version": 1, "payload": {"prereactivity": [s_0, ...]}}

Used by R2/R3 cheap-geometry RL streams (R-stream orchestrator).
"""
from __future__ import annotations
import json
import logging
import os
import signal
import sys
import warnings
from typing import List

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RDK_DEPRECATION_WARNING"] = "off"
logging.disable(logging.CRITICAL)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

RDLogger.DisableLog("rdApp.*")

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
DECAY_DEG = 60.0
PER_MOL_TIMEOUT_S = 3  # ETKDG can hang on pathological inputs


class _TO(Exception):
    pass


def _to_handler(signum, frame):
    raise _TO()


def _score_one(smi: str, patt) -> float:
    if not isinstance(smi, str) or not smi:
        return 0.0
    signal.signal(signal.SIGALRM, _to_handler)
    signal.alarm(PER_MOL_TIMEOUT_S)
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return 0.0
        m = matches[0]
        b_idx, a_idx, c_idx, n_idx = int(m[0]), int(m[1]), int(m[2]), int(m[4])

        mol_h = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        rc = AllChem.EmbedMolecule(mol_h, p)
        if rc != 0:
            p.useRandomCoords = True
            rc = AllChem.EmbedMolecule(mol_h, p)
            if rc != 0:
                return 0.2  # warhead intact, no conformer
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=50)
        except Exception:
            pass
        conf = mol_h.GetConformer()
        d = GetDihedralDeg(conf, b_idx, a_idx, c_idx, n_idx)
        dev = min(abs(d), abs(180.0 - abs(d)))
        s = max(0.0, 1.0 - dev / DECAY_DEG)
        return float(s)
    except _TO:
        return 0.2  # warhead-intact partial-credit when scorer timed out
    except Exception:
        return 0.0
    finally:
        signal.alarm(0)


def main():
    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    if not smiles_list:
        print(json.dumps({"version": 1, "payload": {"prereactivity": []}}))
        return
    patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    scores: List[float] = [_score_one(s, patt) for s in smiles_list]
    print(json.dumps({"version": 1, "payload": {"prereactivity": scores}}))
    sys.stderr.write(f"[prereactivity] scored n={len(scores)} mean={sum(scores)/max(1,len(scores)):.3f}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
        tests = [
            ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "Mol1"),
            ("CCO", "ethanol"),
            ("C=CC(=O)NCC", "simple acrylamide"),
        ]
        for smi, label in tests:
            print(f"  {label}: {_score_one(smi, patt):.3f}  smi={smi}")
    else:
        main()
