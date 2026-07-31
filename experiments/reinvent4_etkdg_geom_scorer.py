#!/usr/bin/env python3
"""REINVENT4 ExternalProcess scorer: ETKDG-derived acrylamide-nucleophile
distance surrogate.

For each SMILES:
  1. Find [CH2]=[CH]C(=O)N pattern.  Atoms: (Cbeta=CH2, Calpha=CH, Ccarbonyl, O, N).
  2. Embed molecule with ETKDG v3 (single conformer, seed=42).
  3. Compute d_b_nuc = |pos(Cbeta) - pos(Ccarbonyl)| in Angstrom.
     (This is the intramolecular Cbeta<->Ccarbonyl distance, a strong proxy for
     the Cbeta<->Cys346 distance once the amide N is anchored.)

Reward transforms (selected by --mode env var):
  mode="gauss"    :  sigmoid(-|d - 3.5| / 1.5)   # covgeom (V2)
  mode="cliff"    :  sigmoid(-(d - 3.0))          # distance-only cliff (V3)
Both are bounded in (0, 1); higher = better warhead pose.

Input  (stdin): one SMILES per line.
Output (stdout JSON): {"version": 1, "payload": {"etkdg_geom": [scores...]}}

Env vars:
  ETKDG_MODE     : "gauss" (default) or "cliff"
  ETKDG_SEED     : int (default 42)
"""
from __future__ import annotations
import json
import logging
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RDK_DEPRECATION_WARNING"] = "off"
logging.disable(logging.CRITICAL)

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH][C](=O)[N]")


def _sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def score_one(smi: str, mode: str = "gauss", seed: int = 42) -> float:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0.0
    matches = m.GetSubstructMatches(ACRYL_SMARTS)
    if not matches:
        return 0.0
    # Use largest fragment if disconnected
    frags = Chem.GetMolFrags(m, asMols=True)
    if len(frags) > 1:
        m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        matches = m.GetSubstructMatches(ACRYL_SMARTS)
        if not matches:
            return 0.0
    a_cbeta, a_calpha, a_ccarb, a_o, a_n = matches[0]
    try:
        mH = Chem.AddHs(m)
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        cid = AllChem.EmbedMolecule(mH, p)
        if cid < 0:
            return 0.0
        conf = mH.GetConformer(cid)
        p_beta = np.array(conf.GetAtomPosition(a_cbeta))
        p_carb = np.array(conf.GetAtomPosition(a_ccarb))
        d_b_nuc = float(np.linalg.norm(p_beta - p_carb))
    except Exception:
        return 0.0
    if not np.isfinite(d_b_nuc):
        return 0.0

    if mode == "gauss":
        # Covgeom: score peaks at d=3.5
        return _sigmoid(-abs(d_b_nuc - 3.5) / 1.5)
    elif mode == "cliff":
        # Distance-only cliff: hard penalty above d=3.0
        return _sigmoid(-(d_b_nuc - 3.0))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    mode = os.environ.get("ETKDG_MODE", "gauss")
    seed = int(os.environ.get("ETKDG_SEED", "42"))
    smiles = [line.strip() for line in sys.stdin if line.strip()]
    scores = [score_one(s, mode=mode, seed=seed) for s in smiles]
    payload = {"version": 1, "payload": {"etkdg_geom": scores}}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
