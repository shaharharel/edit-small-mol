#!/usr/bin/env python3
"""
REINVENT4 external scoring component: warhead → hinge graph distance.

Concept (ZAP70 covalent inhibitor geometry)
-------------------------------------------
For a covalent kinase inhibitor, the acrylamide warhead attacks Cys346 while a
hinge-binding motif (aminopyridine / aminopyrimidine / 2-aminoazine NH) makes
the canonical Met414 backbone H-bonds.  Empirically, the graph distance (bonds)
between the warhead Cβ (the electrophilic CH2 of the acrylamide) and the
nearest hinge nitrogen sits in roughly 6–8 bonds for well-behaved covalent
inhibitors that fit the ZAP70 ATP pocket geometry.

Too short → warhead is rammed up against the hinge ring (geometrically
infeasible).
Too long  → floppy linker, entropy penalty.

Score
-----
Compute the shortest-path bond distance d on the molecular graph between the
warhead's Cβ (CH2 of [CH2]=[CH][C;X3](=O)[N]) and the closest hinge N (we try
several SMARTS patterns and take min over all matches).

score = exp(-((d - d_target)^2) / (2 * sigma^2))
        with d_target = 7, sigma = 1.5

bounds:
    d ∈ [6, 8]  → score ≥ 0.80   (sweet spot)
    d ∈ [4, 10] → score ≥ 0.16   (acceptable tail)
    no warhead match            → score = 0.0
    warhead but no hinge match  → score = 0.10  (small floor, lets the agent
                                                 still feel reward signal from
                                                 other scorers; the warhead
                                                 gate enforces the warhead
                                                 already, and a separate hinge
                                                 scorer can drive it)

I/O contract (REINVENT4 ExternalProcess)
----------------------------------------
stdin : newline-separated SMILES, one per line
stdout: JSON  {"version": 1, "payload": {"graph_distance": [scores...]}}

Usage in TOML:
  [[stage.scoring.component]]
  [stage.scoring.component.ExternalProcess]
  [[stage.scoring.component.ExternalProcess.endpoint]]
  name = "warhead-hinge graph distance"
  weight = 0.20
  params.executable = "<conda>"
  params.args = "run --no-capture-output -n quris python <abs>/reinvent4_graph_distance_scorer.py"
  params.property = "graph_distance"
"""

import sys
import json
import os
import math
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
logging.disable(logging.CRITICAL)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# ----------------------------------------------------------------------------
# SMARTS patterns
# ----------------------------------------------------------------------------
# Warhead: acrylamide CH2=CH-C(=O)-N (match index 0 is the Cβ / CH2)
WARHEAD_SMARTS = "[CH2]=[CH][C;X3](=O)[N]"
WARHEAD_PATTERN = Chem.MolFromSmarts(WARHEAD_SMARTS)

# Hinge patterns (ordered: more specific first). We collect ALL candidate
# nitrogen atom indices and take the minimum graph distance.
#
# Each entry: (name, smarts, N-position-in-match-tuple)
HINGE_PATTERNS = [
    # 2-aminoazine: aromatic carbon - NH2 (afatinib/erlotinib/ibrutinib amino-
    # pyrimidine warhead-distal NH2). N is index 1 in the match.
    ("c-NH2",            "[c][NH2]",           1),
    # secondary aniline / aminopyridine: c-NH-c (osimertinib aniline NH).
    # N is index 1 in the match.
    ("c-NH-c",           "[c][NH;X3][c]",      1),
    # bridging aniline NH to aliphatic: c-NH-[C;!c]
    ("c-NH-C",           "[c][NH;X3][C;!c]",   1),
]
HINGE_COMPILED = [
    (name, Chem.MolFromSmarts(sma), npos) for name, sma, npos in HINGE_PATTERNS
]

# Gaussian target & spread
D_TARGET = 7.0
D_SIGMA  = 1.5

# Floors (let RL still learn something when a constraint is partial)
NO_WARHEAD_SCORE = 0.0   # multiplicative gate handled separately by warhead scorer
NO_HINGE_SCORE   = 0.10


def _all_hinge_nitrogens(mol, exclude_atoms=None):
    """Return set of atom indices that look like hinge donor N's.

    Atoms in *exclude_atoms* (the warhead match itself) are filtered out so
    that the acrylamide amide-N isn't mistaken for a hinge donor.
    """
    if exclude_atoms is None:
        exclude_atoms = set()
    cand = set()
    for name, pat, npos in HINGE_COMPILED:
        if pat is None:
            continue
        for match in mol.GetSubstructMatches(pat):
            if npos < len(match):
                ai = match[npos]
                if ai in exclude_atoms:
                    continue
                if mol.GetAtomWithIdx(ai).GetSymbol() == "N":
                    cand.add(ai)
    return cand


def _warhead_match(mol):
    """Return the full warhead match tuple (Cβ, Cα, C=O, =O, N) or None."""
    matches = mol.GetSubstructMatches(WARHEAD_PATTERN)
    return matches[0] if matches else None


def graph_distance_score(smi: str, return_debug: bool = False):
    """Compute the graph-distance score for a single SMILES.

    Returns:
        float in [0, 1]   if return_debug is False
        (score, debug)    otherwise where debug is a dict
    """
    debug = {"smiles": smi, "cbeta": None, "hinge_n": None,
             "distance": None, "score": 0.0, "reason": ""}

    if not smi:
        debug["reason"] = "empty"
        return (0.0, debug) if return_debug else 0.0

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        debug["reason"] = "invalid_smiles"
        return (0.0, debug) if return_debug else 0.0

    warhead = _warhead_match(mol)
    if warhead is None:
        debug["reason"] = "no_warhead"
        return (NO_WARHEAD_SCORE, debug) if return_debug else NO_WARHEAD_SCORE
    cb = warhead[0]
    debug["cbeta"] = cb

    # Exclude warhead atoms themselves from hinge candidates.  Critically this
    # filters out the acrylamide amide-N (which would otherwise be picked up
    # by the c-NH-C pattern when an aromatic ring is directly conjugated).
    hinge_ns = _all_hinge_nitrogens(mol, exclude_atoms=set(warhead))
    if not hinge_ns:
        debug["reason"] = "no_hinge"
        return (NO_HINGE_SCORE, debug) if return_debug else NO_HINGE_SCORE

    # Compute shortest path to each candidate hinge N; take min.
    best_d = None
    best_n = None
    for ni in hinge_ns:
        path = Chem.GetShortestPath(mol, cb, ni)
        if not path:
            continue
        d = len(path) - 1
        if d <= 0:
            # Should not happen — Cβ is a carbon, hinge is a nitrogen, but
            # guard anyway.
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_n = ni

    if best_d is None:
        debug["reason"] = "no_path"
        return (NO_HINGE_SCORE, debug) if return_debug else NO_HINGE_SCORE

    debug["hinge_n"] = best_n
    debug["distance"] = best_d

    # Gaussian around D_TARGET
    score = math.exp(-((best_d - D_TARGET) ** 2) / (2.0 * D_SIGMA ** 2))
    debug["score"] = score
    debug["reason"] = "ok"
    return (score, debug) if return_debug else score


# ----------------------------------------------------------------------------
# Self-tests (run with `--test`)
# ----------------------------------------------------------------------------
_TEST_MOLS = {
    # Should land near the ideal 6–8 window
    "osimertinib":   ("COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1", 6, 8),
    # Slightly long (piperidine linker → 9-10 bonds)
    "ibrutinib":     ("C=CC(=O)N1CCCC(n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",      8, 11),
    "zanubrutinib":  ("C=CC(=O)N1CCCC1n1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",         8, 10),
    # No warhead → 0.0
    "benzene":       ("c1ccccc1",                                                       None, None),
    # Warhead but no hinge → NO_HINGE_SCORE floor
    "vinyl_amide":   ("C=CC(=O)NCC",                                                    None, None),
}


def _run_tests():
    print("graph_distance_scorer self-test")
    print("=" * 68)
    ok = True
    for name, (smi, lo, hi) in _TEST_MOLS.items():
        score, dbg = graph_distance_score(smi, return_debug=True)
        d = dbg["distance"]
        print(f"  {name:14s}  d={str(d):>4s}  score={score:.3f}  reason={dbg['reason']}")
        if name == "benzene":
            if score != NO_WARHEAD_SCORE:
                print(f"    FAIL: expected NO_WARHEAD_SCORE={NO_WARHEAD_SCORE}")
                ok = False
        elif name == "vinyl_amide":
            if score != NO_HINGE_SCORE:
                print(f"    FAIL: expected NO_HINGE_SCORE={NO_HINGE_SCORE}")
                ok = False
        else:
            if d is None or not (lo <= d <= hi):
                print(f"    FAIL: distance {d} outside expected [{lo},{hi}]")
                ok = False
    print("=" * 68)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.exit(_run_tests())

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    scores = [graph_distance_score(s) for s in smiles_list]
    print(json.dumps({"version": 1,
                      "payload": {"graph_distance": scores}}))
    in_window = sum(1 for s in scores if s >= 0.80)
    print(f"[graph_dist_scorer] {in_window}/{len(scores)} in 6-8 bond window "
          f"(>=0.80); mean={sum(scores)/max(1,len(scores)):.3f}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
