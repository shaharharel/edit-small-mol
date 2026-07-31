#!/usr/bin/env python3
"""
REINVENT4 external scoring component: reactivity-aware warhead scoring.

Replaces the simple "warhead-intact" 0/1 gate with a richer score that:
  - Penalises warheads that are TOO reactive (off-target Cys: BTK Cys481,
    EGFR Cys797, JAK3 Cys909). These are predicted to be promiscuous.
  - Penalises warheads that are NOT reactive enough (will not form
    the ZAP70 Cys346 bond).
  - Rewards warheads in the "sweet spot" of GSH k2 [-2.5, -1.5] log range,
    which has historically tracked covalent kinase inhibitor success.

Inputs:  newline-separated SMILES on stdin.
Outputs: JSON  {"version": 1, "payload": {"reactivity": [s_0, s_1, ...]}}
Each score is in [0,1]; 0 = bad, 1 = ideal.

Scoring schematic (multiplicative gating):
  1.  Warhead must exist.   (0 otherwise)
  2.  No "too-reactive" SMARTS hits.   (multiplier 0.0–0.2)
  3.  No "reversible/over-reactive beta-EWG" hits.   (multiplier 0.5)
  4.  Reward "clean primary acrylamide" or "vinyl-sulfonamide".  (+0.2)
  5.  Predicted log k2(GSH) in [-2.5, -1.5] gives the highest score.
      A simple linear surrogate over the matched-warhead environment is used.

The k2 surrogate is a hand-tuned linear model on Hammett-style EWG/EDG
counts in the beta-position to the Michael acceptor; it is not a ML model
and stays interpretable.

Usage standalone:
    echo "CC(=O)Nc1ccc(/C=C/C(=O)NCCN)cc1" |
       conda run --no-capture-output -n quris python \\
       experiments/reinvent4_reactivity_scorer.py

Usage with REINVENT4 (TOML snippet):

    [[stage.scoring.component]]
    [stage.scoring.component.ExternalProcess]
    [[stage.scoring.component.ExternalProcess.endpoint]]
    name = "Reactivity"
    weight = 0.3
    params.executable = "__CONDA_PATH__"
    params.args = "run --no-capture-output -n quris python \\
                   __PROJECT_ROOT__/experiments/reinvent4_reactivity_scorer.py"
    params.property = "reactivity"
    transform.type = "no_transform"

Self-test:
    python experiments/reinvent4_reactivity_scorer.py --selftest
"""

import sys
import json
import os
import warnings
import logging
from typing import List, Tuple

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
logging.disable(logging.CRITICAL)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


# ----------------------------------------------------------------------
# 1.  Warhead set (covalent-kinase relevant)
#     If none match, score is 0.
# ----------------------------------------------------------------------
WARHEAD_SMARTS = [
    # Primary acrylamide -CH=CH-C(=O)-NH-R   (canonical Cys-binder)
    ("acrylamide_primary", "[CH2]=[CH]-C(=O)-[N;!H2]"),
    # Substituted acrylamide R-CH=CR'-C(=O)-NR'' (still binds, e.g. neratinib)
    ("acrylamide_sub",     "[CH1,CH0]=[CH1,CH0]-C(=O)-[N;!H2]"),
    # Vinyl sulfonamide  CH2=CH-S(=O)(=O)-NR
    ("vinyl_sulfonamide",  "[CH2]=[CH]-S(=O)(=O)-[N]"),
    # Propiolamide  HC#C-C(=O)NR   (faster covalent, used in eg afatinib analogues)
    ("propiolamide",       "[C]#[C]-C(=O)-[N;!H2]"),
]
_WARHEAD_PATS = [(n, Chem.MolFromSmarts(s)) for n, s in WARHEAD_SMARTS]


# ----------------------------------------------------------------------
# 2.  Over-reactive / promiscuous warhead patterns  (multiplicative penalty)
#     These are red-flag SMARTS from the medicinal-chemistry literature
#     (PAINS-like + extra covalent-cys patterns). Each hit drops the
#     reactivity score multiplicatively.
# ----------------------------------------------------------------------
OVER_REACTIVE_SMARTS = [
    # Acyl-imide R-C(=O)-N(C=O)-  (highly electrophilic, hits everything)
    ("acyl_imide",         "[CX3](=O)[NX3]([#1,#6])[CX3](=O)"),
    # Acyl-urea R-C(=O)-N(H)-C(=O)-N(H)-R  (also very reactive)
    ("acyl_urea",          "[CX3](=O)[NX3;H1][CX3](=O)[NX3]"),
    # Alpha-halo carbonyl  X-C-C(=O)-  (alkylating)
    ("alpha_halo_carbonyl","[F,Cl,Br,I][CX4][CX3](=O)"),
    # Beta-cyano acrylamide (reversible, leaks off-target).
    # The cyano is alpha to the carbonyl, vinyl is beta. Two SMARTS catch both
    # SMILES-writing orders.
    ("beta_cyano_acryl",   "N#CC(=C)C(=O)[N]"),
    ("beta_cyano_acryl2",  "N#CC(C(=O)[N])=C"),
    # Beta-nitro / beta-sulfonyl acrylamide (over-reactive, hits BTK Cys481)
    ("beta_nitro_acryl",   "[$([N+](=O)[O-]),$([S](=O)=O)][CH]=CC(=O)[N]"),
    # Aldehyde (broad cys-binder)
    ("aldehyde",           "[CX3H1](=O)[#6]"),
    # Maleimide (extremely reactive, classic warhead but ZERO selectivity)
    ("maleimide",          "O=C1[CH]=[CH]C(=O)N1"),
    # Isocyanate / Isothiocyanate
    ("isocyanate",         "N=C=O"),
    ("isothiocyanate",     "N=C=S"),
    # Chloroacetamide (irreversible alkylation, off-target promiscuity)
    ("chloroacetamide",    "[Cl][CH2]C(=O)[N]"),
    # Epoxide
    ("epoxide",            "C1OC1"),
    # Vinyl ketone (Michael, but no NH to anchor selectivity)
    ("vinyl_ketone",       "[CH2]=[CH]C(=O)[#6;!$([N])]"),
    # Generic nitro group close to acceptor (over-reactive on para position)
    ("para_nitro_acryl",   "[$([N+](=O)[O-])][c]:c(:c):c-[CH]=[CH]C(=O)[N]"),
]
_OVER_PATS = [(n, Chem.MolFromSmarts(s)) for n, s in OVER_REACTIVE_SMARTS]


# ----------------------------------------------------------------------
# 3.  "Sweet-spot" rewards
# ----------------------------------------------------------------------
REWARD_SMARTS = [
    # Clean primary acrylamide -CH=CH-C(=O)-NH-CH2- (the canonical Cys binder
    # used in ibrutinib, acalabrutinib).
    ("clean_acrylamide",   "[CH2]=[CH]C(=O)[NH][CH2,CH1]"),
    # Vinyl sulfonamide attached to a sp3 carbon (somewhat polar, acceptable).
    ("vinyl_sulfonamide_sp3", "[CH2]=[CH]S(=O)(=O)[NH][#6;X4]"),
]
_REWARD_PATS = [(n, Chem.MolFromSmarts(s)) for n, s in REWARD_SMARTS]


# ----------------------------------------------------------------------
# 4.  EWG / EDG SMARTS counted in the beta-position to the Michael acceptor.
#     Used by the linear k2 surrogate.
# ----------------------------------------------------------------------
EWG_BETA_SMARTS = [
    "[F,Cl,Br,I][CH]=[CH]C(=O)[N]",      # halo
    "[CX3](=O)[OX2H][CH]=[CH]C(=O)[N]",  # carboxy ester
    "[C](#N)[CH]=[CH]C(=O)[N]",          # cyano
    "[N+](=O)[O-][CH]=[CH]C(=O)[N]",     # nitro
    "[S](=O)(=O)[CH]=[CH]C(=O)[N]",      # sulfonyl
]
EDG_BETA_SMARTS = [
    "[NH2,NH][CH]=[CH]C(=O)[N]",         # amino
    "[OH][CH]=[CH]C(=O)[N]",             # hydroxyl
    "[OCH3][CH]=[CH]C(=O)[N]",           # methoxy
    "[CH3][CH]=[CH]C(=O)[N]",            # alkyl
]
_EWG_PATS = [Chem.MolFromSmarts(s) for s in EWG_BETA_SMARTS]
_EDG_PATS = [Chem.MolFromSmarts(s) for s in EDG_BETA_SMARTS]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _has_warhead(mol) -> Tuple[bool, str]:
    for name, pat in _WARHEAD_PATS:
        if pat is not None and mol.HasSubstructMatch(pat):
            return True, name
    return False, ""


def _count_over_reactive(mol) -> List[str]:
    return [n for n, p in _OVER_PATS if p is not None and mol.HasSubstructMatch(p)]


def _count_rewards(mol) -> List[str]:
    return [n for n, p in _REWARD_PATS if p is not None and mol.HasSubstructMatch(p)]


def _predict_log_k2_gsh(mol) -> float:
    """Crude Hammett-style linear estimator for log k2(GSH).

    Baseline primary acrylamide log k2(GSH) ~ -2.0 (from Lonsdale 2017,
    Flanagan 2014). Each EWG in the beta-position shifts +0.7; each EDG
    shifts -0.5. Vinyl sulfonamide baseline shifts -0.4 (slower).
    Propiolamide baseline shifts +1.2 (much faster).
    """
    base = -2.0
    ewg = sum(len(mol.GetSubstructMatches(p)) for p in _EWG_PATS if p is not None)
    edg = sum(len(mol.GetSubstructMatches(p)) for p in _EDG_PATS if p is not None)
    delta = 0.7 * ewg - 0.5 * edg
    # warhead-class adjustments
    for name, pat in _WARHEAD_PATS:
        if pat is None or not mol.HasSubstructMatch(pat):
            continue
        if name == "propiolamide":
            delta += 1.2
        elif name == "vinyl_sulfonamide":
            delta -= 0.4
    return base + delta


def _k2_window_score(log_k2: float,
                     low: float = -2.5,
                     high: float = -1.5) -> float:
    """Gaussian-ish window in [low, high]; 1.0 inside, decays outside."""
    if low <= log_k2 <= high:
        return 1.0
    if log_k2 < low:
        return max(0.0, 1.0 - 0.8 * (low - log_k2))
    return max(0.0, 1.0 - 1.2 * (log_k2 - high))


# ----------------------------------------------------------------------
# Main scoring function
# ----------------------------------------------------------------------
def reactivity_score(smiles: str, debug: bool = False) -> float:
    if not smiles:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    has_wh, wh_name = _has_warhead(mol)
    if not has_wh:
        return 0.0

    over_hits = _count_over_reactive(mol)
    reward_hits = _count_rewards(mol)
    log_k2 = _predict_log_k2_gsh(mol)
    window = _k2_window_score(log_k2)

    # multiplicative over-reactive penalty
    over_penalty = 1.0
    for h in over_hits:
        # The really dangerous ones go to 0.0
        if h in ("maleimide", "isocyanate", "isothiocyanate",
                 "acyl_imide", "alpha_halo_carbonyl",
                 "aldehyde", "epoxide", "chloroacetamide"):
            over_penalty *= 0.0
        # Moderately bad: dampens score
        elif h in ("acyl_urea", "vinyl_ketone"):
            over_penalty *= 0.25
        # Reversible / over-reactive beta:
        elif h in ("beta_cyano_acryl", "beta_cyano_acryl2",
                   "beta_nitro_acryl", "para_nitro_acryl"):
            over_penalty *= 0.5
        else:
            over_penalty *= 0.75

    # Reward boost: clamp to keep final in [0,1]
    reward_boost = 1.0
    if reward_hits:
        reward_boost = 1.15  # +15% if a clean-acrylamide pattern present

    raw = window * over_penalty * reward_boost
    score = max(0.0, min(1.0, raw))

    if debug:
        print(f"  warhead={wh_name} over_hits={over_hits} "
              f"rewards={reward_hits} log_k2={log_k2:.2f} "
              f"window={window:.2f} over={over_penalty:.2f} "
              f"reward_boost={reward_boost:.2f} -> {score:.3f}",
              file=sys.stderr)

    return float(score)


# ----------------------------------------------------------------------
# CLI entry
# ----------------------------------------------------------------------
def selftest():
    """Unit test the scorer on known cases."""
    cases = [
        # (label, SMILES, expected_band, comment)
        ("ibrutinib",
         "C=CC(=O)N1CCC[C@@H](C1)n1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
         ("mid", "high"),
         "valid acrylamide warhead, real drug"),
        ("acalabrutinib",
         "CC#CC(=O)N1CCC[C@H]1c1nc(-c2ccc(C(=O)Nc3ccccn3)cc2)c2c(N)ncnn12",
         ("low", "high"),
         "propiolamide -> may be flagged high k2"),
        ("ribose_triphosphate",
         "OCC1OC(n2cnc3c2ncnc3N)C(O)C1O",
         ("zero", "zero"),
         "no warhead at all"),
        ("acyl_urea_overreactive",
         "O=C(NC(=O)c1ccccc1)NC(=O)/C=C/c1ccccc1",
         ("low", "low"),
         "acyl-urea, too reactive"),
        ("maleimide_bad",
         "O=C1C=CC(=O)N1c1ccccc1",
         ("zero", "zero"),
         "maleimide, kill score"),
        ("beta_cyano_reversible",
         "N#C/C(=C\\c1ccc(O)cc1)C(=O)Nc1ccccc1",
         ("low", "low"),
         "beta-cyano acrylamide, reversible"),
        ("benzene",
         "c1ccccc1",
         ("zero", "zero"),
         "no warhead at all"),
        ("vinyl_sulfonamide_ok",
         "C=CS(=O)(=O)NCc1ccccc1",
         ("mid", "high"),
         "acceptable, more polar warhead"),
    ]

    print("Reactivity scorer self-test", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    bands = {
        "zero": (0.0, 0.05),
        "low":  (0.05, 0.45),
        "mid":  (0.30, 0.85),
        "high": (0.50, 1.00),
    }
    passed = 0
    failed = []
    for label, smi, expected, comment in cases:
        s = reactivity_score(smi, debug=True)
        lo_band, hi_band = expected
        lo = bands[lo_band][0]
        hi = bands[hi_band][1]
        ok = lo <= s <= hi
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {label:25s}  score={s:.3f}  "
              f"expected[{lo_band}->{hi_band}={lo:.2f}..{hi:.2f}]  "
              f"({comment})",
              file=sys.stderr)
        if ok:
            passed += 1
        else:
            failed.append((label, s, expected))
    print("=" * 60, file=sys.stderr)
    print(f"  {passed}/{len(cases)} passed", file=sys.stderr)
    return passed, len(cases), failed


def main():
    if "--selftest" in sys.argv:
        passed, total, failed = selftest()
        sys.exit(0 if passed == total else 1)

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    scores = [reactivity_score(s) for s in smiles_list]
    print(json.dumps({"version": 1, "payload": {"reactivity": scores}}))
    n_pass = sum(1 for s in scores if s >= 0.5)
    print(f"[reactivity_scorer] {n_pass}/{len(scores)} >=0.5  "
          f"(mean {sum(scores)/max(1,len(scores)):.3f})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
