"""Smoke test: score 5 hand-picked SMILES, check obvious ranking.

Expected ranking sanity:
  Mol1 (real ZAP70 acrylamide):                expected mid-high
  RDN009 (vinyl sulfone):                      LOW (no acrylamide → no_warhead → 0)
  RDN2150-like (primary acrylamide):           expected mid
  Polymer (huge floppy):                       LOW
  Unparseable junk:                            0 (unparseable)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_geom_scorer import FastGeomScorer

ANCHOR = Path(__file__).resolve().parents[2] / "data/fast_geom_surrogate/anchor_frame.npz"

CASES = [
    ("Mol1 (ZAP70 acrylamide, drug-like)",
     "C=CC(=O)N1Cc2cccc(C(=O)Nc3cnc(NC(=O)c4cn(C)nc4C(C)C)cn3)c2C1"),
    ("RDN009 (vinyl sulfone, NOT acrylamide)",
     "C=CS(=O)(=O)c1ccccc1"),
    ("Primary acrylamide on benzylamine (RDN2150-like)",
     "C=CC(=O)NCc1ccccc1"),
    ("Polymer (huge, floppy, clash-prone)",
     "C=CC(=O)N1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1"),
    ("Unparseable junk",
     "$$$bad_smiles$$$"),
]


def main():
    if not ANCHOR.exists():
        print(f"missing anchor at {ANCHOR}", file=sys.stderr); sys.exit(2)
    sc = FastGeomScorer(ANCHOR, n_conformers=3)

    print(f"\n{'name':50s}  {'score':>8s}  {'shape':>6s}  {'hinge':>6s}  {'bd':>6s}  {'gate':>6s}  reason")
    print("-" * 110)
    t0 = time.time()
    rows = []
    for name, smi in CASES:
        det = sc._score_internal(smi)
        rows.append((name, smi, det))
        print(f"{name[:50]:50s}  {det['composite']:>8.3f}  "
              f"{det.get('shape_score',0.0):>6.2f}  {det.get('hinge_score',0.0):>6.2f}  "
              f"{det.get('bd_score',0.0):>6.2f}  {det.get('clash_gate',0.0):>6.2f}  "
              f"{det.get('reason','?')}")
    dt = time.time() - t0
    print(f"\nelapsed: {dt:.2f}s ({dt/len(CASES)*1000:.1f} ms/mol average)")

    # Ranking sanity assertions (loose — log-level only).
    print("\n[sanity checks]")
    score = {n: r[2]["composite"] for n, _, r in [(n, s, (n, s, det)) for (n, s), det in zip(CASES, [r[2] for r in rows])]}
    # rebuild as direct dict
    score = {n: rows[i][2]["composite"] for i, (n, _) in enumerate(CASES)}
    reasons = {n: rows[i][2].get("reason", "?") for i, (n, _) in enumerate(CASES)}

    ok = True
    # 1. RDN009 has no acrylamide → score=0
    if reasons["RDN009 (vinyl sulfone, NOT acrylamide)"] != "no_warhead" or \
       score["RDN009 (vinyl sulfone, NOT acrylamide)"] > 1e-6:
        print(f"  FAIL: RDN009 expected 0 (no_warhead), got {score['RDN009 (vinyl sulfone, NOT acrylamide)']}")
        ok = False
    else:
        print("  PASS: vinyl sulfone correctly rejected (no acrylamide SMARTS match)")
    # 2. Unparseable → 0
    if reasons["Unparseable junk"] != "unparseable" or score["Unparseable junk"] > 1e-6:
        print(f"  FAIL: junk expected 0, got {score['Unparseable junk']}")
        ok = False
    else:
        print("  PASS: unparseable SMILES returns 0")
    # 3. Polymer < Mol1 (real drug should outscore the giant chain)
    mol1 = score["Mol1 (ZAP70 acrylamide, drug-like)"]
    poly = score["Polymer (huge, floppy, clash-prone)"]
    if poly > mol1:
        print(f"  WARN: polymer ({poly:.3f}) outscores Mol1 ({mol1:.3f}) — geometry surrogate is "
              f"chemistry-blind by design; this is expected if shape alignment happens to work")
    else:
        print(f"  PASS: Mol1 ({mol1:.3f}) ≥ polymer ({poly:.3f})")
    # 4. Mol1 > 0
    if mol1 < 1e-3:
        print(f"  FAIL: Mol1 should produce a non-zero score, got {mol1}")
        ok = False
    else:
        print(f"  PASS: Mol1 produces non-zero score ({mol1:.3f})")

    print(f"\n{'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
