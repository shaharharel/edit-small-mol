import json, os
d = os.path.dirname(os.path.abspath(__file__))
cands = {}
for f in ("group1.json", "group2.json"):
    cands.update(json.load(open(os.path.join(d, f))))
out = {
 "candidates": {k: cands[k] for k in sorted(cands)},
 "panel_verdict": {
   "best_candidates": [8, 9],
   "any_publishable": True,
   "panel_why": ("All ten keep the acrylamide intact and none is alpha-methylated, so the warhead is never the "
     "problem; the failures are attribution and bookkeeping. Deriving the mapping from the input SMILES "
     "(C18H20N4O2) settles almost every call: the N-isopropyl owns exactly three rows - C13/MET414.SD 3.63, "
     "C14/LYS369.CG 3.67, C15/MET414.SD 3.53. Every other distance these candidates credit to 'the isopropyl' "
     "belongs to something the edit never touches: LEU468.CD1 3.56/3.44 are imidazole ring carbons C11/C12 "
     "(cand 02, 05, 07, 10), ALA367.CB 3.33 is the ring nitrogen N3 that stays put (cand 03, 04), and "
     "ALA417.O 3.27 / ALA417.N 2.86 are an isoindoline aryl carbon C8 and the ring N4 (cand 09, 06). Only "
     "cand 01, 06 and 08 attribute correctly, and 7 of 10 get ownership wrong - the dominant failure mode. "
     "Claim-molecule mismatch is the second: cand 01 promises a hydroxyl and delivers plain N-H, cand 02 "
     "promises a methyl and delivers N-H, cand 03 calls an appended CH2CO2Me a 'methylene insertion', and "
     "cand 06 quietly swaps imidazole for a 2-cyanopyrimidine (MCS 17/24) which smuggles a second "
     "cysteine-reactive electrophile into a covalent seed. Five candidates recite the flagged 1.40 A as the "
     "retained bond length; cand 09 alone names it unreliable, which is the behaviour to reward. "
     "I would take two forward. Cand 08 (4-chlorobenzyl) is the only one that is right on both of the things "
     "the filter could not check, and it is a sensible one-step probe of the MET414/LYS369 lipophilic face - "
     "delete the '1.40 A' and it ships. Cand 09 (cyclobutyl) is the best molecule and the best caveat on the "
     "panel, a classic isopropyl isostere, and needs only its contact attribution rewritten from ALA417 to "
     "MET414.SD/LYS369.CG. Cand 05 is a defensible molecule behind a doubly faulty sentence; cand 01/02/07 "
     "are one sound truncation described three different and mostly wrong ways; cand 03, 06 and 10 I would "
     "not make at all - a hydrolysable ester with an unassigned stereocentre, a dual electrophile, and a "
     "tert-alkylation that does not run.")
 }
}
p = os.path.join(os.path.dirname(d), "review_RA.json")
json.dump(out, open(p, "w"), indent=2)
print("wrote", p)
