import json, io
p = "/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/v6/panel_mol1_cand/review_RC.json"
d = json.load(open(p))
d["candidates"].update({
 "cand_06": {
  "molecule_is_sound_analogue": False,
  "claim_matches_molecule": False,
  "ownership_correct": True,
  "caveat_respected": True,
  "would_stand_behind": False,
  "one_line_why": "Undeclared scaffold hop: the imidazole is gone and replaced by a 2-cyanopyrimidine (ring sizes 5/6/6, C17H13N5O2), which the sentence describes only as deleting isopropyl and adding a nitrile, and a 2-cyanopyrimidine is itself a thiol trap -- a second electrophile is the last thing a Cys346 acrylamide programme needs."
 },
 "cand_07": {
  "molecule_is_sound_analogue": True,
  "claim_matches_molecule": True,
  "ownership_correct": False,
  "caveat_respected": True,
  "would_stand_behind": False,
  "one_line_why": "The removal is honestly described and LYS369.CG 3.67 is genuinely isopropyl, but LEU468.CD1 3.44 is ring carbon C12 and is retained, and an N-H imidazole (the same molecule as 01 and 02) is a des-alkyl SAR control that surrenders all three MET414/LYS369 contacts for nothing, not a forward proposal."
 },
 "cand_08": {
  "molecule_is_sound_analogue": True,
  "claim_matches_molecule": True,
  "ownership_correct": True,
  "caveat_respected": False,
  "would_stand_behind": True,
  "one_line_why": "The only candidate whose contact reasoning is wholly correct -- MET414.SD 3.53 (C15) and LYS369.CG 3.67 (C14) are both isopropyl-owned -- and a one-step N-benzylation with 4-chlorobenzyl bromide into that lipophilic face is a real probe; its sole fault is quoting the flagged 1.40 A as a held bond length, a write-up fix."
 },
 "cand_09": {
  "molecule_is_sound_analogue": True,
  "claim_matches_molecule": True,
  "ownership_correct": False,
  "caveat_respected": True,
  "would_stand_behind": True,
  "one_line_why": "Best molecule on the sheet -- isopropyl to cyclobutyl is a textbook same-vector bioisostere, one step, water-stable, acrylamide untouched -- and the only candidate that explicitly calls the 1.40 A unreliable; it misattributes ALA417.O 3.27, which is owned by benzo CH C8, so the rationale needs rewriting but the compound stands."
 },
 "cand_10": {
  "molecule_is_sound_analogue": False,
  "claim_matches_molecule": True,
  "ownership_correct": False,
  "caveat_respected": False,
  "would_stand_behind": False,
  "one_line_why": "A tert-alkyl carbon bonded straight to the imidazole nitrogen is not a one-step N-alkylation (tertiary halides and Mitsunobu both fail; it needs ring construction), and it both credits the isopropyl with LEU468.CD1 3.44 (ring carbon C12) and asserts the 1.40 A length."
 },
})
d["panel_verdict"] = {
 "best_candidates": [8, 9],
 "any_publishable": True,
 "panel_why": ("Atom mapping derived from the input SMILES settles almost every call: the N-isopropyl owns exactly three table rows "
  "(C13 MET414.SD 3.63, C14 LYS369.CG 3.67, C15 MET414.SD 3.53) and nothing else. LEU468.CD1 3.56/3.44 belong to imidazole ring "
  "carbons C11/C12, ALA367.CB 3.33 to the ring nitrogen N3 that survives every N-substituent swap, ALA417.O 3.27 to benzo CH C8, "
  "and ALA417.N 2.86 to ring nitrogen N4. On that basis 7 of 10 credit the isopropyl with a contact it does not own, and the "
  "LEU468/ALA367 pair is the models' favourite fabrication. Four of ten (02, 05, 08, 10) recite the 1.40 A as an established bond "
  "length despite the flag; only 09 states it is unreliable. Claim-molecule mismatches are severe: 01 promises a hydroxyl and "
  "delivers hydrogen, 02 promises a methyl and delivers hydrogen, 04 calls an (S)-3,3,3-trifluoro-2-methylpropyl group "
  "'trifluoroethyl', 03 calls a pendant methyl acetate a methylene insertion, and 06 silently swaps the imidazole for a "
  "2-cyanopyrimidine. Three of the ten (01, 02, 07) are the identical des-isopropyl NH-imidazole, so the effective diversity is "
  "eight molecules, not ten. Credit where due: every candidate keeps an unsubstituted acrylamide, none is alpha-methylated to a "
  "methacrylamide, and none breaks the warhead. I would take 09 (cyclobutyl) and 08 (4-chlorobenzyl) to a team: both are "
  "one-step, water-stable, vector-preserving edits at the one position the pose says is tolerant, 08 is the only one with fully "
  "correct contact ownership and 09 the only one that respects the length caveat. Neither rationale is usable as written -- 08 "
  "needs the 1.40 A struck, 09 needs ALA417 replaced with the MET414/LYS369 contacts it actually perturbs -- so what is "
  "publishable here is the two structures, not the two sentences.")
}
json.dump(d, open(p, "w"), indent=2)
print("written")
