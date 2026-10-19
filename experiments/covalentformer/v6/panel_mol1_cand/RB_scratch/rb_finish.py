import json, io
p = "/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/v6/panel_mol1_cand/review_RB.json"
d = json.load(open(p))
d["candidates"].update({
 "cand_06": {
   "molecule_is_sound_analogue": False,
   "claim_matches_molecule": False,
   "ownership_correct": True,
   "caveat_respected": True,
   "would_stand_behind": False,
   "one_line_why": "Described as 'delete isopropyl and add nitrile' but the imidazole is swapped for a pyrimidine (ring sizes [5,5,6] -> [5,6,6], MCS only 17/24, Murcko scaffold changed), and the nitrile sits between two ring nitrogens: a 2-cyanopyrimidine is itself a cysteine-reactive electrophile, so this ships a second warhead on a covalent seed."
 },
 "cand_07": {
   "molecule_is_sound_analogue": True,
   "claim_matches_molecule": True,
   "ownership_correct": False,
   "caveat_respected": True,
   "would_stand_behind": False,
   "one_line_why": "The des-isopropyl NH-imidazole is a legitimate single-step SAR probe and LYS369.CG 3.67 is correctly its contact, but LEU468.CD1 3.44 is C12, an imidazole ring CH that survives the edit untouched, so half the cited justification is credited to the wrong group."
 },
 "cand_08": {
   "molecule_is_sound_analogue": True,
   "claim_matches_molecule": True,
   "ownership_correct": True,
   "caveat_respected": False,
   "would_stand_behind": True,
   "one_line_why": "The only candidate whose cited contacts are both genuinely isopropyl-owned (MET414.SD 3.53 = C15, LYS369.CG 3.67 = C14) and whose molecule does exactly what it says: a one-step 4-chlorobenzyl N-alkylation with the acrylamide untouched; strike the 'the 1.40 A CYS346.SG covalent bond is held' clause to 'the covalent bond is retained' and I would put it to the team."
 },
 "cand_09": {
   "molecule_is_sound_analogue": True,
   "claim_matches_molecule": True,
   "ownership_correct": False,
   "caveat_respected": True,
   "would_stand_behind": False,
   "one_line_why": "Best-behaved prose of the ten (cites the bond then explicitly calls the 1.40 A length unreliable) and cyclobutyl-for-isopropyl is a classic one-step swap, but its single structural claim is wrong: ALA417.O 3.27 is C8, an aromatic carbon of the benzo ring, so the cyclobutyl cannot be reducing a 'methyl footprint' there."
 },
 "cand_10": {
   "molecule_is_sound_analogue": False,
   "claim_matches_molecule": True,
   "ownership_correct": False,
   "caveat_respected": False,
   "would_stand_behind": False,
   "one_line_why": "N-alkylation at a fully substituted carbon (-C(CH3)2CH2OH on imidazole N3) is not a one-step alkylation (tertiary halides give elimination; it needs a de novo ring build), LEU468.CD1 3.44 is ring carbon C12 and not the isopropyl's 'aliphatic edge', and the 1.40 A anchor is again stated as fact."
 },
})
d["panel_verdict"] = {
 "best_candidates": [8],
 "any_publishable": True,
 "panel_why": ("Deriving the mapping myself settled nearly every call. Element-indexed over the input "
   "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1, the N-isopropyl owns exactly three atoms - C13 "
   "(MET414.SD 3.63), C14 (LYS369.CG 3.67), C15 (MET414.SD 3.53). Everything else the prose reaches "
   "for belongs elsewhere: LEU468.CD1 3.56/3.44 are the imidazole ring carbons C11/C12, ALA367.CB 3.33 "
   "is the ring nitrogen N3, ALA417.O 3.27 is benzo carbon C8, ALA417.N 2.86 is ring nitrogen N4, and "
   "LYS424.NZ 3.02 is the warhead carbonyl O1. Seven of ten candidates therefore credit the isopropyl "
   "with a contact it does not own - the models have evidently learned that the isopropyl is the "
   "editable handle and then attach whichever nearby distance reads well. Three sentences also describe "
   "a molecule that was not delivered: 01 and 02 promise hydroxyl and methyl but hand over plain NH "
   "(identical SMILES to 07; three candidates, one compound), 04 calls a 3,3,3-trifluoro-2-methylpropyl "
   "a 'chiral trifluoroethyl', and 06 silently hops the imidazole to a 2-cyanopyrimidine, which also "
   "installs a second cysteine-reactive electrophile on a covalent seed and fails on chemistry alone. "
   "Five candidates assert the flagged 1.40 A length as measured fact; only 09 handles it as a chemist "
   "should, citing the bond and naming the length unreliable. I stand behind 08 alone. I weigh its fault "
   "differently from the ownership failures on purpose: an overstated bond length quotes a number the "
   "table itself supplies and flags, and striking one clause fixes it without touching the design logic, "
   "whereas a misattributed contact is a wrong hypothesis about the pocket that would send the team "
   "chasing SAR in the wrong subsite. On the chemistry, 09 is the molecule I would most want made - "
   "cyclobutyl for isopropyl is the right first probe of that MET414/LYS369 shelf - so my recommendation "
   "to the programme is 08 as written minus the length claim, plus 09 re-proposed against the MET414.SD "
   "3.53 and LYS369.CG 3.67 contacts it actually perturbs. All ten keep the acrylamide intact and none "
   "is alpha-methylated, so the warhead itself was never the failure mode here.")
}
json.dump(d, open(p,"w"), indent=2, ensure_ascii=False)
