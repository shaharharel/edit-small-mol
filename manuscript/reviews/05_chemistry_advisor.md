# 05 — Structural chemistry advisor

Working memo. Numbers from EVIDENCE.md; pipeline constants cited to code as *methods settings*,
not results (any that enter the paper need a register line).

## 0. One correction to the brief

The table is not `(residue, ligand atom, distance)`. The whitelist actually shown to the model
(`evidence_v4.py:1183-1187`, enforced by `qa_evidence_v7.py` `MODEL_FIELDS`) is 16 fields per
ligand heavy atom: nearest residue atom + distance + **typed** interaction + H-bond angle, a
second *polar* partner/distance/type/angle, `waters_within_3.5A`, `nearest_water_A`, `lining`
(apolar/polar/net charge), `open_fraction`, and a burial `reading`. Water, solvent exposure and
interaction typing are partly expressed. Arguing against a strawman table will get us caught.

## 1. Is the per-atom contact table adequate?

For the *sentence*, nearly. For the *molecule*, no — and the paper should say which.

Cannot be expressed: **growth vectors** (direction and reach of open space from an atom);
**buried volume / slot shape** (whether that space takes a methyl or a biaryl); side-chain and
ligand **flexibility**; water as a *displaceable* entity rather than a count; **protonation and
tautomer state**, hence whether a polar partner donates or accepts; **torsional strain** along the
vector you would grow; and second-shell / paralogue-difference effects, which is most of real
selectivity design.

The decisive omission is (a)+(b), and the reason is sharp: `pocket_room_A`, `room_reading`,
`slot_shape` (growth-cone extents, `evidence_v4.py:460-487`) and `_exit` **are computed** and then
deliberately excluded from the whitelist, because they are computed only at the edit site and
therefore leaked it — `_exit` alone identified the edit site in 93% of rows. The model is handed a
*description of the current pose* and nothing about *where a group could go*. A sentence can be
grounded in that; a molecule cannot be derived from it.

This is the mechanism behind the programme's most honest result: grounded 1.000 vs 0.432 and
selectivity 0.388 vs 0.121, with TC→reference 0.679 vs 0.680 — and behind the wrong pocket costing
0.013 Tanimoto. vdW distance annotations changed nothing for the same reason: more precision on an
axis the molecule never used, no new axis. **Fix, before TBD-2:** compute room/slot-shape at *every*
heavy atom, not only the edit site. Computed everywhere it carries no site information, so it is
leakage-safe, and it is the one field that could make the pocket touch the molecule.

## 2. One diffusion sample — what may be claimed

Claimable: **which residues line the ligand**. Residue composition is far more robust to sampling
than any single distance.

Not claimable: any distance to 0.01 Å; any interaction *type* that depends jointly on distance and
angle (H-bond vs packing); and **any covalent geometry at all** — an unconstrained cofold forms no
bonds, so a warhead lands wherever the sampler puts it.

Sentence the paper must contain:

> Where no crystal structure exists the complex is a single-sample Boltz cofold. Its contact table
> is a hypothesis about which residues line the ligand, not a measurement of distances; no distance
> taken from a cofolded table is quoted as evidence in this paper, and covalent geometry is never
> read from a cofold.

Minimal control — TBD-6 as written ("contact lists stable") is too loose. Re-run a random subset at
≥5 diffusion samples and report three numbers: (i) Jaccard of the residue set per complex across
samples; (ii) per-contact distance spread, median and 90th percentile |Δd|; (iii) the fraction of
interaction *type* labels that flip. Gate on (i) and (iii). Additionally calibrate crystal-vs-cofold
on the 27 complexes already held — the only external check available.

## 3. The 2.7–3.8 Å mislabelling

The structure layer already knows. `IDEAL_BOND['C-S']=1.81`, `COV_TOL=0.35`, `COV_SOFT=0.90`
(`evidence_v4.py:75-78`): ≤2.16 Å is typed `COVALENT BOND`, 2.16–2.71 Å is typed *probable* and
disclosed, and anything beyond 2.71 Å is non-bonded by the pipeline's own rule. The 26–40% band sits
**entirely above** even the soft band. The failure is that the text layer never reads the typing.
7JQ2 VHM–CYS145 at 1.81 Å is exactly the C–S ideal, which makes it the right calibration figure.

Methods sentence:

> A covalent bond is asserted only where the contact table types the row `COVALENT BOND`, requiring
> the modelled heavy-atom separation to lie within 0.35 Å of the ideal length for that element pair
> (1.81 Å for C–S). Separations of 2.16–2.71 Å are typed and disclosed as *probable* and are never
> cited as a bond; any contact beyond 2.71 Å is a non-bonded contact however the sentence describes
> it. Cofolded complexes are not covalently constrained and carry no covalent typing.

Gate for every structural clause: for each clause matching `covalent|bond|anchor|warhead|adduct`,
require a row in **that pair's own** table whose `interaction` starts with `COVALENT BOND` and whose
residue matches the cited residue; else the clause fails. Report the pass rate as a named metric.
`reward_verifiers.py:597` has the hook, but it is folded into `interaction_type`, which is vacuously
1.0 when the sentence cites no distance — so the gate must run on claims **regardless of distance
literals**, since distance-free clauses currently go unscored.

## 4. The groundedness ladder (report all rungs, headline none below L4)

| rung | check | establishes | cannot establish |
|---|---|---|---|
| L0 | cited residue exists in that pair's table (current, 92.5% / grounded 1.000) | the sentence is about this pocket | contact with the edited atom; type; number |
| L1 | (residue, distance) co-occur in **one row** | the number is transcribed, not invented | direction, type, relevance. Denominator is model-controlled — silence scores 1.0 unless coverage-floored |
| L2 | cited interaction type matches the row's typing, honouring the "not a hydrogen bond" negation | geometric compatibility at the structure's resolution | whether that interaction is the one the edit changes |
| L3 | covalent typing gate (§3) | a bond is a bond | nothing about reactivity or kinetics |
| L4 | **site adjacency** — cited residue contacts ≥1 atom of the MCS changed set | the sentence is about *the edit*, not the pocket in general | direction. Stereo-only pairs are unscorable (site_of_change is stereo-blind) and must be reported as unscored, not failed |
| L5 | **directional ownership** — `gives_up` claims map to changed atoms, `retains` to kept atoms | the causal direction matches the edit | that the new group makes the interaction |
| L6 | **atom ownership** — the row's atom *name* maps through the CCD graph to a specific input index, and that index is in the changed set (this is the 2/6 check) | the strongest claim available without new structure | that the analogue realises the interaction — that needs the analogue's own complex, which inference does not have |

Reporting rule: each rung with its **measurable fraction**. The CCD mapping fails on some pairs (no
entry / ambiguous / match failed). The panel's 6/6 against the rigorous 2/6 is partly a coverage
gap, so a bare rate at L6 is not interpretable — report (pass rate, measurable fraction) as a pair.

## 5. Figures that must carry structural evidence

**Fig A — mechanism, mandatory.** 7JQ2 Mpro/VHM in three panels: (i) the pocket with the
CYS145 SG–warhead bond drawn *as a bond*, length labelled; (ii) the same complex as the literal
table rows the model receives, covalent row highlighted; (iii) the same table with the 2.7–3.8 Å
non-bonded rows highlighted differently, captioned with the three thresholds. The reader must *see*
that the bond row and the "anchor" rows are different rows — drawn from the shipped table.

**Fig B — the honest pocket result.** Paired TC→reference, pocket vs blind (0.679/0.680) on one axis
and grounded/selectivity (1.000/0.432, 0.388/0.121) on the other, so the asymmetry is visible rather
than argued, with the 0.013 wrong-pocket cost marked.

**Fig C — ZAP70 Cys346.** Series drawn with the warhead and Cys346 labelled, covalent members
marked, measured vs predicted pIC50, and the honest 0.723 baseline printed beside 0.554 / 0.489.
Never 0.469.

**Fig D — cofold trust.** The §2 control: per-complex residue-set Jaccard and per-contact |Δd| over
5 samples, with the 27 crystal-held complexes as calibration.

Checkability rules for all four: every distance printed; every residue numbered; bonded and
non-bonded drawn with *different line styles*; each panel captioned with its provenance (crystal
PDB, or cofold with sample count). No cartoon pockets; no surface render without a scale bar.

## 6. Claims I will not sign, and the versions I will

1. **"The model reasons about the binding pocket."** → *"Pocket conditioning determines what the
   rationale may cite (grounded 1.000 vs 0.432; selectivity 0.388 vs 0.121) and does not measurably
   change the molecule proposed (TC→reference 0.679 vs 0.680); at cohort scale the wrong pocket
   costs 0.013 Tanimoto."*
2. **"Structurally grounded rationales"** as a headline. → *"Rationales name residues present in
   that complex's own contact table (L0); correct interaction type, contact with the edited atom,
   and atom ownership hold on smaller fractions, reported with their denominators."*
3. **Any clause calling a 2.7–3.8 Å contact a covalent anchor.** → *"the warhead carbon lies 3.1 Å
   from CYS346 SG in the modelled complex, consistent with a pre-reaction pose; no covalent bond is
   modelled."*
4. **"local geometry derived from crystal structures or predicted complexes"** (concept note) —
   implies equivalence. → *"...from crystal structures where available, otherwise from single-sample
   cofolds, whose distances are not quoted."*
5. **"Identifies unoccupied subpockets" / "grows into the pocket."** → *"The representation states
   what each ligand atom currently touches; it does not state where a group could be grown. A
   leakage-safe growth field is the first item of future work."*
6. **ZAP70 numbers framed as pocket-aware generation.** → *"0.489 MAE / 0.792 Spearman from Vina
   scores on 4K2R plus pretraining, which are additive, against an honest 0.723 baseline; at 280
   molecules every more complex architecture hurt."*
