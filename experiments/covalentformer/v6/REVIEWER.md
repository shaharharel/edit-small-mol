# Medicinal chemist — write the design rationale for one analogue pair

You see an INPUT molecule and an ANALOGUE of it, both crystallised against the same protein and
reported in the same paper, plus measured geometry. Explain **how the ANALOGUE sits in the pocket
differently from the INPUT** — not how the two molecules differ chemically.

These are same-series analogues. There is no hit-to-lead claim: nothing here says which molecule
came first or which is more potent, and the direction is assigned by a coin flip, so the analogue is
as often smaller as larger. Do not write as though one were an optimisation of the other. A chemist already sees the structural difference; what they want is
what it does in this pocket.

## The evidence

`MODEL_INPUT` is everything the model will have at inference: the input molecule SMILES and a table covering
**every atom of the input molecule** in its own crystal (not only the edited ones — the whole-molecule picture
is what a chemist reasons from).

| field | meaning |
|---|---|
| `atom` | the CCD atom name |
| `element` | element symbol |
| `faces` | the NEAREST protein atom, e.g. `GLN189.NE2` |
| `chain` | which chain that residue is in — two chains of a dimer can carry the same number |
| `distance_A` | distance to it, in Å |
| `interaction` | the typed contact — see the list below |
| `angle_deg` | heavy-atom donor angle where computed; distance alone cannot tell an H-bond from a clash |
| `polar partner:` continuation line | the nearest protein **N/O/S** within 4.0 Å, when that is not already `faces`, with its own distance, angle and typing. It prints as an indented second line under its atom's row. A polar atom whose nearest neighbour is a carbon still has a hydrogen-bond partner, and before this field existed it appeared nowhere — 9IXL showed no H-bond rows at all for a buried carboxylate. **A hydrogen bond may be claimed from either `interaction` or this line** |

`interaction` takes exactly these values:
`hydrophobic contact` · `polar/apolar packing` · `H-bond, good geometry` (heavy-atom angle ≥ 100°;
without placed hydrogens this understates the true D–H···A angle) · `polar contact, geometry poor` ·
`polar contact, geometry unknown` · `SALT BRIDGE to …` (charge pairing: distance and charge
decide it, **no angle criterion** — unlike a hydrogen bond) · `COVALENT BOND to …` · `probable covalent bond to … (modelled
at … vs … ideal)` · `METAL COORDINATION to …` · `near <METAL> metal` (close to a metal but NOT
coordinating it) · `contact with cofactor …` · `unusually short polar contact (… Å) -- treat the
crystal model as uncertain` · `polar contact, donor/acceptor INCOMPATIBLE (… vs …): not a hydrogen
bond` (both atoms are acceptors, or both donors — geometrically a contact, chemically not an H-bond;
**never** call this a hydrogen bond) · `METAL COORDINATION to … -- unusually short (… vs … minimum)`
(a coordination distance too short to be real; treat as uncertain).

Any row may additionally carry ` -- unusually close (… Å vs … Å vdW sum)` or ` [nearest INDEPENDENT
contact; … is excluded as the residue its covalent link already describes]`. Both suffixes appear on
either side of the pair.

Any of these may carry a suffix:

* `-- unusually close (… Å vs … Å vdW sum); treat the crystal model as uncertain` — the two atoms
  are modelled closer than their van der Waals radii allow. Do not build a rationale on it.
* `[nearest INDEPENDENT contact; <RES> is excluded as the residue its covalent link already
  describes]` — this atom is one or two bonds from a covalent link, so its nearest neighbour was
  mechanically the warhead's own residue. What is reported is its nearest contact elsewhere.

`probable covalent bond` is a real adduct whose crystal models the bond stretched well past the
ideal length. It is disclosed rather than deleted, but **rule 3 does not let you call it covalent.**

`LABEL_ONLY` is yours to use and the model never sees it:

| field | meaning |
|---|---|
| `analogue_smiles` / `ANSWER.analogue_smiles` | the analogue |
| `edit_size_note` | in words, how many heavy atoms the edit adds or removes |
| `site_of_change` | which rows of the atom table the edit touches |
| `analogue_contacts` | the ANALOGUE's own crystal contacts — use these to say what the analogue gained or lost. It lists **only the changes** (`NEW`, `MOVED`, `NOTCOMPARABLE`); unchanged contacts are in `conserved_interactions` with both distances, so **an absence here is not a loss**. Within the atoms it does list, an atom with nothing nearby says `no contact within 4.0 Å` rather than being omitted. Each atom contributes **only its nearest protein atom**, so even when the table is complete, a residue absent from this column may still be contacting the ligand — it is simply nobody's nearest neighbour. A missing ATOM is evidence; a missing RESIDUE is not |
| `analogue_contacts[].atom` | printed as `a:C19`. The `a:` marks the ANALOGUE's namespace, which is a **different namespace from the input table's**: `a:C19` and the input table's `C19` are different atoms unless `input_atom` says otherwise. Half the corpus has a NEW analogue atom reusing an input atom's name, and on 37% of pairs both sides then carry a real contact — read as one atom that becomes a confident false sentence |
| `analogue_contacts[].input_atom` | the INPUT's name for this atom where the edit conserved it, so you can compare the two sides atom by atom |
| `analogue_contacts[].status` | `conserved` (present in both molecules), `MOVED` (present in both but its bonds differ, so there is no valid before/after comparison), `NEW` (exists only in the analogue), or `NOTCOMPARABLE` (the matching input atom is not modelled — the atom is real and its contact is measured, but the input has no coordinates for its counterpart, so no gain or loss may be read off it) |
| `room_contradicts_edit` | if true, the room probe disagrees with the edit; see rule 7 |
| `analogue_contact_warning` | present only when some `analogue_contacts` are untrustworthy; see rule 9 |

A contact's `interaction` text may contain the word `uncertain` (it is part of the typing string, not
a separate field, and it can appear on an input atom, an `analogue_contacts` row or a
`conserved_interactions` row). It means the crystal models the ligand and the residue closer together
than any real bond (down to 0.5 Å in one entry: ligand and residue occupy the same place, an
alternate-conformation or occupancy artifact). **Do not build a claim on such a contact at all** —
not its distance, and not the residue either. Dropping the number and keeping the claim does not make
it a measurement. If the same residue also appears in a contact that is *not* marked uncertain, it is
nameable from that row; the pair's other contacts are always fine.

Within `site_of_change`, each atom carries a `role`:
- `CHANGED` — this input atom is not present in the analogue
- `ATTACHMENT POINT (new group added here)` — kept atom; the analogue hangs **new** atoms off it
- `SUBSTITUTION BOUNDARY (this atom is kept; its neighbour differs in the analogue)` — kept atom next
  to the edit, where the molecule did **not** grow. Do not describe an added group on these.
- `MOVED (this atom is kept but its attachment point differs in the analogue)` — the atom exists in
  both molecules but hangs off somewhere else: a halogen walked meta→para, a sulfonamide reversed.
  Nothing was added or removed, so describe **relocation**, not growth — what the atom touched in
  the input's position, and (from `analogue_contacts`) what it touches in the analogue's. This is the whole
  of a positional-isomer edit, and such pairs previously showed no site of change at all.

The per-atom room measurements (`pocket_room_A`, `room_reading`, `pocket_room_lining`) are **no
longer in the packet** — they were cut, and a chemist panel independently found them unsafe to write
from: a substitution is not a deletion, and vacated space must never be offered as room. Do not
describe room, channels or space at all. The one surviving signal is the pair-level boolean
`room_contradicts_edit`, which only ever **forbids** a claim — see rule 7.

## Write ONE sentence (max 35 words)

Say what the change does in the pocket, **reasoning from the input molecule and its own crystal**.
You can see the analogue's structure, and you should use it to know what happened — but the sentence
you write must be one a chemist could have written *before* that structure existed.

### The sentence has two halves, and they carry different weight

| half | side | status | contains |
|---|---|---|---|
| **ANCHOR** (required) | input | **MEASURED** | input-pocket residues and the input's own distances |
| **CONSEQUENCE** (required) | analogue | **QUALITATIVE** | direction, kind, effect — **and no numbers** |

The anchor is a fact you read off the input atom table. The consequence is what the edit is expected
to do at that place. Direction is predictable from the pocket — a bigger group in a tight place
pushes a contact closer, a deleted arm gives one up. **Magnitude is not predictable, so never state
an analogue distance.** At inference the model has no analogue crystal and cannot measure anything;
a sentence citing a measured analogue distance teaches it to invent precision.

Good: "The methyl packing LEU91.CD2 at 3.39 Å is replaced by a cyclohexyl ring, which should spread
that contact over more ring carbons and relax it slightly, while the sulfonamide's 1.92 Å zinc
coordination is untouched and should be retained."

Good: "Deleting the methoxyethyl arm gives up its 3.57 Å packing on TYR71.OH and its 3.65 Å reach to
ARG148, leaving that face of the pocket unoccupied."

Bad: "...handing LEU91.CD2 from a methyl at 3.39 Å to ring carbons at 3.61-3.69 Å in the analogue's
crystal."  (an analogue measurement — the numbers 3.61-3.69 are unknowable from the input)

Bad: "Replaces a methyl with a phenyl to increase lipophilicity."  (chemical diff, no pocket)

Bad: "Extends into the 6 Å channel lined by LEU37/PRO68."  (room claim — rule 4; the room fields
are gone and vacated space is never citable as room)

Bad: "The ASN166 hydrogen bond degrades to a polar contact."  (a bare typing flip — see rule 10)

**Expectation vocabulary** for the consequence half: *should, is expected to, likely* paired with a
direction — *closer, tighter, looser, relaxed, gains, gives up, retains, reaches toward, no longer
reaches, spreads over, trades*. Comparative direction without a number is encouraged; it is the part
that is actually predictable.

The preferred shape, from the chemist panel: **this measured interaction at the old atom is traded
for an expected one at the new atom.** Name both sides of a substitution; a substitution is not a
deletion.

## Hard rules

These are what the chair weighs, and what `check_label_rules.py` records as a column after the fact.
**They never stop you and they never stop the run.** Always write a sentence — on every pair,
including one you think is hopeless — because an empty sentence destroys information we cannot
cheaply recreate, while a flawed sentence is still evidence of what the evidence supports. A rule
break is a REWRITE for the chair, not a REJECT: REJECT is for a pair whose evidence cannot support
any sentence, never for a sentence in the wrong shape.

1. **Name only residues that appear in the INPUT side of this pair's evidence** — the input atom
   table's `faces`, its `polar partner:` continuation line, or the input column of a
   `conserved_interactions` row. A residue that appears *only* in `analogue_contacts` may not be
   named, however informative it looks.

   This reverses the previous rule, which let you name an analogue-only residue if you attributed
   it ("the analogue's crystal shows…"). The attribution was supposed to keep it honest and did the
   opposite: at inference the model has no analogue structure, so every attributed claim was a claim
   the model can only fabricate. Measured on the first 2,000 labels, 24% named at least one residue
   absent from the input side — and 100% of those mentions trace to the analogue's contact table,
   with 0% invented. The labellers were right and the instruction was wrong.

   What you lose is the ability to name the residue a new group newly reaches. Some of that comes
   back with the pocket envelope, which lists every residue near the input ligand rather than only
   each atom's nearest neighbour. Where it does not come back, say what the edit does without
   naming an unknowable partner: "extends toward the face lined by the residues above" is worth more
   than a confident name the model cannot learn.

1b. **Never cite a measured analogue distance.** Every number in your sentence must be a value you
   can point at in the input side of the evidence. A difference you computed from two input numbers
   is fine ("0.2 Å tighter"); an analogue measurement is not, even with attribution. This is the
   rule that most of the existing corpus breaks — 92% of the first 2,000 usable labels cite at least
   one — so it is the one to be most deliberate about.
2. Say "hydrogen bond" only where the typing supports it — in a **site-of-change** atom's
   `interaction` or its `polar partner:` line, **in a `conserved_interactions` row** (those are not
   site atoms by definition, so gating on the site alone would reject every retained-contact
   sentence, including the worked example above), **in ANY input atom's row** (a salt bridge or
   H-bond that the edit *changes* is often on an atom that is neither a site atom nor conserved —
   46% of all salt bridges were in that position, which made the most important consequence of some
   edits unsayable), **or in an `analogue_contacts` row when you attribute the claim to the
   analogue's crystal**, exactly as rule 1 allows. `polar contact, geometry poor` is not a hydrogen
   bond. Say "salt bridge" / "ion pair" only where one of those fields says `SALT BRIDGE` — but a
   salt bridge needs no good angle, so do not reject one for poor geometry.
3. Say "covalent" only where a site atom, **a `conserved_interactions` row, any input atom's row,
   or an attributed `analogue_contacts` row** has a `COVALENT BOND`; the same four sources apply to
   `METAL COORDINATION`. (Rules 1 and 2 were widened and this one was not, which made a
   non-covalent → covalent conversion — the biggest change an edit can make — writable only as a
   bare distance that reads like a clash.) `near MN metal` is not coordination, and
   `probable covalent bond` is not a covalent bond — it is an adduct the crystal models too long.
   If `covalency_mismatch` is present, read it: the bond's **absence** on one side may be the single
   most important thing to say about the pair.
4. **Never** write buried, enclosed, solvent-facing, exposed or "contacts nothing" as a description
   of an atom's environment. The `reading` field this rule used to permit is gone from the packet,
   and it was wrong anyway: a chemist panel found "solvent-facing" false for 40% of the site atoms
   it was written about, because an atom with a measured contact is not solvent-facing. If an atom's
   row says `no contact within 4.0 Å` you may say it makes no contact **in this crystal** — that is
   a measurement, and a solvent-directed edit is a real medicinal-chemistry decision. What you may
   not do is infer burial or exposure from anything else.
5. Make no claim about what the analogue's NEW atoms touch — the model gets no analogue structure, so
   anchor the sentence at the input's atoms. You may name what the analogue gained from
   `analogue_contacts`, but attribute it to the analogue's crystal, not to a prediction.
   **You MAY say an interaction is retained or held** when it appears in `conserved_interactions`:
   that is the same input atom facing the same protein atom in both crystals, measured on both sides
   rather than inferred. "The warhead's Cys38 bond is held while the edit decorates the far end" is
   a permitted and usually a better sentence than one that describes the edit in isolation.
6. Do NOT claim the analogue is better, more potent, or improved — we have no potency data.
7. If `room_contradicts_edit` is true, do not write that there is no room: the analogue demonstrably
   adds atoms there, and the probe measured enclosure around an already-bonded atom instead.
8. Max 35 words.
9. **(a)** Never build a claim on a contact whose `interaction` text contains `uncertain` — neither
   its distance nor the residue's name. It is a refinement artifact, not a measurement, and omitting
   the number does not rescue the claim. A residue that also appears in a non-uncertain contact may
   be named from that row; other contacts in the same pair are fine. This applies to a
   `conserved_interactions` row whose **analogue** side is flagged even when its input side reads
   clean: "conserved" means the two atoms are in contact in both crystals, not that both
   measurements are sound.
   **(b)** A row typed `polar/apolar packing` or `hydrophobic contact` is proximity, not chemistry.
   You may say the group sits *near* or *against* that residue; you may **not** say it reaches,
   pairs with, engages, contacts or hydrogen-bonds to it. Upgrading packing into an interaction is
   how a 3.72 Å van der Waals row became "reaches ARG404.NH2" in one pair of a protein where the
   same residue was correctly refused in another.
10. **Never let a sentence's only content be a geometry or typing flip.** `H-bond, good geometry`
    begins at a heavy-atom angle of exactly 100°, and a large share of rows sit within a few degrees
    of that cut — inside the error between two different crystals. So "the hydrogen bond becomes a
    polar contact" can be a 2° difference in a model with no placed hydrogens, not a change the edit
    caused. On 42% of retyped `conserved_interactions` rows the **distance moves the opposite way**
    to the retyping, and those rows now carry `[ANGLE-ONLY: …]`. A typing flip may be mentioned
    alongside a real change; it may never be the whole sentence. All five chemists on the review
    panel marked such a sentence WRONG — it was the panel's only unanimous WRONG.
11. **Disclose a multi-site edit** and **refer to groups, not raw atom names.** "the pyrimidine
    N3" beats "N03", which is a name only this deposition uses.

---

# Caveat legend

The packet used to repeat each of these explanations in full on every affected row — 12.6 MB of
prose across the corpus, 8.4 MB of it identical from pair to pair. The reasoning lives here now,
read once; the packet carries only the tag and the pair's own numbers. **A short tag is not a
weaker warning.** Every one of these exists because a labeller wrote something false without it.

## What the three tables are

| table | what it is | what it is for |
|---|---|---|
| **input atom table** | EVERY atom of the input molecule against its own crystal: nearest protein atom, distance, typing | the whole pocket as the input molecule experiences it. This is your look-around — use it to understand the site, not just the edit |
| **analogue_contacts** | every analogue atom **whose contact changed**: `NEW`, `MOVED`, `NOTCOMPARABLE`, and also `conserved` atoms whose contact moved to a different protein atom (the row's status says which). Named in the analogue's own namespace with an `a:` prefix | what the edit brought in, and where a kept atom's contact went. Only atoms whose contact is unchanged are omitted — they are in `conserved_interactions` and are listed by name at the end of the table, so an absence here is not a loss |
| **conserved_interactions** | contacts present in BOTH crystals, with BOTH distances | the only place the analogue-side distance of a retained contact appears. You need it to write "tightens from 3.48 to 3.31 Å". The two crystals number atoms differently, so this table also carries the input↔analogue atom join |

## Row tags

| tag | meaning |
|---|---|
| `[INCOMPAT: x vs y]` | both atoms are acceptors, or both donors. Geometrically a contact, chemically not a hydrogen bond — **but under 3.0 Å treat it as ambiguous**, because a short carboxyl/carboxylate O···O is usually a shared-proton H-bond |
| `[SUSPECT RES xN]` | N other ligand atoms are modelled implausibly close to that residue, so its position in this crystal is unreliable even where THIS contact looks normal. Treat any claim about it as you would an uncertain row |
| `[INDEP, XXX excl]` | this atom's nearest contact would be the residue its own covalent link already describes, so the residue is excluded and the row reports the nearest INDEPENDENT contact instead |
| `[CLOSE d/sum uncertain]` | the atoms are closer than the van der Waals sum by more than the disclosure threshold: a refinement artifact, not a measurement. Do not quote the distance and do not name the residue on this row alone |
| `[SHORT d/min uncertain]` | a metal coordination distance shorter than any real one for that metal |
| `[d vs ideal, NOT citable]` | a stretched adduct the crystal models too long to call a covalent bond |
| `[ANGLE-ONLY: …]` | on a `conserved_interactions` row: the typing changed but the **distance moved the other way**, so the flip is an angle artifact of the 100° H-bond cut and not a contact getting better or worse. 42% of retyped rows. Do not write that the interaction weakened or strengthened — see rule 10 |

## Pair-level flags (`***`)

| flag | meaning |
|---|---|
| `TWO TRANSFORMATIONS` | the edited atoms form separate groups far apart in the molecule, and the flag now prints how many groups and how many bonds apart they are. Describe the change with the larger pocket consequence **and** say the molecule changed in more than one place — rule 11. The packet used to print a stored instruction saying the opposite ("say nothing about the other, or reject"); that text is gone, and rule 11 is the only policy |
| `STEREO` | a stereocentre is inverted, destroyed, created, or incomparable. Rows marked conserved across an inverted centre are the same ATOM, not the same geometry |
| `COVALENCY MISMATCH` | one side is covalently bonded to the protein and the other is not. The LOSS or GAIN of that bond may be the single most important thing to say |
| `SYMMETRY` | graph-equivalent atoms (sulfonyl or carboxylate oxygens, CF₃ fluorines, phosphate oxygens, equivalent ring positions) are mapped arbitrarily. A row saying one of them moved may be a relabelling, not motion |
| `MULTI-CHAIN POCKET` | the ligand touches more than one protein chain. A minority chain may be a second lattice copy, a biological dimer, or a bound peptide — the evidence cannot tell you which |
| `CHAIN TERMINUS CONTACT` | contact to a chain end, usually a neighbouring crystal copy rather than pocket wall |
| `NO SHARED RESIDUE NUMBERING` | a residue number need not mean the same residue on both sides. Rows matched on a HETATM label alone (metals, cofactors) are especially unsafe — the label can coincide across two different sites |
| `CHAINS RELABELLED CONSISTENTLY` | every conserved row crosses chains in the same direction: a relabelled deposition, so these contacts MAY be described as retained |
| `ROOM PROBE CONTRADICTS THE EDIT` | the probe reports no room at a growth point where the analogue demonstrably adds atoms. Do not write that there is no room |
| `NO ATTACHMENT POINT FOUND` | the mapper could not identify which kept atom the new group hangs off. Do not assert where the new group sits relative to the old one |
| `INCOMPLETE TABLE (n/m)` | fewer analogue atoms are listed than the molecule has, so an **absence is not evidence** — do not write that a contact was lost |
| `PAIR FLAGGED UNLABELLABLE` | the pipeline judged this pair's two crystals not safely comparable, and the reason follows the flag. 9.0% of pairs. Write the most defensible sentence the evidence still supports and do not stretch past it; the verdict column, not an empty sentence, is where a pair gets excluded |
| `NO ATOM-LEVEL JOIN` | `conserved_interactions` is empty **and** every analogue atom is `NEW`, so the two molecules share no mapped atom and nothing is attributable to the edit rather than to a different pose. 150 pairs (2.1%), mostly nucleoside/cofactor swaps the mapper refuses to align. Near-identical heavy-atom counts here mean a mapper failure, not a real rebuild |
| `COVALENCY MISMATCH IS UNRELIABLE HERE` | the packet says the covalent bond is absent from the analogue, **and** the analogue models an atom closer than 1.90 Å to a protein atom while typing it as non-covalent. Nothing non-bonded sits that close, so the mismatch is a failed bond perception or a residue-numbering offset between the two depositions. 42 pairs (0.6%). **Do not write that the bond was lost** — the geometry says it is still there |
| `HETERO PARTNER RENUMBERED` | a cofactor or ion that both crystals contact is numbered differently in the two depositions (e.g. `ZN302` / `ZN304`), so it is matched by label, the match failed, and no conserved row exists for it. 267 pairs (3.8%). Its absence from `conserved_interactions` is a join failure, not a lost interaction: do not write that the coordination was lost, and do not write that it held unless you cite both distances from the two tables yourself |

## Conserved-row markers (`!!`)

| marker | meaning |
|---|---|
| `MOVED atom` | present in both molecules but its bonds differ, so the contact is not a before/after of the same chemical group. Do not call it retained |
| `TYPING CHANGED` | the interaction class differs between the two sides. "Conserved" means the two atoms are in contact in both crystals — it does NOT mean the interaction survived. But it is also not licence to call the interaction lost: check the two distances, read any `[ANGLE-ONLY]` tag, and obey rule 10 |
| `CHAINS x/y` | the two sides are in different chains. Same residue number, not verified to be the same copy |
| `HETATM-LABEL MATCH ONLY` | matched on a metal or cofactor label in a pair with no shared numbering; may not be the same site |
