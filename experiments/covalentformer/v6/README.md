# CovalentFormer v6 — pocket-grounded hit→lead pairs from the PDB

A matched-pair corpus mined from crystal structures, where both members of a pair were solved
against the same protein, so every pair carries a real pocket. Not covalent-only: covalent bonds
are one typed interaction among H-bonds, metal coordination and cofactor contacts.

## Provenance of this directory

The pipeline was written on 2026-10-10 into a session scratchpad under `/private/tmp`, and the
machine rebooted before it was committed. macOS clears `/private/tmp` on boot, so every script was
lost while the generated data survived.

The code here was **recovered from the Claude session transcripts** on 2026-10-11 by replaying the
logged `Write`/`Edit` calls and bash heredocs in timestamp order
(`evidence_v3.py` additionally needed a full `cat` dump of the file found in a reviewer subagent's
transcript, plus the 7 `Edit` calls that followed it).

Recovery was **verified, not assumed**: rebuilding 300 stored records with the recovered
`evidence_v3.py` reproduces `MODEL_INPUT`, `LABEL_ONLY.site_of_change` and
`LABEL_ONLY.lead_contacts` **byte-identically, 300/300**. The recovered builder is therefore the
exact code that produced the corpus.

Lesson worth keeping: pipeline code does not belong in a session scratchpad. See the repo
`CLAUDE.md` source-policy section.

## The record format

Every row is one hit→lead pair with a strict train/inference firewall.

| key | contents |
|---|---|
| `pair` | `"<hit pdb> <hit ligand code>-><lead ligand code>"` |
| `target` | UniProt accession |
| `MODEL_INPUT` | `hit_smiles` + one row per **ligand atom** — everything inference will have |
| `ANSWER` | `analogue_smiles` — the lead |
| `tanimoto`, `property_deltas`, `edit_type`, `QA` | pair metadata |
| `LABEL_ONLY` | the **lead's own crystal**. For writing labels only; never shown to the model |

`MODEL_INPUT` is deliberately **frame-free** — no coordinates, no reaction frame. Geometry is
expressed as measured per-atom relationships, which removes a whole class of frame-convention bugs:

| field | meaning |
|---|---|
| `atom`, `element` | CCD atom name and element |
| `faces` | the nearest protein atom, e.g. `LEU137.CD1`, `ZN301.ZN` |
| `distance_A` | distance to it |
| `interaction` | typed contact (see below) |
| `angle_deg` | heavy-atom donor angle where computable — distance alone cannot tell an H-bond from a clash |
| `open_fraction` | fraction of 64 probe directions blocked at 3.5 Å |
| `reading` | `buried` (<0.2) / `partly enclosed` (0.2–0.45) / `solvent-facing` (≥0.45) |

`interaction` vocabulary: `hydrophobic contact`, `polar/apolar packing`, `H-bond, good geometry`
(heavy-atom angle ≥100°), `polar contact, geometry poor`, `polar contact, geometry unknown`,
`COVALENT BOND to …` (with a "geometry soft" note where the deposited bond is long),
`METAL COORDINATION to …`, `near <METAL> metal` (close but **not** coordinating),
`contact with cofactor …`, and
`unusually short polar contact (… Å) -- treat the crystal model as uncertain`.

`LABEL_ONLY` adds `lead_pdb`, `lead_contacts`, `site_of_change` (per-atom `role`, `pocket_room_A`,
`room_reading`, `pocket_room_lining`), `heavy_atoms_added`, `edit_size_note`,
`numbering_consistent`, `numbering_confidence`, `residue_numbering_overlap`,
`room_contradicts_edit`, and — added during recovery — an `uncertain` flag on individual
`lead_contacts` plus a `lead_contact_warning` summary.

`site_of_change.role` is one of `CHANGED`, `ATTACHMENT POINT (new group added here)`, or
`SUBSTITUTION BOUNDARY (this atom is kept; its neighbour differs in the lead)`.

Note on `pocket_room_A`: `8.6` is the measurement **ceiling** ("at least 8.6"), and `0.0` is
ambiguous — read `room_reading` alongside it, because `0.0` means either `blocked by protein` or
`not in a pocket - open to solvent`, which are opposite facts.

## Pipeline order

```
fetch_local.sh                  43,024 ligand-bearing PDB entries -> data/pdb_mirror/cif/   (~12 GB)
build_pairs.py / _fast.py       mine same-protein pairs                     -> v6_pairs.jsonl
filter_stack.py                 pocket >= 15 residues, then Tanimoto >= 0.50
                                  -> v6_pairs_clean.jsonl -> v6_pairs_tc50.jsonl
qa_pairs.py                     pair-level QA
selftest_evidence.py            9 builder self-tests -- RUN BEFORE GENERATING
gen_ev4.py  (uses evidence_v3)  build the evidence; drops edits > 15 heavy atoms
                                  -> v6_evidence_v4.jsonl ... -> v6_evidence_final.jsonl
fix_numbering.py                recover the hit/lead author-numbering offset
                                  -> v6_evidence_aligned.jsonl
dedup_edits.py                  one row per unique chemical edit; redundant crystals to a sidecar
                                  -> v6_labelset.jsonl + v6_evidence_redundant.jsonl
cohort_and_sample.py            8 label-readiness gates     -> v6_labelready.jsonl
annotate_lead_uncertainty.py    flag sub-bond-length lead contacts
                                  -> v6_labelready_annot.jsonl
qa_labelset.py                  corpus QA (blockers / warnings)
render_sample.py                render a stratified audit sample as a labeller sees it
gate.py                         mechanical rule check on a written sentence
```

Self-tests (`selftest_evidence.py`, 9/9 passing) exist because five of one round's seven defects
were the same mistake: *an edit was applied to a string that did not match the file, declared
fixed, and left*. A patch that does not take now fails a test instead of passing a glance.

## Corpus accounting

| stage | rows | note |
|---|---|---|
| unique pairs mined | 54,309 | |
| pocket ≥ 15 residues | 27,716 | not pocket-bound otherwise |
| **Tanimoto ≥ 0.50** | 14,393 | the hit-to-lead similarity floor |
| evidence built, edit ≤ 15 heavy atoms | 13,168 rows | a ±21-atom dimerisation is not one design decision |
| **unique chemical edits** | **9,474** | dedup by `(hit_smiles, analogue_smiles)`; 3,694 redundant crystals kept in `v6_evidence_redundant.jsonl` and pointered via `other_crystals` |
| **label-ready** | **7,478** | 79% of the unique edits |
| final cohort (zero QA blockers) | **7,460** | 728 targets, 2,933 crystals |

The dedup matters: one phosphatidylcholine acyl-chain edit recurs **154 times** in membrane-protein
structures. Paying a panel for 154 rationales on one lipid edit is wasted effort and tilts the
corpus away from hit-to-lead.

Why 9,474 → 7,478 (a row can fail several gates):

| gate | rows | sole reason |
|---|---|---|
| numbering inconsistent → lead contacts not citable | 1,017 | 861 |
| site covers >45% of the molecule (MCS failed to localise) | 510 | 493 |
| room probe contradicts the edit | 304 | 187 |
| edit sits in solvent (no residue within 4.5 Å) | 231 | 123 |
| no `site_of_change` | 153 | 125 |
| no `lead_contacts` | 18 | 0 |

**Open item.** The 861 rows whose *only* failure is numbering are not bad chemistry — they are a
bookkeeping mismatch between two depositors. `fix_numbering.py` recovers a constant offset by
mode-voting over residue-type matches at `conf ≥ 0.5` (613 pairs rescued, real shifts +231, +236,
−5), but the principled fix is a SIFTS → UniProt canonical mapping, which was skipped for time.
That is ~11% of the corpus recoverable without lowering any standard.

## Defects found and fixed (keep this list; several recurred)

| defect | scale | how it was caught |
|---|---|---|
| `_exit`/`room_reading` defined only for site atoms — their presence encoded the answer | 93% of rows | a leak test; closed by whitelisting `MODEL_INPUT` fields, then confirmed by classifier test (every single-feature AUC at chance, base rate 0.102) |
| single **connected** MCS mis-assigned the edit site | `8ARY`: conserved chlorophenyl marked CHANGED, the ring that actually changed unmarked | a chemist reading the rendered pair |
| attachment points computed for pure additions only, but 53% of pairs change **and** add | all such pairs had no anchor | the MCS fix exposed it |
| `ATTACHMENT POINT` claimed on same-size substitutions | 2,171 roles | reading a rendered pair |
| carbon labelled `METAL COORDINATION` (distance-only test, no element guard) | 108 rows / 89 pairs | audit sample, now test T8 |
| flat 2.2 Å clash filter deleted **real covalent bonds** | 737 contacts / 693 pairs | reviewer: *"the lead's 1.78 Å carbon-to-SG against the hit's 2.74 Å is unambiguous"* |
| metals and cofactors absent from the pocket entirely | e.g. a carbonic-anhydrase sulfonamide with no zinc — the dominant interaction invisible | reviewer |
| clash filter applied to the hit side only | 1,217 pairs with impossible lead distances | corpus QA |
| hit/lead author-numbering mismatch (one pair off by **+504**) | 22% of pairs | reviewer |
| ray-casting against every protein atom (41M distances/pair) | 70× slowdown | profiling; restrict to protein within 15 Å (rays reach 9 Å) |
| `qa_labelset.py` stale vs the data model: unconditional 2.2 Å floor, and a `^[A-Z]{3}\d+` residue regex | 1,413 + 806 false blockers — every metal (`ZN301.ZN`) read as "malformed" | found during recovery, 2026-10-11 |
| `lead_contacts` had no short-contact disclosure | 157 contacts / 130 pairs below any bond length, down to 0.52 Å (ligand and residue modelled in the same place) | found during recovery, 2026-10-11 |

## Known limits — not bugs, state them in the paper

- **The evidence supports descriptive rationales, not explanatory ones.** Reviewer, verbatim:
  *"None of my sentences can explain **why** the pocket permits the change, only **that** it does,
  because the evidence has no strain, desolvation, waters, or conformational information."*
- **Lead atom names do not correspond to hit atom names**, so a gain or loss cannot be pinned to
  the specific atoms that changed. Every "the new group now touches X" is inference, not
  measurement. Fixable by MCS-mapping the lead's atoms the way the hit's are.
- `H-bond, good geometry` uses a heavy-atom angle with no placed hydrogens, so it **understates**
  the true D–H···A angle.

## Labelling protocol

Three independent reviewer personas (`potency`, `dmpk`, `covalent` in the v5 round) each write one
sentence from the same evidence, then a **chair** adopts one, rewrites, or rejects the row. The
chair sees the **measured geometry first and the proposals second**, so it is not anchored by the
panel. `REVIEWER.md` and `CHAIR.md` hold the briefs; `gate.py` checks the result mechanically.

Two findings from the v5 labelling round that this round must not repeat:
1. **Chair every row.** v5's v3 labels convened the chair only where reviewer *categories*
   disagreed — but category agreement is not sentence agreement: **all 1,192 "unanimous" rows had
   three different sentences, not one pair matched.** Rows that skipped the chair were not rows
   where labellers agreed.
2. **Do not pick a sentence by rule.** The specificity score discarded real observations — on one
   row it dropped the only sentence noting the growth vector was *sealed*, because a rival named
   more residues.

## Pending improvements (found 2026-10-11, not yet applied to the corpus)

Each needs a full regeneration (~7 min + QA), so they are recorded rather than applied, to keep the
recovered `evidence_v3.py` byte-identical to the code that produced the shipped data.

1. **`faces` should carry the chain.** The parser reads `auth_asym_id` into `ch` (line 134) but
   `faces` is built without it (line 338), so `LEU137.CD1` does not say which chain. Measured on
   250 rebuilt pairs: **11.2% of pockets span more than one protein chain**, but **0% have the same
   residue+number on two chains within one pocket** — so a label naming `LEU137` was never
   ambiguous *within* a pair. It matters for the **hit↔lead comparison**: in `6E7R` (a
   hetero-tetramer) GLN110 sits on chains B/D and THR110 on chains A/C, so a chainless comparison
   reads a two-chain pocket as a point mutation. One-line fix:
   `'faces':'%s%s[%s].%s'%(P[j]['res'],P[j]['rn'],P[j]['ch'],P[j]['name'])` and the same for `P2`;
   the `^([A-Z]{1,3})(\d+)` prefix that `gate.py` and `fix_numbering.py` parse is preserved.
2. **Residue-identity validation belongs in `fix_numbering.py`.** It accepts an offset when the mode
   explains a majority of lead contacts, but never checks that residue TYPES agree at that offset.
   255 pairs (3.4%) carry `numbering_consistent: True` while a shared residue number is a different
   amino acid on each side — `6E7R` GLN110→THR110 at confidence 0.97, `6DH6` VAL50→ILE50 (a real
   construct difference, both chains), `28SE` SER144→ARG144. `check_residue_identity.py` annotates
   these for now; the check should move upstream.
3. **The short-contact disclosure is under-applied.** `unusually short polar contact … uncertain`
   only fires for polar–polar pairs. **3,521 non-bonded contacts sit >0.3 Å inside the van der Waals
   sum typed as ordinary `polar/apolar packing` (2,880) or `hydrophobic contact` (641)**, while only
   314 atoms carry the `uncertain` type. Extend the disclosure to carbon-involving contacts, or
   document the threshold. (Sub-vdW distances on `H-bond, good geometry` atoms are expected and not
   part of this count.)
4. **H-bond typing is conservative and rule 2 compounds it.** The heavy-atom angle is computed
   without placed hydrogens, so real salt bridges type as `polar contact, geometry poor` — e.g. a
   buried 2.70 Å carboxylate–ammonium contact to `LYS12.NZ`, arguably the dominant interaction in
   that complex. Hard rule 2 then forbids the labeller from calling it a hydrogen bond. Either
   relax the angle threshold, add a charged-pair interaction type, or restate rule 2.
