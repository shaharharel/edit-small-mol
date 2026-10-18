# CovalentFormer v7 — status

Living file. Update it in place; do not start a second status document.

Last updated: 2026-10-15.

---

## 1. Dataset generations — what each one is

Every generation re-labels pairs drawn from the same PDB mirror, but the **pair selection criteria**
and the **sentence rules** differ. Mixing generations has already been measured as harmful once:
g1+g2 revived analogue-provenance phrasing on 17% of outputs and dropped SMILES validity from 0.95
to 0.861. Treat the generation field as load-bearing.

**A terminology correction that matters for the writeup: these are NOT matched molecular pairs.**
The selection is *same paper/series + Tanimoto ≥ 0.50 + same protein + unique at the pair level*.
There is no fragmentation step and no single-transformation constraint, so one pair can differ at
several sites at once — `7OAL V62->V5W` changes three things. The correct term is **same-series
analogue pair** or **congeneric pair**. A real MMP filter would shrink the corpus and make "the
edit" well defined; it is not what we have.

| gen | pairs | usable | pair selection | sentence rule | where |
|---|---|---|---|---|---|
| **g1** | 2,000 | 1,714 (85.7%) | same paper, TC ≥ 0.50 | v1 *retrospective*: cited distances from the **product** crystal | `label_out/`, stamped `g1_v1_retrospective` |
| **g2** | 2,000 | 1,747 (87.3%) | same paper, TC ≥ 0.50 | v2 **input-anchored**: measured facts input-side only, consequence qualitative | `label_out/`, `g2_v2_input_anchored` |
| **g3** | 3,069 | 2,644 (86.2%) | same paper, TC ≥ 0.50 | v2 input-anchored **+ pocket envelope** in the packet | `label_out/`, `g3_v2_input_anchored_envelope` |
| **g4** | 7,069 | 6,178 (87.4%) | same paper, TC ≥ 0.50 — **the full corpus** | identical briefs to g3, **different labellers and chairs** | `label_out_g4/`, `g4_v2_input_anchored_envelope_pass2` |
| **g5** | **1,602** | TBD | **same paper** (as g1–g4) but **TC ≥ 0.70**, uncapped per target, both structures must carry the target accession and be distinct | v2 input-anchored + envelope (same as g3/g4) | `label_queue_g5/`, pairs in `data/pdb_mirror/g5_pairs.jsonl` — **evidence not yet built** |
| ~~g6~~ | — | — | **DISCARDED.** Covalent-enriched set; the covalent pairs worth having are already inside g4/g5, and CovPDB adds only ~324 new pairs at TC ≥ 0.70 | — | removed |

g1 is a different *sentence* generation and cannot be mixed with g2/g3/g4 at all (3.1% six-rule
compliance against 90%+). g2/g3/g4 share sentence rules and differ only in packet contents and
labeller identity. **g5 differs in PAIR SELECTION, which is a new axis** — same paper like g1–g4,
but a higher Tanimoto floor (0.70 vs 0.50) and no per-target cap — so it keeps its own directory
and its own stamp.

**Parked, deliberately not a generation:** `deferred_crosspaper/` holds 1,695 pairs whose two
ligands share a target but never co-occur in a paper. Construct, crystal form and residue
numbering may all differ, which is why the v7 builder required same-paper in the first place.
Label g5 first, compare its REJECT rate, then decide. It has no generation id so it cannot be
pooled by accident.

### g5 selection funnel, from 2,727 same-paper candidates

| | |
|---|---|
| only one crystal in that paper holds both ligands | 340 |
| no structure in the shared paper carries the target accession (46 of these are entries with no UniProt at all, deliberately not rescued — that check is what stops unrelated proteins being paired) | 299 |
| edit larger than 15 heavy atoms, which `gen_ev5.py` rejects | 56 |
| duplicate canonical SMILES pairs, deduped to one representative | 314 |
| self-pairs (input ≡ analogue) | 5 |
| **surviving, unique, audited** | **1,602** over 269 targets and 204 papers |

Tanimoto mean 0.815 (min 0.700) and mean heavy-atom change 2.2 — a tighter series than g1–g4,
whose mean Tanimoto is 0.653. Zero duplicate pair ids, zero SMILES-pair duplicates, zero
collisions with g1–g4 or the built corpus, and both cif files present for all 1,602.

### CORRECTION: g5 is NOT same-paper (2026-10-16)

I described g5 as same-paper like g1–g4. **That is wrong for 1,135 of the 1,602 pairs (71%),
1,023 of the 1,249 labellable ones.** `build_g5a_pairs.py` reads the paper as
`v.get('doi') or v.get('pm')`, without the NULLS guard that `build_pairs_v7.cite_of()` applies, so
the literal `?` that mmCIF writes for an absent value is truthy and all **8,356 uncited mirror
entries collapse into one pseudo-paper** — the same bug class v6 had, where 4,045 ligands shared a
single "paper". What still holds, and is the part that mattered in v6: both structures must carry
the target's UniProt accession, so **no pair is two different proteins**. What is lost is the
one-series, one-lab, one-assay guarantee.

**Checked against the deposited cif citation blocks, not just `meta.json` (all 1,602 pairs, both
structures each):**

| | pairs |
|---|---|
| same DOI/PubMed on both cif — provably one publication | 467 |
| **no DOI and no PubMed on either cif**, identical deposition title | 514 |
| **no DOI and no PubMed on either cif**, titles differ | 621 |
| different DOI/PubMed — provably two publications | **0** |

So the right statement is **not** "g5 is cross-paper". It is: 1,135 pairs have **no publication at
all** on either side, so same-paper is unverifiable, and 514 of those share an identical deposition
title, which is weak but real evidence of one depositing series. `meta.json` and the cif files
agree exactly — the `?` is the depositions' own, not a parsing artifact. On the 1,249 labellable
pairs: 226 provably one paper, 445 identical title, 578 unknown — **671 with one-series evidence**.

Attribution of all 1,602 g5 pairs against the original build's filter chain:

| why the original build did not have it | n |
|---|---|
| not same-paper at all — the uncited `?` bucket | 1,135 |
| rotatable bonds > 15 (`build_g5_g6.py` never applied the lipid/detergent cut) | 200 |
| identical ignoring stereochemistry (the original rejects these; g5 kept them) | 176 |
| **genuinely missed: the `PER_TARGET_CAP = 500` overflow** | **91** |

The 91 are the only honestly-missed pairs, and they are two targets: P0DTD1 (SARS-CoV-2 Mpro, 57)
and P03372 (estrogen receptor, 34). Both sit **exactly at 500** in `v7_pairs.jsonl`, and they are
the only two targets that reach the cap.

**So lowering the same-paper Tanimoto floor from 0.70 to 0.50 recovers nothing**: the original
build already ran same-paper at TC ≥ 0.50, so that space is exhausted except for the 91 capped
pairs. The 0.70 floor only ever applied to the non-same-paper space, where tightness was standing
in for the provenance guarantee. Real headroom is target-grouped and cross-paper: 3,276 new pairs
at TC ≥ 0.70, 6,526 at 0.60, 12,153 at 0.50.

The g5 README now states the provenance truthfully to the labellers. Whether to split the queue
into a 226-pair same-paper tier and a 1,023-pair same-protein tier is open; nothing is blocked on
it, and `paper` is on every row in `g5_pairs.jsonl` so the split is recoverable after labelling.

### g5 is packaged and ready to label (2026-10-16)

`~/Downloads/covalentformer_labelling_task/g5/` — 13 batches of 100 (the last holds 49),
**1,249 pairs**, the same briefs plus two new rules, the same `tools/`, an empty `label_out/raw/`.
`tools/verify_output.py` reads the g5 manifest and currently reports all 1,249 unlabelled, which is
the correct state. The briefs, START_HERE and README_G5 are mirrored in `labelling_task_g5/` here;
`batches_g5/` is regenerable and not committed, as with g4.

1,249 of the 1,602 pairs carry evidence (78%); the rest drop at 211 no usable complex, 134 no site
of change, 8 edited atom not modelled.

**New rules 1c and 1d**, from the two blind panels: a trailing "toward ⟨residue⟩" clause is allowed
only if that residue contacts a **changed** atom in the input table, and every changed site must be
named (these are same-series analogue pairs, not MMPs — a pair may change three sites).

**Two build steps were nearly missed.** `gen_ev5` is not the whole pipeline: `clean_v7.py` flags
the unlabellable pairs and repairs water clashes, and `apply_completeness.py` adds the INCOMPLETE
table warning. Skipping them left 399 water clashes, 125 unflagged unlabellable pairs and a dead
`analogue_contact_warning`; `qa_final.py` caught all of it. g5 variants are `clean_g5.py` and
`apply_completeness_g5.py`. **The order is gen_ev5 → clean → apply_completeness → build_envelope →
make_full_batches → qa_final + check_packet_fresh.**

Two `qa_final_g5.py` floors were rescaled for a 1,249-pair corpus, each with the reason in the
check's own note: the analogue-side INDEP-row floor 500 → 50 (g5 has 156 rows over 52 pairs,
0.125/pair against v7's 0.325/pair — lower because g5 holds fewer covalent pairs, but demonstrably
live, which is all that check can establish), and the HETATM-only `numbering_caveat` floor 1 → 0
(21 rows in v7's 7,069 pairs, so ~4 expected here and 0 is unremarkable). Everything else passes
unchanged: QA PASSED on 1,249 pairs, every caveat visible, every batch byte-fresh against the live
corpus and renderer.

**`build_envelope.py` bug fixed.** The non-greedy `_RES` regex split `SF4504.FE2` (comp_id `SF4`,
seq 504) as `('SF','4504')` while the envelope held `('SF4','504')`, so the verification refused to
write g5. The comparison now uses the concatenated token, which is exact for digit-suffixed comp
ids. This touches **only** the verification path — `envelope_for()` never reads the regex — and
that was verified, not asserted: rebuilding `v7_envelope.json` gives all 7,069 keys with
**0 value differences** (the files differ only in key order, which `imap_unordered` never fixed).

### g3 vs g4, measured with identical scoring code

| | g3 | g4 |
|---|---|---|
| usable | 86.2% | 87.4% |
| all six hard rules | 91.5% | 91.1% |
| r2, no analogue distance | 98.7% | 100.0% |
| every cited distance exists in the table | 92.4% | 100.0% |
| (residue, distance) co-occur in one row | 55.2% | 78.6% |
| r1c, contact-table residues only | 89.0% | 80.9% |
| words / distances per sentence | 32.9 / 2.65 | 26.0 / 1.44 |

**Read that 1b row carefully: it is partly measuring verbosity.** g4 cites fewer numbers, so it has
fewer chances to mis-pair. Chemists scored per-claim attribution as a near tie (87.0% vs 88.9%),
and in absolute terms g3 delivers more correct anchors per label — 2.65 × 55.2% ≈ 1.46 against
1.44 × 78.6% ≈ 1.13. A rate whose denominator the labeller controls is not a quality metric.

### Blind paired chemist panels, g3 vs g4 (same pair, two labels, side randomised)

| | run 1 (18 pairs, tables TRUNCATED to 22 rows) | run 2 (16 fresh pairs, FULL tables) |
|---|---|---|
| preference | **g3 42, g4 9**, neither 3 | narrow, A 7 / B 9 per reviewer (unblinding pending 3rd) |
| cited numbers verified | not countable — 10–13 of 18 tables truncated | **66 of 66 correct**, 0 wrong atom, 0 analogue leaks, 0 interaction upgrades |
| dominant defect | wrong-atom numbers (`8P73` ASN142.OD1 3.28 packing sold as "3.30 Å amine H-bond"), analogue leak (`6PJF`) | **none numeric** — all failures in the trailing "toward ⟨residue⟩" clause |

Run 1's truncation was a mistake of mine (trimming tables so agents would stop timing out); the
reviewers scored unverifiable anchors as passing, so its attribution rates are optimistic. Run 2
with full tables is the trustworthy read.

**Both panels agree on the fix**, and it should go into the g5/g6 briefs: a trailing
"toward ⟨residue⟩" clause is only acceptable if that residue contacts a **changed** atom in the
input table, and every changed site must be named.

---

## 2. Model table — all arms on the identical 262-pair benchmark

Benchmark = the g2 val split, copied verbatim into every training arm by `build_sft_arms.py`, so
only the training set varies. ⏳ running · 🔜 queued · ❌ needs a chemist panel.

| arm | data | validity | uniq | echo mol ↓ | TC→in | warhead | scaffold retention | grounded | selectivity | sensible | reasoning | sensible_codex | reasoning_codex | invented chem ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ChemDFM-14B **base** | – | 0.118 | 0.871 | 0.161 | 0.793 | 0.750 | 0.104 | 1.000 | 0.366 | 89.6%¤∅ | 33.3%¤∅ | **97.9%** | **14.6%** | 70.8%◊ |
| ether0-24B **base** | – | 0.225 | 0.847 | 0.678 | 0.884 | 1.000 | 0.247 | – | – | 56.2%¤∅ | 0.0%¤∅ | 85.4% | 12.5% | 37.5%◊‡‡ |
| Llama-3.1-8B-Inst **base** | – | 0.519 | 0.750 | 0.059 | 0.737 | 0.857 | 0.761 | 0.996 | 0.405 | 62.5%¤∅ | 12.5%¤∅ | 85.4% | 6.2% | 85.4%◊ |
| SmileyLlama-8B **base** | – | 0.882 | 0.753 | 0.069 | 0.582 | 0.786 | 0.277 | – | – | ❌ | ❌ | – | – | – |
| ChemDFM-14B | g2 | 0.893 | 0.752 | 0.021 | 0.658 | **0.828** | 0.523 | 1.000 | 0.478 | **81.1%** | **25.6%** | TBD | TBD | TBD |
| **ChemDFM-14B** | **g3** | **0.958** | 0.777 | 0.032 | **0.683** | 0.818 | 0.482 | 1.000 | 0.477 | **100%**† | **31.2%**† | **97.0%** | **23.0%** | 63.2%¶ |
| **ChemDFM-14B** | **g4** | **0.966** | 0.751 | 0.004 | 0.644 | 0.767 | 0.462 | 0.981 | 0.447 | 92.4%¤ | 38.9%¤ | TBD | TBD | 51.5%¶ |
| **ChemDFM-14B** | **g3+g4** | **0.973** | 0.757 | 0.000 | 0.658 | 0.818 | 0.455 | 0.969 | 0.469 | 94.9%¤ | 47.8%¤ | 96.0%◊ | 38.1%◊ | **42.0%**◊ |
| ether0-24B schema A | g3 | 0.828 | 0.770 | 0.014 | 0.560 | 0.808 | 0.223 | 0.996 | **0.537** | **91.7%**§ | **39.6%**§ | 89.7% | 15.8% | 76.6%¶ |
| **ether0-24B schema B** | **g3** | 0.863 | **0.792** | 0.009 | **0.682** | 0.615 | 0.496 | 0.997 | 0.465 | 85.4%§ | 35.4%§ | **98.6%** | 14.7% | 60.6%¶ |
| **ether0-24B (no schema)** | **g2** | 0.897 | 0.753 | 0.009 | **0.692** | 0.800 | **0.532** | **1.000** | 0.448 | 89.6%§ | 29.2%§ | 97.4% | 15.4% | 58.2%¶ |
| **ether0-24B schema B** | **g4** | 0.905 | 0.772 | 0.008 | 0.624 | **0.857** | **0.561** | 0.952 | 0.379 | 86.1%¤ | 37.5%¤ | TBD | TBD | 61.6%¶ |
| **ether0-24B schema B** | **g3+g4** | 0.855 | 0.768 | 0.018 | 0.639 | 0.724 | 0.487 | 0.972 | 0.445 | 93.1%¤ | 19.4%¤ | 95.3%◊ | 29.3%◊ | 45.6%◊ |
| Llama-3.1-8B-Inst | g3 | 0.771 | 0.743 | 0.020 | 0.594 | 0.632 | 0.322 | 1.000 | 0.463 | 75.0%§ | 4.2%§ | TBD | TBD | TBD |
| SmileyLlama-8B | g2 | 0.954 | 0.748 | 0.064 | 0.623 | 0.758 | 0.444 | 1.000 | 0.436 | 64.4% | 15.6% | TBD | TBD | TBD |
| SmileyLlama-8B | g3 | 0.908 | 0.761 | 0.055 | 0.588 | 0.700 | 0.332 | 0.996 | 0.481 | 56.2%§ | 4.2%§ | TBD | TBD | TBD |

Reference values are the human labels' own scores — **not** 1.0 everywhere: the real analogues
keep the Murcko scaffold only 38.9% of the time and the warhead 76% of the time, so TC→input and
scaffold retention are **band** metrics, not higher-is-better.

### Column definitions

- **validity** — SMILES extractable and RDKit-parseable, over all rows.
- **uniq** — distinct canonical SMILES / valid SMILES. Low = canned answers.
- **novelty** — valid generations absent from that arm's training completions.
- **TC→in** — Tanimoto (ECFP4/2048) to the **input**. 1.0 = echo. Reference band ≈ 0.68.
- **heavyΔ MAE** — |heavy atoms the model added − heavy atoms the reference added|.
- **warhead** — of inputs carrying a covalent warhead (17 SMARTS), fraction whose output keeps one.
- **scaffold** — Murcko scaffold preserved. Reference 0.389, so this is a band.
- **grounded** — named residues present in that pair's contact table. Reads 1.000 while chemists
  reject the sentence: necessary, not sufficient, and kept to document exactly that.
- **selectivity** — overlap with the reference sentence's residues. The zero-parameter
  "name the nearest table residue" baseline scores **0.5625**; everything below that is worse than
  a distance sort.
- **attrib 1a / 1b** — cited distance exists in the input table / (residue, distance) co-occur in
  one row. Input-side only, so valid whichever analogue was proposed. 1b has a heuristic noise
  floor (reference labels score 0.709) and is denominator-sensitive — see the g3/g4 note above.
- **elem/dir vs REF** — element-count and direction fidelity of the sentence against the
  **crystallographer's** analogue. Penalises a model for proposing a different VALID analogue, so
  it is not a fair primary metric for a generative task.
- **elem/dir vs OWN** — the same two checks against the **model's own generated molecule**:
  does the prose describe the edit the model actually made? Reference-free, and the honest reading.
  The two families must never be quoted interchangeably. Both are listed because they can
  disagree in sign: against the reference ChemDFM g2 looked *worse* than SmileyLlama on direction
  (0.582 vs 0.598), while against its own output it is far better (0.935 vs 0.782).
- **sensible / reasoning** — chemist columns, from the blind self-consistency panel.
- **† the ChemDFM g3 chemist cells come from a SECOND panel and are not comparable to the g2 row
  above them.** Three blind reviewers, 16 pairs, both arms shown side by side with the side
  randomised and full contact tables (`panel_chemdfm_g2_vs_g3/`, unblinded 2026-10-16). Within
  that panel, across 48 judgements per arm:

  | | ChemDFM g2 | ChemDFM g3 |
  |---|---|---|
  | preferred by the reviewer | 13 | **31** (neither 4) |
  | per reviewer | 4 / 4 / 5 | **11 / 10 / 10** |
  | edit identified correctly | 21 | 21 |
  | atom attribution ok | 14 | **28** |
  | invented chemistry (lower better) | 16 | **13** |
  | chemically sensible edit | 39 (81.2%) | **48 (100%)** |
  | would show a chemist | 6 (12.5%) | **15 (31.2%)** |

  **g3 wins, and all three reviewers agree independently** — no reviewer split the other way.
  The g2 "sensible" rate here, 81.2%, reproduces the 81.1% already in the table, which is why
  those two cells can be read together. The "would show a chemist" rate does **not** reproduce
  (12.5% here against the 25.6% in the g2 row), so the reasoning column is a different
  measurement between panels and **only the within-panel g2-vs-g3 contrast is meaningful**.

  All three reviewers independently flagged the same dominant failure, present on BOTH arms:
  **~28–30% of cited distances are attached to the wrong atom**, nearly always contact theft from
  a group the edit retained. Their proposed mechanical fix, which needs no new labels: reject any
  rationale citing a distance whose owning atom is not in the edited substructure, and reject any
  "hydrogen bond" whose partner atom name begins with C.

### ether0-24B schema B, landed 2026-10-16

`TRAIN rc=0` 21:42, `EVAL rc=0` 23:06, run dir `sft_runs/ether0_schema_g3_v2/`. **schema_ok =
1.000 and format_compliance = 1.000**, against 0/262 before — so the unmerged-adapter fix is
confirmed in the artifact, not just in the diagnostic. Response loss 0.575 against the 0.765 the
merged run reported, i.e. back near the training value.

- **TC→input 0.682 lands exactly on the human band (0.680)** — the joint best in the table with
  ChemDFM g3's 0.683, and it gets there with validity 0.863 and novelty 1.000.
- **attrib-1a 0.985 is above the reference ceiling of 0.962**: nearly every distance it cites
  exists in the input table.
- **attrib-1b 0.330 repeats the field-wide failure** — the cited distance and the residue it is
  attached to co-occur in one table row only a third of the time. ChemDFM g2 0.327, g3 0.290,
  SmileyLlama 0.419, reference 0.709. No arm has solved this.
- **selectivity 0.465 is still below the 0.5625 zero-parameter distance-sort baseline**, as for
  every other arm.
- 12.2% of its sentences exceed the 35-word limit the brief imposes (reference: 0%).

Two numbers were nearly wrong here. `eval_sft.py` writes `eval_summary.json` one level deeper
(`<run>/eval/`), so a `sft_runs/*/eval_summary.json` count sees 8 when there are 24. And a first
`eval_axes.py` pass against the default `sft_out/sft_pocket.jsonl` warned **262 of 262 rows have
no ground-truth match** — every reference-dependent column would have been meaningless. The arm
trains on `sft_out_g3_ether0_v2/sft_pocket.jsonl`; the numbers above come from that, via
`big_table.py` and `table_extra.py` so the column definitions match every other row.

elem/dir **vs REF** stay ⏳: `table_extra.py` emits the self/OWN pair only.

### ether0 re-evals complete, 2026-10-16 — and schema training COSTS attribution

`ALL DONE 04:56`. The probe gate passed first (loss 0.3310 unmerged against 0.7647 merged, training
best 0.3982, schema_ok 8/8), which is what licensed the ~5 GPU-hours. Both old ether0 numbers were
invalid — `merge_and_unload()` on an NF4 base discarded the LoRA, so each scored the base model —
and these replace them. Schema A re-eval took 4h25m against schema B's 1h24m; it was slow, not
stuck, and writes nothing until it finishes. I nearly recommended killing it at the 3-hour mark on
exactly that ambiguity. It would have been wrong.

**The headline is ether0-g2's attrib-1b of 0.551.** That is the metric every arm has failed —
whether a cited distance and the residue it is attached to actually co-occur in one table row:

| arm | attrib 1b |
|---|---|
| *reference ceiling* | 0.709 |
| **ether0 g2 (no schema)** | **0.551** |
| SmileyLlama g2 | 0.419 |
| ether0 schema B g3 | 0.330 |
| ChemDFM g2 | 0.327 |
| ChemDFM g3 | 0.290 |
| **ether0 schema A g3** | **0.183** |

Both ether0 schema arms sit at or below the ChemDFM arms while the **unschema'd** g2 arm clears
them all by a wide margin. The think/answer schema buys format compliance and a much better loss
(schema A 0.379, essentially its training best of 0.3982, against g2's 0.630) **and gives up
attribution**. Reasoning-formatted output that cites the wrong atom is exactly the failure three
chemists independently called dominant, so this is not a cosmetic trade.

Other reads from these two rows:
- **ether0 g2 is the strongest arm overall**: TC→input 0.692 (just over the 0.680 human band),
  validity 0.897, grounded 1.000, scaffold 0.532, elem/dir vs OWN 0.875 / 0.858 — best in table.
- **schema A has the highest selectivity in the table at 0.537**, and it is still **below the
  0.5625 zero-parameter distance-sort baseline**. No arm has beaten that baseline yet.
- schema A's scaffold 0.223 against the reference's 0.389 is the worst in the table; g2's 0.532
  overshoots it the other way.
- `schema_ok = 0.000` for ether0 g2 is **not a failure**: the g2 arm predates the schema and never
  emitted think/answer. Its `format_compliance` is 1.000. Do not read the two as one column.

Columns come from `big_table.py` and `table_extra.py` against each arm's OWN training file
(`sft_out_g3_ether0/` and `sft_out_g2/`), so they mean the same thing as every other row.

### The table is complete, 2026-10-16

Every arm now has a row, computed from generations already on disk — **no new GPU work was
needed**. The last four gaps (Llama-Inst base and g3, ether0 base, ChemDFM g3's chemistry columns)
were all sitting in saved `generations.jsonl` files; the Llama-g3 run was `base_pocket_g3` and
ChemDFM-g3's generations were already local in `sft_2k/`. Machine A was stopped once this was
confirmed.

Every cell comes from the same three scripts so the columns mean the same thing in every row:
`big_table.py` (validity → selectivity), `table_extra.py` (scaffold, attrib, elem/dir **vs OWN**)
and `eval_axes.py --data <that arm's own training file>` (elem/dir **vs REF**). ChemDFM g3
reproduced its previously-recorded 0.525 / 0.529 under this path, which is the check that the
mapping is right.

**‡ Llama-Inst base shows attrib 1a and 1b = 1.000 on a denominator of almost nothing.** The base
model cites barely any distances, so "the ones it cites are right" is close to vacuous. Do not
read it as the best attribution in the table; it is the smallest sample.

**Two ChemDFM g3 cells moved** when recomputed from the local generations: selectivity 0.543 →
**0.477** and uniqueness 0.761 → **0.777**. The earlier pair was recorded from a run on the
second machine, which is no longer ours; the values here are from `sft_2k/chemdfm_pocket_g3` under
the same scripts as every other row, so they are the ones to quote.

Arms whose base model emits no usable sentence (SmileyLlama base at 1.0 word, ether0 base) keep
`–` rather than a number in the sentence columns: there is nothing to score, and a 0 would read
as a measured failure instead of an absence.

**What the completed table says.** No arm beats the 0.5625 zero-parameter distance-sort baseline
on selectivity — the best is ether0 schema A at 0.537. Attribution-1b remains the gap that
matters: ether0 g2 at 0.551 is the only arm within reach of the 0.709 reference, and the two
schema-trained ether0 arms (0.330, 0.183) sit below every ChemDFM and SmileyLlama arm. Fine-tuning
reliably fixes validity and heavy-atom delta — Llama base 12.10 and SmileyLlama base 18.82 fall to
~3.2–3.8 everywhere — but it has not yet fixed which atom a number belongs to.

### § The five-arm chemist panel, 2026-10-16 — the sensible/reasoning columns

Five arms had never had a chemist panel, so those cells sat at ❌. They are now filled from ONE
panel rather than five pairwise ones: 16 cases, all five arms' proposals side by side per case,
slot letters randomised per case, key withheld, three independent blind reviewers
(`panel_5arm/`, 240 slot-judgements). One panel because the ChemDFM g2-vs-g3 panel scored
`would_show_a_chemist` at 12.5% for an arm the earlier panel put at 25.6% — panel-to-panel drift
on these columns exceeds most differences between arms, so only same-panel numbers compare.
**§ marks the five arms scored together here; they are comparable to each other and NOT to the †
or un-marked rows above.**

| arm | edit identified | attribution ok | invented chem | **sensible** | **reasoning** | best-in-case |
|---|---|---|---|---|---|---|
| ether0 schema A | 79.2% | 43.8% | 64.6% | **91.7%** | **39.6%** | 16 |
| ether0 schema B | 79.2% | **54.2%** | **54.2%** | 85.4% | 35.4% | **18** |
| ether0 g2 | 56.2% | 52.1% | 62.5% | 89.6% | 29.2% | 8 |
| Llama-Inst g3 | 35.4% | 22.9% | 87.5% | 75.0% | 4.2% | 1 |
| SmileyLlama g3 | 25.0% | 18.8% | 85.4% | 56.2% | 4.2% | 2 |

**The three ether0 arms beat the two 8B arms on every column, and all three reviewers ordered them
the same way** — the would-show gap is 29–40% against 4.2%, and the 8B arms invent chemistry on
85–88% of slots. R1 scored more generously than R2 and R3 in absolute terms (24, 16 and 14
would-show out of 80) but the ranking is identical across all three, which is the part to trust.

**A caution on reading the panel's own best_overall tallies.** R3 reported that "no slot letter
separates itself", and that is true and misleading: slots are re-randomised per case, so a tally
by LETTER is noise by construction. Only the key-resolved, by-ARM aggregation above is meaningful,
and it separates cleanly.

**The chemists and the computable metric disagree on ether0 g2.** It leads the table on
attribution-1b (0.551, the best of any arm) yet the panel puts its edit-identification at 56.2%,
far below the schema arms' 79.2%. Both can hold: 1b asks whether a cited distance sits in one
table row with its residue, while the panel asks whether the rationale describes the edit that
separates the input from *that model's own molecule*. g2 cites honestly about the wrong edit.

All three reviewers independently converged on the same dominant failure, and it is not
fabrication: of roughly 200–226 cited distances each, about 60–70 were wrong, and **almost all
were real table rows attached to the wrong owning atom** — usually a contact belonging to a group
the edit retained. Three phantom residues were found by all three reviewers independently
(TRP154, GLU148-for-LEU148, ASP119). The cleanest discrimination in the set: in 8FPE four of five
arms deleted the hexafluorocarbinol and still claimed its 2.71 Å HIS407.NE2 hydrogen bond was
retained; the one arm that kept the hydroxyl was the only one whose claim was true.

### sensible_codex / reasoning_codex — the external 165-case panel, kept in its own columns

The `sensible` / `reasoning` columns stay as scored in-house (16 cases, 3 reviewers, 48 judgements
per arm). The external 165-case panel lands in **separate** columns rather than replacing them,
because the two scales do not agree in a way a single column could honestly represent:

| arm | sensible (48) | sensible_codex (495) | reasoning (48) | reasoning_codex (495) |
|---|---|---|---|---|
| ChemDFM g3 | 100%† | 97.0% | 31.2%† | 23.0% |
| ether0 schema A | 91.7%§ | 89.7% | 39.6%§ | 15.8% |
| ether0 schema B | 85.4%§ | 98.6% | 35.4%§ | 14.7% |
| ether0 g2 | 89.6%§ | 97.4% | 29.2%§ | 15.4% |

**The disagreement is not a uniformly higher bar.** The external reviewers are MORE generous on
`sensible` (schema B 85.4 → 98.6) and FAR stricter on `reasoning` (35.4 → 14.7). A stricter panel
cannot move two rates in opposite directions, so this is a calibration difference on two different
questions, not one scorer being harsh.

**And the ranking changed, not just the level.** The in-house numbers put schema A (39.6%) above
ChemDFM g3 (31.2%); the external panel puts ChemDFM (23.0%) above schema A (15.8%). A pure bar
shift preserves rank, so something else is going on — **but the old ordering was never a valid
comparison**: 31.2% came from the g2-vs-g3 panel and 39.6% from the five-arm panel, two different
reviewer sets, which is exactly what the dagger footnote warns against. **The 165-case panel is the
first time ether0 and ChemDFM g3 were scored in the same panel at all.**

A head-to-head is running in-house to separate the two explanations: ChemDFM g3 against ether0
schema B, same panel, 16 cases subsampled from the same 165, scored by the in-house reviewers. If
they also rank ChemDFM above schema B, only the absolute level is scorer-dependent. If they rank
schema B above ChemDFM, the ranking itself depends on who scores, which would mean neither panel
can decide the arm.

### g4 labels improve attribution by ~14% per sentence, NOT 2x — corrected 2026-10-17

ChemDFM-g4 (`TRAIN rc=0` 22:17, `EVAL rc=0` 22:42) and ether0-schema-B-g4 (`rc=0` 04:28 and
05:40, `ALL DONE`). Same models, same schema, same fixed 262-pair benchmark as every other row —
only the labels changed, from the g3 pass to the independent g4 pass.

| attrib 1b | g3 | g4 |
|---|---|---|
| ChemDFM | 0.290 | **0.594** |
| ether0 schema B | 0.330 | **0.693** |
| *reference ceiling* | 0.709 | 0.709 |

**ether0 schema B on g4 reaches 0.693 against a 0.709 ceiling** — read with the denominator
caveat below. attrib-1b is the "real distance, wrong residue" metric — six reviewers across five panels named wrong-owner citation as THE dominant
failure, and no arm had previously cleared 0.551. It turns out the ceiling was in the labels, not
the architecture.

That is consistent with what the label sets themselves measure: the g4 pass scores **100%** on
"every cited distance exists in the input table" and **78.6%** on "(residue, distance) co-occur in
one row", against g3's 92.4% and 55.2%.

**CORRECTION, and it cuts the headline down by a factor of eight.** I first wrote that the models
gained +110% relative on 1b. **attrib-1b is a rate whose denominator the model controls**, and the
g4 arms cite far fewer distances, so they have far fewer chances to mis-attribute:

| arm | distances/sentence | words | 1b | **correct anchors per sentence** |
|---|---|---|---|---|
| ChemDFM g3 | 2.88 | 33.3 | 0.290 | 0.835 |
| ChemDFM g4 | **1.61** | 26.5 | 0.594 | **0.956** (+14%) |
| ether0 schema B g3 | 2.81 | 33.1 | 0.330 | 0.927 |
| ether0 schema B g4 | **1.52** | 26.2 | 0.693 | **1.053** (+14%) |

Multiplied through, g4 delivers about **14% more correct anchors per sentence**, not twice as many.
Real and worth having; not a doubling. The models inherited the g4 labels' style faithfully —
terser (26.5 against 33.3 words), fewer citations, each one better attributed — which is exactly
the pattern the g3-vs-g4 LABEL comparison already showed (g4 labels cite 1.44 distances per
sentence against g3's 2.65). I had flagged that denominator trap for the labels and then walked
into it reading the model rows. **Quote 1b only alongside distances-per-sentence.**

Also improved on both arms: `attrib 1a` is now **1.000** (from 0.996 and 0.985); over-35-word
sentences fell from 12.2% to **0.0%**; elem/dir vs OWN rose (ChemDFM 0.907/0.780 → 0.898/0.914,
ether0 0.844/0.779 → **0.909/0.963**). ether0 also gains on warhead retention (0.615 → 0.857) and
scaffold (0.496 → 0.561, closer to the reference's 0.389 from the other side).

**The cost, stated plainly.** TC→input leaves the human band (ChemDFM 0.683 → 0.644, ether0
0.682 → 0.624), groundedness dips (1.000 → 0.981 and 0.997 → 0.952), and selectivity falls
(0.477 → 0.447, 0.465 → 0.379). So g4 buys attribution and pays in how closely the proposed
molecule tracks the input. Which trade is right depends on what the arm is for; it is not a free win.

**Not yet measured: the chemist columns on either g4 arm.** Both rows carry TBD for sensible and
reasoning. Given that the whole attribution gain is exactly the axis chemists said mattered most,
that panel is the next thing worth running, and it should include the g3 arms in the SAME panel —
cross-panel drift on those columns is larger than most differences between arms.

### ¤ In-house 48-case panel on the two g4 arms — they are indistinguishable, and the chemist attribution audit is NOT reliable

3 reviewers x 48 cases drawn from the same 165 the external g4 package uses, so the two scorer
pools will describe identical molecules. 144 judgements per arm.

| | ChemDFM g4 | ether0 schema B g4 |
|---|---|---|
| edit identified | 65.3% | **70.8%** |
| attribution ok | **59.7%** | 56.2% |
| invented chemistry (bad) | **22.2%** | 31.2% |
| sensible | **92.4%** | 86.1% |
| **reasoning** | 38.9% | 37.5% |
| best-in-case | 67 | **73** (neither 4) |

**The two arms do not separate** — reasoning 38.9% against 37.5%, every other column within a few
points. Same outcome as schema A vs B: a chemist panel of this size cannot tell our arms apart.

**RETRACTION, within the hour.** After two reviewers reported ~50% of cited distances wrong (G1 45
of ~90, G2 47 of 93) I told the user the g4 arms were WORSE at attribution than the g3 arms'
~24%. The third reviewer found **16%** (23 of 146) and explained the gap: **32 of the 48 contact
tables use element-grouped atom names that carry no global ordering**, so atom ownership is often
undecidable from the table. G3 scored those undecidable cases as attribution-OK; G1 and G2 failed
them. The three reviewers also counted different things as a citation — **90, 93 and 146 for the
identical 96 slots**.

Per-reviewer attribution: **48% / 52% / 77%**. That spread is larger than any difference between
arms or between generations, so **the chemist attribution audit cannot be compared across panels**
and must not be used to confirm or refute an attrib-1b change. My "g4 is worse at attribution" was
one scoring convention reported as a finding.

**Where the g4 attribution claim stands after two corrections.** Automated attrib-1b rose
(0.290 -> 0.594, 0.330 -> 0.693) but the model controls its denominator; per sentence g4 delivers
**~14% more correct anchors**, which is the defensible number; and the chemist audit can neither
confirm nor refute a change at this level of reviewer agreement.

`reasoning` here is 37-39% against the external panel's 14.7-23.0% on the g3 arms — consistent with
the 2-3x in-house/external calibration gap already recorded, which is why the external 165-case run
on these two arms is the one that will decide anything.

Two methodology notes from the reviewers, both worth fixing in the next panel: cases 44-48 are
repeat inputs with the same proposal in every slot and every boolean true, so they carry no
discriminating signal, and duplicate-input families (6XOL/6XOM/6XOP, 7I9G) triple-count a single
rationale's errors.

### invented chemistry — the column that matters most, and the most trustworthy

Added to the table 2026-10-18. It asks whether the rationale asserts a contact, bond or interaction
the input table does not support. **It is the least contaminated of the three chemist columns:**

* `would_show_a_chemist` is a comparative judgement, so it moves with which arms sat beside it
  (paired vs single-slot panels).
* `atom_attribution_ok` moves with the undecidable-ownership convention — 16% to 50% on one panel.
* **`invented_chemistry` is an absolute check against the input table.** No convention touches it,
  and it is not a comparison. Two independent pools put ChemDFM g3+g4 at **42.0%** (external, 450
  judgements) and **38.9%** (in-house, 144), which is close enough to trust.

| arm | invented chemistry ↓ |
|---|---|
| Llama-Inst base | 85.4% |
| ether0 schema A g3 | 76.6% |
| ChemDFM base | 70.8% |
| ChemDFM g3 | 63.2% |
| ether0 schema B g4 | 61.6% |
| ether0 schema B g3 | 60.6% |
| ether0 no-schema g2 | 58.2% |
| ChemDFM g4 | 51.5% |
| **ChemDFM g3+g4** | **42.0%** |
| *ether0 base* ‡‡ | *37.5%* |

Across the ChemDFM family it falls **70.8 → 63.2 → 51.5 → 42.0**, which is genuine progress and
still means **roughly two proposals in five invent chemistry that is not in the pocket.** That, not
the reasoning column, is what stands between these models and use.

**‡‡ ether0 base's 37.5% is vacuous**, in the same way as its 100% attribution: it cites no
distances and names no residues, so there is nothing for it to invent. Do not read it as the best
row.

### ◊ ChemDFM g3+g4 chemist panels — and TWO comparisons that are NOT valid, 2026-10-18

Both panels run on the same arm: in-house 48 items (3 reviewers, 144 judgements) and external 150
items (3 reviewers, 450). The 48 were drawn from inside the 150, so the two scorer pools judged
overlapping molecules.

| | in-house (48 / 46 distinct) | external (150 / 126 distinct) |
|---|---|---|
| edit identified | 84.8% | 66.4% |
| attribution, lenient | 65.2% | 68.8% |
| attribution, strict | **~46%** | not computed |
| invented chemistry (bad) | 39.9% | 43.4% |
| **sensible** | **94.9%** | **96.0%** |
| **reasoning** | **47.8%** | **38.1%** |

**Duplicate items, and the fix.** A reviewer caught that the panels contained byte-identical
repeats: 2 of 48 in-house, and **24 of 150 external (126 distinct)**. The cause was deduplicating
on pair id while the same input in the same PDB under a different TARGET ligand code yields a
different id but an identical input and proposal. De-duplicating moves the external reasoning
35.8% → 38.1% and edit identification 62.4% → 66.4%, so the redundant items were the weaker ones.
**Quote the distinct-only figures.** `build_g3g4_single_panel.py` now dedupes on content (255
eligible → 197 distinct); the ether0 g3+g4 package will not have this.

**TWO comparisons in the table are NOT valid, and both flatter this arm.**

| ChemDFM arm | panel design | undecidable-ownership rule | reasoning | attribution |
|---|---|---|---|---|
| base | single-slot | none stated | 14.6% | 29.2% |
| g3 | **paired, 4-arm** | none stated | 23.0% | 32.1% |
| g4 | **paired, 2-arm** | none stated | 22.4% | 34.7% |
| **g3+g4** | single-slot | **undecidable → TRUE** | **38.1%** | **68.8%** |

* **The attribution doubling is substantially MY OWN convention change.** I added "where atom
  ownership is undecidable, score attribution true" to this brief; no earlier panel had it. On the
  in-house g4 panel that same convention moved attribution from 16% to 50%. So 68.8% against
  32-35% is mostly instrument, not model. The in-house strict reading, ~46%, is the honest
  comparator.
* **The reasoning jump is confounded by panel design.** 38.1% is single-slot absolute scoring;
  23.0% and 22.4% came from paired panels where arms sit beside each other and the implied bar
  rises. The ONE clean comparison is against ChemDFM base, also single-slot and also external:
  **14.6% → 38.1%, a genuine 2.6x on an identical instrument.**

**What can be claimed:** g3+g4 is clearly better than untrained ChemDFM on the same measuring
stick. **What cannot:** that it beats g3-alone or g4-alone on the chemist axis. Settling that needs
one paired panel holding g3, g4 and g3+g4 together under a single convention — and nothing short of
that will do, because every cross-panel comparison we have attempted has been confounded.

**In-house vs external converged for the first time**: 47.8% against 38.1%, a 1.25x gap, where
earlier arms ran 2-3x apart. The difference is that this pair shared both the single-slot design and
the stated convention. That is evidence the earlier gaps were instrument, not scorer generosity.

### g3+g4 combined: ChemDFM recovers what g4 alone cost, 2026-10-17

7,692 train rows over 5,407 pairs, the 2,285 shared pairs keeping BOTH labellings (2x sampling
weight on those, recorded in `sft_out_g3g4/build_note.json`). Same fixed 262-pair benchmark.

| | g3 | g4 | **g3+g4** |
|---|---|---|---|
| validity | 0.958 | 0.966 | **0.973** |
| TC→input | **0.683** | 0.644 | 0.658 |
| heavyΔ MAE | 3.31 | 3.67 | **2.84** |
| warhead | 0.818 | 0.767 | **0.818** |
| grounded | **1.000** | 0.981 | 0.969 |
| selectivity | 0.477 | 0.447 | 0.469 |
| attrib 1b | 0.290 | **0.594** | 0.495 |
| distances/sentence | 2.88 | 1.61 | 1.95 |
| correct anchors/sentence | 0.835 | 0.956 | **0.965** |

**Combining recovered the costs g4 alone imposed**: warhead retention is back at g3's 0.818,
TC→input moves part-way back toward the 0.680 human band, and **heavy-atom MAE 2.84 is the best of
any trained arm in the table** (next best ether0 at 3.20). Attribution and verbosity land between
the two parents, which is what mixing two label styles should do. Highest validity of any ChemDFM
arm at 0.973.

**Training behaviour, measured at 0.2-epoch resolution for the first time.** The curve was
0.7865, 0.7023, 0.6737, 0.6378, 0.6330, 0.6305, 0.6290, 0.6282, then **0.6045 at epoch 2.00
(best)**, then 0.6599, 0.6555, 0.7097 by 3.39. Two findings:

* **The optimum is at epoch 2.0, twice as late as the g4-only arms' epoch 1** — the combined
  corpus carries more learnable signal than g4 alone.
* **There is a real stall from epoch 0.8 to 1.6** (0.6378 → 0.6282, under 0.01 across four
  evaluations) that then resolves into a 0.024 improvement. Stopping on that plateau would have
  cost a genuine gain. This is the reason to keep patience above one epoch, and it is invisible at
  epoch-boundary resolution.

The run was stopped by hand at 7 of 10 patience evaluations once the turn was unambiguous; its
adapter was materialised from checkpoint-1920 and byte-verified, with provenance recorded beside
it. **ether0 schema B on the same corpus ran next with patience 5** — one epoch of patience,
which still clears the 0.8-1.6 stall while not spending two hours past a turn as this run did.

**ether0 schema B on g3+g4, completed 2026-10-18 03:48 UTC, patience 5.** Curve at 0.2-epoch
resolution: 0.7755, 0.6901, 0.6494, 0.6203, 0.6083, 0.6189, 0.6136, 0.6079, 0.5934, then
**0.5897 at epoch 2.00 (best, checkpoint-1920)**, then 0.6620, 0.6580, 0.6642, 0.6762, 0.6465 —
five consecutive worse evaluations, so early stopping fired correctly at epoch 3.0 and
`load_best_model_at_end` restored epoch 2.00. Step 2,880 of a 9,620 ceiling. Two things this
confirms independently of the ChemDFM run:

* **The optimum is at epoch 2.0 here too**, and the apparent convergence at epoch 1.00 (0.6083)
  is a false floor — the same shape, on a different model and a different schema. The epoch-2.0
  optimum on this corpus is now a two-model finding, not a one-off.
* **Patience 2 would have stopped at epoch 1.4 and locked in 0.6083**, forfeiting the real 0.0186
  improvement. The 1.0-1.6 region (0.6083, 0.6189, 0.6136, 0.6079) is flat to within 0.011 and
  then drops twice. Patience 5 was the correct setting and patience 2 would have been wrong on
  both arms.

It also beats the ChemDFM g3+g4 arm on eval loss (0.5897 against 0.6045), the first time an ether0
schema arm has led its ChemDFM counterpart on this corpus. That is a loss comparison across two
different tokenizers and is **not** a quality claim; the chemist panels decide that.

### ether0 g3+g4 chemist panels, 2026-10-18 — the first cross-pool calibration on identical molecules

Both pools scored the SAME arm with the SAME single-slot instrument and the SAME undecidable-
ownership convention, and the in-house 48 was drawn as a **strict subset** of the external 150
(`--subset-of`, verified). So for the first time the two scales can be compared on identical
molecules rather than on two different samples — last round's pools overlapped only 33/48 by
accident, which is why that gap was uninterpretable.

Restricting the external reviewers to the in-house 48:

| field | external (same 48) | in-house (48) | gap |
|---|---|---|---|
| chemically_sensible_edit | 93.1% | 93.1% | **0.0** |
| would_show_a_chemist | 22.2% | 19.4% | 2.8 |
| invented_chemistry | 44.4% | 31.9% | 12.5 |
| edit_identified_correctly | 39.6% | 50.7% | 11.1 |
| atom_attribution_ok | 65.3% | 47.9% | 17.4 |

**The two pools agree exactly on `sensible` and to within 3 points on `would_show`.** That is the
instrument validating itself, and it retires the worry that the earlier external/in-house gaps were
scorer bias: on the two headline fields, with convention and format held fixed, there is no bias to
find. The scales are not interchangeable on the attribution fields, where the pools still differ by
12-17 points — those remain convention-sensitive and should not be read across pools.

**A batch effect worth not over-reading.** Within each in-house batch the three reviewers agree
almost exactly (would_show: batch 1 = 33%/33%/33%, batch 2 = 4%/8%/4%), but the two batches differ
sharply (would_show 33.3% vs 5.6%, edit 63.9% vs 37.5%, invented 15.3% vs 48.6%). The batches are
random halves of a shuffled pool, so this is item difficulty, not calibration. Per-item majority
gives 8/24 against 1/24, two-sided Fisher **p = 0.023** — suggestive, but across five fields it is
within multiple-comparison noise. The honest reading is that **n=48 is unstable at this resolution**
and the 150-item external number is the one to quote. It is also why the in-house panel is a
calibration check and not the primary measurement.

### Echo mol replaces novelty in the big table, 2026-10-19

`novelty` is gone: it was blank for all four base arms by construction (measured against that
arm's own training completions, of which an untrained arm has none) and read 1.000 for every
trained arm, so it discriminated nothing. It also cannot do the job it looks like it does —
novelty asks whether a molecule is absent from the training *targets*, so an arm that hands back
its *input* every time scores ~1.000.

`echo mol` is the fraction of an arm's VALID generations whose canonical SMILES equals the input
ligand's. Computed two independent ways — the stored `echo` field on each generation row, and a
fresh recomputation against `MODEL_INPUT.input_smiles` from the evidence file — which agree
exactly on all 16 arms. Rows were matched to generation files by their validity fingerprint, each
of the 16 matching uniquely.

| arm | echo mol |
|---|---|
| **ether0-24B base** | **0.678** |
| ChemDFM-14B base | 0.161 |
| SmileyLlama-8B base | 0.069 |
| SmileyLlama-8B g2 | 0.064 |
| Llama-3.1-8B base | 0.059 |
| SmileyLlama-8B g3 | 0.055 |
| ChemDFM-14B g3 | 0.032 |
| ChemDFM-14B g2 | 0.021 |
| Llama-3.1-8B g3 | 0.020 |
| ether0-24B schema B g3+g4 | 0.018 |
| ether0-24B schema A g3 | 0.014 |
| ether0-24B schema B g3 | 0.009 |
| ether0-24B (no schema) g2 | 0.009 |
| ether0-24B schema B g4 | 0.008 |
| ChemDFM-14B g4 | 0.004 |
| **ChemDFM-14B g3+g4** | **0.000** |

Two readings. **ether0 base echoing 0.678 of its valid output explains its entire chemist-panel
signature** — 100% attribution in both scorer pools (an echo cites no distances, so nothing can be
attributed wrongly), 0% would-show, 12.5% edit-identified, and the 12/62/94 reviewer split on
`sensible`. It was not scoring well on attribution; it was declining to make a claim.

**Fine-tuning removes echoing almost entirely**: every trained arm sits in 0.000–0.064, and both
ChemDFM arms trained on g4 reach 0.004 and 0.000.

Low echo is necessary, not sufficient. SmileyLlama base echoes only 0.069 while naming a residue
in 0 of 262 rows — different molecules, no explanation at all. This column cannot see that failure;
`grounded` and the chemist columns are what catch it.

A bug worth recording: a first pass at this reported 0.0% echo for every arm, because it compared
against an `input_smiles` field that does not exist on generation rows — `None` on both sides,
silently vacuous. Same failure class as the `'None'`-string truthiness bug. Always diff a new
metric against an independently computed one before believing a uniform result.

### Base-arm in-house panel, 2026-10-19 — the cells exist, and they measure the convention

Six reviewer-runs over the SAME 48 items (16 per arm) the external pool scored, so the two scales
are comparable item-for-item. They are not close:

| arm | field | in-house | external |
|---|---|---|---|
| ChemDFM base | invented chem | 18.8% | 70.8% |
| Llama base | invented chem | 29.2% | 85.4% |
| ether0 base | invented chem | 0.0% | 37.5% |
| ChemDFM base | sensible | 89.6% | 97.9% |
| ether0 base | sensible | 56.2% | 85.4% |
| Llama base | sensible | 62.5% | 85.4% |

**The 40-55 point `invented chem` gap is a definition, not a disagreement about chemistry.** The
external pool counted a rambling monologue asserting unsupported interactions as invention; the
in-house pool counted an output that returns its input unchanged as asserting nothing. On the
trained arms both readings coincide because the models actually propose edits and actually cite
numbers; on base arms they diverge because most outputs do neither.

**`sensible` is close to vacuous on base arms.** 11 of 24 items in one batch returned the input
byte-identical (RDKit-confirmed), and an echo is trivially "a reasonable analogue of the input"
unless the rubric forbids it. In-house reviewers split **12% / 62% / 94%** on ether0 base for
exactly that reason — an 82-point spread on 16 items, where the same reviewers agreed to within
1 point on `edit_identified_correctly` (all three 12%). That is the clearest demonstration so far
that the field needs "must differ from the input" written into it.

**ether0 base's 100% attribution in BOTH pools** is the same artefact from the other side: it cites
no distances, so there is nothing to attribute wrongly. Already flagged `‡‡` on its invented-chem
cell; it applies to attribution too.

**n=16 per arm is the data ceiling, not a choice.** Of 262 benchmark pairs, only 136 / 59 / 31
yield a sentence plus a parseable molecule for Llama / ether0 / ChemDFM base, and the three overlap
on exactly one pair. Each rate carries roughly ±12 points before any convention effect. SmileyLlama
base stays excluded: its outputs average 1.0 word, so any rate would describe the parser.

Conclusion: the six base chemist cells are filled and marked `∅`. They establish that a floor
exists. They should not be used to rank the base models against each other, and the `invented chem`
column should not be compared across the two pools at all.

### Base-model chemist panel, 2026-10-17 — fine-tuned ether0 is no better than untrained ChemDFM

48 single-slot items per arm (16 cases x 3 reviewers), scored by the SAME external reviewers as the
165-case panel, so these numbers sit on the `_codex` scale and are directly comparable to it.

| arm | edit identified | attribution‡‡ | invented | sensible | **reasoning** |
|---|---|---|---|---|---|
| ChemDFM base | **56.2%** | 29.2% | 70.8% | **97.9%** | **14.6%** |
| ether0 base | 12.5% | 100.0% | **37.5%** | 85.4% | 12.5% |
| Llama-Inst base | 31.2% | 35.4% | 85.4% | 85.4% | 6.2% |

**The result that matters, and it is uncomfortable.** Untrained ChemDFM scores **14.6%** on
reasoning. The same reviewers scored fine-tuned **ether0 schema B at 14.7%**, schema A at 15.8% and
ether0 g2 at 15.4%. **Fine-tuning ether0 on our labels produced rationales chemists rate no better
than an off-the-shelf ChemDFM with no training at all.** Per-reviewer counts are tight (2/3/2,
2/2/2, 1/1/1), so this is not one scorer. It does not say fine-tuning is worthless — the automated
columns show large gains in validity, heavy-atom error and groundedness — but it does say the
chemist-facing quality of ether0's reasoning is not where the gain landed.

**‡‡ `attribution` is NOT the same measurement on these rows.** All three base arms cite **zero
distances** in the sampled items (ChemDFM base 0.0 per item with 2.3 residues named, ether0 base 0.0
and 0.0 across 111.7 words, Llama base 0.0 and 0.9). ether0 base's 100% is therefore vacuous —
nothing is cited, so nothing is miscited — while ChemDFM and Llama score 29–35% because they name
residues *without* a distance and reviewers counted that as unsupported. Do not read this column
across the base and trained rows.

Raw scores: `~/Downloads/covalentformer_labelling_task/base_model_eval/score_out/` (9 files,
verified complete: 720 of 720 booleans); rates in `base_eval_results.json`; key in
`panel_base_KEY_DO_NOT_SHIP.json`.

### ¶ The 165-case chemist panel, 2026-10-16 — and it REVERSES the 48-judgement read

External reviewers, 165 cases, 4 arms side by side per case, letters reshuffled, 3 independent
reviewers, **495 judgements per arm** and 9,900 booleans with zero missing slots.

| arm | edit identified | attribution | invented | sensible | **reasoning** | best-in-case |
|---|---|---|---|---|---|---|
| **ChemDFM g3** | **66.9%** | 32.1% | 63.2% | 97.0% | **23.0%** | **88** |
| ether0 schema A | 43.0% | 21.2% | 76.6% | 89.7% | 15.8% | 31 |
| ether0 g2 | 49.9% | **38.6%** | **58.2%** | 97.4% | 15.4% | 55 |
| ether0 schema B | 51.5% | 34.3% | 60.6% | **98.6%** | 14.7% | 41 |

**What the small panel got wrong.** At 48 judgements the ether0 arms read 39.6 / 35.4 / 29.2% on
reasoning and appeared ordered A > B > g2. At 495 they are **15.8 / 14.7 / 15.4% — a dead heat** —
and the absolute rates were inflated about twofold. The earlier ordering was noise, and the lead-arm
recommendation built on it (schema B) does not survive. All three reviewers agree here
(per-reviewer would-show: 12/18/18%, 13/15/16%, 13/18/15%), so this is not one harsh scorer.

**ChemDFM g3 is the best arm on the chemist axis** and it was never in the five-arm panel, which is
why this is new information rather than a revision: 23.0% reasoning against ~15%, best-in-case 88
against 31–55, and edit identification 15 points clear.

**What got STRONGER.** The chemists and the computable attrib-1b now agree on attribution ordering —
ether0 g2 (38.6% / 1b 0.551) > schema B (34.3% / 0.330) > schema A (21.2% / 0.183). The
contradiction flagged after the small panel was an artefact of n=48. **The think/answer schema
costs attribution, and that result survives tenfold more data.**

**The number to lead with: `best_overall` was "none" in 280 of 495 judgements — 57%.** More than half
the time a chemist saw four proposals and judged none worth putting in front of an audience. Every
between-arm difference above sits inside that fact.

Raw scores: `~/Downloads/covalentformer_labelling_task/metrics_eval/score_out/` (21 files, verified
complete before unblinding); per-arm rates in `metrics_eval_results.json`; key in
`panel_metrics_eval_KEY_DO_NOT_SHIP.json`.

### Known caveats on this table

- Losses are **not comparable across bases**: Llama 128k vocab, Qwen 152k, Mistral 131k.
- Every arm trained with `max_epochs=10`, `early_stopping_patience=2`,
  `load_best_model_at_end=True`, `metric_for_best_model='eval_loss'`, `save_total_limit=2`. So the
  adapter is the **best-by-cross-entropy** checkpoint and the later epochs were **deleted**.
  Cross-entropy is the criterion we established is nearly blind to format and completely blind to
  attribution; whether a later epoch writes better-attributed sentences is **unanswerable on these
  runs**. Decision taken: finish the current experiments under this config for comparability, and
  revisit selection afterwards.
- Every arm is prompted with the **Alpaca** template, which is no model's native format (Llama
  `<|start_header_id|>`, Qwen `<|im_start|>`, Mistral `[INST]`). This may confound the
  "chemistry post-training degrades instruction-following" reading of the zero-shot row.
- ether0's two existing evals were invalid (`merge_and_unload` silently discards a LoRA on a 4-bit
  base); re-evals are queued behind the fixed code.

---

## 3. Corpus expansion — measured headroom

Mirror: 39,711 PDB entries (1999 → 2026-09-25), 51,094 ligands → 8,359 pairs → 7,069 with evidence.

Dropping the same-paper constraint (group by target instead), keeping every other filter:

| Tanimoto | total pairs | NEW vs our 8,359 | with the 500/target cap | targets we do not have |
|---|---|---|---|---|
| 0.50 | 33,584 | 12,153 | 10,138 | 441 |
| 0.60 | 17,681 | 6,526 | 6,189 | 353 |
| **0.70** | 8,196 | **3,276** | 3,276 (cap never binds) | **254** |

This buys **breadth**, not only depth: at TC 0.70 it adds 254 targets absent from our current 775.
Expect ~85% to survive evidence-building, less an unquantified loss from cross-paper residue
numbering.

### Is Tanimoto >= 0.70 enough to mean "same series"? Yes — measured.

| | mean TC | identical Murcko | scaffold-TC | MCS coverage (p10) |
|---|---|---|---|---|
| existing corpus (same paper, TC >= 0.50) | 0.653 | 37.8% | 0.765 | 0.890 (0.696) |
| **g5 candidates (cross-paper, TC >= 0.70)** | 0.805 | **49.0%** | **0.865** | **0.937 (0.800)** |
| cross-paper, TC >= 0.60 | 0.734 | 42.5% | 0.788 | 0.917 (0.775) |

**Do not add a scaffold-identity filter.** The corpus we already trust shares an identical Murcko
scaffold only 37.8% of the time -- these edits routinely touch the ring system -- so the filter
would reject ~62% of the pairs everything is built on. g5 at TC >= 0.70 is already a TIGHTER
series than the baseline on every measure, including a p10 MCS coverage of 0.800. If a
belt-and-braces check is ever wanted, use **MCS coverage >= 0.70**, not scaffold identity.

### External sources

There is **no MMP + crystal-structure database**.

**CovPDB adds real structures, contrary to an earlier assumption here.** Of its 2,261 complexes
only **334 (14.8%) are in our mirror**; **1,927 are not**, and just 43 appear among our 7,069
labelled pairs. Our 39,711-entry mirror is therefore a SELECTIVE sweep, not a comprehensive
protein-ligand one, so fetching those 1,927 entries is a genuine covalent expansion for g6.
CovBinderInPDB (3,555 structures, to Jan 2022) has not been intersected yet. Fragalysis-XChem is the one place with possibly
undeposited structures, but fragments mostly fall under our `HEAVY_MIN = 10` cut. PDBbind,
BindingMOAD and PoseBusters are PDB subsets; PDB-REDO is re-refinement of the same entries; the CSD
has no protein pocket; predicted complexes would break the premise that the contact table is
*measured*.

---

## 4. Infrastructure

- `ai-gpu-a100` (us-central1-a) and `ai-gpu-a100-b` (us-central1-b), both A100 40 GB, **both
  preemptible** — five reclaims so far, each costing ≤1 epoch thanks to `save_strategy="epoch"`.
- ether0's 88 GB fp32 weights live on A's 1.5 TB second disk and do not fit B, so all ether0 work
  is pinned to A.
- Direct SSH is unreliable; use `--tunnel-through-iap`.
- Installing transformers 5.9.0 on B pulled huggingface-hub 1.33, which forces `click>=8.4.2` and
  **breaks boltz 2.2.1's `click==8.1.7` pin** on that machine.
