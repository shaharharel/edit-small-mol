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

| arm | data | validity | uniq | novelty | TC→in | heavyΔ MAE | warhead | scaffold | grounded | selectivity | attrib 1a | attrib 1b | **elem vs REF** | **dir vs REF** | **elem vs OWN** | **dir vs OWN** | sensible | reasoning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Inst **base** | – | 0.519 | 0.750 | – | 0.737 | 12.10 | 0.857 | 0.761 | 0.996 | 0.405 | 1.000‡ | 1.000‡ | 0.238 | 0.577 | 0.584 | 0.941 | ❌ | ❌ |
| Llama-3.1-8B-Inst | g3 | 0.771 | 0.743 | 1.000 | 0.594 | 3.32 | 0.632 | 0.322 | 1.000 | 0.463 | 0.977 | 0.384 | 0.441 | 0.553 | 0.809 | 0.765 | 75.0%§ | 4.2%§ |
| SmileyLlama-8B **base** | – | 0.882 | 0.753 | – | 0.582 | 18.82 | 0.786 | – | – | – | – | – | 0.000 | – | – | – | ❌ | ❌ |
| SmileyLlama-8B | g2 | 0.954 | 0.748 | 1.000 | 0.623 | 3.40 | 0.758 | 0.444 | 1.000 | 0.436 | 0.958 | 0.459 | 0.419 | 0.598 | 0.794 | 0.782 | 64.4% | 15.6% |
| SmileyLlama-8B | g3 | 0.908 | 0.761 | 1.000 | 0.588 | 3.77 | 0.700 | 0.332 | 0.996 | 0.481 | 1.000 | 0.268 | 0.425 | 0.505 | 0.709 | 0.748 | 56.2%§ | 4.2%§ |
| ChemDFM-14B **base** | – | 0.118 | 0.871 | – | 0.793 | 2.84 | 0.750 | 0.104 | 1.000 | 0.366 | – | – | 0.253 | 0.516 | 0.706 | 1.000 | ❌ | ❌ |
| ChemDFM-14B | g2 | 0.893 | 0.752 | 0.994 | 0.658 | 3.74 | **0.828** | 0.523 | 1.000 | 0.478 | 0.927 | 0.327 | 0.422 | 0.582 | 0.800 | **0.935** | **81.1%** | **25.6%** |
| **ChemDFM-14B** | **g3** | **0.958** | 0.777 | 1.000 | **0.683** | 3.31 | 0.818 | 0.482 | 1.000 | 0.477 | **0.996** | 0.290 | **0.525** | 0.529 | **0.907** | 0.780 | **100%**† | **31.2%**† |
| ether0-24B **base** | – | 0.225 | 0.847 | – | 0.884 | 4.31 | 1.000 | 0.247 | – | – | 0.000 | – | 0.547 | 0.625 | 0.809 | 0.000 | ❌ | ❌ |
| ether0-24B schema A | g3 | 0.828 | 0.770 | **1.000** | 0.560 | 3.62 | 0.808 | 0.223 | 0.996 | **0.537** | 0.969 | 0.183 | 0.491 | 0.527 | 0.849 | 0.853 | **91.7%**§ | **39.6%**§ |
| **ether0-24B schema B** | **g3** | 0.863 | **0.792** | **1.000** | **0.682** | **3.20** | 0.615 | 0.496 | 0.997 | 0.465 | **0.985** | 0.330 | 0.463 | 0.564 | 0.844 | 0.779 | 85.4%§ | 35.4%§ |
| **ether0-24B (no schema)** | **g2** | 0.897 | 0.753 | **1.000** | **0.692** | **3.20** | 0.800 | **0.532** | **1.000** | 0.448 | 0.931 | **0.551** | 0.391 | 0.559 | **0.875** | **0.858** | 89.6%§ | 29.2%§ |
| *reference (ceiling)* | – | 1.000 | 1.000 | – | **0.680** | 0 | **0.760** | **0.389** | 1.000 | 1.000 | 0.962 | 0.709 | **0.825** | **0.933** | **0.825** | **0.933** | – | – |
| *trivial baseline* | – | – | – | – | 1.000 echo | – | 1.000 | 1.000 | – | **0.5625** | – | – | – | – | – | – |

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
