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
| Llama-3.1-8B-Inst **base** | – | 0.519 | ⏳ | – | ⏳ | ⏳ | ⏳ | ⏳ | 0.998 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| Llama-3.1-8B-Inst | g3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| SmileyLlama-8B **base** | – | 0.882 | ⏳ | – | ⏳ | ⏳ | ⏳ | ⏳ | – | – | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| SmileyLlama-8B | g2 | 0.954 | 0.748 | 1.000 | 0.623 | 3.40 | 0.758 | 0.444 | 1.000 | 0.436 | 0.958 | 0.459 | 0.419 | 0.598 | 0.794 | 0.782 | 64.4% | 15.6% |
| SmileyLlama-8B | g3 | 0.908 | 0.761 | 1.000 | 0.588 | 3.77 | 0.700 | 0.332 | 0.996 | 0.481 | 1.000 | 0.268 | 0.425 | 0.505 | 0.709 | 0.748 | ❌ | ❌ |
| ChemDFM-14B **base** | – | 0.118 | 0.871 | – | 0.793 | 2.84 | 0.750 | 0.104 | 1.000 | 0.275 | – | – | – | – | ❌ | ❌ |
| ChemDFM-14B | g2 | 0.893 | 0.752 | 0.994 | 0.658 | 3.74 | **0.828** | 0.523 | 1.000 | 0.478 | 0.927 | 0.327 | 0.422 | 0.582 | 0.800 | **0.935** | **81.1%** | **25.6%** |
| **ChemDFM-14B** | **g3** | **0.958** | 0.761 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | 1.000 | **0.543** | ⏳ | ⏳ | **0.525** | 0.529 | **0.907** | 0.780 | ❌ | ❌ |
| ether0-24B **base** | – | ⏳ | ⏳ | – | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| ether0-24B schema A | g3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| ether0-24B schema B | g3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
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
