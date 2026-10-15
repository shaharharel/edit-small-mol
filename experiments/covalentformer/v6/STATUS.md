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
| **g5** | ~3,276 | TBD | **cross-paper**, same protein, **TC ≥ 0.70** — NEW criteria | v2 input-anchored + envelope (same as g3/g4) | `label_queue_g5/` — to be labelled |
| **g6** | TBD | TBD | **covalent-enriched**: input ligand covalently bound or carrying a warhead | v2 input-anchored + envelope | `label_queue_g6/` — to be labelled |

g1 is a different *sentence* generation and cannot be mixed with g2/g3/g4 at all (3.1% six-rule
compliance against 90%+). g2/g3/g4 share sentence rules and differ only in packet contents and
labeller identity. **g5 and g6 differ in PAIR SELECTION, which is a new axis** — they must stay in
their own directories with their own stamps so a future training run cannot silently pool a
cross-paper pair with a same-paper one.

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

| arm | data | validity | uniq | novelty | TC→in | heavyΔ MAE | warhead | scaffold | grounded | selectivity | attrib 1a | attrib 1b | elem self | dir self | sensible | reasoning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Inst **base** | – | 0.519 | ⏳ | – | ⏳ | ⏳ | ⏳ | ⏳ | 0.998 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| Llama-3.1-8B-Inst | g3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| SmileyLlama-8B **base** | – | 0.882 | ⏳ | – | ⏳ | ⏳ | ⏳ | ⏳ | – | – | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| SmileyLlama-8B | g2 | 0.954 | 0.748 | 1.000 | 0.623 | 3.40 | 0.758 | 0.444 | 1.000 | 0.436 | 0.958 | 0.459 | 0.794 | 0.782 | 64.4% | 15.6% |
| SmileyLlama-8B | g3 | 0.908 | 0.761 | 1.000 | 0.588 | 3.77 | 0.700 | 0.332 | 0.996 | 0.481 | 1.000 | 0.268 | 0.709 | 0.748 | ❌ | ❌ |
| ChemDFM-14B **base** | – | 0.118 | 0.871 | – | 0.793 | 2.84 | 0.750 | 0.104 | 1.000 | 0.275 | – | – | – | – | ❌ | ❌ |
| ChemDFM-14B | g2 | 0.893 | 0.752 | 0.994 | 0.658 | 3.74 | **0.828** | 0.523 | 1.000 | 0.478 | 0.927 | 0.327 | **0.800** | **0.935** | **81.1%** | **25.6%** |
| ChemDFM-14B | g3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| ether0-24B **base** | – | ⏳ | ⏳ | – | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| ether0-24B schema A | g3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| ether0-24B schema B | g3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ❌ | ❌ |
| *reference (ceiling)* | – | 1.000 | 1.000 | – | **0.680** | 0 | **0.760** | **0.389** | 1.000 | 1.000 | 0.962 | 0.709 | **0.825** | **0.933** | – | – |
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
- **elem self / dir self** — element-count and direction fidelity of the sentence against the
  **model's own** molecule, not the reference. Reference-free.
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
