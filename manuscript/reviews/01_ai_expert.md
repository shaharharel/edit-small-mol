# Memo 01 — ML expert: making Methods defensible

Numbers quoted only from `manuscript/EVIDENCE.md`; implementation files read for configuration
only. Two code-vs-evidence conflicts are flagged at the end — resolve before Methods freezes.

## 1. Formal setup

`x` = input molecule, `c` = its per-atom contact table in the complex, `z` = design-intent
sentence, `y` = analogue:

`p(y, z | x, c) = p_R(z | x, c) · p_M(y | x, c, z)`

This is the chain rule, so the factorisation costs nothing in expressiveness. What differs is the
**parameterisation**: R and M are separate adapters with separate losses, so `z` is a bottleneck the
molecule is routed through, not a decoration emitted beside it. In direct joint training one decoder
emits `z` then `y` in one sequence; nothing forces `y` to depend on `z`, and the rationale is free to
be post-hoc. Three things this buys:

1. **Separate supervision.** R trains on stated intent, M on measured pairs — signals with different
   noise processes (reasons 90% honest but 60.7% restating an outcome; molecules 3.3% contradicted),
   so one objective leaves neither correctable.
2. **Independent identifiability.** Each factor becomes testable by intervening on `z`: chemists'
   sentence (TC→reference **0.919**, median 1.000), another pair's (**0.541**), none (**0.684**). A
   joint model offers no such handle.
3. **Human intent at inference.** A chemist types the intent and M returns the analogue — the oracle
   arm *is* that condition, so the product claim is already validated.

State the cost in the same paragraph: `R→M` reaches **0.574**, far under oracle and barely above
mismatch; R is the bottleneck by **18.6×**, and splitting costs about **2×** regardless.

## 2. Training, as Methods must state it

- **Bases.** ChemDFM-v2.0-14B primary; ether0-24B, Llama-3.1-8B-Instruct, SmileyLlama-8B, DESRO-8B
  as comparators. DESRO is 8B vs ChemDFM's 14B, so cross-base rows are confounded with scale — the
  clean contrast is each base **against itself**, with and without our adapter.
- **LoRA.** r=16, α=32, dropout 0.05, all seven projections; lr 1e-4 cosine, warmup 0.03; batch 1 ×
  grad-accum 8; bf16, gradient checkpointing; `max_length` 2304 enforced as a **refusal threshold,
  not truncation** (truncating deletes contact-table rows from the arm the pocket ablation favours).
  One shared config across every cell of an ablation.
- **Serialisation.** Alpaca format in every arm — never per-model chat templates, which confound
  template with checkpoint. Input block: `TARGET`, `INPUT SMILES`, then the contact table as
  fixed-width text, one row per ligand heavy atom (nearest protein atom within 4.0 Å, chain,
  distance, angle, untruncated interaction typing with uncertainty flags). Blind arm substitutes
  "No pocket structure is provided."
- **Masked loss.** Response-only. The prompt is a median ≈960 tokens against ≈100 for the answer, so
  an unmasked objective spends ~90% of gradient regenerating a table the model is always given; the
  mask is built from prompt ids, making the prompt a prefix by construction.
- **Leakage controls, reported not assumed.** Counterfactual fields out of render scope plus a string
  assertion that the analogue SMILES, analogue PDB id and analogue-only residue tokens are absent;
  for M, the SMILES line stripped from the given reason and checked per row (truncation-substring
  trap handled); special-token ids pinned per vocabulary and re-verified at load.
- **Checkpoint selection.** ≤10 epochs, early stop on val loss, patience 2, best checkpoint kept,
  stopping epoch reported. Then, verbatim, in Methods *and* every val caption: *"Because both the
  stopping epoch and the retained checkpoint were selected on the validation split, every validation
  number reported here is a model-selection score, not a held-out test score, and is optimistically
  biased by an unknown amount."*

## 3. Splits

- **Molecule-level holdout is the floor.** For the 21k arms neither input nor reference analogue
  appears anywhere in training: 1,469 unseen-molecule rows (553 never-trained targets, 916 trained
  targets with new molecules), 400 prompts per arm.
- **Unseen-target holdout is infeasible**: target space repeats across papers, leaving **21
  targets** — variance would be dominated by which handful of proteins landed in the holdout.
- **Against random pair splits.** Sibling pairs off one complex share pocket, numbering and
  near-identical tables; a random split puts them on both sides, so the holdout measures
  memorisation of pockets already seen — the sibling project measured **71% exact duplicate pairs**.
  For the PDB corpora the split runs over connected components of the target↔PDB bipartite graph, so
  zero target and zero PDB overlap hold by construction, capped at 20% of the val budget per component.

## 4. The ablation table

| arm | isolates | decided by |
|---|---|---|
| pocket-21k vs blind-21k | whether the table carries signal, and into *what* | grounded 1.000 vs 0.432; selectivity 0.388 vs 0.121; **TC→ref 0.679 vs 0.680** |
| wrong-pocket swap (cohort) | whether the *molecule* uses the pocket | ΔTanimoto vs true-analogue noise floor: **0.013** |
| M oracle | ceiling of M given a human intent | TC→ref **0.919**, median 1.000 |
| M ← R | the deployable pipeline | TC→ref **0.574**; uniqueness 0.238 |
| M mismatch | that M *follows* the instruction | oracle−mismatch **+0.379** (p=9.6×10⁻⁵⁰); mismatch−none **−0.147** (p=6.9×10⁻⁴⁰) |
| M none | *control only, not a baseline* | TC→input **1.000**; every property delta exactly 0 |
| direct joint vs R/M | whether factorising helps or costs | paired TC→ref + reasoning-sound rate, same prompts — **[TBD-2]** |
| corpus scaling g2→g3→g4→9k | what data buys, and what it does not | validity 0.893→0.958→0.966→**0.973** (saturates) vs reasoning 22.9→29.9→37.6→**46.5%** (climbing) |
| base vs finetuned, per base | the adapter's contribution, free of base size | ChemDFM validity 0.118→0.973, reasoning 3.9%→**46.5%**; same contrast within each base |
| property-directed response | does a stated intent move the stated property | TPSA +3.62 (n=259), cLogP −0.63 (n=146), MW −31.2 Da (n=75); basicity flat; size↑/rigidity↓ wrong-signed — **[TBD-5]** |

`M none` stays in the table with a footnote: a reviewer who sees only oracle vs `none` reads the
wrong comparison.

## 5. Metric definitions

- **validity** — SMILES extracted *and* RDKit-parsed; denominator all rows, so emitting no molecule
  scores 0, not NaN. Format compliance reported separately (format failure ≠ bad chemistry).
- **uniqueness** — distinct canonical / valid SMILES; always reported, since the 21k model's
  **0.173** is a real cost of the potency corpus.
- **TC→input** — mean Tanimoto (ECFP4, 2048 bits) to the input; 1.0 = echo. References sit near
  0.68 — a **band**, not a maximum.
- **TC→reference** — mean Tanimoto to the analogue actually made. **Necessary, not sufficient**: it
  penalises a different *valid* analogue, so a low value is ambiguous (wrong edit vs other good
  edit). Used only in paired same-input arm contrasts, never as a standalone quality score.
- **heavy-atom Δ MAE** — |heavy atoms added by model − by reference|; catches right-size/wrong-edit.
- **groundedness** — named residues present in that pair's own table; also necessary-not-sufficient,
  since it reads 1.000 on sentences chemists reject.
- **selectivity** — overlap of residues named by the sentence vs the reference, printed beside its
  zero-parameter "nearest table residue" baseline, **[TBD-8]** (new register line; hypothesis: arms
  near that baseline are no better than a distance sort).
- **invented chemistry** — sentences asserting a contact absent from that pair's table.
- **Never reported: pooled multi-target R².** Between-target potency offsets inflate it even when
  within-target ranking is random — an artefact of prediction shrinkage. Per-target averaged
  Pearson/Spearman/R² only, with n per target.
- **Panels** carry their seeded controls and no comparison crosses panels: panel-to-panel drift
  exceeds the between-arm differences being measured.

## 6. Three attacks, pre-empted

**(a) "Your val set is your test set."**
> *"The validation split chose both the stopping epoch and the retained checkpoint, so we label every
> validation figure a model-selection score rather than a held-out result; the held-out claims rest on
> a separate molecule-level holdout in which neither the input nor the reference analogue appears
> anywhere in training."*

**(b) "The pocket conditioning does nothing — your own ablation says so."** Lead with it.
> *"Withholding the contact table leaves the molecule essentially unchanged (TC→reference 0.680 vs
> 0.679) while collapsing the prose (groundedness 0.432 vs 1.000, selectivity 0.121 vs 0.388), and the
> wrong pocket costs only 0.013 Tanimoto against the true-analogue noise floor: the pocket governs what
> the model may assert, not yet what it builds, and closing that gap is the objective of the next round."*

**(c) "Oracle TC is not a result — the intent leaks the answer."**
> *"A sentence drawn from another pair is not merely uninformative but actively harmful — 0.147 below
> the no-sentence control, p=6.9×10⁻⁴⁰, with validity falling from 100% to 91.5% — so M follows the
> instruction rather than decoding a leaked target; and since the no-sentence arm echoes the input on
> 100% of rows (TC→input 1.000, zero property change), it is a degenerate control and the mismatched
> arm is the honest baseline against which oracle's +0.379 is measured."*

## Flags for the lead

1. **Split conflict.** `build_rm_arms_21k.py` says the 21k splits are inherited **by target** (0.15
   val fraction, 32 val targets); EVIDENCE §4b describes a **molecule-level** holdout and puts
   unseen-target holdout at **21** targets. Methods needs one authoritative sentence for (i) the
   split the 21k model was *trained* under and (ii) the holdout §4b was *scored* on, with 32 vs 21
   reconciled.
2. **Groundedness ceiling conflict.** EVIDENCE §4a gives the chemists' own sentences **1.000**;
   `eval_sft.py` argues the ceiling is ≈0.90 because the analogue's new atoms reach residues the
   input never touched. If the ceiling is under 1.0, every groundedness figure must be printed
   against it, or we invite "10% hallucination" misreadings. One recomputation, one number.
