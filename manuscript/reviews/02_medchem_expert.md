# Memo 02 — med-chem expert review

Sources: `manuscript/EVIDENCE.md` + PI concept note. All numbers from EVIDENCE.md.
Perspective: the chemist who would have to run this on a live programme.

## 1. What would change how I work, ranked

**A1 — the intent sentence is an executable specification (§4b).** Given the chemists' own
sentence, M reproduces the analogue they actually made at TC→reference **0.919 mean, 1.000 median**;
oracle − mismatch **+0.379 paired, p = 9.6×10⁻⁵⁰**; a *wrong* sentence is worse than none (0.541 vs
0.684, p = 6.9×10⁻⁴⁰) and costs validity 100% → 91.5%. If I can write the hypothesis, the machine
renders it as structures. That is the division of labour I want.

**A2 — it replicates (§4a).** PocketPair-14k: oracle 0.698, R→M 0.518, foreign sentence 0.503,
+0.180 at p = 3×10⁻²⁷. Two corpora, two splits, same ordering — which is why I believe A1.

**A3 — ZAP70, a real small programme (§5).** MAE **0.489**, Spearman **0.792** with Vina on 4K2R
plus pretraining, additive — and *at 280 molecules every more complex architecture hurt*. The most
directly actionable sentence in the pack: at my data volumes, add structure, not parameters.

**A4 — property dials with honest failures (§4c).** TPSA +3.62 (n=259), cLogP −0.63 (n=146), MW
−31.2 Da (n=75) move as asked; basicity flat; size↑ and rigidity↓ **wrong-signed** [TBD-5]. I'd use
the first three tomorrow. Publish the failures in the same table — a dial I know is broken beats
four I must test myself.

**A5 — uniqueness collapse, as a deployment warning.** 21k uniqueness **0.173**, R's intents 0.238.
Seventeen distinct ideas per hundred asks is not a generator; use it as a *renderer* of my
hypotheses, not an idea source.

**Interesting-but-useless.** Validity 0.118 → 0.973 is plumbing, not chemistry: one sentence, then
SI. Base-model rows — nobody picks a backbone from them, and they are internally incomparable (§4).
Model-labeller panels (`would_make` 0.61, sound 0.08, `sensible` 92.4%, `reasoning` 46.5%) are
atmosphere, not evidence while caveat 6.1 stands. Exception: **invented chem 42.0%** — sentences
asserting a contact absent from that pair's own table — disqualifies any grounded claim and cannot
sit silently beside "groundedness 1.000" from another arm without a reconciliation sentence.

## 2. The pocket result — framing I would believe

1. The contact table **decides what the sentence may cite**: groundedness 1.000 vs 0.432,
   specificity 0.388 vs 0.121. Large, clean effect.
2. It **does not measurably decide the molecule**: 0.679 vs 0.680; a deliberately wrong pocket costs
   **0.013** against the true-analogue noise floor; distance annotations changed nothing.
3. Then say why 2 is *partly* a measurement problem, without overselling the rescue. A single-edit
   analogue lives at TC→input ≈ 0.68–0.75, where whole-molecule Tanimoto has almost no dynamic
   range — a poor instrument for "did the pocket move the edit". The discriminating test is whether
   the edit lands on the atoms the table flags: site-of-change vs cited-residue agreement. **That
   experiment is not in the pack.** Register it [TBD-12]; until it exists, point 2 stands unmitigated.

**Does the premise collapse?** The *pocket-conditioned generator* premise collapses — no evidence
the pocket steers structure, one negative result that it does not. The *pocket-grounded rationale*
premise stands and is the real contribution. Retitle and reframe around learned, auditable design
intent, with the pocket as what makes a rationale checkable and target-specific. Claiming
pocket-aware *generation* on this evidence is how a referee kills us, and this programme has paid
that price twice already (caveat 6.5; ZAP70 0.469).

Why the reduced claim still earns a paper: enumeration is not the bottleneck — any MMP tool floods
me with analogues. Choosing five to make and defending it on Thursday is. A falsifiable one-line
hypothesis per analogue, checkable against a contact table, is exactly what a project team consumes.

## 3. The echo failure

**Artifact of pair training, mislabelled as a property.** Trained on (A→B) pairs with mostly small
edits, p(B|A) with no instruction peaks at B = A: the model learned *copy, then perturb where told*.
It is not refusing — there is no refusal mechanism — so "safety" is flattery.

It is a fine **diagnostic**: zero intent → zero edit (TC→input 1.000, 100% of rows, every property
delta exactly zero) proves the sentence is the sole driver of structural change. The paper should
(i) keep `none` out of the baseline role, per caveat 6.2, and use mismatch; (ii) report zero-intent
echo as instruction-dependence evidence, not performance; (iii) state the engineering consequence —
**no sentence-free fallback mode** in deployment; (iv) rename, since the main table's `echo` column
(0.000–0.678, measured *with* an intent) is a different quantity, or readers will average them.

## 4. Metrics: trusted vs vanity

**Lead with.** **TC→reference on molecule-level holdout** — the only metric with a real human design
decision as ground truth; always paired, always beside mismatch. **Heavy-atom Δ MAE** — 0.374 vs
2.622 reads instantly as "one atom moved" vs "drifting three atoms into another series"; badly
underrated, promote it into the main figure. **Invented-chemistry rate** — the one rationale metric
needing no panel, and monotone: 0.0% (chemists) / 13.1% (R) / 27.1% (R→M) / 55.7% (mismatch).
**Warhead retention** on the covalent arm — binary and decisive, a lost warhead is a dead compound;
no number exists in the pack, so register [TBD-8] and compute it on the **largest fragment only**
(anywhere-in-SMILES inflates it).

**Keep conditionally.** **Scaffold retention** is the "still the same series" metric, which is what
hit-to-lead means — but it, like every conditional-on-validity metric, is selection-biased in the
base rows (Llama base: 0.761 retention over 51.9% valid output). Caption that or drop those rows.

**Vanity → SI.** **Validity**: a gate, not a result. **Uniqueness**: it *anti-correlates with
quality here* — mismatch has the highest uniqueness (0.937) and the worst molecules (0.541); demote
to a collapse detector, its one good job. **TC→input**: not a quality metric at all, 0.68 is the
reference analogues' own band; use it as a three-way window check only — in-band / echo (1.000) /
runaway.

## 5. Results order for a chemistry audience

(a)…(g) buries the engine and hides the uncomfortable result at (f). Reorder:

1. **(b) does the reason decide the molecule** — the engine. Show the holdout definition, four arms
   with paired statistics and CIs, heavy-atom Δ, echo disclosed up front.
2. **(f) what the pocket means** — immediately, while the reader still trusts us: the prose/structure
   split, wrong-pocket 0.013, and [TBD-12] or an explicit admission it is absent.
3. **(c) retrospective hit-to-lead** — the only evidence of prospective value; must beat *both* a
   pocket-blind and a plain MMP-enumeration baseline. PENDING [TBD-3]: the load-bearing gap.
4. **(d) ZAP70** — one programme end-to-end on measured data; must show it does not rest on 0.469.
5. **(e) agentic vs literature reasoning** — must show literature intents beat agent labels on
   usefulness [TBD-1/2]; if not, cut to a paragraph plus the corpus-defects list, itself a
   contribution.
6. **(a) generation with a reason** — half-page usability gate; the interesting part is the
   uniqueness cost of the potency corpus.
7. **(g) LLM ablation** — SI table; base-vs-tuned and corpus-size scaling only.

## 6. Three figures

**Fig 1 — "the sentence is the specification" (Mpro, 7JQ2).** One hit; the contact table as a
residue × ligand-atom strip; four molecules from the *same* input under four sentences — chemists'
(recovers the published analogue), R's, a foreign one (edit at the wrong site), none (byte-identical
input). Each panel shows its sentence with cited residues highlighted, TC→reference, heavy-atom Δ.
Inset: paired oracle − mismatch distribution. *Reader learns what the system is, what each control
does, and why `none` is not a baseline.*

**Fig 2 — "pocket governs prose, not structure".** Two rows, same input. Row 1: sentences, true vs
wrong pocket, residues colour-coded green (in that pair's table) / red (absent). Row 2: the two
molecules overlaid with their Tanimoto. The punchline is visual — unrecognisable prose,
near-superimposable structures. Right panel: TC→reference histograms, true vs wrong pocket, the
0.013 shift against the noise floor. *Reader learns the honest claim and its exact limit — this
figure is why a referee trusts the rest.*

**Fig 3 — ZAP70 / Cys346 worked programme.** (a) the Cys346 contact environment with the acrylamide
anchor marked; (b) five analogues with stated intents, warhead-retained Y/N, measured pIC50 where
held; (c) predicted-vs-measured scatter at MAE 0.489 / Spearman 0.792, the honest baseline 0.723 as
a reference line, 0.469 absent and annotated as retracted. *Reader learns it works on a real
covalent programme at realistic n (280), and that we police our own numbers.*

## 7. What a chemist will say is missing

- **Synthetic accessibility: nothing at all** — no SA, retro-route or triage number anywhere, while
  the concept note promises SA-directed RL. My first question on any analogue is whether my CRO can
  make it in two weeks. Needs SA + retro hit-rate on ~100, chemist-triaged [TBD-9]. Without it this
  is a structure-drawing exercise.
- **"Selectivity" is the wrong word** — 0.388 is residue specificity of a *sentence*, not off-target
  selectivity. Chemists and referees will read the latter and demand a kinase panel. Rename throughout.
- **ADMET absent** — TPSA/cLogP/MW are not a property profile (no hERG, solubility, permeability,
  soft spots). Hit-to-lead is multiparameter; one dial is a cartoon of the job.
- **Novelty vs prior art: zero**, and this is the sharp edge of our own headline. TC→reference median
  **1.000** means we reproduce compounds already in the patent literature. Needs a prior-art /
  date-cutoff check on generated non-reference molecules [TBD-10] and an explicit "recovery ≠
  novelty".
- **Is 0.919 a trivial edit?** Partly — quantify it before being asked. The oracle arm sits at
  TC→input 0.677 with heavy-atom Δ MAE 0.374: roughly one heavy atom moves, with the target residue
  named in the prompt. That is close to multiple choice with the answer supplied. Control: enumerate
  all single-heavy-atom edits of the input, score them by the same sentence, report set size and
  recovery by edit size (0–1 / 2–3 / 4+) [TBD-11]. The number survives as *faithful rendering of a
  stated intent*, not as *design*.
- **No measured potency gain for any generated molecule** — everything is similarity to something a
  human already made. Claim *recapitulation*, never improvement; hold that line in the abstract.
- **Stereochemistry invisible** (caveat 6.6); eutomer selection is routine med chem.
- **One idea per hypothesis** — R's uniqueness 0.238, no demonstration of k contrasting intents on
  one input [TBD-4]. A chemist wants three options.
- **No chemist panel exists yet** [TBD-1], so every "usefulness" claim is unsupported; the only
  sound-explanation figure we hold is 0.08 from model labellers.

**Register additions proposed:** TBD-8 warhead retention (largest-fragment); TBD-9 SA / retro
accessibility; TBD-10 prior-art novelty of generated non-reference molecules; TBD-11 enumeration
baseline + recovery by edit size; TBD-12 site-of-change vs cited-residue agreement.
