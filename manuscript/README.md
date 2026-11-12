# manuscript/

Started fresh 2026-11-12. The covaGen paper in `paper/manuscript/` is a different paper and is
untouched.

```
EVIDENCE.md          every number the paper is allowed to use, with status and the hard caveats.
                     If a number is not in here it does not go in the paper -- write [TBD-n].
main.tex             10pt article; build with `latexmk -pdf main.tex`
sections/            00 abstract, 01 intro, 02 methods, 03 results, 03b related work,
                     04 discussion + limitations + ethics, 05 appendix (placeholder register)
references.bib       every entry carries note = {VERIFIED | PARTIAL | UNVERIFIED}.
                     Nothing marked UNVERIFIED may be typeset in a submission.
reviews/             the five expert memos the draft was assembled from
```

## The three decisions the cohort converged on

1. **The title claim changed.** "Pocket-aware" promises a molecule-level effect our own ablation
   does not support (groundedness 1.000 vs 0.432, but TC→reference 0.679 vs 0.680 and a wrong
   pocket costs 0.013 Tc). The paper claims **grounded attribution** and leads with that asymmetry
   instead of hiding it.
2. **The oracle result is reported as an upper bound, not an end-to-end gain.** R→M retains 8.7% of
   the oracle-over-mismatch gap. The leakage decomposition (TBD-17) decides what the 0.919 means and
   is the highest-value open experiment; it needs no training and no new structures.
3. **Everything pending is a placeholder with a stated hypothesis** (appendix register), so an
   experiment that fails costs a revert rather than a rewrite.

## Open before submission

- Every `note = {UNVERIFIED}` in `references.bib` must be opened or the entry removed.
- TBD-14: the split sentence. The corpus builder inherits a target-level val fraction (32 val
  targets); the holdout the arms were scored on is molecule-level. One authoritative statement for
  each is required and the 32-vs-21 target counts must be reconciled.
- The groundedness ceiling: EVIDENCE §4a puts the chemists' own sentences at 1.000, while the
  scorer argues the ceiling is ≈0.90 because an analogue's new atoms reach residues the input never
  touched. One recomputation, one number, printed against every groundedness figure.
- Figures are not drawn yet. Three are specified in `reviews/02_medchem_expert.md` §6 and four in
  `reviews/05_chemistry_advisor.md` §5; they overlap into roughly four. Budget ~1.5 pages.

## Length: decided, do not re-cut

Body is 12 pages, ~13.5 with figures, against a nominal 10. **The decision is to keep all seven
results subsections at full length** rather than move ZAP70 and the base-model ablation to the SI,
which is what the area chair's cut order would have done. That fits arXiv and a journal with a
generous main text, and is over for a 10-page ML venue; if a venue forces the issue, the cut order
in `reviews/04_area_chair.md` §5 is the one to follow (ZAP70 -> SI, LLM ablation -> SI,
agentic-vs-extracted -> SI), and never cut the pocket null, mismatch < `none`, or the 60.7%
caveat.
