# Covalent Generation — Multi-Day Status (2026-05-24 → 25)

## TL;DR

Goal: build a generative model that produces covalent inhibitors with the warhead chemistry preserved END-TO-END (no post-hoc patches), with **soft geometric constraints** (chemistry-aware D-arm) + **chemistry conditioning** (C-arm).

After ~3 model architectures × 4+ variants each, the headline is:

| Model | Inpaint | Atom-aware? | Bond-aware? | As-output acrylamide | Why |
|---|---|---|---|---|---|
| **PocketFlow** | warhead-seeded autoregressive | ✅ atom-by-atom | ✅ bonds-as-tokens | **16.7%** (2/12 valid) | Architecture supports it E2E |
| DrugFlow | flow-matching + Markov bridge bond | ✅ via hooks | ❌ stripped by `build_molecule` | 0% (3/25 sanitize) | Postproc re-infers bonds from geometry |
| DiffSBDD | atom diffusion + OpenBabel post-hoc | ✅ via inpaint | ❌ post-hoc inference | 0–4% (post-patch: 37.5%) | Architecture limited |

**PocketFlow is the working covalent-aware path.** DrugFlow has the architectural support but its `build_molecule` step needs patching. DiffSBDD's atom-only formulation is fundamentally limited (confirmed by our earlier experiments).

## Comparison table — all 11 methods, no post-hoc patch

| Method | N | Skeleton % | As-output acrylamide % |
|---|---|---|---|
| DiffSBDD M0 vanilla | 25 | 0% | 0% |
| DiffSBDD M1 hard inpaint | 24 | 21% | 0% |
| DiffSBDD M2 iterated inpaint (resamplings=50) | 24 | 42% | 4.2% |
| DiffSBDD M3 baseline finetune (CovBinder) | 24 | 8% | 0% |
| DiffSBDD M3 jitter_small | 24 | 0% | 0% |
| DiffSBDD M3 jitter_med | 25 | 4% | 0% |
| DiffSBDD M3 jitter_large | 25 | 4% | 0% |
| DrugFlow M0 vanilla | 25 | 0% | 0% |
| PocketFlow M0 ZINC-pretrained | 25 | 8% | 0% |
| **PocketFlow Inpaint** ★ | 12 | 8% | **16.7%** |
| DrugFlow Inpaint v1 (hooks) | 25 | 0% | 0% (build_molecule strips) |

★ = covalent-aware method with internal bond chemistry, not post-hoc patches.

## What's running in background

| Where | What | Status |
|---|---|---|
| V100 | DrugFlow C+D finetune (10 epochs CovBinder) | step 125, loss 0.48, **43% skip rate** (concerning) |
| T4 | PocketFlow extended smoke (8000 steps single example) | running, ~22 min total |
| Local | QA agent reviewing both inpaint implementations | running |

## Day 2 priorities

1. **Patch DrugFlow's `build_molecule`** to accept `force_bonds` list for warhead → unlock the bond-aware advantage
2. **Build real CovBinder→PocketFlow LMDB** preprocessing — current PocketFlow training is single-example smoke, not generalizing
3. **Diagnose DrugFlow training skip rate** (43% is too high — likely PDB parsing fails on most CovBinder records)
4. **Multi-warhead chemistry conditioning at inference** — currently inpaint hardcoded to acrylamide; let user choose acrylamide/chloroacetamide/vinyl-sulfone via C-arm token override
5. **Boltz cofold + Tier 2 eval** on top PocketFlow Inpaint candidates (d_SG, Bürgi-Dunitz, hinge H-bond, pocket fit)

## Honest limitations

- Sample sizes are smoke (25 mols each) — need 200-500 for stat significance
- "12/25 valid" for PocketFlow Inpaint means 48% completion. Need to diagnose the other 13 failures
- PocketFlow Inpaint C-arm bias is broadcast over all ligand atoms — not per-atom targeted (Day 2)
- D-arm in both implementations is data-side jitter (regularization), not a proper MDN log-prob penalty (Day 2)
- No real CovBinder generalization yet — both background trainings are still "extended smoke"
- No multi-target validation (only ZAP70 Cys346 tested so far)

## Architectural conclusion

**Bond awareness must live INSIDE the model AND survive through to the output.** DiffSBDD fails at the architecture level (no bonds in the latent). DrugFlow fails at the I/O level (bonds in the latent but stripped by `build_molecule`). PocketFlow succeeds because its autoregressive nature commits bond decisions atomically alongside atoms — there's no "rebuild from geometry" step.

For Day 2, the highest-ROI engineering move is patching DrugFlow's `build_molecule` — that should unlock DrugFlow's flow-matching efficiency + native bonds and probably beat PocketFlow's 16.7% number significantly.
