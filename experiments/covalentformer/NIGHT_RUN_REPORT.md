# CovalentFormer night run — results

All numbers recompute from artifacts in `experiments/covalentformer/data/` and
`.../data/replicates/`. Anything that samples was seeded and replicated; single-draw numbers are
marked as superseded.

---

## The finding

**A discrete conditioning bit steers generation decisively. A continuous distance at the same
injection point does not.**

| | seeds | mean | significance |
|---|---|---|---|
| **H1** geometry shuffle | 4 | **+1.5 pts** | all p ≥ 0.33 — **NULL** |
| **H3** label r=1 vs r=0 | 3 | **+19.6 pts** | all p < 1e-4 — **HOLDS** |

13× effect-size gap, same two-row cross-attention site, same model, same endpoint.

> **THIS SECTION IS STALE (mtime Sep 13 11:04) AND IS CORRECTED BELOW.** It is left in place
> because it is what was believed at the time, but nothing in it should be quoted. Three specific
> defects, all found by later recomputation from disk:
>
> 1. **These are NOT Phase A numbers.** The sweep loads a `tierA_v3` GeomEncoderDecoder
>    (`ckpt_v3*`, Sep 13 02:25–03:41); the first Phase A role checkpoint does not exist until
>    Sep 14. The number predates Phase A by ~18–28h and is a different architecture (#94).
> 2. **"+0.510" is 97.9% pure length.** It is the mean of `q1_rho_reach` over
>    `data/replicates/sweep_rep_{101,202,303}.json` (0.50981/0.51188/0.50732 → +0.50967). The
>    matched `q1_rho_bonds` mean is +0.49917, so requested-d→reach exceeds requested-d→BOND COUNT
>    by only +0.0105. `sweep_test`'s own docstring labels this Q1 as "(WEAK — could be a pure
>    length lookup table)". It is.
> 3. **"+0.280 achievable ceiling" IS UNSOURCED.** It appears in this file and NOWHERE ELSE — no
>    script computes it, no artifact records it. It is a THIRD ceiling denominator after the
>    retracted +0.9555 and the derived +0.9860 (#132), and the "50% of achievable" reading was
>    already WITHDRAWN by task #25 — the withdrawal simply never reached this file. **Do not use
>    +0.280 as the ceiling end of the achievable range until something computes it.**
>
> The shape statistic is Q2 (`pooled_within_rho`), mean **+0.1389 (sd 0.0175)**, and per #154 its
> standing verdict against the length-matched null is **WITHIN / unresolvable**, not "above".

| H2 sweep (STALE — see box above) | 3 seeds | |
|---|---|---|
| Q1 requested d → achieved reach | +0.510 ± 0.002 | tierA_v3, not Phase A; 97.9% length |
| Q1 requested d → bond count | +0.505 | nearly identical to Q1 |
| Q2 shape at fixed length | +0.139 ± 0.018 | ceiling denominator UNSOURCED — do not quote a % |

Q1 and Q2 being nearly equal means the response is carried almost entirely by **linker length**.
Changing length changes *which* molecule is written but not *whether* it reaches, because ~28% of
the fragment vocabulary satisfies any given gap.

**Relevance to CovaCraft-v2-cond.** That architecture conditions on a near-constant continuous pose
vector (d ≈ 1.85 Å — a *formed* C–S bond, i.e. the product) through this same injection design, with
no ablation. These measurements show that even a **varying, informative** continuous vector at that
site yields no behavioural control.

---

## Five architectural hypotheses, all refuted by measurement

| hypothesis | verdict | evidence |
|---|---|---|
| attention starvation | REFUTED | geometry attended **more** than the label (0.72×) |
| influence weakness | REFUTED | geometry shifts logits **more** (0.883 vs 0.597) |
| input encoding (RBF) | REFUTED | Q1 +0.204, Q2 +0.019 — worse on **both** axes at matched loss |
| "length is loss-optimal" | REFUTED | only ~40% of reaching fragments within ±1 bond at d = 6–7 Å |
| objective is the wall | REFUTED | NLL prefers reaching targets 61.3%, **4.1 σ** (sampling-independent) |

`k=8` (conditioning rows repeated 8×) met its pre-registered bar — GAP +42% — but the advantage
*closed* across training (2.1× → 1.4×) and its shuffled arm (43.5%) beat k=1's **true** arm (41.0%),
so most of its gain was geometry-independent.

---

## The largest remaining lever is decode-time

**94.5%** of 8-candidate pools already contain a reaching molecule. The model emits one **~41%** of
the time. It generates reaching molecules almost always and cannot identify which.

Oracle reranking closes this at **zero property cost** — every descriptor within noise of a random
pick from the same pool, Tanimoto 0.54 (i.e. it selects a genuinely different molecule). It is
circular for evaluation, so it is a **deployment recipe, not evidence**.

---

## Verified infrastructure

- **Dataset (tierA_v3)**: fingerprint-alone AUROC 0.483, geometry-alone 0.492, **joint 0.913 on a
  fragment-disjoint split**. The task is extractable; neither channel leaks alone.
- **Conditioning information**: 0.00% of anchors have a single target (v1 had 99.49%); 95,830
  (anchor, gap) pairs carry both labels and 100% have differing targets.
- **Oracle**: non-degenerate (0/214 decoys pass), spatial resolution **4.0 Å** half-max.
- **Endpoint floor**: a random in-vocabulary fragment reaches **24.6%** of the time.

---

## My own errors, and what caused them

| reported | corrected | cause |
|---|---|---|
| shuffle "+4.3 pts, near-miss" | **+1.5, clean null** | unseeded single draw |
| Q2 "1% of variance" | ~~50% of achievable~~ **ALSO WRONG — withdrawn (#25)** | the *correction* used a third unsourced denominator (+0.280); Q2 is +0.1389 and its ceiling is uncomputed |
| best-of-N "+16.5 pts, p=0.0011" | **unstable, +4.1 on rerun** | unseeded single draw |
| "attention starvation" | **misdirected, not starved** | attention mass ≠ influence |
| MW cap 520 "fixes drift" | **overcorrected; 600 is right** | fixed toward *better*, not toward *target* |

Two root causes, both now written up as rules (task #23):

1. **Measure both ends of the achievable range** — floor *and* ceiling — not just the null. The
   project did this correctly for the oracle (69.7% recall vs a 71.6% ceiling = 97.3% of
   achievable) and I failed to apply it to new endpoints.
2. **Seed torch and replicate anything that samples.** Every generation-based number was one draw
   until late in the run. I also twice mis-scoped which results were affected — audit by reading
   the call graph, not by recalling what feels sampled.

Seven bugs total; four produced *plausible wrong numbers* rather than errors. The worst:
`load_model()` ignored `cond_repeat`, so every k=8 evaluation would have silently measured a k=1
configuration — `load_state_dict` succeeds because it is not an `nn.Parameter`.

---

## Open — these need a decision, not another autonomous tick

1. **Property drift I introduced.** `build_cond.py` cross-products give QED 0.28 / MW 578 against
   real CovInDB parents at 0.39 / 494. The model reproduces its targets exactly, so this is a data
   defect. `--max-mw 600` is implemented and calibrated (→ 486 / 0.38) and does not reintroduce a
   label leak. Rebuilding discards 42% of recombinations and invalidates direct comparison with
   every number above.
2. **Addressable programs: 6, not 244** — 4 of them EGFR, across 3 targets. Will not support the
   cluster bootstrap as planned. 81% of real covalent programs improved by *core hop*, which a
   warhead-directed editor structurally cannot reproduce.
3. **DPO pairs: 764** survive the 4.0 Å resolution filter, **1** of them in the 3–4 bond bin where
   64/139 real inhibitors sit.
4. **Oracle tolerance.** Measured (null 24.1% → 9.1% as it tightens); deliberately unchanged. The
   cost side — how many real crystal poses each setting rejects — is unquantified, and must be
   before any change.

---

## What is *not* claimed

No statement about molecule quality, potency, or binding. Every endpoint here is scored by the
reachability oracle the model was trained against; these measurements show whether the
**conditioning is wired**, not whether the molecules are good. With the property drift above, no
absolute claim about generated chemistry is supportable at all.
