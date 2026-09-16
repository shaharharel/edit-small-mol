# CovalentFormer — HANDOFF

**Written at commit `d31402d`. Read this before touching anything.**

The goal is not sophisticated covalent descriptors. It is a **generative editor that follows a
steering instruction** a medicinal chemist would actually give during hit-to-lead. Every decision
below was made against that test, and several params were killed by it.

---

## 1. STATE: what is on disk right now

| artifact | path | rows | status |
|---|---|---|---|
| ligand pairs | `data/covalent_final/pairs_mw800.jsonl` | 1,470,338 | **CONTAMINATED — rebuild first** |
| pocket pairs | `data/pocket_pairs/pairs.jsonl` | 5,758 | ready to label |
| planarity labels | `data/labels/planarity_by_molecule.csv` | 36,446 molecules | valid, but re-derive on the clean pool |
| covalent filter | `stage0/covalent_filter.py` | — | **validated 9/9, ready to use** |
| manifest | `data/CANONICAL_DATASETS.json` | — | D1–D11, read it |

**Nothing is running. No GPU is booked. `ai-chem`, `ai-chem2`, `ai-gpu-a100` are TERMINATED.**

Everything under `data/chembl36_pairs_v3/` and `data/chembl36_pairs_v4*` is **SUPERSEDED**. Do not
train new arms on it.

---

## 2. THE BLOCKER: the corpus is 37.7% non-covalent

Measured on 16,348 unique molecules of `pairs_mw800.jsonl`:

- **15.16%** genuine terminal acrylamide
- **37.69%** match a contaminant class — 2.5 contaminants per real warhead

It contains **sunitinib** (a marketed *non-covalent* kinase inhibitor), nintedanib's oxindole,
rhodanines/TZDs (textbook PAINS), and N-ethylmaleimide (a thiol-capping *reagent*).

Cause: `build_covalent_union.py` used `[CX3]=[CX3][CX3](=O)[NX3,OX2]`, which does not separate a
**terminal vinyl** (designed TCI warhead) from a **β-substituted or aryl-conjugated** alkene.
Patterns were written without anyone looking at what they matched.

**Fix is built and tested:** `stage0/covalent_filter.py`, `PANEL_VERSION cf-warheads-v1-2026-09-15`.
15 accepted warhead classes, 7 rejected contaminant motifs, conjugation status returned separately.
Validated on named drugs — ibrutinib / osimertinib / afatinib / sotorasib pass; nirmatrelvir /
sunitinib / cinnamamide / N-ethylmaleimide / rhodanine rejected, each with the reason named.

**The panel version IS part of every param definition.** Two corpora built under different
`PANEL_VERSION` are not poolable. Bump it on any edit.

---

## 3. PARAM STATUS — what to build, what is dead

### BUILD THESE

| param | what it is | why |
|---|---|---|
| **warhead_planarity** | Michael-acceptor dihedral C=C–C(=O)–N | **The only param with a demonstrated steering effect**: unconditioned 61.8–63.5° → conditioned **2.62° median**. Producer pinned to `scripts/compute_planar_2d_e2.py`, verified to reproduce the stored manuscript column to **1e-6 on 215/215 rows** |
| **acyl_N_motif** | categorical {ArNH-, alkyl-NH-, N-Me-aryl, endocyclic ring-4/5/6, fused/indoline} | The real named move: exocyclic NH → endocyclic ring N. This is the ibrutinib / sotorasib / futibatinib warhead in one variable |
| **linker_atom_count** | atoms between the electrophilic C and the first ring, walking **through the carbonyl and acyl heteroatom toward the Murcko scaffold** | "Reading A done correctly" — the direction restriction makes reference-jumping *impossible by construction* rather than filtered at 97% cost |
| **michael_subst_class** | categorical {terminal, β-aryl/conjugated, β-amino, α-cyano, α-substituted, ring-embedded} | Separates ibrutinib from a rhodanine. Currently collapsed *into* span, which is why span appeared to move |
| **pocket ×3** | buried SASA, pocket occupancy, shape complementarity | On the 5,758 pocket pairs. **Buried SASA first** — it's the robust one; occupancy needs a pocket detector, complementarity needs surface normals |

**Precondition on every linker/placement instruction: warhead class AND conjugation status must be
IDENTICAL between A and B.** In a sampled DOWN set, 16/20 pairs flipped conjugation. Without this
you are training warhead swap under a linker instruction.

### DEAD — do not rebuild, do not train

**`warhead_span`** (bond path to nearest ring system) and **`warhead_linker_flex`**.

Killed by a covalent medicinal chemist on four measurements:

1. **Not a length — a 3-level categorical.** span=1 40.4%, span=3 27.5%, span=4 25.7% = 93.6%.
   There is no ladder to climb.
2. **The span=1 bucket is not covalent chemistry** — 78% arylidene/β-aryl Michael acceptors.
3. **Every approved TCI sits in one unit of it.** ibrutinib/zanubrutinib/acalabrutinib/ritlecitinib/
   sotorasib/adagrasib/futibatinib = 3; osimertinib/afatinib/neratinib = 4; dacomitinib = 2.
4. **The path points the wrong way half the time.** For span ≥ 2 it runs toward the scaffold in only
   **49.6%** of molecules; the rest runs *outward along the warhead tail*. `min` over directions
   silently switches between two opposite quantities. **The defect is in the definition, not the data.**

The unanswerable demonstration: afatinib (span 4) and dacomitinib (span 2) are the same EGFR series
making the same move — a basic amine tail on the β-carbon. One tail is acyclic, one cyclic. The param
disagrees with itself inside a single marketed drug series. **You cannot filter your way out of a
definition.**

The *design axis* those params reached for is real and survives as `linker_atom_count` +
`acyl_N_motif`. Two formulas were lost, not two directions.

### Also dead
- `extent` / warhead_reach_3d — six independent constructions put it at or below its size-matched control
- `d_cys` — it is the covalent **bond length**. CYS 1.800 Å, sd 0.106, and every residue type hits its
  textbook value. Nothing to steer
- `wclass` — user's call; warhead swap is the role-token machinery that already works

---

## 4. ORDER OF OPERATIONS

1. **Rebuild the corpus under `cf-warheads-v1`** (~10 min). Re-run union → all-vs-all → filter with
   `covalent_filter.classify()` gating **both** sides. Expect the 37,035-molecule pool and the 1.47M
   pairs to drop substantially. **Smaller and actually covalent is the correct direction.**
2. **Verify contamination is near zero** on the new set *before labelling anything* (~10 min).
3. **Label in parallel** (boot the machines first):
   - **local** — pocket params on 5,758 pairs, buried SASA first
   - **ai-chem** — `acyl_N_motif` + `michael_subst_class` (pure SMARTS, cheap)
   - **ai-chem2** — `linker_atom_count` + planarity relabel
4. **Split** — molecule-disjoint, **both endpoints unseen**, stereo/isotope-blind key.
5. **Train** on ai-gpu / ai-gpu2 / a100.
6. **GENERATE AND SCORE.** Never once run. The only test that answers whether anything steers.

Steps 1–4 are ~3 hours of mechanical execution. The judgement calls are made and recorded.

---

## 5. FIXES OWED, INDEPENDENT OF EVERYTHING ABOVE

- **Rotor SMARTS counts the amide C–N as rotatable.** `C=CC(=O)Nc1ccccc1` → 3 by ours, 2 by RDKit
  default *and* Strict. Every acrylamide carries +1. Cancels in deltas; inflates every **level** quoted.
- **`flex` uses one arbitrary shortest path.** 12.83% of molecules have >1 equal-length path; flex
  changes with the choice in 6.72%, mean spread 1.10 against a median of 3 — ~30% of the median,
  decided by RDKit traversal order. Fix: **union of all shortest paths**.
- **Tautomers decide whether the anchor EXISTS** — the span value is robust (0.73%) but 12.27% of
  molecules lose the warhead or ring under canonical tautomer. 228,445 pairs are cross-source
  (ChEMBL × CovInDB), each database recording its curator's tautomer, so this is a **silent coverage
  bias correlated with `pair_source`**. Canonicalise once at build; stamp the enumerator.
- **`valid_frac_realised`** in `build_steer_v4.py` measures post-stratification row share, not the
  split fraction. Two of three params read *below* the request, which an overshoot cannot produce.

---

## 6. TRAPS SPECIFIC TO THIS CODEBASE

**Nine bugs tonight were the same shape: code that runs, writes a valid-looking file, and reports a
confident wrong number.** Not crashes. Assume this is the default failure mode.

- **Smoke-test everything on 200 rows and READ THE OUTPUT.** The planarity labeller reported
  *100.00% acrylamide, 100.00% embed_ok* — a dict unpacked as a tuple. A perfect rate on two
  independent quantities is a parse error, not a result. Four seconds to catch.
- **If a result looks too good, hunt the artifact.** CYS burial sd 58.8 against mean 31.8 was 18 NMR
  files whose MODELs were being stacked.
- **A stalled job looks identical to a slow one.** Check CPU time, not log mtime — several producers
  buffer all output until exit. And check the *right PID*: I once declared a healthy job dead by
  piping `ps` through `tail -1` and reading the wrong row.
- **Never default a failed label to 0.** Return `None` and filter. A bare `except` emitting 0.0 turns
  an RDKit failure into a confident "perfectly planar" and a cohort of those reads as a real
  distribution.
- **Do not edit a producer while its arms are mid-flight** — that is an estimator straddle. Stamp
  instead, and re-push to every remote so md5s match before any chained job fires.
- **`git add` under `experiments/covalentformer/data/` is gitignored.** It silently drops the file if
  you swallow stderr. Use `-f` for the manifest.
- **`SendMessage` to `"main"` is REJECTED** — *"You are the main conversation."* Use `"team-lead"`.
  This lost five QA reports.
- **A comment asserting what the code does is not evidence it does it.** This happened ten times
  tonight, including in a stamp added to prevent exactly that (it reported a level *count* and
  asserted a value *set*; the values were wrong).

---

## 7. NUMBERS THAT ARE SETTLED — quote these, not older ones

- **Obedience floor (v3 attach_path, full valid, all arms converged): 0.668173** — A4, not A.
  The on-disk `floor_to_quote` of 0.6550 and filed 0.6544 are the superseded rule.
- **GAP_det, v3, full valid:** attach_path +0.0387/+0.0504/+0.0575 · wclass +0.0039/+0.0188/+0.0272 ·
  attach_flex +0.0053/+0.0099/+0.0121. **Dependence, not benefit.** Do **not** quote the ep0→ep2 rise
  as a decomposition — that is an arithmetic identity, and it reverses on attach_flex.
- **Planarity: 2.62° is the MEDIAN.** The mean is 26.02, the distribution is heavily skewed, and
  `embed_ok` is 7,174/10,000. Always quote median + embed-failure rate.
- **v4e leaks 0.00%** on every channel with working controls — but it is **core-disjoint, not
  protein-disjoint** (99.77% of valid targets appear in train). No new-pocket claim.
- **Scope, unavoidable:** v4e is **4.77% acrylamide**. It is generic ChEMBL matched-pair medicinal
  chemistry. The new covalent corpus exists to fix exactly this.

---

## 8. WHAT NOT TO CLAIM

- Do not call anything "linker steering" until a param survives chemist review. Currently none has.
- Do not describe the corpus as "covalent" until the filter has been applied and verified.
- Do not quote a pooled contamination rate: reference jumping is ~2% pooled and **38% on the rows
  that carry training signal**. Anyone quoting the pooled figure concludes the corpus is safe.

---

# ADDENDUM — 2026-09-15, corpus v2 (commits 5936ce7, 3adf0b2)

## DONE
1. **Filter wired into BOTH branches** (5936ce7). The CovInDB branch was writing pairs
   unconditionally — covalent by annotation, never checked by structure. Half of CovInDB
   (78,678 / 154,895) does not carry a structurally covalent warhead.
2. **Corpus rebuilt** (`stage0/rebuild_v2.sh`, ~90 s end-to-end):
   `data/covalent_final_v2/pairs.jsonl` — **1,101,089 pairs / 19,538 molecules**.
3. **Step-2 gate PASSED**: contaminant match **37.69% → 0.00%**, terminal acrylamide
   **15.16% → 43.22%**, sunitinib/NEM gone, MW median 482, 1.7% under 250 Da.
   NOTE: "100% accepted warhead" is TAUTOLOGICAL — the filter gated the build.
4. **Molecule params labelled**, 19,538 rows → `data/labels/molecule_params_v2.csv`.

## PARAM VERDICTS FROM THE LABEL DISTRIBUTIONS
- `acyl_N_motif` — **9 levels, well spread.** Exocyclic↔endocyclic populated both sides. BUILD.
- `linker_atom_count` — 15 levels, 84% in {2,3,0}. Directed; span's flip is unreachable. BUILD.
- `michael_subst_class` — **99.3% in two values.** Same degeneracy that killed `warhead_span`.
  Use as a pair PRECONDITION, do NOT train as a steering target.

## OPEN, AND IT BLOCKS THE PLANARITY ARM
`label_planarity.py` on the v2 corpus prints `acryl %` and `embed_ok %` **identical to 3 s.f.
at every checkpoint** (81.3/81.3, 90.7/90.7, 93.8/93.8). embed_ok is a SUBSET of acryl_match,
so equality means ETKDG never failed — against a filed prior of **71.7% embed_ok**. Either
embedding genuinely always succeeds on this cleaner pool, or embed_ok is being set from the
acrylamide match without the embed being checked. **RESOLVE BEFORE QUOTING ANY PLANARITY
NUMBER.** Read `_compute_2d_one` in `scripts/compute_planar_2d_e2.py` and confirm embed_ok is
written from an actual EmbedMolecule return code.

## NEXT
1. Resolve the embed_ok question above.
2. Pocket params on the 5,758 pocket pairs — buried SASA first. **Add θ_BD (Bürgi–Dunitz
   approach angle).** The manuscript's pose vector is (d_Sγ–Cβ, θ_BD, φ_planar); d_Sγ–Cβ is the
   formed C–S bond (1.797 Å, sd 0.106) so it is a constant, φ is validated, and **θ_BD has never
   been measured or killed** — it is the one live geometry channel with no verdict.
3. Pair-level deltas + molecule-disjoint split (both endpoints unseen).
4. Train: `acyl_N_motif`, `linker_atom_count`, planarity. NOT span, NOT flex,
   NOT michael_subst_class.
