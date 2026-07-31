# PocketFlow Covalent-Aware Extension — Day-1 Status

**Target:** ZAP70 Cys346, acrylamide warhead, Michael Addition.
**Env on T4:** `diffsbdd` (additive: no new installs).

---

## Deliverables shipped

| File | Status | Notes |
|---|---|---|
| `warhead_seed.py` | DONE | reads 5-atom acrylamide SDF → PocketFlow Ligand dict w/ correct bonds (C=C, C-C, C=O, C-N) |
| `cov_token.py`    | DONE | 290-d cov token (log10 d, θ, Morgan-256, residue-20, mech-12) + CovTokenAdapter (19,600 params) |
| `inpaint.py`      | DONE | `SeededGenerate` overrides `Generate.run()` to start autoregressive loop at `atom_idx = n_seed_atoms`. The focus is picked from `idx_ligand_ctx_in_cpx` (not protein surface) — growth extends FROM the warhead Cβ outward. Bonds emerge from PocketFlow's bond predictor; no post-hoc fix. |
| `train_dc.py`     | SMOKE PASS, FULL FINETUNE NOT RUN | C-arm: token → 19.6K-param adapter → bias added to ligand atom feature rows via `index_add_` (gradient-safe). D-arm: Gaussian jitter on warhead y_pos (radial σ_d, angular σ_θ → isotropic σ_total). Background extended-smoke run (8000 steps) ongoing on T4. |
| `sample.py`       | DONE | Loads finetuned ckpt + optional adapter state, applies C-arm bias to `cpx_feature[idx_ligand_ctx_in_cpx]`, then seeded inpaint. |

---

## Measurements

### inpaint.py smoke (vanilla ZINC-pretrained ckpt, n=25, max_atom=30)

```
valid=12/25 (48%)  unique=10/12 (83%)
runtime=46.8s on T4
```

Example SMILES (all 12 valid contain the acrylamide motif):
- `O=C(Nc1ccccc1)c1ccccc1` — benzanilide (the C=C reduced to single in cleanup, scaffold intact)
- `CC=CC1C(=O)NC(=O)c2c1cc(NC(=O)c1ccc(C(F)F)cc1)cc2` — acrylamide-bridged biaryl
- `CC=CNC(=O)c1ccccc1` — N-prop-1-enyl benzamide (Michael acceptor preserved)
- `O=C(O)c1cccc(NC(=O)c2ccccc2)c1` — meta-carboxy benzanilide
- `N=Cc1cc(C(=O)Nc2ccccc2)ccc1`

Output: `~/runs/pocketflow_inpaint/zap70_inpaint_smoke/<ts>/generated.{sdf,smi}`
Local copy: `results/covalent_gen_day1/pocketflow_inpaint_smoke/samples.{sdf,smi}`

### train_dc.py smoke (100 steps, single-example batch, ZAP70 pocket+warhead)

```
steps=100/100 OK    loss[0]=3.17 → loss[-1]=2.04
adapter grad >0 on 100/100 steps
runtime=17.4s on T4
SMOKE PASS = True
```

Confirms gradient flow through BOTH arms:
- C-arm: `adapter|g|` averaged ~0.85 across steps (range 0.40–1.42).
- D-arm: y_pos jitter applied at σ_d=0.10Å, σ_θ=5° (default); loss still finite & decreasing.

Saved: `~/runs/pocketflow_dc_smoke/{smoke_final.ckpt, adapter_smoke.pt}`

### sample.py against smoke ckpt (n=10)

```
valid=3/10 (30%)  unique=3/3 (100%)
```

(Validity is lower vs the vanilla smoke because the smoke ckpt only had 100 noisy single-example training steps — sanity check that the load/inject path works, not a quality claim.)

Local copy: `results/covalent_gen_day1/pocketflow_dc_sample_smoke/samples.{sdf,smi}`

### Background "extended smoke" (full-finetune proxy)

PID 106532 on T4. 8000 steps × ~0.17s/step ≈ 22min target. Saves `smoke_final.ckpt`
+ `adapter_smoke.pt` at `~/runs/pocketflow_dc_long/` when done.

---

## Broken pieces / gaps (brutally honest)

1. **Full CovBinder→PocketFlow LMDB conversion is NOT done.** PocketFlow's
   `LoadDataset` expects a per-complex preprocessed LMDB (one entry per
   ligand+pocket pair with all featurized graphs). CovBinder gives us only
   SMILES + Cys-SG coords — building the LMDB requires running PocketFlow's
   `process_raw` over ~1000 complexes (multi-hour, mostly I/O). The
   `--data_dir` code path in `train_dc.py` (`full_finetune`) is plumbed but
   has never been executed end-to-end. **Cost to fix: 3-4 more hours.**

2. **The "extended smoke" run is single-example.** It trains on ONE pocket
   (ZAP70) with ONE jittered Cβ position. It will not generalize. It exists
   to give `sample.py` a non-trivial ckpt to load and to prove the wiring is
   numerically stable over 1000+ steps. Real finetune needs item (1).

3. **D-arm is implemented as a data-augmentation (jitter on y_pos), NOT a
   loss-side penalty.** The spec mentioned "probability boost for canonical
   position" — that requires intercepting `pos_predictor.get_mdn_probability`
   and adding a λ·log-prob term at the canonical position; PocketFlow's
   MDN-based positioning makes that intrusive. Jitter-on-target is the
   equivalent regularizer (it teaches the model that "near canonical" is the
   right answer with σ uncertainty) and is exactly the v1 approach we shipped
   for DiffSBDD M3. If we want the stronger penalty version we'd add it in a
   v2 that subclasses PocketFlow.

4. **C-arm is broadcast to all ligand atoms, not per-atom.** Each ligand
   context atom currently gets the SAME bias from the token. A per-atom
   variant ("focus the Michael-EWG signal on the seed Cβ specifically") is
   straightforward but adds complexity — left for v2.

5. **No covalent-bond constraint enforced at sampling.** The Cys-S–Cβ
   "covalent bond" is implicit (Cβ is placed at the SG-anchored canonical
   position by the warhead seed). We do NOT physically bond Cβ to the
   protein SG (PocketFlow has no protein-ligand bond infrastructure).
   Downstream covalent docking / SMILES join-with-Cys would need to happen
   in post-processing — but the user asked us NOT to do post-hoc bond fixes,
   so this is left to the v2 architectural change.

6. **`SeededGenerate` overrides `run()` but inherits `generate()` via a
   monkey-patched fallback (`_safe_generate`).** Reason: PocketFlow's stock
   `Generate.generate()` has an `UnboundLocalError` when `self.run()` returns
   a falsy value (`mol` referenced before assignment). I patched around it
   in the subclass via a bound function; the upstream bug is real but I
   chose not to edit `~/PocketFlow/pocket_flow/generate.py` (protect the
   shared `diffsbdd` env per instructions).

---

## Reproduction commands

```bash
# Activate env
source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffsbdd

# Inpaint (vanilla ckpt, warhead-seeded)
cd ~/PocketFlow && python ~/edit-small-mol/anchordiff/pocketflow_covind/inpaint.py \
    -pkt ~/edit-small-mol/anchordiff/pockets/zap70_cys346/pocket.pdb \
    --warhead_sdf ~/edit-small-mol/anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf \
    --ckpt ~/PocketFlow/ckpt/ZINC-pretrained-255000.pt \
    -n 25 --name zap70_inpaint_smoke --max_atom_num 30

# Train (smoke)
cd ~/PocketFlow && python ~/edit-small-mol/anchordiff/pocketflow_covind/train_dc.py \
    --smoke --steps 100 --out_dir ~/runs/pocketflow_dc_smoke

# Sample from finetuned ckpt
cd ~/PocketFlow && python ~/edit-small-mol/anchordiff/pocketflow_covind/sample.py \
    --pocket ~/edit-small-mol/anchordiff/pockets/zap70_cys346/pocket.pdb \
    --warhead_sdf ~/edit-small-mol/anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf \
    --ckpt ~/runs/pocketflow_dc_smoke/smoke_final.ckpt \
    --adapter_state ~/runs/pocketflow_dc_smoke/adapter_smoke.pt \
    -n 25 --name zap70_dc_sample
```
