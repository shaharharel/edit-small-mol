"""Sample a clean POST-RL cohort from the distilled-iptm RL checkpoint, then
score and stratify by iptm to test the 'fewer but better' hypothesis.

The 21,274-mol cohort previously analyzed for ExpB was the DURING-RL training
stream (samples drawn from the policy at each of 50 steps).  This script
produces an honest POST-RL cohort: 5,500 mols drawn from the final policy
checkpoint using the same 220-seed pool (+ Mol1 anchor) used for all other
PPO/DPO post-RL cohorts.

Outputs:
  - data/exp_ppo_v2plus/cohort_distilled_iptm_postRL.csv
  - data/exp_ppo_v2plus/cohort_distilled_iptm_postRL.csv.summary.json
  - results/paper_evaluation/exp_distilled_iptm_stratified.md
"""
from __future__ import annotations

import argparse
import csv as csvmod
import gc
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_fast_geom_surrogate"))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

# REINVENT4 sampler primitives
from reinvent.runmodes.create_adapter import create_adapter
from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.chemistry import conversions
import torch.utils.data as tud

# Local scorers (in-process, no REST server overhead)
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP
from fast_geom_scorer import FastGeomScorer


# -------------------- Constants --------------------
DEFAULT_CKPT = PROJECT_ROOT / "models" / "rl_checkpoints_b" / "distilled_iptm_stage1.chkpt"
DEFAULT_SEEDS = PROJECT_ROOT / "data" / "mol1_rl_seeds" / "seed_zap70_all_plus_mol1_clean.smi"
DEFAULT_FILM_PT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
DEFAULT_ANCHOR = PROJECT_ROOT / "data" / "fast_geom_surrogate" / "anchor_frame.npz"

ANCHOR_MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

LADDER_PATS_DEF = [
    ("C=CC(=O)N1Cc2ccccc2C1", 1.00),       # THIQ-acrylamide (exact)
    ("C(=O)C=C[#7]",          0.75),       # vinyl-amide-N
    ("C=CC(=O)N",             0.50),       # generic acrylamide
    ("[CX3]=[CX3][CX3]=[OX1]", 0.25),      # any enone
]
THIQ_SMARTS = "C=CC(=O)N1Cc2ccccc2C1"

# Composite-v2 sigmoid + geometric mean weights (must match reward used during RL)
SIGMOID_LOW, SIGMOID_HIGH, SIGMOID_K = 5.5, 7.5, 0.5
W_FILM, W_THIQ, W_QED = 0.5, 0.4, 0.1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------- Sampling --------------------
def load_seeds(path: Path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split()[0])
    return out


def standardize_list(smilies, randomize=True):
    out = []
    for s in smilies:
        try:
            s_std = conversions.convert_to_standardized_smiles(s)
        except Exception:
            s_std = s
        if randomize:
            try:
                mol = conversions.smile_to_mol(s_std)
                if mol is not None:
                    s_std = conversions.mol_to_random_smiles(mol, isomericSmiles=True)
            except Exception:
                pass
        out.append(s_std)
    return out


def sample_n(adapter, seeds, n_total, device, batch_size=64, randomize=True):
    out_inputs, out_outputs, out_nlls = [], [], []
    n_done = 0
    adapter.set_mode("inference")
    tokenizer = SMILESTokenizer()
    while n_done < n_total:
        bsz = min(batch_size, n_total - n_done)
        seed_subset = list(np.random.choice(seeds, size=bsz, replace=True))
        proc = standardize_list(seed_subset, randomize=randomize)
        dataset = Dataset(proc, adapter.get_vocabulary(), tokenizer)
        loader = tud.DataLoader(dataset, batch_size=bsz, shuffle=False,
                                collate_fn=Dataset.collate_fn)
        for src, src_mask in loader:
            src = src.to(device)
            src_mask = src_mask.to(device)
            sb = adapter.sample(src, src_mask, "multinomial")
            out_inputs.extend(list(sb.input))
            out_outputs.extend(list(sb.output))
            nlls = sb.nlls
            if isinstance(nlls, torch.Tensor):
                nlls = nlls.detach().cpu().numpy()
            out_nlls.extend(list(nlls))
            n_done += len(sb.input)
        if n_done % (batch_size * 4) < batch_size or n_done >= n_total:
            log(f"  sampled {n_done}/{n_total}")
    return out_inputs[:n_total], out_outputs[:n_total], out_nlls[:n_total]


# -------------------- FiLM scorer (in-process) --------------------
class FilmScorer:
    def __init__(self, model_pt: Path):
        log(f"loading FiLM model from {model_pt}")
        ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
        self.model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256],
                                  dropout=0.2)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.scaler = StandardScaler()
        self.scaler.mean_ = ckpt["scaler_mean"]
        self.scaler.scale_ = ckpt["scaler_scale"]
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.scaler.mean_)
        self.anchor_embs = ckpt["anchor_embs"]
        self.anchor_pic50 = ckpt["anchor_pIC50"]
        log(f"  FiLM ready: {len(self.anchor_pic50)} anchors")

    def score_batch(self, smiles_list):
        scores = [0.0] * len(smiles_list)
        valid_idx, valid_fps = [], []
        for i, smi in enumerate(smiles_list):
            if not smi:
                continue
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros(2048, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                valid_fps.append(arr)
                valid_idx.append(i)
            except Exception:
                continue
        if not valid_fps:
            return scores
        fps_arr = np.asarray(valid_fps)
        batch_embs = torch.FloatTensor(self.scaler.transform(fps_arr))
        n_anchors = len(self.anchor_pic50)
        with torch.no_grad():
            for j, oi in enumerate(valid_idx):
                tgt = batch_embs[j:j + 1].expand(n_anchors, -1)
                deltas = self.model(self.anchor_embs, tgt).numpy().flatten()
                scores[oi] = float(np.mean(self.anchor_pic50 + deltas))
        return scores


# -------------------- Composite & helpers --------------------
def _dsig(x):
    if x is None or not np.isfinite(x):
        return 0.0
    mid = 0.5 * (SIGMOID_LOW + SIGMOID_HIGH)
    span = SIGMOID_HIGH - SIGMOID_LOW
    z = (x - mid) / (span / 4.0)
    return 1.0 / (1.0 + math.exp(-z * 4.0 * SIGMOID_K))


def _gmean(vals, ws):
    ws_sum = sum(ws); eps = 1e-9
    log_acc = 0.0
    for v, w in zip(vals, ws):
        if v <= 0.0:
            return 0.0
        log_acc += (w / ws_sum) * math.log(max(v, eps))
    return math.exp(log_acc)


# -------------------- Main pipeline --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    ap.add_argument("--film_pt", default=str(DEFAULT_FILM_PT))
    ap.add_argument("--anchor", default=str(DEFAULT_ANCHOR))
    ap.add_argument("--n_samples", type=int, default=5500)
    ap.add_argument("--seed_subsample_n", type=int, default=25)
    ap.add_argument("--anchor_mol1", default=ANCHOR_MOL1)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_conf", type=int, default=3)
    ap.add_argument("--out_csv",
                    default=str(PROJECT_ROOT / "data" / "exp_ppo_v2plus"
                                / "cohort_distilled_iptm_postRL.csv"))
    ap.add_argument("--out_md",
                    default=str(PROJECT_ROOT / "results" / "paper_evaluation"
                                / "exp_distilled_iptm_stratified.md"))
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device selection: MPS for sampling, CPU for FiLM/fast_geom (scoring is CPU anyway)
    if args.device == "mps" and not torch.backends.mps.is_available():
        log("MPS not available, falling back to CPU")
        sample_device = torch.device("cpu")
    else:
        sample_device = torch.device(args.device if args.device != "cuda"
                                     or torch.cuda.is_available() else "cpu")
    log(f"sample device: {sample_device}")

    # ---- 1. Sample 5500 mols ----
    log(f"loading checkpoint {args.ckpt}")
    adapter, _, mt = create_adapter(args.ckpt, "inference", sample_device)
    assert mt == "Mol2Mol", f"expected Mol2Mol, got {mt}"

    seeds_full = load_seeds(Path(args.seeds))
    log(f"loaded {len(seeds_full)} seeds; subsampling {args.seed_subsample_n} + Mol1")
    chosen = list(np.random.choice(seeds_full, size=args.seed_subsample_n, replace=False))
    if args.anchor_mol1 not in chosen:
        chosen.insert(0, args.anchor_mol1)
    log(f"{len(chosen)} effective seeds")

    t0 = time.time()
    inputs, outputs, nlls = sample_n(adapter, chosen, args.n_samples, sample_device,
                                     batch_size=args.batch_size, randomize=True)
    log(f"sampled {len(outputs)} mols in {time.time() - t0:.1f}s")

    # Release sampler GPU memory before loading scorers
    del adapter
    gc.collect()
    if sample_device.type == "mps":
        torch.mps.empty_cache()

    # ---- 2. Local descriptors ----
    LADDER_PATS = [(Chem.MolFromSmarts(s), v) for s, v in LADDER_PATS_DEF]
    THIQ_PAT = Chem.MolFromSmarts(THIQ_SMARTS)

    valid_smiles, valid_idx, valid_mols = [], [], []
    for i, s in enumerate(outputs):
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        valid_idx.append(i)
        valid_smiles.append(s)
        valid_mols.append(m)

    n_total = len(outputs)
    n_valid = len(valid_smiles)
    log(f"valid {n_valid}/{n_total} ({100 * n_valid / max(1, n_total):.1f}%)")

    thiq_exact = np.zeros(n_total, dtype=bool)
    warhead_any = np.zeros(n_total, dtype=bool)
    ladder_val = np.zeros(n_total, dtype=np.float32)
    qed_val = np.zeros(n_total, dtype=np.float32)
    scaffolds = []
    fps = []
    for s, m, i in zip(valid_smiles, valid_mols, valid_idx):
        best = 0.0
        for pat, v in LADDER_PATS:
            if pat is None:
                continue
            if m.HasSubstructMatch(pat) and v > best:
                best = v
        ladder_val[i] = best
        warhead_any[i] = best > 0
        if THIQ_PAT is not None and m.HasSubstructMatch(THIQ_PAT):
            thiq_exact[i] = True
        try:
            qed_val[i] = QED.qed(m)
        except Exception:
            qed_val[i] = 0.0
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            sc = ""
        scaffolds.append(sc)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))

    # ---- 3. FiLM scoring (in-process, batched) ----
    film_scorer = FilmScorer(Path(args.film_pt))
    film_pic50 = np.zeros(n_total, dtype=np.float32)
    CHUNK = 256
    t0 = time.time()
    for start in range(0, len(valid_smiles), CHUNK):
        chunk = valid_smiles[start:start + CHUNK]
        chunk_i = valid_idx[start:start + CHUNK]
        sc = film_scorer.score_batch(chunk)
        for j, oi in enumerate(chunk_i):
            film_pic50[oi] = sc[j]
        if (start // CHUNK) % 4 == 0:
            log(f"  FiLM scored {start + len(chunk)}/{len(valid_smiles)}")
    log(f"FiLM done in {time.time() - t0:.1f}s")

    # ---- 4. Distilled iptm via FastGeomScorer ----
    log(f"loading FastGeomScorer from {args.anchor}")
    iptm_scorer = FastGeomScorer(args.anchor, n_conformers=args.n_conf)
    iptm_val = np.zeros(n_total, dtype=np.float32)
    t0 = time.time()
    for j, (s, oi) in enumerate(zip(valid_smiles, valid_idx)):
        try:
            v = iptm_scorer.score(s)
            if v is None or not np.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        iptm_val[oi] = float(v)
        if j and j % 250 == 0:
            log(f"  iptm scored {j}/{len(valid_smiles)} "
                f"(avg {(time.time() - t0) / j:.2f}s/mol)")
    log(f"iptm done in {time.time() - t0:.1f}s")

    # ---- 5. Composite v2 (no Tanimoto tail, slight under-estimate) ----
    composite_v2 = np.zeros(n_total, dtype=np.float32)
    for i in valid_idx:
        f = _dsig(float(film_pic50[i])) if film_pic50[i] > 0 else 0.0
        composite_v2[i] = _gmean([f, float(ladder_val[i]), float(qed_val[i])],
                                 [W_FILM, W_THIQ, W_QED])

    # ---- 6. Write per-row CSV ----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csvmod.writer(fh)
        w.writerow(["SMILES", "Input_SMILES", "NLL", "valid",
                    "ladder", "thiq_exact", "warhead_any",
                    "film_pIC50", "QED", "distilled_iptm", "composite_v2"])
        valid_set = set(valid_idx)
        for i in range(n_total):
            w.writerow([
                outputs[i], inputs[i], float(nlls[i]),
                int(i in valid_set), float(ladder_val[i]),
                int(bool(thiq_exact[i])), int(bool(warhead_any[i])),
                float(film_pic50[i]), float(qed_val[i]),
                float(iptm_val[i]), float(composite_v2[i]),
            ])
    log(f"wrote {out_csv}")

    # ---- 7. Summary metrics ----
    valid_film = film_pic50[valid_idx]
    valid_qed = qed_val[valid_idx]
    valid_ladder = ladder_val[valid_idx]
    valid_warhead = warhead_any[valid_idx]
    valid_thiq = thiq_exact[valid_idx]
    valid_iptm = iptm_val[valid_idx]
    valid_composite = composite_v2[valid_idx]

    n_uniq_scaff = len(set(s for s in scaffolds if s))
    scaff_per_mol = (n_uniq_scaff / max(1, len(scaffolds))) if scaffolds else 0.0

    if len(fps) > 500:
        sub_idx = np.random.choice(len(fps), 500, replace=False)
        sub_fps = [fps[i] for i in sub_idx]
    else:
        sub_fps = fps
    tans = []
    for i in range(len(sub_fps)):
        if not sub_fps[i + 1:]:
            break
        sims = DataStructs.BulkTanimotoSimilarity(sub_fps[i], sub_fps[i + 1:])
        tans.extend(sims)
    internal_mean_tc = float(np.mean(tans)) if tans else 0.0

    summary = {
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_rate": n_valid / max(1, n_total),
        "thiq_exact_rate": float(np.mean(valid_thiq)) if len(valid_thiq) else 0.0,
        "warhead_any_rate": float(np.mean(valid_warhead)) if len(valid_warhead) else 0.0,
        "mean_ladder": float(np.mean(valid_ladder)) if len(valid_ladder) else 0.0,
        "mean_film_pIC50": float(np.mean(valid_film)) if len(valid_film) else 0.0,
        "median_film_pIC50": float(np.median(valid_film)) if len(valid_film) else 0.0,
        "max_film_pIC50": float(np.max(valid_film)) if len(valid_film) else 0.0,
        "frac_film_ge_7": float(np.mean(valid_film >= 7.0)) if len(valid_film) else 0.0,
        "frac_film_ge_8": float(np.mean(valid_film >= 8.0)) if len(valid_film) else 0.0,
        "mean_QED": float(np.mean(valid_qed)) if len(valid_qed) else 0.0,
        "mean_distilled_iptm": float(np.mean(valid_iptm)) if len(valid_iptm) else 0.0,
        "median_distilled_iptm": float(np.median(valid_iptm)) if len(valid_iptm) else 0.0,
        "max_distilled_iptm": float(np.max(valid_iptm)) if len(valid_iptm) else 0.0,
        "frac_iptm_ge_0p85": float(np.mean(valid_iptm >= 0.85)) if len(valid_iptm) else 0.0,
        "mean_composite_v2": float(np.mean(valid_composite)) if len(valid_composite) else 0.0,
        "n_unique_scaffolds": int(n_uniq_scaff),
        "scaffolds_per_mol_valid": float(scaff_per_mol),
        "internal_mean_tc_sub500": internal_mean_tc,
        "ckpt": str(args.ckpt),
        "seeds": str(args.seeds),
        "stream": "post_RL",
    }
    if int(np.sum(valid_warhead)) > 0:
        wf = valid_film[valid_warhead]
        wq = valid_qed[valid_warhead]
        wi = valid_iptm[valid_warhead]
        summary["warhead_positive"] = {
            "n": int(np.sum(valid_warhead)),
            "mean_film_pIC50": float(np.mean(wf)),
            "mean_QED": float(np.mean(wq)),
            "mean_distilled_iptm": float(np.mean(wi)),
            "frac_film_ge_7": float(np.mean(wf >= 7.0)),
        }

    with open(str(out_csv) + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"wrote {out_csv}.summary.json")

    # ---- 8. Stratified analysis ----
    strat_rows = []
    for thr in (0.6, 0.7, 0.8, 0.85):
        mask = valid_iptm >= thr
        n = int(mask.sum())
        if n == 0:
            strat_rows.append({"iptm_thr": thr, "n": 0,
                               "acryl_pct": None, "thiq_pct": None,
                               "FiLM_mean": None, "QED_mean": None})
            continue
        strat_rows.append({
            "iptm_thr": thr,
            "n": n,
            "acryl_pct": float(100 * valid_warhead[mask].mean()),
            "thiq_pct": float(100 * valid_thiq[mask].mean()),
            "FiLM_mean": float(valid_film[mask].mean()),
            "QED_mean": float(valid_qed[mask].mean()),
            "iptm_mean": float(valid_iptm[mask].mean()),
        })

    # Top-N vs random-N comparison (N=82, matching prior during-RL analysis)
    N = 82
    top_n_cmp = None
    if n_valid >= N:
        top_idx = np.argsort(valid_iptm)[::-1][:N]
        rng = np.random.RandomState(args.seed + 17)
        rand_idx = rng.choice(n_valid, size=N, replace=False)
        top_n_cmp = {
            "N": N,
            "top_iptm": {
                "iptm_mean": float(valid_iptm[top_idx].mean()),
                "acryl_pct": float(100 * valid_warhead[top_idx].mean()),
                "thiq_pct": float(100 * valid_thiq[top_idx].mean()),
                "FiLM_mean": float(valid_film[top_idx].mean()),
                "QED_mean": float(valid_qed[top_idx].mean()),
            },
            "random": {
                "iptm_mean": float(valid_iptm[rand_idx].mean()),
                "acryl_pct": float(100 * valid_warhead[rand_idx].mean()),
                "thiq_pct": float(100 * valid_thiq[rand_idx].mean()),
                "FiLM_mean": float(valid_film[rand_idx].mean()),
                "QED_mean": float(valid_qed[rand_idx].mean()),
            },
        }

    # "Fewer but better binders" survivor counts (warhead AND FiLM>=7 AND QED>=0.4 AND iptm>=0.7)
    surv_mask = valid_warhead & (valid_film >= 7.0) & (valid_qed >= 0.4) & (valid_iptm >= 0.7)
    n_survivors = int(surv_mask.sum())
    # And a lenient version (iptm gate only)
    surv_mask_loose = valid_warhead & (valid_film >= 7.0) & (valid_qed >= 0.4)
    n_survivors_loose = int(surv_mask_loose.sum())

    summary["stratified"] = strat_rows
    summary["top_vs_random"] = top_n_cmp
    summary["survivors"] = {
        "criteria": "warhead_any AND FiLM_pIC50>=7 AND QED>=0.4 AND distilled_iptm>=0.7",
        "n": n_survivors,
        "criteria_loose": "warhead_any AND FiLM_pIC50>=7 AND QED>=0.4 (no iptm gate)",
        "n_loose": n_survivors_loose,
    }

    # Refresh summary on disk with stratification block
    with open(str(out_csv) + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- 9. Markdown report ----
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# ExpB — Distilled-IPTM Post-RL Cohort Stratified Analysis\n")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.\n")
    lines.append("## Source\n")
    lines.append(f"- Checkpoint: `{args.ckpt}`\n")
    lines.append(f"- Seeds: `{args.seeds}` ({len(seeds_full)} total, "
                 f"{args.seed_subsample_n} sampled + Mol1 anchor → {len(chosen)} effective)\n")
    lines.append(f"- N sampled: {n_total} | N valid: {n_valid} "
                 f"({100 * n_valid / max(1, n_total):.1f}%)\n")
    lines.append(f"- Sampler device: {sample_device}, seed={args.seed}\n")
    lines.append("\n## Cohort-level (valid mols)\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| acrylamide_any % | {100 * summary['warhead_any_rate']:.2f} |")
    lines.append(f"| thiq_exact % | {100 * summary['thiq_exact_rate']:.2f} |")
    lines.append(f"| FiLM mean / median / max | "
                 f"{summary['mean_film_pIC50']:.3f} / "
                 f"{summary['median_film_pIC50']:.3f} / "
                 f"{summary['max_film_pIC50']:.3f} |")
    lines.append(f"| FiLM frac >=7 / >=8 | "
                 f"{100 * summary['frac_film_ge_7']:.2f}% / "
                 f"{100 * summary['frac_film_ge_8']:.2f}% |")
    lines.append(f"| QED mean | {summary['mean_QED']:.3f} |")
    lines.append(f"| distilled_iptm mean / median / max | "
                 f"{summary['mean_distilled_iptm']:.3f} / "
                 f"{summary['median_distilled_iptm']:.3f} / "
                 f"{summary['max_distilled_iptm']:.3f} |")
    lines.append(f"| iptm frac >= 0.85 | {100 * summary['frac_iptm_ge_0p85']:.2f}% |")
    lines.append(f"| internal Tanimoto (sub500) | {internal_mean_tc:.3f} |")
    lines.append(f"| scaffolds/valid_mol | {scaff_per_mol:.3f} |")

    lines.append("\n## Stratified by distilled_iptm\n")
    lines.append("| iptm threshold | n | acryl % | thiq % | FiLM mean | QED mean | iptm mean |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in strat_rows:
        if r["n"] == 0:
            lines.append(f"| ≥{r['iptm_thr']:.2f} | 0 | — | — | — | — | — |")
        else:
            lines.append(f"| ≥{r['iptm_thr']:.2f} | {r['n']} | "
                         f"{r['acryl_pct']:.1f}% | {r['thiq_pct']:.1f}% | "
                         f"{r['FiLM_mean']:.3f} | {r['QED_mean']:.3f} | "
                         f"{r['iptm_mean']:.3f} |")

    if top_n_cmp is not None:
        lines.append(f"\n## Top-{N} by iptm vs random-{N} (enrichment check)\n")
        lines.append("| slice | iptm mean | acryl % | thiq % | FiLM mean | QED mean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        t = top_n_cmp["top_iptm"]; rr = top_n_cmp["random"]
        lines.append(f"| top-{N} iptm | {t['iptm_mean']:.3f} | "
                     f"{t['acryl_pct']:.1f}% | {t['thiq_pct']:.1f}% | "
                     f"{t['FiLM_mean']:.3f} | {t['QED_mean']:.3f} |")
        lines.append(f"| random-{N} | {rr['iptm_mean']:.3f} | "
                     f"{rr['acryl_pct']:.1f}% | {rr['thiq_pct']:.1f}% | "
                     f"{rr['FiLM_mean']:.3f} | {rr['QED_mean']:.3f} |")

    lines.append("\n## 'Fewer but better' survivor count\n")
    lines.append("Survivor filter: `warhead_any AND FiLM_pIC50>=7 AND QED>=0.4 AND distilled_iptm>=0.7`.\n")
    lines.append(f"- ExpB post-RL survivors: **{n_survivors}** "
                 f"({100 * n_survivors / max(1, n_valid):.2f}% of valid)\n")
    lines.append(f"- Loose (no iptm gate): {n_survivors_loose} "
                 f"({100 * n_survivors_loose / max(1, n_valid):.2f}%)\n")
    lines.append("\n## Hypothesis verdict\n")
    lines.append("The 'fewer but better binders' hypothesis predicts that the IPTM term, "
                 "while dragging the cohort-level acrylamide retention down, should "
                 "concentrate the surviving warhead mols in the high-iptm region — "
                 "and that the high-iptm slice should be ENRICHED in acrylamide/thiq, "
                 "not depleted. See the stratified table and top-vs-random comparison "
                 "above.\n")
    md_path.write_text("\n".join(lines))
    log(f"wrote {md_path}")

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("stratified",)}, indent=2,
                     default=lambda o: float(o) if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
