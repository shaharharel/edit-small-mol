"""OptionA — 5-cell ablation on held-out test targets.

Cells:
    1. BASELINE_covFT: sample from frozen covFT prior only (no coord conditioning).
    2. OURS_predicted: Stage 1 predicts pose from real test-target pocket → Stage 2 samples.
    3. ABL_shuffled: predict pose from a RANDOM OTHER test-target's pocket → Stage 2 samples.
    4. ABL_zero: zero the 5-vector (in normalized space) → Stage 2 samples.
    5. ABL_oracle: use PDB-truth pose directly → Stage 2 samples (upper bound).

For each cell we sample N (default 40, so ~64 * 40 = ~2560 total per target;
we sub-sample down to `--n-samples` per cell across all test targets for eval speed).

Metrics per cell (using paper's compute_planar_dihedral_and_prereact + acrylamide checks):
    - validity_frac
    - planar_dihedral_median_deg (lower = better, planar torsion)
    - acryl_largest_frag_pct
    - bd_angle_deviation_from_105 (mean |angle - 105|, from ETKDG conformer)
    - prereact_frac_le_30

Output: data/optionA/ablation_results.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
from optionA_mol2mol import (
    CoordConditionedMol2Mol,
    load_mol2mol_prior,
    subsequent_mask,
    tokenize_smiles,
    detokenize,
)
from optionA_stage1 import Stage1Model, PocketDataset, collate as stage1_collate, load_cache, compute_target_stats as stage1_target_stats

OPT = REPO / "data" / "optionA"
MODELS = REPO / "models"
PRIOR = MODELS / "reinvent4_mol2mol_covalent_ft.prior"

ACRYL_SMARTS = Chem.MolFromSmarts("[C;X3]=[C;X3]-[C;X3](=O)-[N;X3]")


def canonicalize(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def largest_fragment_smi(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return None
    largest = max(frags, key=lambda mm: mm.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest)


def acryl_on_largest(smi):
    lf = largest_fragment_smi(smi)
    if lf is None:
        return False
    m = Chem.MolFromSmiles(lf)
    if m is None:
        return False
    return bool(m.GetSubstructMatch(ACRYL_SMARTS))


def compute_planar_and_bd(smi_list, seed=42):
    """For each SMILES compute (planar_dihedral_deg, bd_angle_deg) using ETKDG conformer.

    Returns arrays: dihedrals_deg (NaN for failure), bd_angle_deg (NaN for failure).
    """
    n = len(smi_list)
    dihs = np.full(n, np.nan, dtype=np.float64)
    bds = np.full(n, np.nan, dtype=np.float64)
    for i, smi in enumerate(smi_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        matches = m.GetSubstructMatches(ACRYL_SMARTS)
        if not matches:
            continue
        a1, a2, a3, _o, a4 = matches[0]
        try:
            mH = Chem.AddHs(m)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            cid = AllChem.EmbedMolecule(mH, params)
            if cid < 0:
                continue
            conf = mH.GetConformer(cid)
            phi = AllChem.GetDihedralDeg(conf, a1, a2, a3, a4)
            phi_wrap = ((phi + 180.0) % 360.0) - 180.0
            d = min(abs(phi_wrap), abs(180.0 - abs(phi_wrap)))
            dihs[i] = d
            # BD angle = angle at C=C-C(=O) i.e. a1-a2-a3
            ang = AllChem.GetAngleDeg(conf, a1, a2, a3)
            bds[i] = ang
        except Exception:
            continue
    return dihs, bds


def eval_cohort(smi_list, tag, seed=42):
    canon = [canonicalize(s) for s in smi_list]
    canon_valid = [s for s in canon if s is not None]
    n_raw = len(smi_list)
    validity = len(canon_valid) / max(n_raw, 1)
    # Dedup, plot
    canon_uniq = sorted(set(canon_valid))
    dihs, bds = compute_planar_and_bd(canon_uniq, seed=seed)
    finite = np.isfinite(dihs)
    dihs_f = dihs[finite]
    bd_dev = np.abs(bds[np.isfinite(bds)] - 105.0)
    acryl = [acryl_on_largest(s) for s in canon_uniq]
    return {
        "cell": tag,
        "n_raw": int(n_raw),
        "n_valid": int(len(canon_valid)),
        "n_unique": int(len(canon_uniq)),
        "validity_frac": float(validity),
        "planar_dihedral_n_computed": int(finite.sum()),
        "planar_dihedral_median_deg": float(np.median(dihs_f)) if len(dihs_f) else float("nan"),
        "planar_dihedral_mean_deg": float(np.mean(dihs_f)) if len(dihs_f) else float("nan"),
        "prereact_le_30_frac": float((dihs_f <= 30.0).mean()) if len(dihs_f) else float("nan"),
        "acryl_largest_frag_pct": float(np.mean(acryl)) if acryl else float("nan"),
        "bd_dev_from_105_mean": float(bd_dev.mean()) if len(bd_dev) else float("nan"),
    }


def load_stage1(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = Stage1Model().to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, np.array(ck["y_mean"]), np.array(ck["y_std"])


def load_stage2(ckpt_path, prior_path, device):
    base, vocab, _max_len, _m, _u = load_mol2mol_prior(prior_path, device=device)
    model = CoordConditionedMol2Mol(base).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, vocab, np.array(ck["y_mean"]), np.array(ck["y_std"])


def sample_cell(model, vocab, anchors, coord_norm, n_per_anchor, temperature, device, use_conditioning=True):
    """Sample n_per_anchor SMILES per anchor.

    If use_conditioning=False, coord is zeroed and coord_mlp output multiplied by 0
    (behaves like baseline covFT).
    """
    tokens = vocab["tokens"]; inv = {v: k for k, v in tokens.items()}
    pad = vocab["pad_token"]; bos = vocab["bos_token"]; eos = vocab["eos_token"]
    max_len = 96
    out = []
    for i, (smi, cvec) in enumerate(zip(anchors, coord_norm)):
        toks = tokenize_smiles(smi, tokens, bos, eos, pad, max_len)
        src = torch.tensor([toks], dtype=torch.long, device=device).expand(n_per_anchor, -1)
        src_mask = (src != pad).unsqueeze(1)
        cv = torch.tensor(cvec, dtype=torch.float32, device=device).unsqueeze(0).expand(n_per_anchor, -1)
        if not use_conditioning:
            # bypass conditioning: use base directly (no coord tokens prepended)
            with torch.no_grad():
                # Manual sample loop with base.encode
                memory = model.base.encode(src, src_mask)
                ys = torch.full((n_per_anchor, 1), bos, dtype=torch.long, device=device)
                done = torch.zeros(n_per_anchor, dtype=torch.bool, device=device)
                for step in range(max_len - 1):
                    tm = subsequent_mask(ys.size(1)).to(device)
                    h = model.base.decode(memory, src_mask, ys, tm)
                    logp = model.base.generator(h[:, -1])
                    if temperature != 1.0:
                        logp = logp / temperature
                    probs = logp.exp()
                    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs)).clamp_min(0)
                    rs = probs.sum(-1, keepdim=True)
                    probs = torch.where(rs > 0, probs, torch.ones_like(probs) / probs.size(-1))
                    nxt = torch.multinomial(probs, 1)
                    nxt = torch.where(done.unsqueeze(-1), torch.full_like(nxt, eos), nxt)
                    ys = torch.cat([ys, nxt], dim=1)
                    done = done | (nxt.squeeze(-1) == eos)
                    if done.all():
                        break
                ids_batch = ys
        else:
            with torch.no_grad():
                ids_batch = model.sample(src, src_mask, cv, bos, eos, max_len=max_len, temperature=temperature)
        for k in range(n_per_anchor):
            out.append(detokenize(ids_batch[k].tolist(), inv, bos, eos, pad))
    return out


def predict_coords(stage1, y_mean_s1, y_std_s1, cache, test_df, device):
    """Run Stage 1 on each test row; return raw coord predictions (list of (5,)) aligned with test rows."""
    sid_to_idx = {s: i for i, s in enumerate(cache["struct_ids"])}
    preds = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            sid = row["struct_id"]
            if sid not in sid_to_idx:
                # fallback: use zeros in normalized space
                preds.append(np.array([0.0, 0.0, 0.0, 105.0, 3.14], dtype=np.float32))
                continue
            cache_i = sid_to_idx[sid]
            seq_i = int(cache["row_seq_idx"][cache_i])
            emb = cache["residues_emb"][seq_i]
            mask = cache["residues_mask"][seq_i]
            from optionA_stage1 import NUC2ID
            nuc = row.get("nucleophile_resname", "XXX")
            nuc_id = NUC2ID.get(nuc, NUC2ID["XXX"])
            emb_t = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask.astype(np.bool_)).unsqueeze(0).to(device)
            nuc_t = torch.tensor([nuc_id], dtype=torch.long, device=device)
            pred_norm = stage1(emb_t, mask_t, nuc_t)
            pred_raw = pred_norm.cpu().numpy()[0] * y_std_s1 + y_mean_s1
            preds.append(pred_raw)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(MODELS / "optionA_stage1.pt"))
    ap.add_argument("--stage2", default=str(MODELS / "optionA_stage2.pt"))
    ap.add_argument("--prior", default=str(PRIOR))
    ap.add_argument("--n-per-anchor", type=int, default=8)
    ap.add_argument("--max-anchors", type=int, default=250)  # 250 * 8 = 2000
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out-csv", default=str(OPT / "ablation_results.csv"))
    ap.add_argument("--out-samples", default=str(OPT / "ablation_samples.parquet"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[abl] device={device}", flush=True)

    # Load models
    print("[abl] loading stage1...", flush=True)
    stage1, y_mean_s1, y_std_s1 = load_stage1(args.stage1, device)
    print("[abl] loading stage2...", flush=True)
    stage2, vocab, y_mean_s2, y_std_s2 = load_stage2(args.stage2, args.prior, device)

    print("[abl] loading test set...", flush=True)
    te = pd.read_parquet(OPT / "test.parquet")
    cache = load_cache()

    # Subsample test rows: max_anchors
    if len(te) > args.max_anchors:
        te = te.sample(n=args.max_anchors, random_state=0).reset_index(drop=True)
    print(f"[abl] using {len(te)} test rows", flush=True)

    anchors = te["canon_smi"].tolist()

    # Predict coords for each test row (Stage 1)
    print("[abl] running Stage 1 predictions...", flush=True)
    pred_raw_list = predict_coords(stage1, y_mean_s1, y_std_s1, cache, te, device)

    # Build normalized-space coord vectors for each cell
    def to_norm(raw):
        return ((np.array(raw, dtype=np.float32) - y_mean_s2) / np.where(y_std_s2 > 1e-6, y_std_s2, 1.0)).astype(np.float32)

    # cell 2: OURS_predicted
    c_pred_norm = [to_norm(r) for r in pred_raw_list]
    # cell 3: ABL_shuffled — permute pred_raw_list
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(te))
    # ensure no fixed points (shuffled truly different)
    for i in range(len(perm)):
        if perm[i] == i:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]
    c_shuf_norm = [to_norm(pred_raw_list[j]) for j in perm]
    # cell 4: ABL_zero — zero in normalized space (mean prediction)
    c_zero_norm = [np.zeros(5, dtype=np.float32) for _ in range(len(te))]
    # cell 5: ABL_oracle — truth from test row
    c_oracle_norm = []
    for _, row in te.iterrows():
        bc = row["bc_local_xyz"]
        raw = np.array([bc[0], bc[1], bc[2], float(row["bd_angle"]), float(row["phi_planar"])], dtype=np.float32)
        c_oracle_norm.append(to_norm(raw))

    results = []
    all_samples = []

    print("[abl] sampling cells...", flush=True)
    t0 = time.time()

    def run_cell(tag, coords, use_cond=True):
        s = sample_cell(stage2, vocab, anchors, coords, args.n_per_anchor, args.temperature, device, use_conditioning=use_cond)
        print(f"[abl] {tag}: sampled {len(s)} [{time.time()-t0:.0f}s]", flush=True)
        for anchor_i, batch_start in enumerate(range(0, len(s), args.n_per_anchor)):
            for k in range(args.n_per_anchor):
                if batch_start + k < len(s):
                    all_samples.append({"cell": tag, "anchor_idx": anchor_i, "sample_idx": k, "smiles": s[batch_start + k]})
        m = eval_cohort(s, tag)
        results.append(m)
        print(f"[abl] {tag}: validity={m['validity_frac']:.3f} planar_med={m['planar_dihedral_median_deg']:.1f}° acryl={m['acryl_largest_frag_pct']:.3f}", flush=True)

    run_cell("BASELINE_covFT", c_zero_norm, use_cond=False)
    run_cell("OURS_predicted", c_pred_norm, use_cond=True)
    run_cell("ABL_shuffled", c_shuf_norm, use_cond=True)
    run_cell("ABL_zero", c_zero_norm, use_cond=True)
    run_cell("ABL_oracle", c_oracle_norm, use_cond=True)

    df_r = pd.DataFrame(results)
    df_r.to_csv(args.out_csv, index=False)
    pd.DataFrame(all_samples).to_parquet(args.out_samples)
    print(f"[abl] wrote {args.out_csv} and {args.out_samples}", flush=True)

    # Win conditions
    def get(cell, col):
        row = df_r[df_r["cell"] == cell]
        if len(row) == 0:
            return float("nan")
        return float(row.iloc[0][col])

    base_plan = get("BASELINE_covFT", "planar_dihedral_median_deg")
    ours_plan = get("OURS_predicted", "planar_dihedral_median_deg")
    shuf_plan = get("ABL_shuffled", "planar_dihedral_median_deg")
    print()
    print("=== WIN CONDITIONS ===")
    print(f"1. OURS < BASELINE by >=20°: BASELINE={base_plan:.1f}° OURS={ours_plan:.1f}° (Δ={base_plan-ours_plan:+.1f}°) {'PASS' if (base_plan - ours_plan) >= 20 else 'FAIL'}")
    print(f"2. OURS − ABL_shuffled gap >=30°: OURS={ours_plan:.1f}° SHUF={shuf_plan:.1f}° (Δ={shuf_plan-ours_plan:+.1f}°) {'PASS' if (shuf_plan - ours_plan) >= 30 else 'FAIL'}")

    # win-conditions JSON
    (OPT / "ablation_win_conditions.json").write_text(json.dumps({
        "cond1_planar_gain_deg": float(base_plan - ours_plan),
        "cond1_pass": bool((base_plan - ours_plan) >= 20),
        "cond2_pocket_use_gap_deg": float(shuf_plan - ours_plan),
        "cond2_pass": bool((shuf_plan - ours_plan) >= 30),
        "baseline_planar": base_plan,
        "ours_planar": ours_plan,
        "shuffled_planar": shuf_plan,
    }, indent=2))
    print(f"[abl] done", flush=True)


if __name__ == "__main__":
    main()
