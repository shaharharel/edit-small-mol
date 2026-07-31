"""GPU-batched scorer for Mol1-anchored cohort sampling CSVs.

Same output schema as `score_mol1_cohort_fast.py` but uses CUDA for FiLM scoring
with proper batching (50K mols x 280 anchors). All FiLM inference in a few
batched forward passes instead of 50K separate forward calls.

Usage:
    python experiments/score_mol1_cohort_gpu.py <input.csv> <output.csv> \
        --tag <cohort_tag> [--seed <seed_smi>] [--smi-col SMILES]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (AllChem, Crippen, Descriptors, FilterCatalog, Lipinski,
                        QED, RDConfig, rdMolDescriptors)
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Mol1 reference
MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
_m1 = Chem.MolFromSmiles(MOL1)
MOL1_CANON = Chem.MolToSmiles(_m1)
MOL1_MURCKO = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(_m1))
MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(_m1, 2, 2048)
THIQ_ACRYL = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)[N]")


def _load_zap70():
    df = pd.read_csv(PROJECT / "data/docking_chembl_zap70/docking_results.csv")
    canon = set()
    fps = []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        canon.add(Chem.MolToSmiles(m))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    return canon, fps


def _build_filter_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


def score_panel(smi: str, zap70_canon, zap70_fps, filter_catalog, sascorer) -> Optional[dict]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    canon = Chem.MolToSmiles(m)

    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    tpsa = Descriptors.TPSA(m)
    hba = Lipinski.NumHAcceptors(m)
    hbd = Lipinski.NumHDonors(m)
    rotb = Lipinski.NumRotatableBonds(m)
    hatoms = m.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(m)
    qed = QED.qed(m)
    fsp3 = Lipinski.FractionCSP3(m)
    lipinski_viol = int((mw > 500) + (logp > 5) + (hba > 10) + (hbd > 5))

    warhead_intact = bool(m.HasSubstructMatch(ACRYL))
    thiq_core = bool(m.HasSubstructMatch(THIQ_ACRYL))
    murcko_match = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) == MOL1_MURCKO

    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    tc_mol1 = DataStructs.TanimotoSimilarity(MOL1_FP, fp)

    sims = DataStructs.BulkTanimotoSimilarity(fp, zap70_fps)
    sims_sorted = sorted(sims, reverse=True)
    max_tc_train = sims_sorted[0] if sims_sorted else 0.0
    mean_top10 = float(np.mean(sims_sorted[:10])) if sims_sorted else 0.0

    novel = canon not in zap70_canon

    brenk_alerts = 0
    pains_alerts = 0
    for m_match in filter_catalog.GetMatches(m):
        cat = m_match.GetProp("FilterSet")
        if cat == "BRENK":
            brenk_alerts += 1
        elif cat == "PAINS":
            pains_alerts += 1

    sa = float(sascorer.calculateScore(m)) if sascorer is not None else None

    return {
        "smiles": canon,
        "MW": round(mw, 3),
        "LogP": round(logp, 3),
        "TPSA": round(tpsa, 2),
        "HBA": hba,
        "HBD": hbd,
        "RotBonds": rotb,
        "HeavyAtoms": hatoms,
        "Rings": rings,
        "QED": round(qed, 4),
        "fsp3": round(fsp3, 3),
        "Lipinski_violations": lipinski_viol,
        "warhead_intact": warhead_intact,
        "thiq_core": thiq_core,
        "mol1_murcko_match": murcko_match,
        "Tc_to_Mol1": round(tc_mol1, 4),
        "max_Tc_train": round(max_tc_train, 4),
        "mean_top10_Tc_train": round(mean_top10, 4),
        "Brenk_alerts": brenk_alerts,
        "PAINS_alerts": pains_alerts,
        "SAScore": round(sa, 3) if sa is not None else None,
        "novel_vs_chembl_zap70": novel,
    }


def smiles_to_fp_array(smiles_list: List[str]) -> np.ndarray:
    """Compute Morgan FPs as float32 [N, 2048]."""
    out = np.zeros((len(smiles_list), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def load_film_model_gpu(device: torch.device):
    """Load cached FiLM model onto GPU. Returns model, scaler arrays, anchor tensors."""
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    cache = PROJECT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    if not cache.exists():
        raise FileNotFoundError(f"FiLM model checkpoint not found: {cache}")
    ckpt = torch.load(cache, map_location="cpu", weights_only=False)

    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    scaler_mean = torch.from_numpy(ckpt["scaler_mean"].astype(np.float32)).to(device)
    scaler_scale = torch.from_numpy(ckpt["scaler_scale"].astype(np.float32)).to(device)

    anchor_embs = ckpt["anchor_embs"]
    if isinstance(anchor_embs, np.ndarray):
        anchor_embs = torch.from_numpy(anchor_embs).float()
    anchor_embs = anchor_embs.float().to(device)
    anchor_pIC50 = ckpt["anchor_pIC50"]
    if isinstance(anchor_pIC50, np.ndarray):
        anchor_pIC50 = torch.from_numpy(anchor_pIC50.astype(np.float32))
    anchor_pIC50 = anchor_pIC50.float().to(device)

    print(
        f"[scorer-gpu] FiLM loaded: {anchor_embs.shape[0]} anchors, device={device}",
        file=sys.stderr,
    )
    return model, scaler_mean, scaler_scale, anchor_embs, anchor_pIC50


@torch.no_grad()
def score_film_gpu(
    fps_np: np.ndarray,
    model,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    anchor_embs: torch.Tensor,
    anchor_pIC50: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Batched anchor-mean prediction on GPU.

    For each target mol i:
        pred[i] = mean_a (anchor_pIC50[a] + FiLM(anchor[a], target[i]))

    Implementation: per target batch B, expand to (B, n_anchors) pairs,
    call model.forward once per batch.

    Returns: float32 array [N].
    """
    n = fps_np.shape[0]
    n_anchors = anchor_embs.shape[0]
    fps = torch.from_numpy(fps_np).to(device)
    # standardize on GPU
    fps_std = (fps - scaler_mean) / scaler_scale

    preds = np.empty(n, dtype=np.float32)

    # Note: FiLMDeltaMLP.forward(emb_a, emb_b) computes pred_b - pred_a
    # We want pred(j) where anchor = a, target = j. The training "delta" was
    # pIC50[b] - pIC50[a]; abs_pred[j] = pIC50[a] + model(emb_a, emb_b=target_j).
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start
        tgt = fps_std[start:end]  # [b, 2048]
        # Expand: pair every target with every anchor
        # shape: anchors [n_anchors, 2048] x targets [b, 2048] -> [b*n_anchors, 2048]
        emb_a = anchor_embs.unsqueeze(0).expand(b, n_anchors, -1).reshape(-1, anchor_embs.shape[1])
        emb_b = tgt.unsqueeze(1).expand(b, n_anchors, -1).reshape(-1, tgt.shape[1])
        deltas = model(emb_a, emb_b).view(b, n_anchors)
        abs_pred = anchor_pIC50.unsqueeze(0) + deltas  # broadcast [b, n_anchors]
        preds[start:end] = abs_pred.mean(dim=1).cpu().numpy()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--smi-col", default="SMILES")
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--film-batch", type=int, default=512,
                    help="target-mol batch size for FiLM scoring (each batch = batch*n_anchors pairs)")
    ap.add_argument("--device", default="auto", help="cuda/cpu/auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[scorer-gpu] device: {device}", file=sys.stderr)

    t0 = time.time()
    print(f"[scorer-gpu] Loading ZAP70 anchors / filter catalog / SAScore...", file=sys.stderr)
    zap70_canon, zap70_fps = _load_zap70()
    print(f"[scorer-gpu] Loaded ZAP70: {len(zap70_canon)} unique, {len(zap70_fps)} fps",
          file=sys.stderr)
    filter_catalog = _build_filter_catalog()
    try:
        sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer  # type: ignore
    except Exception:
        sascorer = None
        print("[scorer-gpu] WARN: sascorer unavailable", file=sys.stderr)

    print(f"[scorer-gpu] Reading {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    smi_col = args.smi_col if args.smi_col in df.columns else (
        "SMILES" if "SMILES" in df.columns else "smiles"
    )
    if smi_col not in df.columns:
        print(f"[scorer-gpu] ERROR: no SMILES column. Columns: {list(df.columns)}")
        sys.exit(1)
    print(f"[scorer-gpu] Scoring {len(df):,} mols (SMILES col: '{smi_col}')")

    # RDKit panel (this is the slow CPU part; ~5-10 min for 50K)
    t_panel = time.time()
    rows: List[dict] = []
    n = len(df)
    for i, smi in enumerate(df[smi_col]):
        rec = score_panel(str(smi), zap70_canon, zap70_fps, filter_catalog, sascorer)
        if rec is None:
            continue
        rec["row_id"] = f"{args.tag}_{i}"
        rec["method"] = args.tag
        if args.seed:
            rec["seed_smi"] = args.seed
        rows.append(rec)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t_panel
            rate = (i + 1) / elapsed
            print(f"  scored {i+1:,}/{n:,}  ({rate:.0f} mol/s, ETA "
                  f"{(n-i-1)/max(rate,1):.0f}s)")
    out = pd.DataFrame(rows)
    print(f"[scorer-gpu] RDKit panel done in {time.time()-t_panel:.1f}s. "
          f"Valid mols: {len(out):,}/{n:,}")

    # FiLM batch scoring on GPU
    if not args.no_film and len(out):
        t_film = time.time()
        print(f"[scorer-gpu] FiLM pIC50 scoring (GPU batched)...", file=sys.stderr)
        try:
            model, mu, sigma, anchor_embs, anchor_pIC50 = load_film_model_gpu(device)
            fps = smiles_to_fp_array(out["smiles"].tolist())
            preds = score_film_gpu(
                fps, model, mu, sigma, anchor_embs, anchor_pIC50, device,
                batch_size=args.film_batch,
            )
            out["pIC50_film"] = [round(float(s), 4) for s in preds]
            out["pIC50_mean"] = out["pIC50_film"]
            out["delta_vs_mol1"] = [round(float(s) - 6.59, 4) for s in preds]
            print(f"[scorer-gpu] FiLM done in {time.time()-t_film:.1f}s. "
                  f"Mean pIC50: {np.nanmean(out['pIC50_film']):.3f}")
        except Exception as e:
            import traceback
            print(f"[scorer-gpu] FiLM scoring failed: {e}", file=sys.stderr)
            traceback.print_exc()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[scorer-gpu] Wrote {args.output_csv}  (total {time.time()-t0:.1f}s)")

    # Summary
    print()
    print(f"=== {args.tag} summary ===")
    print(f"  N: {len(out):,}")
    if "warhead_intact" in out.columns:
        print(f"  Acrylamide intact: {out['warhead_intact'].mean()*100:.1f}%")
    if "thiq_core" in out.columns:
        print(f"  THIQ-acryl core:   {out['thiq_core'].mean()*100:.1f}%")
    if "mol1_murcko_match" in out.columns:
        print(f"  Mol1 Murcko match: {out['mol1_murcko_match'].mean()*100:.2f}%")
    if "Tc_to_Mol1" in out.columns:
        print(f"  Median Tc-to-Mol1: {out['Tc_to_Mol1'].median():.3f}")
        print(f"  Tc>=0.5: {(out['Tc_to_Mol1']>=0.5).mean()*100:.1f}%")
    if "QED" in out.columns:
        print(f"  Median QED: {out['QED'].median():.3f}")
    if "pIC50_film" in out.columns:
        print(f"  Median FiLM pIC50: {out['pIC50_film'].median():.2f}")
    if "novel_vs_chembl_zap70" in out.columns:
        print(f"  Novel vs ChEMBL ZAP70: {out['novel_vs_chembl_zap70'].mean()*100:.1f}%")
    if "Brenk_alerts" in out.columns:
        print(f"  Median Brenk alerts: {out['Brenk_alerts'].median():.1f}")


if __name__ == "__main__":
    main()
