#!/usr/bin/env python3
"""
EXP6 Retrospective LO — Phase 1 preparation (LOCAL ONLY).

For each of 3 kinase targets (EGFR T790M, BTK, KRAS G12C):
  1) Extract within-assay pairs (target_pairs.csv).
  2) Build "future set" exclude_set.csv = drug + Tc>=0.6 analogs + manual
     clinical successors.
  3) Build train_pairs.csv = target_pairs minus any pair touching exclude_set.
  4) Build anchor_pool_strategy_b.csv = 100 Murcko-stratified diverse anchors
     from train pool (early-lead first).
  5) Define warhead_smarts.json (strict + generic) per target.
  6) Train kinase-pretrained -> target-finetuned FiLMDelta and DirectAbsoluteMLP;
     report 10% held-out MAE/Spearman.
  7) Write per-target summary.md.

Top-level summary written to results/paper_evaluation/exp6_phase1_prep_summary.md.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
os.environ["RDK_DEPRECATION_WARNING"] = "off"
torch.backends.mps.is_available = lambda: False  # CPU only
torch.set_num_threads(4)  # cap CPU thread usage (less overhead)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.film_delta_predictor import (  # noqa: E402
    FiLMDeltaPredictor,
    FiLMDeltaMLP,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SHARED_PAIRS_CSV = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
ALL_PAIRS_CSV = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "all_pairs_within_assay.csv"
# Per CLAUDE.md & spec: shared_pairs_deduped is the canonical dataset.
PAIRS_CSV = SHARED_PAIRS_CSV
PIC50_CSV = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "molecule_pIC50_minimal.csv"
OUT_BASE = PROJECT_ROOT / "data" / "exp6_retrospective"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
TOP_SUMMARY = RESULTS_DIR / "exp6_phase1_prep_summary.md"

N_BITS = 2048
RADIUS = 2
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.2
LR = 1e-3
PRETRAIN_LR = 5e-4
FINETUNE_LR = 5e-4
BATCH_SIZE = 256
MAX_EPOCHS = 20
PATIENCE = 5
PRETRAIN_EPOCHS = 3
MAX_KINASE_PAIRS = 30000  # cap kinase pretrain pool for speed
TC_DRUG_THRESHOLD = 0.6
N_DIVERSE_ANCHORS = 100
HELDOUT_FRAC = 0.10
SEED = 42

# --------------------------------------------------------------------------
# Target definitions
# --------------------------------------------------------------------------
TARGETS: Dict[str, Dict] = {
    "egfr_t790m": {
        "label": "EGFR T790M",
        "target_chembl_ids": ["CHEMBL203", "CHEMBL5145"],
        "anchor_chembl": "CHEMBL1229592",
        "anchor_name": "WZ4002",
        "anchor_smiles": "C=CC(=O)Nc1cccc(Oc2nc(Nc3ccc(N4CCN(C)CC4)cc3OC)ncc2Cl)c1",
        "drug_chembl": "CHEMBL3353410",
        "drug_name": "osimertinib",
        "drug_smiles": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
        # Manual clinical successors (name -> SMILES; ChEMBL IDs optional, recovered later)
        "manual_successors": {
            "rociletinib": "C=CC(=O)Nc1cc(Nc2nc(Nc3ccc(N4CCN(C)C(=O)C4)cc3)nc(C(F)(F)F)c2)c(OC)cc1N1CCOCC1",
            "naquotinib": "C=CC(=O)N(C)c1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N1CCN(C(C)C)CC1",
            "lazertinib":  "C=CC(=O)Nc1cc(Nc2ncc(Cl)c(Oc3ccc(N4CCN(C)CC4)c(OC)c3)n2)c(OC)cc1N1CCOCC1",
            "mavelertinib": "C=CC(=O)Nc1cccc(Nc2nccc(-c3cccnc3)n2)c1",
        },
        "warhead_anchor_role": "acrylamide-on-aniline",
        "warhead_smarts": {
            "strict": "[NH]([c])C(=O)C=C",       # anilino-acrylamide (anchor & drug share)
            "generic": "C=CC(=O)N",                # any acrylamide
        },
    },
    "btk": {
        "label": "BTK",
        "target_chembl_ids": ["CHEMBL5251"],
        "anchor_chembl": "CHEMBL4072833",
        "anchor_name": "evobrutinib",
        # Evobrutinib: pyrimidine-anilino + vinyl ketone (Michael acceptor) — NOT acrylamide
        "anchor_smiles": "C=CC(=O)N1CCC(n2nc(-c3ccc(Oc4ccccn4)cc3)c3c(N)ncnc32)CC1",
        "drug_chembl": "CHEMBL1873475",
        "drug_name": "ibrutinib",
        "drug_smiles": "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
        "manual_successors": {
            "acalabrutinib": "CC#CC(=O)N1CCC[C@H]1c1ncc(-c2ccc(C(=O)Nc3ccccn3)cc2)n1-c1ncccn1",
            "zanubrutinib": "C=CC(=O)N1CCC[C@H]1Cn1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
            # evobrutinib is the anchor — DO NOT exclude
            "tirabrutinib": "C=CC(=O)N1CCCC1Cn1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
            "spebrutinib": "C=CC(=O)Nc1ccc(C(=O)NC2CCCCC2)cc1Nc1ncc(Cl)c(Nc2ccccc2)n1",
        },
        "warhead_anchor_role": "acrylamide-on-piperidine (evobrutinib & ibrutinib both share)",
        "warhead_smarts": {
            # Plain piperidine N-acrylamide — matches both evobrutinib (C-4 sub) and ibrutinib (C-3 sub)
            "strict": "C=CC(=O)N1CCCCC1",
            "generic": "C=CC(=O)N",                  # any acrylamide
        },
    },
    "kras_g12c": {
        "label": "KRAS G12C",
        "target_chembl_ids": ["CHEMBL2189121", "CHEMBL5658"],
        "anchor_chembl": "CHEMBL4214264",
        "anchor_name": "ARS-1620",
        "anchor_smiles": "C=CC(=O)N1CCN(c2ncnc3c(F)c(-c4c(O)cccc4F)c(Cl)cc23)CC1",
        "drug_chembl": "CHEMBL4535757",
        "drug_name": "sotorasib (AMG-510)",
        "drug_smiles": "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C(C)C)c3nc(-c4c(O)cccc4F)c(F)cc23)[C@@H](C)C1",
        "manual_successors": {
            "adagrasib": "CN1CCN([C@H]2CCCN2C(=O)C2=CN(C3CCC3)C(=O)N(c3nc(N4CCC[C@H](OCC#N)C4)c4cc(C)c(F)cc4n3)C2)CC1",
            "divarasib": "CC#CC(=O)N1CCN(c2nc3c(F)cc(Cl)c(-c4c(O)cccc4F)c3c(=O)n2C2(C)CCCN2C)CC1",
            "MRTX1133":  "Cn1c(=O)c2cc(F)c(-c3ccc4c(c3O)C(=O)N(C)CC4)c(F)c2nc1N1CCC[C@H](OCC#N)C1",
            "glecirasib": "C=CC(=O)N1CCN(c2nc(=O)n(-c3cc(C)cnc3C(C)C)c3nc(-c4cccnc4O)c(F)cc23)[C@@H](C)C1",
            "JDQ443":     "C=CC(=O)N1CCN(c2cc(C)c(-c3c(O)cccc3F)c3cc(F)c(N4CCC[C@H](OCC#N)C4)nc23)C1",
        },
        "warhead_anchor_role": "acrylamide-on-piperazine",
        "warhead_smarts": {
            "strict": "C=CC(=O)N1CCN([c,n])CC1",
            "generic": "C=CC(=O)N",
        },
    },
}


# --------------------------------------------------------------------------
# Morgan FP utilities
# --------------------------------------------------------------------------
def smi_to_morgan(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(N_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smi_to_fp_obj(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Exclude set construction
# --------------------------------------------------------------------------
def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def find_chembl_id_for_smiles(smi: str, can_to_id_local: Dict[str, str],
                              can_to_id_global: Dict[str, str] | None) -> str | None:
    """Look up CHEMBL ID by canonical SMILES match, prebuilt maps."""
    target_can = _canon(smi)
    if target_can is None:
        return None
    if target_can in can_to_id_local:
        return can_to_id_local[target_can]
    if can_to_id_global is not None and target_can in can_to_id_global:
        return can_to_id_global[target_can]
    return None


def build_exclude_set(target_pairs: pd.DataFrame, target_cfg: Dict,
                      pic50_can_to_id: Dict[str, str] | None,
                      anchor_chembl: str) -> Tuple[pd.DataFrame, Dict]:
    """Returns (exclude_df, report)."""
    drug_smi = target_cfg["drug_smiles"]
    drug_id = target_cfg["drug_chembl"]
    drug_fp = smi_to_fp_obj(drug_smi)

    # Unique mol pool in target_pairs
    rows = []
    rows.extend(zip(target_pairs["mol_a_id"], target_pairs["mol_a"]))
    rows.extend(zip(target_pairs["mol_b_id"], target_pairs["mol_b"]))
    pool = pd.DataFrame(rows, columns=["chembl_id", "smiles"]).drop_duplicates("chembl_id")

    # Build a local canonical->id map for fast lookup of manual successors
    local_can_to_id: Dict[str, str] = {}
    for cid, smi in zip(pool["chembl_id"], pool["smiles"]):
        c = _canon(smi)
        if c is not None and c not in local_can_to_id:
            local_can_to_id[c] = cid

    # Tc analogs: compute Tc(drug, each mol in pool)
    rows_excl = []
    for cid, smi in zip(pool["chembl_id"], pool["smiles"]):
        if cid == anchor_chembl:
            continue  # do not exclude anchor
        fp = smi_to_fp_obj(smi)
        if fp is None:
            continue
        tc = DataStructs.TanimotoSimilarity(drug_fp, fp)
        if tc >= TC_DRUG_THRESHOLD:
            kind = "drug" if cid == drug_id else "tc_analog"
            rows_excl.append({"chembl_id": cid, "smiles": smi,
                              "drug_or_analog": kind, "tc_to_drug": float(tc)})
    # Make sure drug itself is in the set even if not in pool
    if not any(r["chembl_id"] == drug_id for r in rows_excl):
        rows_excl.append({"chembl_id": drug_id, "smiles": drug_smi,
                          "drug_or_analog": "drug", "tc_to_drug": 1.0})

    # Manual successors: check whether each is already caught (by SMILES match)
    successor_status = []
    target_can_set = {Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"])): r["chembl_id"]
                      for r in rows_excl
                      if Chem.MolFromSmiles(r["smiles"]) is not None}
    for name, smi in target_cfg["manual_successors"].items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            successor_status.append({"name": name, "smiles": smi, "caught": False,
                                     "reason": "invalid_smiles", "tc_to_drug": None})
            continue
        can = Chem.MolToSmiles(mol)
        # Try to find CHEMBL id via prebuilt canonical maps
        cid = find_chembl_id_for_smiles(smi, local_can_to_id, pic50_can_to_id)
        fp = smi_to_fp_obj(smi)
        tc = DataStructs.TanimotoSimilarity(drug_fp, fp) if fp is not None else None
        caught = can in target_can_set
        if caught:
            successor_status.append({"name": name, "smiles": smi,
                                     "chembl_id": target_can_set[can],
                                     "caught": True, "reason": "tc_filter",
                                     "tc_to_drug": float(tc) if tc is not None else None})
        else:
            # Add manually
            rows_excl.append({"chembl_id": cid or f"MANUAL_{name}",
                              "smiles": smi,
                              "drug_or_analog": f"manual_successor:{name}",
                              "tc_to_drug": float(tc) if tc is not None else None})
            successor_status.append({"name": name, "smiles": smi,
                                     "chembl_id": cid,
                                     "caught": False, "reason": "added_manually",
                                     "tc_to_drug": float(tc) if tc is not None else None})

    excl_df = pd.DataFrame(rows_excl).drop_duplicates("chembl_id").reset_index(drop=True)
    report = {
        "n_exclude": int(len(excl_df)),
        "n_tc_analogs": int(((excl_df["drug_or_analog"] == "tc_analog")).sum()),
        "n_manual_added": int(excl_df["drug_or_analog"].str.startswith("manual_successor:").sum()),
        "successor_status": successor_status,
        "tc_threshold": TC_DRUG_THRESHOLD,
    }
    return excl_df, report


# --------------------------------------------------------------------------
# Murcko-stratified diverse anchor sampling
# --------------------------------------------------------------------------
def diverse_anchor_pool(target_pairs: pd.DataFrame, exclude_ids: set,
                        anchor_chembl: str, anchor_smiles: str, n: int) -> pd.DataFrame:
    rows = []
    rows.extend(zip(target_pairs["mol_a_id"], target_pairs["mol_a"], target_pairs["value_a"]))
    rows.extend(zip(target_pairs["mol_b_id"], target_pairs["mol_b"], target_pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["chembl_id", "smiles", "pIC50"])
    # Mean pIC50 across appearances
    agg = df.groupby("chembl_id", as_index=False).agg({"smiles": "first", "pIC50": "mean"})
    # Exclude future set
    agg = agg[~agg["chembl_id"].isin(exclude_ids)].reset_index(drop=True)

    # Compute Murcko scaffold per mol
    agg["scaffold"] = agg["smiles"].apply(murcko_scaffold)

    # Group by scaffold, then stratified-sample
    rng = np.random.RandomState(SEED)
    by_scaff = agg.groupby("scaffold")
    scaffolds = list(by_scaff.groups.keys())
    rng.shuffle(scaffolds)

    selected = []
    selected_ids = set()
    # Put anchor first (force-include even if scaffold is shared with others)
    anchor_row = agg[agg["chembl_id"] == anchor_chembl]
    if len(anchor_row) > 0:
        ar = anchor_row.iloc[0]
        selected.append({
            "chembl_id": ar["chembl_id"], "smiles": ar["smiles"],
            "scaffold": ar["scaffold"], "pIC50": float(ar["pIC50"]),
        })
        selected_ids.add(ar["chembl_id"])
    else:
        # Anchor not in target_pairs (rare); add it anyway with NaN pIC50
        selected.append({
            "chembl_id": anchor_chembl, "smiles": anchor_smiles,
            "scaffold": murcko_scaffold(anchor_smiles), "pIC50": float("nan"),
        })
        selected_ids.add(anchor_chembl)

    # Round-robin across scaffolds, picking highest-pIC50 mol per scaffold each pass
    scaff_to_sorted = {sc: gp.sort_values("pIC50", ascending=False).to_dict("records")
                       for sc, gp in by_scaff}
    pass_idx = 0
    while len(selected) < n:
        added_this_pass = 0
        for sc in scaffolds:
            if len(selected) >= n:
                break
            bucket = scaff_to_sorted[sc]
            if pass_idx >= len(bucket):
                continue
            cand = bucket[pass_idx]
            if cand["chembl_id"] in selected_ids:
                continue
            selected.append({
                "chembl_id": cand["chembl_id"], "smiles": cand["smiles"],
                "scaffold": cand["scaffold"], "pIC50": float(cand["pIC50"]),
            })
            selected_ids.add(cand["chembl_id"])
            added_this_pass += 1
        pass_idx += 1
        if added_this_pass == 0:
            break

    return pd.DataFrame(selected)


# --------------------------------------------------------------------------
# Direct-absolute model
# --------------------------------------------------------------------------
class DirectAbsoluteMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_loop_generic(model: nn.Module, train_loader: DataLoader,
                       val_loader: DataLoader | None, lr: float,
                       max_epochs: int, patience: int, forward_fn) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        for batch in train_loader:
            pred, y = forward_fn(model, batch)
            opt.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val_loader is not None:
            model.eval()
            vl = []
            with torch.no_grad():
                for batch in val_loader:
                    pred, y = forward_fn(model, batch)
                    vl.append(loss_fn(pred, y).item())
            v = float(np.mean(vl))
            sched.step(v)
            if v < best - 1e-5:
                best = v
                best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------
# Kinase pretraining (per-target, excluding the target itself)
# --------------------------------------------------------------------------
def build_kinase_pretrain_pool(all_pairs: pd.DataFrame, exclude_targets: List[str],
                               max_pairs: int) -> pd.DataFrame:
    # Use shared_pairs preferred (smaller, cleaner) but expand to all_pairs to fill.
    pool = all_pairs[(all_pairs["is_within_assay"] == True) &
                     (~all_pairs["target_chembl_id"].isin(exclude_targets))]
    # Take a kinase-y subset by filtering to other known kinase targets if available.
    # Heuristic: pick known kinase target IDs already in the dataset (CHEMBL2971=JAK2, etc.)
    kinase_targets = {"CHEMBL1862", "CHEMBL258", "CHEMBL2599", "CHEMBL267",
                      "CHEMBL2971", "CHEMBL5251", "CHEMBL203", "CHEMBL5145",
                      "CHEMBL2803", "CHEMBL2189121", "CHEMBL5658", "CHEMBL4296",
                      "CHEMBL3717", "CHEMBL279", "CHEMBL4005", "CHEMBL2842",
                      "CHEMBL5407", "CHEMBL2492", "CHEMBL1075091", "CHEMBL2842"}
    kinase_targets -= set(exclude_targets)
    pool = pool[pool["target_chembl_id"].isin(kinase_targets)]
    if len(pool) > max_pairs:
        pool = pool.sample(n=max_pairs, random_state=SEED).reset_index(drop=True)
    return pool


def pretrain_filmdelta(kinase_df: pd.DataFrame, fp_cache: Dict[str, np.ndarray],
                       epochs: int) -> dict:
    A = np.stack([fp_cache[s] for s in kinase_df["mol_a"]]).astype(np.float32)
    B = np.stack([fp_cache[s] for s in kinase_df["mol_b"]]).astype(np.float32)
    d = kinase_df["delta"].values.astype(np.float32)
    n_val = max(500, int(0.1 * len(d)))
    A_tr, A_v = A[:-n_val], A[-n_val:]
    B_tr, B_v = B[:-n_val], B[-n_val:]
    d_tr, d_v = d[:-n_val], d[-n_val:]

    m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(A_tr), torch.from_numpy(B_tr), torch.from_numpy(d_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(A_v), torch.from_numpy(B_v), torch.from_numpy(d_v)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    def fwd(model, batch):
        a, b, y = [t.float() for t in batch]
        return model(a, b), y
    m = train_loop_generic(m, train_loader, val_loader, PRETRAIN_LR, epochs, patience=3, forward_fn=fwd)
    return {k: v.cpu().clone() for k, v in m.state_dict().items()}


def pretrain_directabs(kinase_df: pd.DataFrame, fp_cache: Dict[str, np.ndarray],
                       epochs: int) -> dict:
    rows = []
    rows.extend(zip(kinase_df["mol_a_id"], kinase_df["mol_a"], kinase_df["value_a"]))
    rows.extend(zip(kinase_df["mol_b_id"], kinase_df["mol_b"], kinase_df["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id", "smiles", "value"]) \
            .groupby("mol_id", as_index=False).agg({"smiles": "first", "value": "mean"})
    X = np.stack([fp_cache[s] for s in df["smiles"]]).astype(np.float32)
    y = df["value"].values.astype(np.float32)
    n_v = max(200, int(0.1 * len(y)))
    X_tr, X_v = X[:-n_v], X[-n_v:]
    y_tr, y_v = y[:-n_v], y[-n_v:]
    m = DirectAbsoluteMLP(N_BITS, HIDDEN_DIMS, DROPOUT)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    def fwd(model, batch):
        x, y = [t.float() for t in batch]
        return model(x), y
    m = train_loop_generic(m, train_loader, val_loader, PRETRAIN_LR, epochs, patience=3, forward_fn=fwd)
    return {k: v.cpu().clone() for k, v in m.state_dict().items()}


# --------------------------------------------------------------------------
# Target finetune + 10% held-out eval
# --------------------------------------------------------------------------
def finetune_and_eval_filmdelta(train_pairs: pd.DataFrame, fp_cache, pretrained_state: dict,
                                save_path: Path) -> Dict:
    # Mol-disjoint 10% held-out
    mols = sorted(set(train_pairs["mol_a_id"]).union(train_pairs["mol_b_id"]))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(mols))
    n_test_mols = max(1, int(HELDOUT_FRAC * len(mols)))
    test_mols = set(mols[i] for i in perm[:n_test_mols])
    test_mask = train_pairs["mol_a_id"].isin(test_mols) & train_pairs["mol_b_id"].isin(test_mols)
    train_mask = ~train_pairs["mol_a_id"].isin(test_mols) & ~train_pairs["mol_b_id"].isin(test_mols)
    train_split = train_pairs[train_mask].reset_index(drop=True)
    test_split = train_pairs[test_mask].reset_index(drop=True)

    if len(train_split) < 20 or len(test_split) < 5:
        return {"n_train": int(len(train_split)), "n_test": int(len(test_split)),
                "mae": float("nan"), "spearman": float("nan"), "pearson": float("nan"),
                "note": "insufficient_data"}

    A_tr = np.stack([fp_cache[s] for s in train_split["mol_a"]]).astype(np.float32)
    B_tr = np.stack([fp_cache[s] for s in train_split["mol_b"]]).astype(np.float32)
    d_tr = train_split["delta"].values.astype(np.float32)
    A_te = np.stack([fp_cache[s] for s in test_split["mol_a"]]).astype(np.float32)
    B_te = np.stack([fp_cache[s] for s in test_split["mol_b"]]).astype(np.float32)
    d_te = test_split["delta"].values.astype(np.float32)

    # Internal val (10% of train)
    n_val = max(20, int(0.1 * len(train_split)))
    rng2 = np.random.RandomState(SEED + 1)
    perm2 = rng2.permutation(len(train_split))
    val_idx = perm2[:n_val]
    fit_idx = perm2[n_val:]
    A_fit, B_fit, d_fit = A_tr[fit_idx], B_tr[fit_idx], d_tr[fit_idx]
    A_val, B_val, d_val = A_tr[val_idx], B_tr[val_idx], d_tr[val_idx]

    m = FiLMDeltaMLP(input_dim=N_BITS, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    if pretrained_state is not None:
        m.load_state_dict(pretrained_state)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(A_fit), torch.from_numpy(B_fit), torch.from_numpy(d_fit)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(A_val), torch.from_numpy(B_val), torch.from_numpy(d_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    def fwd(model, batch):
        a, b, y = [t.float() for t in batch]
        return model(a, b), y
    lr_use = FINETUNE_LR if pretrained_state is not None else LR
    m = train_loop_generic(m, train_loader, val_loader, lr_use, MAX_EPOCHS, PATIENCE, fwd)

    # Test
    m.eval()
    with torch.no_grad():
        d_pred = m(torch.from_numpy(A_te), torch.from_numpy(B_te)).cpu().numpy()
    mae = float(np.mean(np.abs(d_pred - d_te)))
    if np.std(d_pred) < 1e-9:
        pr, sr = 0.0, 0.0
    else:
        pr = float(scipy_stats.pearsonr(d_pred, d_te)[0])
        sr = float(scipy_stats.spearmanr(d_pred, d_te)[0])

    # Save checkpoint
    torch.save({
        "model_state_dict": m.state_dict(),
        "hyperparameters": {
            "input_dim": N_BITS, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
            "spectral": False, "modulation_strength": 1.0,
            "learning_rate": FINETUNE_LR,
        },
        "fp_config": {"n_bits": N_BITS, "radius": RADIUS, "type": "morgan"},
    }, save_path)

    return {
        "n_train_pairs_fit": int(len(fit_idx)),
        "n_train_pairs_val": int(len(val_idx)),
        "n_test_pairs": int(len(test_split)),
        "n_test_mols_held_out": int(n_test_mols),
        "delta_mae": mae, "delta_pearson": pr, "delta_spearman": sr,
        "model_path": str(save_path),
    }


def finetune_and_eval_directabs(train_pairs: pd.DataFrame, fp_cache, pretrained_state: dict,
                                save_path: Path) -> Dict:
    # Mol-disjoint 10% held-out on UNIQUE MOLS (predict pIC50)
    rows = []
    rows.extend(zip(train_pairs["mol_a_id"], train_pairs["mol_a"], train_pairs["value_a"]))
    rows.extend(zip(train_pairs["mol_b_id"], train_pairs["mol_b"], train_pairs["value_b"]))
    df = pd.DataFrame(rows, columns=["mol_id", "smiles", "value"]) \
            .groupby("mol_id", as_index=False).agg({"smiles": "first", "value": "mean"})

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_test = max(5, int(HELDOUT_FRAC * len(df)))
    test_df = df.iloc[perm[:n_test]].reset_index(drop=True)
    train_df = df.iloc[perm[n_test:]].reset_index(drop=True)

    X_tr = np.stack([fp_cache[s] for s in train_df["smiles"]]).astype(np.float32)
    y_tr = train_df["value"].values.astype(np.float32)
    X_te = np.stack([fp_cache[s] for s in test_df["smiles"]]).astype(np.float32)
    y_te = test_df["value"].values.astype(np.float32)

    # Internal val
    n_val = max(20, int(0.1 * len(train_df)))
    rng2 = np.random.RandomState(SEED + 1)
    perm2 = rng2.permutation(len(train_df))
    val_idx, fit_idx = perm2[:n_val], perm2[n_val:]
    X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]

    m = DirectAbsoluteMLP(N_BITS, HIDDEN_DIMS, DROPOUT)
    if pretrained_state is not None:
        m.load_state_dict(pretrained_state)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    def fwd(model, batch):
        x, y = [t.float() for t in batch]
        return model(x), y
    lr_use = FINETUNE_LR if pretrained_state is not None else LR
    m = train_loop_generic(m, train_loader, val_loader, lr_use, MAX_EPOCHS, PATIENCE, fwd)

    m.eval()
    with torch.no_grad():
        y_pred = m(torch.from_numpy(X_te)).cpu().numpy()
    mae = float(np.mean(np.abs(y_pred - y_te)))
    if np.std(y_pred) < 1e-9:
        pr, sr = 0.0, 0.0
    else:
        pr = float(scipy_stats.pearsonr(y_pred, y_te)[0])
        sr = float(scipy_stats.spearmanr(y_pred, y_te)[0])

    torch.save({
        "model_state_dict": m.state_dict(),
        "hyperparameters": {
            "input_dim": N_BITS, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
        },
        "fp_config": {"n_bits": N_BITS, "radius": RADIUS, "type": "morgan"},
        "training_value_mean": float(np.mean(y_tr)),
        "training_value_std": float(np.std(y_tr)),
    }, save_path)

    return {
        "n_train_mols_fit": int(len(fit_idx)),
        "n_train_mols_val": int(len(val_idx)),
        "n_test_mols": int(len(test_df)),
        "abs_mae": mae, "abs_pearson": pr, "abs_spearman": sr,
        "model_path": str(save_path),
    }


# --------------------------------------------------------------------------
# Per-target main
# --------------------------------------------------------------------------
def process_target(target_key: str, cfg: Dict,
                   all_pairs: pd.DataFrame,
                   pic50_can_to_id: Dict[str, str] | None) -> Dict:
    out_dir = OUT_BASE / target_key
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(s: str):
        print(f"[{target_key}] {s}", flush=True)
        log_lines.append(s)

    t0 = time.time()
    log(f"=== {cfg['label']} ===")

    # 1) Target pairs
    target_pairs = all_pairs[(all_pairs["is_within_assay"] == True) &
                             (all_pairs["target_chembl_id"].isin(cfg["target_chembl_ids"]))].copy()
    # Construct a unified assay_id (within-assay -> assay_id_a == assay_id_b)
    target_pairs["assay_id"] = target_pairs["assay_id_a"]
    target_pairs = target_pairs[["assay_id", "mol_a", "mol_b", "mol_a_id", "mol_b_id",
                                 "value_a", "value_b", "delta", "target_chembl_id"]].reset_index(drop=True)
    target_pairs.to_csv(out_dir / "target_pairs.csv", index=False)
    n_mols = len(set(target_pairs["mol_a_id"]).union(target_pairs["mol_b_id"]))
    n_assays = target_pairs["assay_id"].nunique()
    log(f"  target_pairs: {len(target_pairs)} pairs, {n_mols} mols, {n_assays} assays")

    if len(target_pairs) < 50:
        log(f"  WARNING: insufficient target_pairs ({len(target_pairs)} < 50)")

    # 2) Exclude set
    excl_df, excl_report = build_exclude_set(target_pairs, cfg, pic50_can_to_id, cfg["anchor_chembl"])
    excl_df.to_csv(out_dir / "exclude_set.csv", index=False)
    log(f"  exclude_set: {len(excl_df)} mols "
        f"(tc_analogs={excl_report['n_tc_analogs']}, manual_added={excl_report['n_manual_added']})")
    caught = [s for s in excl_report["successor_status"] if s["caught"]]
    missed = [s for s in excl_report["successor_status"] if not s["caught"]
              and s["reason"] != "invalid_smiles"]
    log(f"  successors caught by Tc>={TC_DRUG_THRESHOLD}: "
        f"{[s['name'] for s in caught]}")
    log(f"  successors added manually:      "
        f"{[(s['name'], round(s['tc_to_drug'], 3) if s['tc_to_drug'] else None) for s in missed]}")

    # 3) Train pairs (exclude any pair touching excluded mol)
    excl_ids = set(excl_df["chembl_id"])
    keep_mask = (~target_pairs["mol_a_id"].isin(excl_ids)) & (~target_pairs["mol_b_id"].isin(excl_ids))
    train_pairs = target_pairs[keep_mask].reset_index(drop=True)
    train_pairs.to_csv(out_dir / "train_pairs.csv", index=False)
    pct_lost = 100.0 * (1.0 - len(train_pairs) / max(1, len(target_pairs)))
    log(f"  train_pairs: {len(train_pairs)} (lost {pct_lost:.1f}%)")

    if len(train_pairs) < 50:
        log(f"  CRITICAL: <50 train pairs after exclusion — Phase 2 NO-GO")
        # Still write what we can
        summary = {
            "target_key": target_key, "label": cfg["label"],
            "n_target_pairs": int(len(target_pairs)),
            "n_exclude": int(len(excl_df)),
            "n_train_pairs": int(len(train_pairs)),
            "pct_lost": float(pct_lost),
            "go_no_go": "NO-GO",
            "reason": "insufficient train pairs after exclusion",
            "exclude_report": excl_report,
        }
        with open(out_dir / "prep_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        write_target_summary_md(target_key, cfg, summary, log_lines, out_dir)
        return summary

    # 4) Diverse anchor pool
    anchor_df = diverse_anchor_pool(target_pairs, excl_ids,
                                    cfg["anchor_chembl"], cfg["anchor_smiles"],
                                    N_DIVERSE_ANCHORS)
    anchor_df.to_csv(out_dir / "anchor_pool_strategy_b.csv", index=False)
    log(f"  anchor_pool: {len(anchor_df)} mols, "
        f"unique_scaffolds={anchor_df['scaffold'].nunique()}, "
        f"pIC50 median={anchor_df['pIC50'].median():.2f} "
        f"[{anchor_df['pIC50'].min():.2f}, {anchor_df['pIC50'].max():.2f}]")

    # 5) Warhead SMARTS
    warhead_json = {
        "target_label": cfg["label"],
        "anchor": cfg["anchor_name"],
        "anchor_smiles": cfg["anchor_smiles"],
        "anchor_warhead_role": cfg["warhead_anchor_role"],
        "smarts_strict": cfg["warhead_smarts"]["strict"],
        "smarts_generic": cfg["warhead_smarts"]["generic"],
        "tc_threshold_for_exclude_analog": TC_DRUG_THRESHOLD,
    }
    # Validate anchor matches strict & generic SMARTS
    anchor_mol = Chem.MolFromSmiles(cfg["anchor_smiles"])
    for tag in ["strict", "generic"]:
        patt = Chem.MolFromSmarts(cfg["warhead_smarts"][tag])
        warhead_json[f"anchor_matches_{tag}"] = (
            bool(anchor_mol.HasSubstructMatch(patt)) if patt is not None else False
        )
    with open(out_dir / "warhead_smarts.json", "w") as f:
        json.dump(warhead_json, f, indent=2)
    log(f"  warhead: strict={warhead_json['smarts_strict']} (anchor matches: {warhead_json['anchor_matches_strict']}), "
        f"generic={warhead_json['smarts_generic']} (anchor matches: {warhead_json['anchor_matches_generic']})")

    # 6) Train models — build FP cache first
    log("  building Morgan FP cache for train + kinase pretrain...")
    kinase_pool = build_kinase_pretrain_pool(all_pairs, cfg["target_chembl_ids"], MAX_KINASE_PAIRS)
    log(f"  kinase_pretrain_pool: {len(kinase_pool)} pairs across "
        f"{kinase_pool['target_chembl_id'].nunique()} targets")
    smis = set()
    for df in [train_pairs, kinase_pool]:
        smis.update(df["mol_a"].tolist())
        smis.update(df["mol_b"].tolist())
    fp_cache = {s: smi_to_morgan(s) for s in smis}

    # Kinase pretrain (per-target, target-leakage-free)
    log("  pretraining FiLMDelta on kinase pool...")
    film_pre = pretrain_filmdelta(kinase_pool, fp_cache, PRETRAIN_EPOCHS)
    log("  pretraining DirectAbs on kinase pool...")
    dabs_pre = pretrain_directabs(kinase_pool, fp_cache, PRETRAIN_EPOCHS)

    log("  finetuning FiLMDelta on target train_pairs...")
    film_path = out_dir / "filmdelta.pt"
    film_res = finetune_and_eval_filmdelta(train_pairs, fp_cache, film_pre, film_path)
    log(f"  FiLMDelta held-out: MAE={film_res.get('delta_mae', float('nan')):.3f} "
        f"Spearman={film_res.get('delta_spearman', float('nan')):.3f} "
        f"(n_test_pairs={film_res.get('n_test_pairs', 0)}); saved {film_path.exists()}")

    log("  finetuning DirectAbs on target train_pairs...")
    dabs_path = out_dir / "directabs.pt"
    dabs_res = finetune_and_eval_directabs(train_pairs, fp_cache, dabs_pre, dabs_path)
    log(f"  DirectAbs held-out: MAE={dabs_res.get('abs_mae', float('nan')):.3f} "
        f"Spearman={dabs_res.get('abs_spearman', float('nan')):.3f} "
        f"(n_test_mols={dabs_res.get('n_test_mols', 0)}); saved {dabs_path.exists()}")

    # 7) Summary
    elapsed = time.time() - t0
    summary = {
        "target_key": target_key,
        "label": cfg["label"],
        "target_chembl_ids": cfg["target_chembl_ids"],
        "anchor": {"chembl_id": cfg["anchor_chembl"], "name": cfg["anchor_name"], "smiles": cfg["anchor_smiles"]},
        "drug": {"chembl_id": cfg["drug_chembl"], "name": cfg["drug_name"], "smiles": cfg["drug_smiles"]},
        "n_target_pairs": int(len(target_pairs)),
        "n_unique_mols": int(n_mols),
        "n_assays": int(n_assays),
        "n_exclude": int(len(excl_df)),
        "n_train_pairs": int(len(train_pairs)),
        "pct_lost_to_exclusion": float(pct_lost),
        "successors_caught": [s["name"] for s in caught],
        "successors_added_manually": [s["name"] for s in missed],
        "anchor_pool_size": int(len(anchor_df)),
        "anchor_pool_scaffolds": int(anchor_df["scaffold"].nunique()),
        "anchor_pool_pIC50_median": float(anchor_df["pIC50"].median()),
        "anchor_pool_pIC50_min": float(anchor_df["pIC50"].min()),
        "anchor_pool_pIC50_max": float(anchor_df["pIC50"].max()),
        "warhead": warhead_json,
        "filmdelta_eval": film_res,
        "directabs_eval": dabs_res,
        "kinase_pretrain_n_pairs": int(len(kinase_pool)),
        "elapsed_sec": round(elapsed, 1),
        "go_no_go": "GO" if len(train_pairs) >= 500 else ("CONDITIONAL" if len(train_pairs) >= 100 else "NO-GO"),
        "files_written": [
            "target_pairs.csv", "exclude_set.csv", "train_pairs.csv",
            "anchor_pool_strategy_b.csv", "warhead_smarts.json",
            "filmdelta.pt", "directabs.pt", "summary.md", "prep_summary.json",
        ],
        "exclude_report": excl_report,
    }
    with open(out_dir / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_target_summary_md(target_key, cfg, summary, log_lines, out_dir)
    log(f"  DONE in {elapsed/60:.1f} min — verdict: {summary['go_no_go']}")

    # cleanup
    del fp_cache, kinase_pool
    gc.collect()
    return summary


# --------------------------------------------------------------------------
# Per-target summary markdown
# --------------------------------------------------------------------------
def write_target_summary_md(target_key: str, cfg: Dict, summary: Dict,
                            log_lines: List[str], out_dir: Path) -> None:
    lines = [f"# {cfg['label']} prep summary\n\n"]
    lines.append(f"- Target ChEMBL IDs: {', '.join(cfg['target_chembl_ids'])}\n")
    lines.append(f"- Anchor: {cfg['anchor_name']} ({cfg['anchor_chembl']})\n")
    lines.append(f"- Drug: {cfg['drug_name']} ({cfg['drug_chembl']})\n")
    lines.append(f"- n target pairs: {summary.get('n_target_pairs', 'n/a')} "
                 f"({summary.get('n_unique_mols', 'n/a')} mols, "
                 f"{summary.get('n_assays', 'n/a')} assays)\n")
    lines.append(f"- n exclude set: {summary.get('n_exclude', 'n/a')} "
                 f"(drug + Tc>={TC_DRUG_THRESHOLD} analogs + manual successors)\n")
    lines.append(f"- Clinical successors caught (Tc>={TC_DRUG_THRESHOLD}): "
                 f"{summary.get('successors_caught', [])}\n")
    lines.append(f"- Clinical successors added manually:               "
                 f"{summary.get('successors_added_manually', [])}\n")
    lines.append(f"- n train pairs after exclusion: {summary.get('n_train_pairs', 'n/a')} "
                 f"(lost {summary.get('pct_lost_to_exclusion', 0):.1f}%)\n")
    if "anchor_pool_size" in summary:
        lines.append(f"- 100 diverse anchors: Murcko scaffold count = "
                     f"{summary['anchor_pool_scaffolds']}\n")
        lines.append(f"- Anchor pIC50 distribution: median {summary['anchor_pool_pIC50_median']:.2f}, "
                     f"range [{summary['anchor_pool_pIC50_min']:.2f}, "
                     f"{summary['anchor_pool_pIC50_max']:.2f}]\n")
    if "filmdelta_eval" in summary:
        fe = summary["filmdelta_eval"]
        lines.append(f"- FiLMDelta MAE on 10% held-out: {fe.get('delta_mae', float('nan')):.3f} "
                     f"(Spearman {fe.get('delta_spearman', float('nan')):.3f}, "
                     f"n={fe.get('n_test_pairs', 0)} pairs)\n")
    if "directabs_eval" in summary:
        de = summary["directabs_eval"]
        lines.append(f"- DirectAbs MAE on 10% held-out: {de.get('abs_mae', float('nan')):.3f} "
                     f"(Spearman {de.get('abs_spearman', float('nan')):.3f}, "
                     f"n={de.get('n_test_mols', 0)} mols)\n")
    if "warhead" in summary:
        wh = summary["warhead"]
        lines.append(f"- Warhead SMARTS (strict / generic): "
                     f"`{wh['smarts_strict']}` / `{wh['smarts_generic']}`\n")
        lines.append(f"  - Anchor matches strict: {wh.get('anchor_matches_strict', False)}; "
                     f"generic: {wh.get('anchor_matches_generic', False)}\n")
    lines.append(f"- Files written: {summary.get('files_written', [])}\n")
    lines.append(f"\n## GO/NO-GO: **{summary.get('go_no_go', 'unknown')}**\n")

    lines.append("\n## Run log\n```\n")
    for ll in log_lines:
        lines.append(f"{ll}\n")
    lines.append("```\n")
    with open(out_dir / "summary.md", "w") as f:
        f.writelines(lines)


# --------------------------------------------------------------------------
# Top-level summary
# --------------------------------------------------------------------------
def write_top_summary(per_target_summaries: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# EXP6 Phase 1 prep — top-level summary\n\n"]
    lines.append("Local retrospective LO setup across 3 kinase targets.\n\n")
    lines.append("## Table\n\n")
    lines.append("| Target | n_target_pairs | n_excluded | n_train_pairs | %lost | "
                 "scaffolds (anchors) | FiLM Δ-MAE (10%) | FiLM Δ-Spr | "
                 "DAbs abs-MAE (10%) | DAbs abs-Spr | GO/NO-GO |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for s in per_target_summaries:
        fe = s.get("filmdelta_eval", {}) or {}
        de = s.get("directabs_eval", {}) or {}
        lines.append(
            f"| {s['label']} | {s.get('n_target_pairs', '—')} | "
            f"{s.get('n_exclude', '—')} | {s.get('n_train_pairs', '—')} | "
            f"{s.get('pct_lost_to_exclusion', 0):.1f}% | "
            f"{s.get('anchor_pool_scaffolds', '—')} | "
            f"{fe.get('delta_mae', float('nan')):.3f} | "
            f"{fe.get('delta_spearman', float('nan')):.3f} | "
            f"{de.get('abs_mae', float('nan')):.3f} | "
            f"{de.get('abs_spearman', float('nan')):.3f} | "
            f"**{s.get('go_no_go', '?')}** |\n"
        )
    lines.append("\n## Per-target notes\n\n")
    for s in per_target_summaries:
        lines.append(f"### {s['label']}\n")
        lines.append(f"- Anchor: {s['anchor']['name']} ({s['anchor']['chembl_id']})\n")
        lines.append(f"- Drug: {s['drug']['name']} ({s['drug']['chembl_id']})\n")
        lines.append(f"- Warhead (strict / generic): "
                     f"`{s['warhead']['smarts_strict']}` / `{s['warhead']['smarts_generic']}`\n")
        lines.append(f"- Anchor matches strict: {s['warhead'].get('anchor_matches_strict', False)}; "
                     f"generic: {s['warhead'].get('anchor_matches_generic', False)}\n")
        caught = s.get("successors_caught", [])
        added = s.get("successors_added_manually", [])
        lines.append(f"- Successors caught by Tc>={TC_DRUG_THRESHOLD}: {caught}\n")
        lines.append(f"- Successors added manually:                {added}\n")
        lines.append(f"- Kinase pretrain pool: {s.get('kinase_pretrain_n_pairs', 0)} pairs "
                     f"(target itself excluded from pretrain)\n\n")
    # Recommended order
    lines.append("## Recommended Phase 2 RL order\n\n")
    ranked = sorted(per_target_summaries, key=lambda s: -s.get("n_train_pairs", 0))
    for i, s in enumerate(ranked, 1):
        lines.append(f"{i}. **{s['label']}** — {s.get('n_train_pairs', 0)} train pairs "
                     f"(verdict {s.get('go_no_go', '?')})\n")
    lines.append("\n## Concerns / flags\n\n")
    any_flag = False
    for s in per_target_summaries:
        if s.get("go_no_go") == "NO-GO":
            lines.append(f"- **{s['label']}**: NO-GO — insufficient train pairs.\n"); any_flag = True
        elif s.get("go_no_go") == "CONDITIONAL":
            lines.append(f"- **{s['label']}**: CONDITIONAL — only "
                         f"{s.get('n_train_pairs', 0)} train pairs; "
                         f"rely on kinase pretrain.\n"); any_flag = True
        # Flag if anchor doesn't match generic warhead
        if not s.get("warhead", {}).get("anchor_matches_generic", True):
            lines.append(f"- **{s['label']}**: anchor does NOT match generic warhead SMARTS — verify.\n")
            any_flag = True
    if not any_flag:
        lines.append("- None — all 3 targets ready for Phase 2.\n")
    with open(TOP_SUMMARY, "w") as f:
        f.writelines(lines)
    print(f"PROGRESS: wrote {TOP_SUMMARY}", flush=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PROGRESS: loading {PAIRS_CSV} ...", flush=True)
    t0 = time.time()
    all_pairs = pd.read_csv(PAIRS_CSV)
    print(f"PROGRESS: {len(all_pairs)} pairs loaded in {time.time()-t0:.1f}s", flush=True)

    pic50_can_to_id: Dict[str, str] | None = None
    if PIC50_CSV.exists():
        try:
            pic50_df = pd.read_csv(PIC50_CSV, usecols=["molecule_chembl_id", "canonical_smiles"])
            print(f"PROGRESS: pIC50 table: {len(pic50_df)} rows; building canonical map ...", flush=True)
            t1 = time.time()
            # Dedup by canonical_smiles first then canonicalize
            pic50_df = pic50_df.drop_duplicates("canonical_smiles")
            pic50_can_to_id = {}
            for cid, smi in zip(pic50_df["molecule_chembl_id"], pic50_df["canonical_smiles"]):
                c = _canon(smi)
                if c is not None and c not in pic50_can_to_id:
                    pic50_can_to_id[c] = cid
            print(f"PROGRESS: canonical map: {len(pic50_can_to_id)} entries in {time.time()-t1:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"PROGRESS: could not load/build pIC50 map: {e}", flush=True)
            pic50_can_to_id = None

    summaries = []
    for tkey, cfg in TARGETS.items():
        try:
            s = process_target(tkey, cfg, all_pairs, pic50_can_to_id)
            summaries.append(s)
        except Exception as e:
            import traceback
            print(f"PROGRESS: ERROR on {tkey}: {e}\n{traceback.format_exc()}", flush=True)
            summaries.append({"target_key": tkey, "label": cfg["label"],
                              "go_no_go": "ERROR", "error": str(e)})

    # Top-level summary
    write_top_summary(summaries)
    # Save combined JSON
    with open(RESULTS_DIR / "exp6_phase1_prep_summary.json", "w") as f:
        json.dump({"targets": summaries}, f, indent=2)
    print(f"PROGRESS: TOTAL TIME {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
