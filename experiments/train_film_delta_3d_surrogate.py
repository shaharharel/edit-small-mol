"""
Train the 3D SchNet siamese surrogate to imitate the cached FiLMDelta predictor.

Pipeline
--------
1. Load 280 ZAP70 anchors (from ``run_zap70_v3.load_zap70_molecules``).
2. Generate one ETKDGv3 conformer per anchor; cache to
   ``data/embedding_cache/zap70_3d_conformers.pkl``.
3. Build all 280×279 = 78,120 ordered pairs and compute the FiLMDelta
   *predicted* delta for each (target = imitation, NOT raw labels).
4. Hold out 10% of pairs for validation (random split on pairs).
5. Train ``FiLMDelta3DSurrogate`` with MSE; early-stop when val Pearson r >
   0.85 or after a max number of epochs.
6. Save to ``results/paper_evaluation/film_delta_3d_surrogate.pt``.

CPU only (forced).
"""

from __future__ import annotations

import os
# Force CPU before any torch import that touches MPS detection.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gc
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

# Force CPU for torch (MPS crashes after prolonged use with attention models).
torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
DEVICE = torch.device("cpu")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from experiments.run_zap70_v3 import load_zap70_molecules, compute_fingerprints  # noqa: E402
from src.models.predictors.film_delta_predictor import FiLMDeltaMLP  # noqa: E402
from src.models.predictors.film_delta_3d_surrogate import FiLMDelta3DSurrogate  # noqa: E402

CONFORMER_CACHE = PROJECT_ROOT / "data" / "embedding_cache" / "zap70_3d_conformers.pkl"
TEACHER_CACHE = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
SURROGATE_OUT = PROJECT_ROOT / "results" / "paper_evaluation" / "film_delta_3d_surrogate.pt"


# ---------------------------------------------------------------------------
# Conformer generation
# ---------------------------------------------------------------------------

def generate_conformer(smi: str, seed: int = 42):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        # Fallback: try with random coords
        params2 = AllChem.ETKDGv3()
        params2.useRandomCoords = True
        params2.randomSeed = seed
        cid = AllChem.EmbedMolecule(mol, params2)
    if cid < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    conf = mol.GetConformer()
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    pos = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    return {"z": z, "pos": pos, "smiles": smi, "n_atoms": int(z.shape[0])}


def build_or_load_conformers(smiles_list):
    if CONFORMER_CACHE.exists():
        print(f"[conf] Loading cached conformers from {CONFORMER_CACHE}")
        with open(CONFORMER_CACHE, "rb") as f:
            cache = pickle.load(f)
        # Verify coverage
        if all(s in cache for s in smiles_list):
            return cache
        print("[conf] Cache miss for some SMILES, regenerating delta...")
    else:
        cache = {}
    print(f"[conf] Generating conformers for {len(smiles_list)} molecules...")
    t0 = time.time()
    fail = 0
    for i, smi in enumerate(smiles_list):
        if smi in cache:
            continue
        c = generate_conformer(smi, seed=42 + i)
        if c is None:
            fail += 1
            continue
        cache[smi] = c
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(smiles_list)} done, {fail} fails, "
                  f"{time.time() - t0:.1f}s")
    CONFORMER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFORMER_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"[conf] {len(cache)}/{len(smiles_list)} cached, {fail} failed in "
          f"{time.time() - t0:.1f}s")
    return cache


# ---------------------------------------------------------------------------
# Teacher predictions
# ---------------------------------------------------------------------------

def load_teacher():
    print(f"[teacher] Loading FiLMDelta from {TEACHER_CACHE}")
    ckpt = torch.load(TEACHER_CACHE, map_location="cpu", weights_only=False)
    teacher = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return teacher, scaler


def teacher_deltas_all_pairs(teacher, scaler, smiles_list, batch_size=512):
    """Return dict[(i,j)->delta] for all i!=j. Predicts in batches."""
    fps = compute_fingerprints(smiles_list, "morgan", radius=2, n_bits=2048)
    if isinstance(fps, tuple):
        fps = fps[0]
    fps = np.array(fps)
    embs = scaler.transform(fps).astype(np.float32)
    embs_t = torch.from_numpy(embs)

    n = len(smiles_list)
    pair_idx_a, pair_idx_b = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_idx_a.append(i)
                pair_idx_b.append(j)
    pair_idx_a = np.array(pair_idx_a)
    pair_idx_b = np.array(pair_idx_b)
    n_pairs = len(pair_idx_a)
    print(f"[teacher] Predicting {n_pairs:,} pairs...")
    deltas = np.zeros(n_pairs, dtype=np.float32)
    teacher.eval()
    with torch.no_grad():
        for s in range(0, n_pairs, batch_size):
            ia = pair_idx_a[s:s + batch_size]
            ib = pair_idx_b[s:s + batch_size]
            xa = embs_t[ia]
            xb = embs_t[ib]
            d = teacher(xa, xb).numpy()
            deltas[s:s + batch_size] = d
    return pair_idx_a, pair_idx_b, deltas


# ---------------------------------------------------------------------------
# Batching for surrogate
# ---------------------------------------------------------------------------

def collate_pairs(idx_pairs, conformers, smiles_list, device=DEVICE):
    """Build batched (z, pos, batch) for a list of (i, j) pair indices.

    Returns tensors for the A side and B side.
    """
    za_list, pa_list, ba_list = [], [], []
    zb_list, pb_list, bb_list = [], [], []
    for graph_idx, (i, j) in enumerate(idx_pairs):
        ca = conformers[smiles_list[i]]
        cb = conformers[smiles_list[j]]
        za_list.append(torch.from_numpy(ca["z"]))
        pa_list.append(torch.from_numpy(ca["pos"]))
        ba_list.append(torch.full((ca["z"].shape[0],), graph_idx, dtype=torch.long))
        zb_list.append(torch.from_numpy(cb["z"]))
        pb_list.append(torch.from_numpy(cb["pos"]))
        bb_list.append(torch.full((cb["z"].shape[0],), graph_idx, dtype=torch.long))
    za = torch.cat(za_list, dim=0).to(device)
    pa = torch.cat(pa_list, dim=0).to(device)
    ba = torch.cat(ba_list, dim=0).to(device)
    zb = torch.cat(zb_list, dim=0).to(device)
    pb = torch.cat(pb_list, dim=0).to(device)
    bb = torch.cat(bb_list, dim=0).to(device)
    return za, pa, ba, zb, pb, bb


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(seed: int = 42, max_epochs: int = 40, batch_size: int = 32,
          lr: float = 1e-3, target_r: float = 0.85, val_frac: float = 0.10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- 1. Load anchors -----------------------------------------------------
    smiles_df, _ = load_zap70_molecules()
    smiles_list = smiles_df["smiles"].tolist()
    pIC50 = smiles_df["pIC50"].values.astype(np.float32)
    print(f"[anchors] {len(smiles_list)} ZAP70 molecules, pIC50 mean={pIC50.mean():.2f}")

    # -- 2. Conformers -------------------------------------------------------
    conformers = build_or_load_conformers(smiles_list)
    valid_smiles = [s for s in smiles_list if s in conformers]
    if len(valid_smiles) < len(smiles_list):
        # Re-index smiles_list / pIC50 to keep only molecules with conformers
        keep_idx = [i for i, s in enumerate(smiles_list) if s in conformers]
        smiles_list = [smiles_list[i] for i in keep_idx]
        pIC50 = pIC50[keep_idx]
        print(f"[anchors] After conformer filter: {len(smiles_list)} mols")
    n = len(smiles_list)

    # -- 3. Teacher predictions ---------------------------------------------
    teacher, scaler = load_teacher()
    pair_a, pair_b, target_deltas = teacher_deltas_all_pairs(
        teacher, scaler, smiles_list
    )
    n_pairs = len(pair_a)
    print(f"[teacher] target delta: mean={target_deltas.mean():.3f} "
          f"std={target_deltas.std():.3f} "
          f"min={target_deltas.min():.3f} max={target_deltas.max():.3f}")

    # -- 4. Train/val split (random over pairs) -----------------------------
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_pairs)
    n_val = int(n_pairs * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[split] {len(train_idx):,} train pairs, {len(val_idx):,} val pairs")

    # -- 5. Build the surrogate ---------------------------------------------
    surrogate = FiLMDelta3DSurrogate(
        hidden=128, num_interactions=3, cutoff=6.0,
        feature_dim=128, head_hidden=256, dropout=0.1,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in surrogate.parameters())
    print(f"[model] FiLMDelta3DSurrogate, {n_params:,} parameters")

    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    crit = nn.MSELoss()

    # Cache pair tuples
    train_pair_tuples = [(int(pair_a[i]), int(pair_b[i])) for i in train_idx]
    val_pair_tuples = [(int(pair_a[i]), int(pair_b[i])) for i in val_idx]
    train_targets = torch.from_numpy(target_deltas[train_idx])
    val_targets = torch.from_numpy(target_deltas[val_idx])

    best_val_r = -np.inf
    best_state = None
    best_val_mae = np.inf
    history = []

    for epoch in range(max_epochs):
        t0 = time.time()
        surrogate.train()
        order = rng.permutation(len(train_pair_tuples))
        losses = []
        for s in range(0, len(order), batch_size):
            bi = order[s:s + batch_size]
            pairs = [train_pair_tuples[k] for k in bi]
            targets = train_targets[bi].to(DEVICE)
            za, pa, ba, zb, pb, bb = collate_pairs(pairs, conformers, smiles_list)
            optimizer.zero_grad()
            pred = surrogate(za, pa, ba, zb, pb, bb)
            loss = crit(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surrogate.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        # Validation
        surrogate.eval()
        val_preds = np.zeros(len(val_pair_tuples), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(val_pair_tuples), 64):
                pairs = val_pair_tuples[s:s + 64]
                za, pa, ba, zb, pb, bb = collate_pairs(pairs, conformers, smiles_list)
                pred = surrogate(za, pa, ba, zb, pb, bb).cpu().numpy()
                val_preds[s:s + 64] = pred

        val_targets_np = val_targets.numpy()
        val_mae = float(np.mean(np.abs(val_preds - val_targets_np)))
        val_mse = float(np.mean((val_preds - val_targets_np) ** 2))
        if val_preds.std() < 1e-6 or val_targets_np.std() < 1e-6:
            val_r = 0.0
            val_sp = 0.0
        else:
            val_r = float(pearsonr(val_preds, val_targets_np)[0])
            val_sp = float(spearmanr(val_preds, val_targets_np)[0])

        scheduler.step(val_mse)
        train_loss = float(np.mean(losses))
        history.append({
            "epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_mse,
            "val_mae": val_mae, "val_pearson": val_r, "val_spearman": val_sp,
        })
        print(f"[ep {epoch + 1:02d}] train_mse={train_loss:.4f} "
              f"val_mse={val_mse:.4f} val_mae={val_mae:.4f} "
              f"val_r={val_r:.4f} val_sp={val_sp:.4f}  "
              f"({time.time() - t0:.0f}s)")

        if val_r > best_val_r:
            best_val_r = val_r
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in surrogate.state_dict().items()}

        if val_r >= target_r:
            print(f"[ok] Hit target Pearson r >= {target_r:.2f} "
                  f"at epoch {epoch + 1}.")
            break

    if best_state is not None:
        surrogate.load_state_dict(best_state)

    # -- 6. Save -------------------------------------------------------------
    SURROGATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": surrogate.state_dict(),
        "config": {
            "hidden": 128, "num_interactions": 3, "cutoff": 6.0,
            "feature_dim": 128, "head_hidden": 256, "dropout": 0.1,
        },
        "best_val_pearson": best_val_r,
        "best_val_mae": best_val_mae,
        "anchor_smiles": smiles_list,
        "anchor_pIC50": pIC50,
        "history": history,
    }, SURROGATE_OUT)
    print(f"[save] surrogate -> {SURROGATE_OUT}")
    print(f"[done] best val Pearson r = {best_val_r:.4f}, MAE = {best_val_mae:.4f}")
    return best_val_r, best_val_mae


if __name__ == "__main__":
    train()
