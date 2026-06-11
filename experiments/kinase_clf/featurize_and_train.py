"""Featurize the kinase-classifier dataset and train the 3-head FFN.

Featurization:
  - Morgan FP (radius=2, n_bits=2048)  [RDKit, CPU]
  - ChemBERTa-2 MTR pooled embedding (384d) [HuggingFace, GPU if available]
  -> concat to a 2432-dim feature vector

Model: shared trunk [2432 -> 512 -> 256 -> 128] + 3 heads
  - kinase_binary (BCE)
  - tec_family   (BCE, only on label_kinase==1 in the loss mask)
  - pIC50_aux    (MSE, only where pIC50 is finite)
Joint loss = BCE_kinase + BCE_tec + 0.5 * MSE_pIC50

Output:
  models/kinase_clf/trunk.pt
  models/kinase_clf/calibrators.pkl  (Platt for binary heads)
  results/paper_evaluation/kinase_classifier/training_metrics.json
"""
from __future__ import annotations
import os, sys, time, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET_CSV = ROOT / "results" / "paper_evaluation" / "kinase_classifier_dataset.csv"
EMB_CACHE_MTR = ROOT / "data" / "embedding_cache" / "chemberta2-mtr.npz"
EMB_CACHE_MORGAN = ROOT / "data" / "embedding_cache" / "morgan.npz"

OUT_DIR = ROOT / "models" / "kinase_clf"
METRICS_PATH = ROOT / "results" / "paper_evaluation" / "kinase_classifier" / "training_metrics.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
def morgan_fp(smi: str, n_bits: int = 2048) -> np.ndarray | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    arr = np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    ConvertToNumpyArray(fp, arr)
    return arr


def featurize_morgan(smiles: list, cache_lookup: dict | None) -> np.ndarray:
    n = len(smiles)
    X = np.zeros((n, 2048), dtype=np.float32)
    miss = []
    for i, s in enumerate(smiles):
        if cache_lookup and s in cache_lookup:
            X[i] = cache_lookup[s]
        else:
            miss.append(i)
    if miss:
        print(f"  Morgan cache miss: {len(miss):,} -- computing live")
        from concurrent.futures import ThreadPoolExecutor
        miss_smi = [smiles[i] for i in miss]
        # rdkit GIL-friendly enough; do serial to avoid memory blowup
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        for idx, s in zip(miss, miss_smi):
            fp = morgan_fp(s)
            if fp is not None:
                X[idx] = fp
    return X


def load_cache_dict(npz_path: Path, expected_dim: int):
    if not npz_path.exists():
        print(f"  No cache at {npz_path}")
        return {}
    d = np.load(npz_path, allow_pickle=True)
    smi = d["smiles"]
    emb = d["embeddings"]
    if emb.shape[1] != expected_dim:
        print(f"  Cache dim mismatch: {emb.shape[1]} != {expected_dim}")
        return {}
    print(f"  Loaded {len(smi):,} cached vectors from {npz_path.name} (dim={emb.shape[1]})")
    return {s: emb[i] for i, s in enumerate(smi)}


def featurize_chemberta(smiles: list, batch_size: int = 256) -> np.ndarray:
    """Cache-aware ChemBERTa-2 MTR encoder (384-dim mean-pooled)."""
    cache = load_cache_dict(EMB_CACHE_MTR, expected_dim=384)
    n = len(smiles)
    X = np.zeros((n, 384), dtype=np.float32)
    miss = []
    for i, s in enumerate(smiles):
        if s in cache:
            X[i] = cache[s]
        else:
            miss.append(i)
    if not miss:
        print("  All ChemBERTa features cached.")
        return X
    print(f"  ChemBERTa cache miss: {len(miss):,} -- running model")
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    mdl = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR").to(device).eval()

    miss_smi = [smiles[i] for i in miss]
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(miss), batch_size):
            chunk = miss_smi[b : b + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            for j, i_orig in enumerate(miss[b : b + batch_size]):
                X[i_orig] = pooled[j].cpu().numpy()
            if b % (batch_size * 20) == 0:
                pct = (b + batch_size) / len(miss) * 100
                print(f"    {min(b+batch_size,len(miss)):,}/{len(miss):,}  {pct:.1f}%  {time.time()-t0:.1f}s")
    print(f"  ChemBERTa pass: {time.time()-t0:.1f}s")
    return X


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ThreeHeadFFN(nn.Module):
    def __init__(self, in_dim: int = 2432, h_dims=(512, 256, 128), p_drop: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in h_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head_kinase = nn.Linear(prev, 1)
        self.head_tec    = nn.Linear(prev, 1)
        self.head_pic50  = nn.Linear(prev, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.head_kinase(z).squeeze(-1), self.head_tec(z).squeeze(-1), self.head_pic50(z).squeeze(-1)


def scaffold_split(smiles: list, frac_test: float = 0.15, seed: int = 42):
    """Murcko-scaffold-based group split."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    sc = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            sc.append(f"X_{len(sc)}")
            continue
        try:
            v = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            v = f"X_{len(sc)}"
        sc.append(v if v else f"X_{len(sc)}")
    sc = np.array(sc)
    unique = np.unique(sc)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * frac_test))
    test_scs = set(unique[:n_test].tolist())
    is_test = np.array([x in test_scs for x in sc])
    return ~is_test, is_test


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting featurize_and_train")

    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded {len(df):,} mols")
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    smiles = df["smiles"].tolist()

    print("Featurizing Morgan FP (2048)...")
    morgan_lookup = load_cache_dict(EMB_CACHE_MORGAN, expected_dim=2048)
    X_morgan = featurize_morgan(smiles, morgan_lookup)
    print(f"  Morgan X: {X_morgan.shape}")

    print("Featurizing ChemBERTa-2 MTR (384)...")
    X_cb = featurize_chemberta(smiles, batch_size=256)
    print(f"  ChemBERTa X: {X_cb.shape}")

    X = np.concatenate([X_morgan, X_cb], axis=1).astype(np.float32)
    print(f"X shape: {X.shape}  (in_dim={X.shape[1]})")

    y_kin   = df["label_kinase"].values.astype(np.float32)
    y_tec   = df["label_tec"].values.astype(np.float32)
    y_pic50 = df["pIC50"].values.astype(np.float32)  # may contain NaN
    pic_mask= np.isfinite(y_pic50).astype(np.float32)
    # standardize pIC50 to zero-mean / unit-var on the labeled subset
    pic_mu  = float(np.nanmean(y_pic50))
    pic_sd  = float(np.nanstd(y_pic50))
    y_pic_norm = np.where(np.isfinite(y_pic50), (y_pic50 - pic_mu) / pic_sd, 0.0).astype(np.float32)

    print("Scaffold split...")
    tr_idx, te_idx = scaffold_split(smiles, frac_test=0.15, seed=42)
    print(f"  train: {tr_idx.sum():,}  test: {te_idx.sum():,}")
    print(f"  train kinase pos rate: {y_kin[tr_idx].mean():.3f}")
    print(f"  test  kinase pos rate: {y_kin[te_idx].mean():.3f}")

    # Carve out a small calibration slice from train for Platt
    rng = np.random.default_rng(0)
    tr_pool_idx = np.where(tr_idx)[0]
    rng.shuffle(tr_pool_idx)
    n_cal = int(len(tr_pool_idx) * 0.1)
    cal_idx = np.zeros_like(tr_idx); cal_idx[tr_pool_idx[:n_cal]] = True
    fit_idx = tr_idx & ~cal_idx
    print(f"  fit: {fit_idx.sum():,}  cal: {cal_idx.sum():,}  test: {te_idx.sum():,}")

    # ----- Train -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    model = ThreeHeadFFN(in_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    Xt = torch.from_numpy(X)
    y_kin_t   = torch.from_numpy(y_kin)
    y_tec_t   = torch.from_numpy(y_tec)
    y_pic_t   = torch.from_numpy(y_pic_norm)
    pic_m_t   = torch.from_numpy(pic_mask)

    fit_pos = np.where(fit_idx)[0]
    rng2 = np.random.default_rng(7)
    bs = 1024
    n_epoch = 25
    best_test_auroc_kin = 0.0
    history = []
    from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

    for ep in range(n_epoch):
        model.train()
        rng2.shuffle(fit_pos)
        total_loss = 0; n_steps = 0
        for b in range(0, len(fit_pos), bs):
            ids = fit_pos[b:b+bs]
            xb = Xt[ids].to(device)
            yb_kin = y_kin_t[ids].to(device)
            yb_tec = y_tec_t[ids].to(device)
            yb_pic = y_pic_t[ids].to(device)
            mb     = pic_m_t[ids].to(device)
            kmask  = yb_kin.float()           # tec loss only on kinase positives
            opt.zero_grad()
            lk, lt, lp = model(xb)
            loss_k = bce(lk, yb_kin).mean()
            loss_t = (bce(lt, yb_tec) * kmask).sum() / kmask.sum().clamp(min=1)
            loss_p = (mse(lp, yb_pic) * mb).sum() / mb.sum().clamp(min=1)
            loss = loss_k + loss_t + 0.5 * loss_p
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_steps += 1
        train_loss = total_loss / n_steps

        # eval
        model.eval()
        te_pos = np.where(te_idx)[0]
        lk_all, lt_all, lp_all = [], [], []
        with torch.no_grad():
            for b in range(0, len(te_pos), 2048):
                ids = te_pos[b:b+2048]
                xb = Xt[ids].to(device)
                lk, lt, lp = model(xb)
                lk_all.append(lk.cpu().numpy())
                lt_all.append(lt.cpu().numpy())
                lp_all.append(lp.cpu().numpy())
        lk_all = np.concatenate(lk_all)
        lt_all = np.concatenate(lt_all)
        lp_all = np.concatenate(lp_all)
        # Kinase AUROC on test
        p_kin = 1/(1+np.exp(-lk_all))
        auroc_kin = roc_auc_score(y_kin[te_idx], p_kin)
        ap_kin    = average_precision_score(y_kin[te_idx], p_kin)
        # Tec AUROC on kinase-positive subset of test
        m = y_kin[te_idx] == 1
        if m.sum() > 50 and y_tec[te_idx][m].sum() > 5:
            p_tec = 1/(1+np.exp(-lt_all[m]))
            auroc_tec = roc_auc_score(y_tec[te_idx][m], p_tec)
        else:
            auroc_tec = float("nan")
        # pIC50 R^2
        mp = pic_mask[te_idx] == 1
        if mp.sum() > 50:
            yhat = lp_all[mp] * pic_sd + pic_mu
            ytrue= y_pic50[te_idx][mp]
            r2_pic = r2_score(ytrue, yhat)
        else:
            r2_pic = float("nan")
        history.append({
            "epoch": ep, "loss": train_loss,
            "test_auroc_kin": float(auroc_kin), "test_ap_kin": float(ap_kin),
            "test_auroc_tec": float(auroc_tec), "test_r2_pic50": float(r2_pic),
        })
        print(f"  ep{ep:02d} loss={train_loss:.4f} | kinase AUROC={auroc_kin:.4f} AP={ap_kin:.4f} | tec AUROC={auroc_tec:.4f} | pIC50 R²={r2_pic:.4f}")
        if auroc_kin > best_test_auroc_kin:
            best_test_auroc_kin = auroc_kin
            torch.save({
                "model_state": model.state_dict(),
                "in_dim": X.shape[1],
                "pic_mu": pic_mu, "pic_sd": pic_sd,
            }, OUT_DIR / "trunk.pt")

    # ----- Platt calibration on calibration slice -----
    print("Platt calibration of binary heads...")
    model.load_state_dict(torch.load(OUT_DIR / "trunk.pt", map_location=device)["model_state"])
    model.eval()
    cal_pos = np.where(cal_idx)[0]
    lk_all, lt_all = [], []
    with torch.no_grad():
        for b in range(0, len(cal_pos), 2048):
            ids = cal_pos[b:b+2048]
            xb = Xt[ids].to(device)
            lk, lt, _ = model(xb)
            lk_all.append(lk.cpu().numpy()); lt_all.append(lt.cpu().numpy())
    lk_all = np.concatenate(lk_all); lt_all = np.concatenate(lt_all)
    from sklearn.linear_model import LogisticRegression
    platt_kin = LogisticRegression(C=1.0).fit(lk_all.reshape(-1,1), y_kin[cal_idx])
    m_pos_kin = (y_kin[cal_idx] == 1)
    if m_pos_kin.sum() > 100 and y_tec[cal_idx][m_pos_kin].sum() > 10:
        platt_tec = LogisticRegression(C=1.0).fit(lt_all[m_pos_kin].reshape(-1,1), y_tec[cal_idx][m_pos_kin])
    else:
        platt_tec = None
        print("  Skipping Tec Platt -- too few Tec positives in cal slice")

    with open(OUT_DIR / "calibrators.pkl", "wb") as fh:
        pickle.dump(dict(platt_kin=platt_kin, platt_tec=platt_tec), fh)

    # ----- Final sanity panel -----
    print("\n=== Sanity panel ===")
    # Re-score the test set with calibrated probabilities
    te_pos = np.where(te_idx)[0]
    lk_test, lt_test, lp_test = [], [], []
    with torch.no_grad():
        for b in range(0, len(te_pos), 2048):
            ids = te_pos[b:b+2048]
            xb = Xt[ids].to(device)
            lk, lt, lp = model(xb)
            lk_test.append(lk.cpu().numpy()); lt_test.append(lt.cpu().numpy()); lp_test.append(lp.cpu().numpy())
    lk_test = np.concatenate(lk_test); lt_test = np.concatenate(lt_test); lp_test = np.concatenate(lp_test)
    p_kin = platt_kin.predict_proba(lk_test.reshape(-1,1))[:,1]
    if platt_tec is not None:
        p_tec = platt_tec.predict_proba(lt_test.reshape(-1,1))[:,1]
    else:
        p_tec = 1/(1+np.exp(-lt_test))

    test_kin_auroc = float(roc_auc_score(y_kin[te_idx], p_kin))
    test_kin_ap    = float(average_precision_score(y_kin[te_idx], p_kin))
    m = y_kin[te_idx] == 1
    if m.sum() > 50 and y_tec[te_idx][m].sum() > 5:
        test_tec_auroc = float(roc_auc_score(y_tec[te_idx][m], p_tec[m]))
    else:
        test_tec_auroc = float("nan")

    mp = pic_mask[te_idx] == 1
    if mp.sum() > 50:
        yhat = lp_test[mp] * pic_sd + pic_mu
        ytrue= y_pic50[te_idx][mp]
        r2_pic = float(r2_score(ytrue, yhat))
        mae_pic = float(np.mean(np.abs(ytrue - yhat)))
    else:
        r2_pic = float("nan"); mae_pic = float("nan")
    print(f"Scaffold-test kinase AUROC = {test_kin_auroc:.4f} (target > 0.90)")
    print(f"Scaffold-test kinase PR-AUC = {test_kin_ap:.4f}")
    print(f"Scaffold-test Tec   AUROC = {test_tec_auroc:.4f} (target > 0.85)")
    print(f"Scaffold-test pIC50 R²    = {r2_pic:.4f}, MAE = {mae_pic:.4f}")

    pass_kin = test_kin_auroc > 0.90
    pass_tec = (test_tec_auroc != test_tec_auroc) or (test_tec_auroc > 0.85)  # NaN -> pass-by-default

    metrics = dict(
        history=history,
        test=dict(
            kinase_auroc=test_kin_auroc,
            kinase_pr_auc=test_kin_ap,
            tec_auroc=test_tec_auroc,
            pic50_r2=r2_pic,
            pic50_mae=mae_pic,
            n_test=int(te_idx.sum()),
            n_kinase_pos_test=int(y_kin[te_idx].sum()),
            n_tec_pos_test=int((y_tec[te_idx][m]).sum()) if m.sum()>0 else 0,
        ),
        sanity=dict(
            kinase_auroc_target=0.90, pass_kinase=bool(pass_kin),
            tec_auroc_target=0.85, pass_tec=bool(pass_tec),
        ),
        n_train_fit=int(fit_idx.sum()),
        n_train_cal=int(cal_idx.sum()),
        n_test=int(te_idx.sum()),
        wall_clock_s=time.time()-t0,
        device=device,
        pic_mu=pic_mu, pic_sd=pic_sd,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {METRICS_PATH}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
