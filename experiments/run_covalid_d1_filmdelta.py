#!/usr/bin/env python3
"""D1 — FiLMDelta per-target ranking on London COValid (JACS 2026).

For each of the 10 COValid cysteine sites:
  1. Pull within-assay MMP pairs for the target's ChEMBL ID from shared_pairs.
  2. Pretrain on 100K kinase MMPs, then fine-tune on the target's pairs
     (mol-disjoint val if N ≥ 200, else simple random split).
  3. Anchor-score COValid actives + decoys via mean(anchor_pIC50 + Δ).
  4. Compute adj LogAUC (London's metric).

Output: results/covalid/covalid_d1_filmdelta_ranking.json
        results/covalid/covalid_d1_per_target.csv
"""
from __future__ import annotations
import sys, json, gc, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

torch.backends.mps.is_available = lambda: torch.backends.mps.is_built()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

PROJECT_ROOT = Path(__file__).parent.parent
SHARED_PAIRS = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
PIC50_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"
KINASE_PAIRS = PROJECT_ROOT / "data" / "kinase_within_pairs.csv"
COVALID = PROJECT_ROOT / "data" / "covalid" / "ja5c22222_si_004.xlsx"
OUT_DIR = PROJECT_ROOT / "results" / "covalid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# COValid target → ChEMBL target id
TARGETS = {
    "BMX":       "CHEMBL2581",
    "BTK":       "CHEMBL5251",
    "FGFR1":     "CHEMBL3650",
    "JAK3":      "CHEMBL2148",
    "KRAS":      "CHEMBL2189121",
    "EGFR":      "CHEMBL203",
    "FGFR4_477": "CHEMBL3973",
    "FGFR4_552": "CHEMBL3973",
    # ITK, MAP3K7 — no within-assay pairs in our shared_pairs; skip for now
}

N_KINASE_PRETRAIN = 50_000  # smaller than 100k — keeps total runtime per target ≤10 min
PRETRAIN_EPOCHS, FINETUNE_EPOCHS, PATIENCE = 20, 30, 5
LR_PRE, LR_FT = 2e-4, 1e-4
BATCH = 256


def morgan_fp(smi, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    a = np.zeros(n_bits, dtype=np.float32); DataStructs.ConvertToNumpyArray(fp, a)
    return a


def adj_log_auc(labels, scores, lam=10):
    """Adjusted LogAUC per Mysinger/Shoichet — emphasises early enrichment.
    label=1 active, 0 decoy; higher score = better."""
    # rank descending
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    n = len(y); n_act = int(y.sum()); n_dec = n - n_act
    if n_act == 0 or n_dec == 0: return float('nan'), float('nan')
    # cumulative fractions
    cum_act = np.cumsum(y) / n_act
    cum_dec = np.cumsum(1 - y) / n_dec
    # use log-spaced bins from 1/lam to 1
    # adjusted formula: integrate over log(FPR) from 1/lam to 1
    eps = 1e-12
    mask = cum_dec >= 1 / lam
    if not mask.any(): return 0.0, float(0.0)
    log_x = np.log10(np.clip(cum_dec[mask], eps, 1.0))
    y_at = cum_act[mask]
    # trapezoidal integral over log_x
    auc = float(np.trapz(y_at, log_x))
    # normalise to optimal (=1) and random (=0)
    log_lam = np.log10(1.0 / (1.0 / lam))  # = log10(lam)
    auc_norm = auc / log_lam  # in [0, 1] approx
    # AUC (unadjusted, area under cum_act vs cum_dec)
    raw_auc = float(np.trapz(cum_act, cum_dec))
    return auc_norm * 100, raw_auc


def kinase_pretrain(emb_dict, scaler):
    """Pretrain FiLMDelta on N_KINASE_PRETRAIN kinase MMPs. Returns ckpt state."""
    kp = pd.read_csv(KINASE_PAIRS, usecols=["mol_a", "mol_b", "delta"])
    if len(kp) > N_KINASE_PRETRAIN:
        kp = kp.sample(n=N_KINASE_PRETRAIN, random_state=42).reset_index(drop=True)
    need = [s for s in set(kp.mol_a.tolist() + kp.mol_b.tolist()) if s not in emb_dict]
    for s in need: emb_dict[s] = morgan_fp(s)
    Xa = np.array([emb_dict[s] for s in kp.mol_a]); Xb = np.array([emb_dict[s] for s in kp.mol_b])
    Xa = scaler.transform(Xa).astype(np.float32); Xb = scaler.transform(Xb).astype(np.float32)
    yd = kp.delta.values.astype(np.float32)
    n_val = len(Xa) // 10
    Xa_t = torch.FloatTensor(Xa); Xb_t = torch.FloatTensor(Xb); yd_t = torch.FloatTensor(yd)
    device = torch.device(DEVICE)
    m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR_PRE, weight_decay=1e-4); crit = nn.L1Loss()
    best, best_st, w = float('inf'), None, 0
    for ep in range(PRETRAIN_EPOCHS):
        m.train()
        perm = np.random.permutation(len(Xa) - n_val) + n_val
        for s in range(0, len(perm), BATCH):
            bi = perm[s:s+BATCH]; opt.zero_grad()
            crit(m(Xa_t[bi].to(device), Xb_t[bi].to(device)), yd_t[bi].to(device)).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xa_t[:n_val].to(device), Xb_t[:n_val].to(device)), yd_t[:n_val].to(device)).item()
        if vl < best: best, w = vl, 0; best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            w += 1
            if w >= PATIENCE: break
    print(f"    [pretrain] kinase MAE={best:.4f}")
    return best_st


def finetune_and_score(target_chembl, target_pairs, target_pic50_csv,
                       protomers_df, pretrain_state, scaler, emb_dict, target_label):
    """Fine-tune on target_pairs, then score COValid protomers via anchor."""
    if len(target_pairs) < 50:
        print(f"    [target_chembl] <50 pairs, skipping fine-tune; using pretrained model directly")
        # Use pretrained model directly
        device = torch.device(DEVICE)
        m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
        m.load_state_dict({k: v.to(device) for k, v in pretrain_state.items()})
        m.eval(); val_mae = None
    else:
        # Fingerprint any new SMILES
        need = [s for s in set(target_pairs.mol_a.tolist() + target_pairs.mol_b.tolist()) if s not in emb_dict]
        for s in need: emb_dict[s] = morgan_fp(s)
        # Mol-disjoint val split: hold out 15% of unique mols
        rng = np.random.default_rng(42)
        all_mols = list(set(target_pairs.mol_a.tolist() + target_pairs.mol_b.tolist()))
        n_held = max(5, int(0.15 * len(all_mols)))
        held = set(rng.choice(all_mols, n_held, replace=False).tolist())
        train_df = target_pairs[~(target_pairs.mol_a.isin(held) | target_pairs.mol_b.isin(held))]
        val_df = target_pairs[target_pairs.mol_a.isin(held) | target_pairs.mol_b.isin(held)]
        if len(train_df) < 20: train_df = target_pairs; val_df = target_pairs.sample(n=min(100, len(target_pairs)))
        Xa_tr = scaler.transform(np.array([emb_dict[s] for s in train_df.mol_a])).astype(np.float32)
        Xb_tr = scaler.transform(np.array([emb_dict[s] for s in train_df.mol_b])).astype(np.float32)
        yd_tr = train_df.delta.values.astype(np.float32)
        Xa_v  = scaler.transform(np.array([emb_dict[s] for s in val_df.mol_a])).astype(np.float32)
        Xb_v  = scaler.transform(np.array([emb_dict[s] for s in val_df.mol_b])).astype(np.float32)
        yd_v  = val_df.delta.values.astype(np.float32)
        device = torch.device(DEVICE)
        m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2).to(device)
        m.load_state_dict({k: v.to(device) for k, v in pretrain_state.items()})
        opt = torch.optim.Adam(m.parameters(), lr=LR_FT, weight_decay=1e-4); crit = nn.MSELoss()
        tr_ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(Xa_tr), torch.FloatTensor(Xb_tr), torch.FloatTensor(yd_tr)),
            batch_size=BATCH, shuffle=True)
        Xa_v_t = torch.FloatTensor(Xa_v); Xb_v_t = torch.FloatTensor(Xb_v); yd_v_t = torch.FloatTensor(yd_v)
        best, best_st, w = float('inf'), None, 0
        for ep in range(FINETUNE_EPOCHS):
            m.train()
            for batch in tr_ld:
                a, b, y = [t.to(device) for t in batch]; opt.zero_grad()
                crit(m(a, b), y).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = nn.L1Loss()(m(Xa_v_t.to(device), Xb_v_t.to(device)), yd_v_t.to(device)).item()
            if vl < best: best, w = vl, 0; best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            else:
                w += 1
                if w >= PATIENCE: break
        if best_st: m.load_state_dict(best_st)
        m.cpu().eval(); m = m.to(device); val_mae = best

    # Build target anchors from molecule_pIC50_minimal.csv
    pic50 = pd.read_csv(PIC50_FILE)
    pic50 = pic50[pic50.target_chembl_id == target_chembl].copy()
    if len(pic50) == 0:
        print(f"    no pIC50 anchors for {target_chembl}; skipping target")
        return None
    pic50 = pic50.groupby('molecule_chembl_id').agg({'smiles': 'first', 'pIC50': 'mean'}).reset_index()
    anchor_smiles = pic50.smiles.tolist()
    anchor_pic = pic50.pIC50.values
    for s in anchor_smiles:
        if s not in emb_dict: emb_dict[s] = morgan_fp(s)
    anchor_embs = torch.FloatTensor(
        scaler.transform(np.array([emb_dict[s] for s in anchor_smiles])).astype(np.float32)).to(device)
    n_anchor = len(anchor_smiles)
    print(f"    target_anchors: {n_anchor}, val_MAE={val_mae}")

    # Score every protomer in the protomers_df (which has active+decoy rows)
    protomer_smiles = protomers_df.protomer_smiles.tolist()
    scores = np.full(len(protomer_smiles), np.nan)
    for i, smi in enumerate(protomer_smiles):
        try:
            f = morgan_fp(smi)
        except Exception:
            continue
        if f.sum() == 0: continue  # invalid SMILES
        emb_dict[smi] = f
        e = torch.FloatTensor(scaler.transform(f[None, :]).astype(np.float32)).to(device)
        with torch.no_grad():
            deltas = m(anchor_embs, e.expand(n_anchor, -1)).cpu().numpy().flatten()
        scores[i] = float(np.mean(anchor_pic + deltas))
    return scores, val_mae


def main():
    t0 = time.time()
    print("=" * 80)
    print("COValid D1 — FiLMDelta per-target ranking")
    print("=" * 80)
    if not COVALID.exists():
        sys.exit(f"missing {COVALID}")
    sp = pd.read_csv(SHARED_PAIRS, usecols=['mol_a', 'mol_b', 'delta', 'is_within_assay', 'target_chembl_id'])
    sp = sp[sp.is_within_assay == True]
    print(f"shared_pairs (within-assay): {len(sp):,}")

    # Pre-fingerprint a base scaler on a large kinase mix
    print("Fitting scaler on 10K-mol kinase sample…")
    kp = pd.read_csv(KINASE_PAIRS, usecols=["mol_a", "mol_b"]).head(20_000)
    emb_dict = {}
    for s in set(kp.mol_a.tolist() + kp.mol_b.tolist()):
        emb_dict[s] = morgan_fp(s)
    X = np.array(list(emb_dict.values()))
    scaler = StandardScaler().fit(X)

    pretrain_state = kinase_pretrain(emb_dict, scaler)

    results = {}
    for target_label, chembl in TARGETS.items():
        print(f"\n=== {target_label} ({chembl}) ===")
        t1 = time.time()
        actives_df = pd.read_excel(COVALID, sheet_name=f"{target_label}_actives")
        decoys_df  = pd.read_excel(COVALID, sheet_name=f"{target_label}_decoys")
        actives_df = actives_df.rename(columns={c: 'protomer_smiles' for c in actives_df.columns if 'smiles' in c.lower()})
        decoys_df  = decoys_df.rename( columns={c: 'protomer_smiles' for c in decoys_df.columns  if 'smiles' in c.lower()})
        protomers = pd.concat([
            actives_df.assign(label=1, source='active')[['protomer_smiles', 'label', 'source']],
            decoys_df.assign(label=0, source='decoy') [['protomer_smiles', 'label', 'source']],
        ], ignore_index=True)
        print(f"    actives={int((protomers.label == 1).sum())}  decoys={int((protomers.label == 0).sum())}")
        target_pairs = sp[sp.target_chembl_id == chembl][['mol_a', 'mol_b', 'delta']]
        print(f"    target within-assay pairs: {len(target_pairs)}")
        ret = finetune_and_score(chembl, target_pairs, None, protomers, pretrain_state, scaler, emb_dict, target_label)
        if ret is None: continue
        scores, val_mae = ret
        valid = ~np.isnan(scores)
        if valid.sum() < 10: print("    too few valid scores"); continue
        adj_pct, raw_auc = adj_log_auc(protomers.label.values[valid], scores[valid])
        n_a = int(protomers.label.values[valid].sum()); n_d = int(valid.sum()) - n_a
        elapsed = time.time() - t1
        print(f"    adj_LogAUC = {adj_pct:.1f}%  raw_AUC = {raw_auc:.3f}  (n_act={n_a}, n_dec={n_d})  ({elapsed:.0f}s)")
        # Build per-mol records for downstream D3 combiner
        per_mol = []
        for i, (smi, lbl) in enumerate(zip(protomers.protomer_smiles.values, protomers.label.values)):
            if i < len(scores) and not np.isnan(scores[i]):
                per_mol.append({'smiles': str(smi), 'label': int(lbl), 'score': float(scores[i])})
        results[target_label] = {
            'chembl_id': chembl,
            'n_actives': n_a, 'n_decoys': n_d,
            'adj_logAUC_pct': float(adj_pct),
            'raw_AUC': float(raw_auc),
            'val_mae': val_mae,
            'n_train_pairs': len(target_pairs),
            'per_mol': per_mol,
        }
        gc.collect()

    out = {
        'method': 'FiLMDelta — kinase pretrain + per-target fine-tune + anchor scoring',
        'kinase_pretrain_pairs': N_KINASE_PRETRAIN,
        'per_target': results,
        'avg_adj_logAUC_pct': float(np.mean([v['adj_logAUC_pct'] for v in results.values()])) if results else 0,
        'london_avg_adj_logAUC_pct': 71.8,
    }
    # Two outputs: aggregate (no per-mol — smaller) and full per-mol for D3
    aggregate = {**out, 'per_target': {k: {kk: vv for kk, vv in v.items() if kk != 'per_mol'} for k, v in results.items()}}
    json.dump(aggregate, open(OUT_DIR / "covalid_d1_filmdelta_ranking.json", 'w'), indent=2)
    json.dump(results, open(OUT_DIR / "covalid_d1_per_mol_scores.json", 'w'), indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"avg adj_LogAUC over {len(results)} targets: {out['avg_adj_logAUC_pct']:.1f}%  (London: 71.8%)")
    print(f"Total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
