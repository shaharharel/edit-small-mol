"""Analyze COValid Boltz cofold results: extract mPAE per compound + compute adj_LogAUC per target.

Output:
  results/covalid/covalid_d2_boltz_mpae.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "covalid_boltz_results"
MANIFEST = PROJECT_ROOT / "experiments" / "boltz_inputs" / "covalid_top100_aligned" / "manifest.csv"
OUT = PROJECT_ROOT / "results" / "covalid" / "covalid_d2_boltz_mpae.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def chain_split(cif_path):
    """Count atoms in chain A (protein) and chain B (ligand)."""
    lines = open(cif_path).read().splitlines()
    cols, n_A, n_B = [], 0, 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            j = i + 1; col_local = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                col_local.append(lines[j].strip().removeprefix("_atom_site.")); j += 1
            if col_local:
                cols = col_local
                try: chain_idx = cols.index("auth_asym_id")
                except ValueError: chain_idx = cols.index("label_asym_id")
                i = j
                while i < len(lines) and not lines[i].startswith("#") and not lines[i].startswith("loop_") and lines[i].strip():
                    parts = lines[i].split()
                    if len(parts) == len(cols):
                        if parts[chain_idx] == "A": n_A += 1
                        elif parts[chain_idx] == "B": n_B += 1
                    i += 1
                continue
        i += 1
    return n_A, n_B


def mpae_from_npz(pae_npz, n_ligand):
    if not pae_npz.exists(): return None
    try:
        d = np.load(pae_npz)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception:
        return None
    if pae.ndim != 2: return None
    N = pae.shape[0]; lig_lo = N - n_ligand
    if lig_lo <= 0 or lig_lo >= N: return float(np.min(pae))
    cross = pae[:lig_lo, lig_lo:]
    return float(np.min(cross)) if cross.size else None


def confidence_score(conf_json):
    if not conf_json.exists(): return None
    try:
        d = json.loads(conf_json.read_text())
        return {
            'confidence_score': d.get('confidence_score'),
            'iptm': d.get('iptm'),
            'ligand_iptm': d.get('ligand_iptm'),
            'complex_plddt': d.get('complex_plddt'),
            'complex_pde': d.get('complex_pde'),
        }
    except Exception:
        return None


def adj_log_auc(labels, scores_ascending, lam=1000):
    """labels: 1=active 0=decoy. scores_ascending: LOWER is better (e.g. mPAE).
    We invert internally for ranking."""
    order = np.argsort(np.asarray(scores_ascending))  # ascending (low first = best)
    y = np.asarray(labels)[order]
    n_act = int(y.sum()); n_dec = len(y) - n_act
    if n_act == 0 or n_dec == 0: return float('nan'), float('nan')
    cum_act = np.cumsum(y) / n_act
    cum_dec = np.cumsum(1 - y) / n_dec
    mask = cum_dec >= 1/lam
    if not mask.any(): return 0.0, 0.0
    log_x = np.log10(np.clip(cum_dec[mask], 1e-12, 1.0))
    auc = float(np.trapz(cum_act[mask], log_x))
    raw = float(np.trapz(cum_act, cum_dec))
    return (auc / np.log10(lam)) * 100, raw


def main():
    print("Loading manifest…")
    man = pd.read_csv(MANIFEST)
    print(f"  {len(man)} YAMLs, {(man.is_active==1).sum()} actives, {(man.is_active==0).sum()} decoys")
    print(f"\nExtracting mPAE per cofold…")
    rows = []
    for _, r in man.iterrows():
        target = r['target']
        name = r['name']
        # Boltz writes outputs at: <out_dir>/boltz_results_<name>/predictions/<name>/<name>_model_0.cif
        cif = RESULTS_DIR / target / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.cif"
        pae = RESULTS_DIR / target / f"boltz_results_{name}" / "predictions" / name / f"pae_{name}_model_0.npz"
        conf = RESULTS_DIR / target / f"boltz_results_{name}" / "predictions" / name / f"confidence_{name}_model_0.json"
        if not cif.exists():
            continue
        n_A, n_B = chain_split(cif)
        if n_B == 0: continue
        mpae = mpae_from_npz(pae, n_B)
        conf_d = confidence_score(conf)
        rows.append({
            'name': name, 'target': target, 'is_active': int(r['is_active']),
            'smiles': r['smiles'], 'mPAE': mpae,
            **(conf_d or {}),
        })
    df = pd.DataFrame(rows)
    print(f"  Got {len(df)} cofolds")
    print(f"  per-target counts: {df.groupby('target').size().to_dict()}")
    print(f"\nComputing adj_LogAUC per target…")
    result = {'london_avg': 71.8, 'per_target': {}}
    for target, g in df.groupby('target'):
        gs = g.dropna(subset=['mPAE'])
        if gs['is_active'].sum() == 0 or (gs['is_active']==0).sum() == 0:
            print(f"  {target}: insufficient class balance (n_act={int(gs['is_active'].sum())}, n_dec={int((gs['is_active']==0).sum())})"); continue
        adj_pct, raw_auc = adj_log_auc(gs['is_active'].values, gs['mPAE'].values)
        # Also rank by iptm (HIGHER better, so we negate)
        if gs['ligand_iptm'].notna().any():
            gs2 = gs.dropna(subset=['ligand_iptm'])
            adj_iptm, _ = adj_log_auc(gs2['is_active'].values, -gs2['ligand_iptm'].values)
        else:
            adj_iptm = None
        print(f"  {target:10s}  n_act={int(gs['is_active'].sum())}  n_dec={int((gs['is_active']==0).sum())}  adj_LogAUC(mPAE)={adj_pct:5.1f}%  adj_LogAUC(lig_iPTM)={adj_iptm if adj_iptm is None else f'{adj_iptm:.1f}%'}")
        result['per_target'][target] = {
            'n_actives_with_cofold': int(gs['is_active'].sum()),
            'n_decoys_with_cofold': int((gs['is_active']==0).sum()),
            'adj_logAUC_pct_mPAE': float(adj_pct),
            'raw_AUC_mPAE': float(raw_auc),
            'adj_logAUC_pct_iptm': float(adj_iptm) if adj_iptm is not None else None,
            'median_mPAE_actives': float(gs.loc[gs.is_active==1, 'mPAE'].median()),
            'median_mPAE_decoys': float(gs.loc[gs.is_active==0, 'mPAE'].median()),
        }
    vals = [v['adj_logAUC_pct_mPAE'] for v in result['per_target'].values() if not np.isnan(v['adj_logAUC_pct_mPAE'])]
    result['our_avg_adj_logAUC_mPAE'] = float(np.mean(vals)) if vals else 0.0
    print(f"\nOur avg adj_LogAUC (mPAE): {result['our_avg_adj_logAUC_mPAE']:.1f}%  (London: 71.8%)")
    json.dump(result, open(OUT, 'w'), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
