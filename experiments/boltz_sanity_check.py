"""Sanity check on Boltz-2 cofold metrics: detect leakage/artifacts driving 98% adj_LogAUC."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SK = True
except ImportError:
    HAS_SK = False

METRICS = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/covalid_mv_cofolds/warhead_metrics.csv")
MANIFEST = Path("/Users/shaharharel/Documents/github/edit-small-mol/experiments/boltz_inputs/covalid_minimum_viable/manifest.csv")

m = pd.read_csv(METRICS)
lab = pd.read_csv(MANIFEST)
print(f"metrics: {len(m)} rows  manifest: {len(lab)} rows")
print(f"metrics cols: {list(m.columns)}")
print(f"manifest cols: {list(lab.columns)}")

# Merge on (target, name)
df = m.merge(lab[["target", "name", "is_active", "smiles"]], on=["target", "name"], how="left")
print(f"\nmerged rows: {len(df)}, missing labels: {df['is_active'].isna().sum()}")
df = df.dropna(subset=["is_active", "mpae_london_min"]).copy()
df["is_active"] = df["is_active"].astype(int)
print(f"after dropna: {len(df)}  actives={df['is_active'].sum()}  decoys={(df['is_active']==0).sum()}")
print(f"targets: {df['target'].nunique()} → {sorted(df['target'].unique().tolist())}")

# === 1) Distribution of mpae_london_min per target by active/decoy ===
print("\n" + "=" * 80)
print("[1] mpae_london_min distribution per target (lower = better contact w/ Cys)")
print("=" * 80)
print(f"{'target':<10} {'class':<6} {'n':>4} {'p5':>7} {'med':>7} {'p95':>7} {'IQR':>7}")
for tgt, sub in df.groupby("target"):
    for cls in [1, 0]:
        v = sub.loc[sub["is_active"] == cls, "mpae_london_min"].values
        if len(v) == 0:
            continue
        p5, p25, med, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"{tgt:<10} {'act' if cls else 'dec':<6} {len(v):>4} {p5:>7.3f} {med:>7.3f} {p95:>7.3f} {p75 - p25:>7.3f}")

# Bimodality test: gap between active p95 and decoy p5
print("\n[1b] Gap analysis: decoy_p5 - active_p95 (positive = clean separation)")
for tgt, sub in df.groupby("target"):
    a = sub.loc[sub["is_active"] == 1, "mpae_london_min"].values
    d = sub.loc[sub["is_active"] == 0, "mpae_london_min"].values
    if len(a) < 5 or len(d) < 5:
        continue
    a_p95 = np.percentile(a, 95)
    d_p5 = np.percentile(d, 5)
    gap = d_p5 - a_p95
    overlap_pct = 100 * np.mean((d >= np.percentile(a, 5)) & (d <= a_p95))
    print(f"  {tgt}: act_p95={a_p95:.3f}  dec_p5={d_p5:.3f}  gap={gap:+.3f}  decoy in active range: {overlap_pct:.1f}%")

# === 2) Correlations of mpae_london_min with confounds ===
print("\n" + "=" * 80)
print("[2] Correlations: mpae_london_min vs (n_lig, complex_plddt, complex_iptm, ligand_iptm, complex_ipde)")
print("=" * 80)
for col in ["n_lig", "complex_plddt", "complex_iptm", "ligand_iptm", "complex_ipde"]:
    if col not in df.columns:
        print(f"  {col}: NOT IN DATA")
        continue
    sub = df.dropna(subset=[col, "mpae_london_min"])
    if len(sub) < 10:
        continue
    pr = sub["mpae_london_min"].corr(sub[col], method="pearson")
    sp = sub["mpae_london_min"].corr(sub[col], method="spearman")
    print(f"  mpae_london_min vs {col:<16}  pearson={pr:+.3f}  spearman={sp:+.3f}  n={len(sub)}")

# Per-target too (in case global is washed by target differences)
print("\n[2b] Per-target spearman(mpae_london_min, n_lig) and spearman(mpae_london_min, complex_plddt)")
for tgt, sub in df.groupby("target"):
    if len(sub) < 20:
        continue
    sp_nlig = sub["mpae_london_min"].corr(sub["n_lig"], method="spearman")
    sp_plddt = sub["mpae_london_min"].corr(sub["complex_plddt"], method="spearman")
    sp_iptm = sub["mpae_london_min"].corr(sub["complex_iptm"], method="spearman")
    print(f"  {tgt}: vs n_lig={sp_nlig:+.3f}  vs plddt={sp_plddt:+.3f}  vs iptm={sp_iptm:+.3f}  n={len(sub)}")

# === 3) Ligand size confound ===
print("\n" + "=" * 80)
print("[3] n_lig distribution per target × class")
print("=" * 80)
print(f"{'target':<10} {'act_mean':>9} {'act_std':>8} {'dec_mean':>9} {'dec_std':>8} {'diff':>7} {'KS_p':>10}")
for tgt, sub in df.groupby("target"):
    a = sub.loc[sub["is_active"] == 1, "n_lig"].values
    d = sub.loc[sub["is_active"] == 0, "n_lig"].values
    if len(a) < 3 or len(d) < 3:
        continue
    ks_p = stats.ks_2samp(a, d).pvalue if HAS_SCIPY else float("nan")
    print(f"{tgt:<10} {a.mean():>9.2f} {a.std():>8.2f} {d.mean():>9.2f} {d.std():>8.2f} {a.mean() - d.mean():>+7.2f} {ks_p:>10.2e}")

# === 4) BMX active-vs-decoy SMILES inspection ===
print("\n" + "=" * 80)
print("[4] BMX shortcut: top-5 actives w/ LOWEST mpae and top-5 decoys w/ HIGHEST mpae")
print("=" * 80)
bmx = df[df["target"] == "BMX"].copy()
if len(bmx):
    a_best = bmx[bmx["is_active"] == 1].nsmallest(5, "mpae_london_min")
    d_worst = bmx[bmx["is_active"] == 0].nlargest(5, "mpae_london_min")
    print("\nBMX best actives (lowest mpae):")
    for _, r in a_best.iterrows():
        smi = (r["smiles"] or "")[:120]
        print(f"  mpae={r['mpae_london_min']:.3f}  n_lig={r['n_lig']:>3}  plddt={r['complex_plddt']:.2f}  {r['name']}  {smi}")
    print("\nBMX worst decoys (highest mpae):")
    for _, r in d_worst.iterrows():
        smi = (r["smiles"] or "")[:120]
        print(f"  mpae={r['mpae_london_min']:.3f}  n_lig={r['n_lig']:>3}  plddt={r['complex_plddt']:.2f}  {r['name']}  {smi}")
    # Also: median active vs median decoy SMILES
    print("\nBMX median active mpae vs median decoy mpae:")
    a_all = bmx[bmx["is_active"] == 1]["mpae_london_min"].values
    d_all = bmx[bmx["is_active"] == 0]["mpae_london_min"].values
    print(f"  actives:  n={len(a_all)} median={np.median(a_all):.3f} mean={a_all.mean():.3f}")
    print(f"  decoys:   n={len(d_all)} median={np.median(d_all):.3f} mean={d_all.mean():.3f}")
else:
    print("No BMX rows after merge.")

# === 5) Same-mol-same-target / cross-contamination ===
print("\n" + "=" * 80)
print("[5] Duplicates and cross-contamination by raw SMILES")
print("=" * 80)
# Use raw SMILES (canonicalize without RDKit would require it; raw is a decent proxy)
df["smiles_str"] = df["smiles"].astype(str)
# 5a) within a target × class: duplicate SMILES
print("\n[5a] Within-target duplicate SMILES (same target + same class)")
for tgt, sub in df.groupby("target"):
    for cls in [1, 0]:
        s = sub.loc[sub["is_active"] == cls, "smiles_str"]
        dup = s[s.duplicated(keep=False)]
        if len(dup):
            print(f"  {tgt} ({'act' if cls else 'dec'}): {len(dup)} dup rows / {dup.nunique()} unique dup SMILES")

# 5b) SMILES appearing as BOTH active and decoy (anywhere)
both = df.groupby("smiles_str")["is_active"].nunique()
ambiguous = both[both > 1].index.tolist()
print(f"\n[5b] SMILES appearing as BOTH active AND decoy somewhere: {len(ambiguous)}")
for s in ambiguous[:10]:
    rows = df[df["smiles_str"] == s][["target", "name", "is_active"]].values
    print(f"  '{s[:80]}'  → {rows.tolist()}")

# 5c) Same SMILES across multiple targets
n_targets = df.groupby("smiles_str")["target"].nunique()
multi = n_targets[n_targets > 1]
print(f"\n[5c] Unique SMILES appearing in >1 target: {len(multi)} (of {df['smiles_str'].nunique()} unique)")

# === 6) Sanity baselines: ROC AUC using n_lig and plddt ALONE ===
print("\n" + "=" * 80)
print("[6] ROC AUC of trivial predictors (per-target, pooled label)")
print("=" * 80)
def safe_auc(y, score):
    if not HAS_SK:
        return float("nan")
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return roc_auc_score(y[mask], score[mask])

# Per-target AUC for n_lig, complex_plddt, mpae_london_min itself, complex_ipde
print(f"{'target':<10} {'AUC(-mpae)':>11} {'AUC(n_lig)':>11} {'AUC(-n_lig)':>12} {'AUC(plddt)':>11} {'AUC(iptm)':>11} {'AUC(-ipde)':>11}")
for tgt, sub in df.groupby("target"):
    y = sub["is_active"].values
    if len(np.unique(y)) < 2:
        continue
    auc_mpae = safe_auc(y, -sub["mpae_london_min"].values)
    auc_nlig_pos = safe_auc(y, sub["n_lig"].values)
    auc_nlig_neg = safe_auc(y, -sub["n_lig"].values)
    auc_plddt = safe_auc(y, sub["complex_plddt"].values)
    auc_iptm = safe_auc(y, sub["complex_iptm"].values)
    auc_ipde = safe_auc(y, -sub["complex_ipde"].fillna(sub["complex_ipde"].median()).values)
    print(f"{tgt:<10} {auc_mpae:>11.3f} {auc_nlig_pos:>11.3f} {auc_nlig_neg:>12.3f} {auc_plddt:>11.3f} {auc_iptm:>11.3f} {auc_ipde:>11.3f}")

# Pooled (across targets)
y_all = df["is_active"].values
print("\nPooled across targets:")
for name, score in [
    ("-mpae_london_min", -df["mpae_london_min"].values),
    ("n_lig", df["n_lig"].values),
    ("-n_lig", -df["n_lig"].values),
    ("complex_plddt", df["complex_plddt"].values),
    ("complex_iptm", df["complex_iptm"].values),
    ("ligand_iptm", df["ligand_iptm"].fillna(df["ligand_iptm"].median()).values),
    ("-complex_ipde", -df["complex_ipde"].fillna(df["complex_ipde"].median()).values),
]:
    print(f"  pooled AUC({name}) = {safe_auc(y_all, score):.3f}")

# === 7) Cohen's d per target ===
print("\n" + "=" * 80)
print("[7] Cohen's d: (mean_decoy - mean_active) / pooled_std for mpae_london_min")
print("    Larger positive d = actives have lower mpae (cleaner Cys contact)")
print("=" * 80)
print(f"{'target':<10} {'n_a':>4} {'n_d':>4} {'mean_a':>8} {'sd_a':>7} {'mean_d':>8} {'sd_d':>7} {'cohen_d':>8} {'AUC':>6}")
for tgt, sub in df.groupby("target"):
    a = sub.loc[sub["is_active"] == 1, "mpae_london_min"].values
    d = sub.loc[sub["is_active"] == 0, "mpae_london_min"].values
    if len(a) < 3 or len(d) < 3:
        continue
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(d) - 1) * d.var(ddof=1)) / (len(a) + len(d) - 2))
    cd = (d.mean() - a.mean()) / sp if sp > 0 else float("nan")
    y = sub["is_active"].values
    auc = safe_auc(y, -sub["mpae_london_min"].values)
    print(f"{tgt:<10} {len(a):>4} {len(d):>4} {a.mean():>8.3f} {a.std(ddof=1):>7.3f} {d.mean():>8.3f} {d.std(ddof=1):>7.3f} {cd:>+8.3f} {auc:>6.3f}")

# === 8) Multivariate: does mpae add ANYTHING over plddt+n_lig? ===
print("\n" + "=" * 80)
print("[8] Conditional analysis: residual mpae after regressing out n_lig+plddt")
print("=" * 80)
if HAS_SK:
    from sklearn.linear_model import LinearRegression
    sub = df.dropna(subset=["mpae_london_min", "n_lig", "complex_plddt"]).copy()
    X = sub[["n_lig", "complex_plddt"]].values
    y = sub["mpae_london_min"].values
    lr = LinearRegression().fit(X, y)
    sub["mpae_resid"] = y - lr.predict(X)
    auc_full = safe_auc(sub["is_active"].values, -sub["mpae_london_min"].values)
    auc_resid = safe_auc(sub["is_active"].values, -sub["mpae_resid"].values)
    auc_size_plddt = safe_auc(sub["is_active"].values, lr.predict(X) - sub["mpae_london_min"].values)  # not super meaningful
    print(f"  pooled AUC(-mpae_london_min)           = {auc_full:.3f}")
    print(f"  pooled AUC(-mpae_resid | n_lig,plddt)  = {auc_resid:.3f}")
    print(f"  R² of mpae ~ n_lig + plddt              = {lr.score(X, y):.3f}")
    print(f"  coef(n_lig)={lr.coef_[0]:+.5f}   coef(plddt)={lr.coef_[1]:+.5f}   intercept={lr.intercept_:+.3f}")

    # Per-target residual AUC
    print("\n  Per-target AUC(-mpae) vs AUC(-mpae_resid):")
    for tgt, ssub in sub.groupby("target"):
        y2 = ssub["is_active"].values
        if len(np.unique(y2)) < 2:
            continue
        a_full = safe_auc(y2, -ssub["mpae_london_min"].values)
        a_res = safe_auc(y2, -ssub["mpae_resid"].values)
        print(f"    {tgt}: AUC_full={a_full:.3f}  AUC_resid={a_res:.3f}  drop={a_full - a_res:+.3f}")

print("\nDONE.")
