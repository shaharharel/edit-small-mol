"""Phase A — recompute COValid D3 with London's per-compound protocol.

London JACS 2026 Table S2 evaluates at the **compound level**: each compound has
several protomer states; they reduce mPAE across protomers and rank unique
compounds. Our previous numbers ranked at the protomer level (1 YAML = 1 row),
which gives larger n and an easier task.

Hypothesis: collapsing our 35 actives + 199 decoys for BMX to compound-level
(via worst-case-protomer = MAX mPAE per compound) drops the adj_LogAUC from
~99% to roughly London's reported 56.4%.

This script:
  1. Loads MV manifest + BMX-strat manifest + warhead metrics (MV + strat).
  2. Maps every protomer → parent compound:
      - decoys: si_004 has `active_compound_name` (the parent active)
      - actives: si_004 has its own naming (typically by `compound_name` or
        the protomer_ind groups by parent). We try multiple columns.
  3. Per (target, compound), reduces mPAE / iPDE / ligand_iptm with both
     `min` (best-protomer = easiest task) and `max` (worst-protomer = London).
  4. Computes adj_LogAUC at compound level and prints the comparison.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent
SI4 = PROJECT_ROOT / "data/covalid/ja5c22222_si_004.xlsx"
MV_MANIFEST = PROJECT_ROOT / "experiments/boltz_inputs/covalid_minimum_viable/manifest.csv"
STRAT_MANIFEST = Path("/tmp/bmx_stratified_yamls/manifest.csv")
WARHEAD_MV = PROJECT_ROOT / "data/covalid_mv_cofolds/warhead_metrics.csv"
WARHEAD_STRAT = PROJECT_ROOT / "data/covalid_mv_cofolds/warhead_metrics_bmx_strat.csv"

LONDON_S2 = {
    "BMX": 56.4, "FGFR1": 79.4, "FGFR4_477": 83.8, "FGFR4_552": 82.5,
    "JAK3": 71.6, "EGFR": 68.1, "MAP3K7": 78.4, "KRAS": 74.7,
    "BTK": 72.3, "ITK": 65.8,
}


def canon(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except Exception:
        return None


def auc(y, s):
    y = np.asarray(y).astype(float)
    s = np.asarray(s, dtype=float)
    m = ~np.isnan(s)
    y = y[m]
    s = s[m]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n = sum((neg < p).sum() + 0.5 * (neg == p).sum() for p in pos)
    return float(n / (len(pos) * len(neg)))


def adj_log_auc(y, s, lam=1000):
    order = np.argsort(-np.asarray(s))
    y = np.asarray(y)[order].astype(float)
    n_act = y.sum()
    n_dec = len(y) - n_act
    if n_act == 0 or n_dec == 0:
        return 0.0
    tpr = np.concatenate([[0.0], np.cumsum(y) / n_act])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / n_dec])
    lo = 1.0 / lam
    mask = fpr >= lo
    if not mask.any():
        return 0.0
    idx = np.searchsorted(fpr, lo, side="left")
    if idx == 0:
        tpr_lo = 0.0
    elif idx >= len(fpr):
        tpr_lo = tpr[-1]
    else:
        x0, x1 = fpr[idx - 1], fpr[idx]
        y0, y1 = tpr[idx - 1], tpr[idx]
        tpr_lo = y0 + (y1 - y0) * (lo - x0) / (x1 - x0) if x1 > x0 else y0
    fpr_w = np.concatenate([[lo], fpr[mask]])
    tpr_w = np.concatenate([[tpr_lo], tpr[mask]])
    x = np.log10(fpr_w.clip(min=1e-12))
    yv = tpr_w
    auc_lam = float(np.sum((yv[1:] + yv[:-1]) * 0.5 * (x[1:] - x[:-1])))
    return 100.0 * (auc_lam - (1 - lo) / np.log(10)) / (-np.log10(lo) - (1 - lo) / np.log(10))


def load_si4_compound_map() -> dict[tuple[str, str], dict]:
    """Build (target, smiles_canonical) → {compound_id, role, target}."""
    xl = pd.ExcelFile(SI4)
    out = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(SI4, sheet_name=sheet)
        if "protomer_smiles" not in df.columns:
            continue
        # parse target + role from sheet name like 'BMX_actives' / 'BMX_decoys' / 'FGFR4_552_actives'
        target = sheet.rsplit("_", 1)[0]
        role = sheet.rsplit("_", 1)[1]  # 'actives' or 'decoys'
        for _, r in df.iterrows():
            smi = canon(r["protomer_smiles"])
            if smi is None:
                continue
            if role == "actives":
                # Actives have a parent compound name — protomers of the same compound share it.
                comp = (r.get("compound_name") or r.get("active_compound_name")
                        or r.get("compound") or f"act_{r.get('protomer_ind', '?')}")
                comp = f"act::{comp}"
            else:
                # Decoys: each decoy IS a distinct ZINC molecule with multiple protomers.
                # The unique compound id is the ZINC id. The `active_compound_name` column
                # is the PARENT active the decoy was drawn for — collapsing on that would
                # merge unrelated decoys into one row.
                z = r.get("decoy_zinc_id") or r.get("zinc_id")
                if pd.isna(z) or not z:
                    # fallback: group by parent + matched_decoy_protomer (protomers of one decoy)
                    z = f"{r.get('active_compound_name', 'unknown')}__matched_{r.get('matched_decoy_protomer', '?')}"
                comp = f"dec::{z}"
            out[(target, smi)] = {
                "compound_id": str(comp),
                "role": role,
                "target": target,
                "protomer_ind": r.get("protomer_ind"),
            }
    return out


def main():
    print("=" * 100)
    print("PHASE A — Compound-level adj_LogAUC (London protocol)")
    print("=" * 100)

    # 1. Load manifests + warhead
    mv = pd.read_csv(MV_MANIFEST)
    strat = pd.read_csv(STRAT_MANIFEST) if STRAT_MANIFEST.exists() else pd.DataFrame()
    mv["source"] = "mv"
    if len(strat):
        strat["source"] = "strat"
    labels = pd.concat([mv[["target", "name", "is_active", "smiles", "source"]],
                        strat[["target", "name", "is_active", "smiles", "source"]] if len(strat) else mv.iloc[0:0]],
                       ignore_index=True)

    warhead = pd.concat(
        [pd.read_csv(WARHEAD_MV)]
        + ([pd.read_csv(WARHEAD_STRAT)] if WARHEAD_STRAT.exists() else []),
        ignore_index=True, sort=False)
    df = labels.merge(warhead, on=["target", "name"], how="inner")
    df["smi_canon"] = df["smiles"].apply(canon)
    df = df.dropna(subset=["smi_canon", "mpae_london_min"])
    print(f"\nProtomer-level rows after join: {len(df)}")
    print(df.groupby(["target", "is_active"]).size().unstack(fill_value=0).to_string())

    # 2. Map protomers → parent compounds via si_004
    print("\nBuilding si_004 compound map …")
    si4_map = load_si4_compound_map()
    print(f"  {len(si4_map)} (target, canonical_smiles) entries from si_004")

    def lookup_compound(target, smi):
        info = si4_map.get((target, smi))
        if info is not None:
            return info["compound_id"]
        # fallback: use canonical smiles as the compound id
        return smi

    df["compound_id"] = df.apply(lambda r: lookup_compound(r["target"], r["smi_canon"]), axis=1)

    # 3. Per (target, compound), reduce mPAE/iPDE/lig_iptm with both min and max
    print("\nProtomer-level → compound-level reduction …")
    rows = []
    for (tgt, comp), g in df.groupby(["target", "compound_id"]):
        rows.append({
            "target": tgt,
            "compound_id": comp,
            "is_active": int(g["is_active"].max()),  # any active protomer → active compound
            "n_protomers": len(g),
            # London = worst-case-protomer (HIGHEST mPAE; harder = larger value)
            "mpae_max": g["mpae_london_min"].max(),
            # Best-case-protomer (lowest mPAE; this is what our protomer-level eval effectively uses)
            "mpae_min": g["mpae_london_min"].min(),
            "mpae_mean": g["mpae_london_min"].mean(),
            "ipde_max": g["complex_ipde"].max(),
            "ipde_min": g["complex_ipde"].min(),
            "ligand_iptm_min": g["ligand_iptm"].min(),  # for HIGHER-is-better, take min as "worst"
            "ligand_iptm_max": g["ligand_iptm"].max(),
        })
    comp_df = pd.DataFrame(rows)
    print(f"  {len(comp_df)} compound-level rows")
    print(comp_df.groupby(["target", "is_active"]).size().unstack(fill_value=0).to_string())

    # 4. Compute adj_LogAUC per target × reduction
    print()
    print("=" * 100)
    print("BMX comparison: protomer-level vs compound-level (worst-case) vs London")
    print("=" * 100)

    # Reference: protomer-level (what we did before)
    bmx_proto = df[df.target == "BMX"]
    y_p = bmx_proto.is_active.values
    print(f"\nPROTOMER-LEVEL (n={len(bmx_proto)}, {y_p.sum()} act, {(y_p == 0).sum()} dec):")
    for col, neg in [("mpae_london_min", True), ("complex_ipde", True), ("ligand_iptm", False)]:
        s = -bmx_proto[col].values if neg else bmx_proto[col].values
        print(f"  {col:25s}  AUC={auc(y_p, s):.3f}  adj_LogAUC={adj_log_auc(y_p, s):5.1f}%")

    # Compound-level: best-case (min mPAE = easiest)
    bmx_c = comp_df[comp_df.target == "BMX"]
    y_c = bmx_c.is_active.values
    print(f"\nCOMPOUND-LEVEL best-case (min mPAE) — n={len(bmx_c)}, {y_c.sum()} act, {(y_c == 0).sum()} dec:")
    for col, neg in [("mpae_min", True), ("ipde_min", True), ("ligand_iptm_max", False)]:
        s = -bmx_c[col].values if neg else bmx_c[col].values
        print(f"  {col:25s}  AUC={auc(y_c, s):.3f}  adj_LogAUC={adj_log_auc(y_c, s):5.1f}%")

    # Compound-level: WORST-case (max mPAE = London protocol)
    print(f"\nCOMPOUND-LEVEL WORST-case (max mPAE; LONDON PROTOCOL) — n={len(bmx_c)}:")
    for col, neg in [("mpae_max", True), ("ipde_max", True), ("ligand_iptm_min", False)]:
        s = -bmx_c[col].values if neg else bmx_c[col].values
        print(f"  {col:25s}  AUC={auc(y_c, s):.3f}  adj_LogAUC={adj_log_auc(y_c, s):5.1f}%")

    # Compound-level: mean (average)
    print(f"\nCOMPOUND-LEVEL mean mPAE — n={len(bmx_c)}:")
    s = -bmx_c["mpae_mean"].values
    print(f"  {'mpae_mean':25s}  AUC={auc(y_c, s):.3f}  adj_LogAUC={adj_log_auc(y_c, s):5.1f}%")

    print(f"\nLondon Table S2 (AF3 + covalent constraint, BMX): adj_LogAUC = {LONDON_S2['BMX']:.1f}%")

    # 5. Protomer breakdown — how many protomers per compound on each side?
    print()
    print("=" * 100)
    print("Protomer/compound breakdown (BMX)")
    print("=" * 100)
    for cls, name in [(1, "actives"), (0, "decoys")]:
        sub = bmx_c[bmx_c.is_active == cls]["n_protomers"]
        if len(sub):
            print(f"  {name}: n_compounds={len(sub)}  protomers/cmpd  median={sub.median():.1f}  mean={sub.mean():.2f}  min={sub.min()} max={sub.max()}")

    # 6. Save a JSON
    out = {
        "bmx_protomer_level": {
            "n_total": int(len(bmx_proto)),
            "n_act": int(y_p.sum()),
            "n_dec": int((y_p == 0).sum()),
            "mpae_auc": auc(y_p, -bmx_proto["mpae_london_min"].values),
            "mpae_adj_logauc": adj_log_auc(y_p, -bmx_proto["mpae_london_min"].values),
        },
        "bmx_compound_level_worst_protomer": {
            "n_total": int(len(bmx_c)),
            "n_act": int(y_c.sum()),
            "n_dec": int((y_c == 0).sum()),
            "mpae_auc": auc(y_c, -bmx_c["mpae_max"].values),
            "mpae_adj_logauc": adj_log_auc(y_c, -bmx_c["mpae_max"].values),
        },
        "bmx_compound_level_best_protomer": {
            "n_total": int(len(bmx_c)),
            "mpae_auc": auc(y_c, -bmx_c["mpae_min"].values),
            "mpae_adj_logauc": adj_log_auc(y_c, -bmx_c["mpae_min"].values),
        },
        "london_S2_bmx": LONDON_S2["BMX"],
    }
    out_path = PROJECT_ROOT / "results" / "covalid" / "phase_a_compound_collapse.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
