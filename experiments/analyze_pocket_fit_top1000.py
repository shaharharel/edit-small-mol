#!/usr/bin/env python3
"""
E1 + E2 — pocket-fit analysis on the 997 Boltz cofolds.

For each cofolded complex:
  - extract H-bond contacts (protein N/O <-> ligand N/O at 2.5-3.5 A)
  - count by ZAP70 functional region (hinge / Cys560 / αC / P-loop / activation loop)
  - measure ligand-Cys560.SG distance (proxy for warhead reach)
  - load Boltz metrics (mPAE, iptm, plddt) and FiLMDelta pIC50 from manifest

Output:
  results/paper_evaluation/pocket_fit_analysis.csv
  results/paper_evaluation/pocket_fit_analysis.json (top picks)
  prints E1 (top-20 ranking comparison) and E2 (correlation table)
"""
import json
from pathlib import Path
from collections import defaultdict
import sys

import numpy as np
import pandas as pd
import gemmi
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "data" / "boltz_poses" / "boltz_results_top1000" / "predictions"
MANIFEST = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest.json"
OUT_CSV = PROJECT_ROOT / "results" / "paper_evaluation" / "pocket_fit_analysis.csv"
OUT_JSON = PROJECT_ROOT / "results" / "paper_evaluation" / "pocket_fit_analysis.json"

# ZAP70 functional regions (from kinase fold + structure analysis)
HINGE_RES = set(range(411, 421))         # M414 area + neighbors
GATEKEEPER = 414                           # M414
CYS560_AREA = set(range(556, 566))        # activation-loop covalent target
ACTIVATION_LOOP = set(range(479, 540))    # DFG to APE
P_LOOP = set(range(348, 358))             # G-rich loop
AC_HELIX = set(range(370, 388))           # αC + catalytic K (~K376)
CATALYTIC_LYS = 376                        # K376 — essential salt bridge


def analyze_cif(cif_path: Path):
    """Returns dict with pocket-fit metrics for one cofold."""
    s = gemmi.read_structure(str(cif_path))[0]
    if "A" not in [ch.name for ch in s] or "B" not in [ch.name for ch in s]:
        return None
    prot, lig = s["A"], s["B"]
    lig_atoms = []
    for r in lig:
        for a in r:
            if a.element.name == "H":
                continue
            lig_atoms.append((a.name, a.element.name, np.array([a.pos.x, a.pos.y, a.pos.z])))

    h_bond_counts = defaultdict(int)
    h_bonds_to_hinge = []
    h_bonds_total = 0
    cys560_min_d_to_C = float("inf")
    pocket_residues_5A = set()
    contacts_4A = 0  # close-contact count proxy for pocket fit

    for r in prot:
        rn = r.seqid.num
        rname = r.name
        for a in r:
            if a.element.name == "H":
                continue
            ap = np.array([a.pos.x, a.pos.y, a.pos.z])
            for la_name, la_elem, la_pos in lig_atoms:
                d = np.linalg.norm(ap - la_pos)
                if d <= 5.0:
                    pocket_residues_5A.add(rn)
                if d <= 4.0:
                    contacts_4A += 1

                # H-bond candidate
                if a.element.name in ("N", "O") and la_elem in ("N", "O") and 2.5 < d < 3.5:
                    h_bonds_total += 1
                    if rn in HINGE_RES:
                        h_bond_counts["hinge"] += 1
                        h_bonds_to_hinge.append((rname, rn, a.name, la_name, round(d, 2)))
                    elif rn in CYS560_AREA:
                        h_bond_counts["cys560_area"] += 1
                    elif rn in ACTIVATION_LOOP:
                        h_bond_counts["activation_loop"] += 1
                    elif rn in P_LOOP:
                        h_bond_counts["p_loop"] += 1
                    elif rn in AC_HELIX:
                        h_bond_counts["ac_helix"] += 1
                    else:
                        h_bond_counts["other"] += 1

                # Cys560 SG distance
                if rn == 560 and a.name == "SG" and la_elem == "C":
                    cys560_min_d_to_C = min(cys560_min_d_to_C, d)

    # Hinge "canonical" pattern: at least 2 H-bonds in residues 414-417
    canonical_hinge = h_bond_counts["hinge"] >= 2
    # Catalytic Lys salt bridge: K376.NZ <-> ligand O within 3.5 A
    has_klys_contact = False
    for r in prot:
        if r.seqid.num != CATALYTIC_LYS:
            continue
        for a in r:
            if a.name == "NZ":
                ap = np.array([a.pos.x, a.pos.y, a.pos.z])
                for la_name, la_elem, la_pos in lig_atoms:
                    if la_elem != "O":
                        continue
                    if np.linalg.norm(ap - la_pos) < 3.5:
                        has_klys_contact = True
                        break

    return {
        "h_bonds_total": h_bonds_total,
        "h_bonds_hinge": h_bond_counts["hinge"],
        "h_bonds_cys560_area": h_bond_counts["cys560_area"],
        "h_bonds_activation_loop": h_bond_counts["activation_loop"],
        "h_bonds_p_loop": h_bond_counts["p_loop"],
        "h_bonds_ac_helix": h_bond_counts["ac_helix"],
        "h_bonds_other": h_bond_counts["other"],
        "canonical_hinge_binding": int(canonical_hinge),
        "klys_contact": int(has_klys_contact),
        "warhead_to_cys560_sg_min_d": (cys560_min_d_to_C if cys560_min_d_to_C != float("inf") else None),
        "n_pocket_residues_5A": len(pocket_residues_5A),
        "contacts_4A": contacts_4A,
        "hinge_hbonds_detail": h_bonds_to_hinge,
    }


def main():
    manifest = json.load(open(MANIFEST))
    print(f"Manifest entries: {len(manifest)}")

    rows = []
    for i, (rid_str, m) in enumerate(manifest.items()):
        cif = PRED_DIR / m["yaml_name"] / f"{m['yaml_name']}_model_0.cif"
        if not cif.exists():
            continue
        try:
            d = analyze_cif(cif)
        except Exception as e:
            print(f"  failed {m['yaml_name']}: {e}")
            continue
        if d is None:
            continue
        # Merge with manifest metadata
        rows.append({
            "row_id": int(rid_str),
            "method": m["method"],
            "smiles": m["smiles"],
            "MW": m["MW"],
            "rank_score": m["rank_score"],
            "pIC50_method": m["pIC50_method"],
            "Tc_to_Mol1": m["Tc_to_Mol1"],
            "max_Tc_train": m["max_Tc_train"],
            "mPAE": m["mPAE"],
            "iptm": m["iptm"],
            "complex_plddt": m["complex_plddt"],
            "complex_pde": m["complex_pde"],
            "combined_score": m["combined_score"],
            **{k: v for k, v in d.items() if k != "hinge_hbonds_detail"},
            "hinge_hbonds_detail": json.dumps(d["hinge_hbonds_detail"]),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(manifest)}")

    df = pd.DataFrame(rows)
    print(f"\nAnalyzed {len(df)} cofolds")

    # ── Define a pocket-fit score ──
    # Higher = better. Components:
    #  + canonical hinge binding (2 H-bonds to 411-420)
    #  + Klys catalytic interaction
    #  + many close contacts (well-packed pocket)
    #  + warhead reaches Cys560 (low SG-C distance)
    #  - high mPAE
    pocket_fit = (
        2.0 * df["canonical_hinge_binding"]
        + 1.0 * df["klys_contact"]
        + 0.05 * df["contacts_4A"].clip(0, 80)
        - 0.3 * df["mPAE"].fillna(df["mPAE"].median())
        + 0.3 * df["warhead_to_cys560_sg_min_d"].apply(
            lambda x: 1.0 if (x is not None and x < 3.5) else 0.0
        )
    )
    df["pocket_fit_score"] = pocket_fit

    # ── E2: Correlation matrix ──
    print("\n" + "=" * 80)
    print("E2: Correlations vs FiLMDelta predicted pIC50 (rank_score)")
    print("=" * 80)
    print(f"{'metric':<28}{'Pearson r':>12}{'p-val':>12}{'Spearman ρ':>14}{'p-val':>12}")
    print("-" * 80)
    for col in [
        "h_bonds_total", "h_bonds_hinge", "h_bonds_cys560_area",
        "h_bonds_activation_loop", "h_bonds_p_loop", "h_bonds_ac_helix",
        "canonical_hinge_binding", "klys_contact",
        "n_pocket_residues_5A", "contacts_4A",
        "warhead_to_cys560_sg_min_d",
        "mPAE", "iptm", "complex_plddt", "complex_pde",
        "pocket_fit_score", "combined_score",
    ]:
        sub = df[["rank_score", col]].dropna()
        if len(sub) < 30:
            continue
        p_r, p_p = pearsonr(sub["rank_score"], sub[col])
        s_r, s_p = spearmanr(sub["rank_score"], sub[col])
        print(f"{col:<28}{p_r:>+12.4f}{p_p:>12.3g}{s_r:>+14.4f}{s_p:>12.3g}")

    # ── E1: Top-20 ranking comparison ──
    print("\n" + "=" * 80)
    print("E1: Top-20 ranking comparison")
    print("=" * 80)
    top_filmdelta = df.nlargest(20, "rank_score")["row_id"].tolist()
    top_combined  = df.nlargest(20, "combined_score")["row_id"].tolist()
    top_pocketfit = df.nlargest(20, "pocket_fit_score")["row_id"].tolist()
    overlap_fc = len(set(top_filmdelta) & set(top_combined))
    overlap_fp = len(set(top_filmdelta) & set(top_pocketfit))
    overlap_cp = len(set(top_combined) & set(top_pocketfit))
    print(f"FiLMDelta top-20  ∩  Combined top-20  : {overlap_fc} / 20")
    print(f"FiLMDelta top-20  ∩  PocketFit top-20 : {overlap_fp} / 20")
    print(f"Combined top-20   ∩  PocketFit top-20 : {overlap_cp} / 20")

    print("\n=== Top 10 by PocketFit score (3D-derived) ===")
    cols = ["row_id", "method", "rank_score", "mPAE", "iptm",
            "h_bonds_hinge", "h_bonds_cys560_area",
            "warhead_to_cys560_sg_min_d", "pocket_fit_score", "combined_score"]
    print(df.nlargest(10, "pocket_fit_score")[cols].round(3).to_string())

    print("\n=== Hinge-binding statistics ===")
    print(f"Molecules with ≥1 hinge H-bond: {(df['h_bonds_hinge']>=1).sum()} / {len(df)} ({100*(df['h_bonds_hinge']>=1).mean():.1f}%)")
    print(f"Molecules with ≥2 hinge H-bonds (canonical): {df['canonical_hinge_binding'].sum()} / {len(df)} ({100*df['canonical_hinge_binding'].mean():.1f}%)")
    print(f"Molecules with K376 contact: {df['klys_contact'].sum()} / {len(df)} ({100*df['klys_contact'].mean():.1f}%)")
    print(f"Molecules with warhead within 3.5 Å of Cys560.SG: {(df['warhead_to_cys560_sg_min_d']<3.5).sum()} / {len(df)}")
    print(f"Mean H-bonds to hinge: {df['h_bonds_hinge'].mean():.2f}")
    print(f"Mean total H-bonds: {df['h_bonds_total'].mean():.2f}")

    # ── Per-method ──
    print("\n=== Per-method hinge-binding rates ===")
    for method, sub in df.groupby("method"):
        ch = sub["canonical_hinge_binding"].mean() * 100
        klys = sub["klys_contact"].mean() * 100
        warh = (sub["warhead_to_cys560_sg_min_d"] < 3.5).mean() * 100
        print(f"  {method[:50]:<50}  N={len(sub):>4}  canonical_hinge={ch:>5.1f}%  K376={klys:>5.1f}%  warhead<3.5Å={warh:>5.1f}%")

    # ── Save ──
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    summary = {
        "n_total": len(df),
        "top20_filmdelta_rowids": top_filmdelta,
        "top20_combined_rowids": top_combined,
        "top20_pocketfit_rowids": top_pocketfit,
        "overlap": {
            "filmdelta_combined": overlap_fc,
            "filmdelta_pocketfit": overlap_fp,
            "combined_pocketfit": overlap_cp,
        },
        "hinge_canonical_rate": float(df["canonical_hinge_binding"].mean()),
        "klys_contact_rate": float(df["klys_contact"].mean()),
        "mean_hinge_hbonds": float(df["h_bonds_hinge"].mean()),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
