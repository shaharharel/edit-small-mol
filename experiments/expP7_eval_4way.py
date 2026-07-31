"""Exp-P7 4-way eval: Boltz + covalent + ligand metrics × 4 methods on ZAP70.

Reads Boltz cofold outputs from two local dirs (batch_a + batch_b_selective),
merges with the SMILES manifest, computes metrics, and prints a per-method
comparison table + KS pairwise p-values. Also writes per-mol CSV.

Methods: mol2mol_baseline (142), P5-1v2_base (300), P6-POSENOISE_s03 (299),
RL_GEOMEAN (300). Total 1041 cofolds.

Metrics:
  Boltz confidence: iptm, ligand_iptm, complex_pde, confidence_score
  London mPAE:      min over cross-chain PAE block (protein × ligand)
  Covalent geom:    d_SG (Cys346 SG - warhead Cβ), Burgi-Dunitz |θ-107°|
  Pose quality:     n_h_bonds, n_stabilizing_contacts, pocket_occupancy_pct
  Ligand-only:      MW, LogP, QED, has_acryl, Tc_to_Mol1

Usage:
  conda run -n quris python experiments/expP7_eval_4way.py
"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
import gemmi

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
P7 = ROOT / "data/exp_P7"
MANIFEST = ROOT / "experiments/boltz_inputs_p7/manifest.csv"
OUT_CSV = P7 / "per_mol_metrics.csv"
SUMMARY_CSV = P7 / "per_method_summary.csv"
KS_CSV = P7 / "ks_pairwise.csv"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cnc(NC(=O)c4cn(C)nc4C(C)C)cn3)c2C1"
ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
CYS_RESI = 346
BURGI_DUNITZ_DEG = 107.0
HB_MIN_A, HB_MAX_A = 2.5, 3.5
VDW_MIN_A, VDW_MAX_A = 3.5, 4.5
ATP_POCKET_RESIS = {344, 345, 346, 347, 348, 349, 350, 352, 367, 369, 399,
                    414, 415, 416, 417, 418, 420, 421, 424,
                    461, 465, 466, 468, 479, 482}


def build_uid_to_dir():
    """Map uid → (predictions_dir Path). Scans both batch_a and batch_b_selective."""
    idx = {}
    for base in [P7 / "results_a" / "results", P7 / "results_b_selective"]:
        if not base.exists():
            continue
        for boltz_dir in base.iterdir():
            if not boltz_dir.name.startswith("boltz_results_"):
                continue
            uid = boltz_dir.name[len("boltz_results_"):]
            pred_dir = boltz_dir / "predictions" / uid
            if pred_dir.is_dir():
                idx[uid] = pred_dir
    return idx


def load_confidence(pred_dir: Path, uid: str) -> dict:
    p = pred_dir / f"confidence_{uid}_model_0.json"
    if not p.exists():
        return {"iptm": None, "ligand_iptm": None, "confidence_score": None,
                "complex_pde": None, "protein_iptm": None}
    try:
        d = json.loads(p.read_text())
        return {k: d.get(k) for k in ["iptm", "ligand_iptm", "confidence_score",
                                       "complex_pde", "protein_iptm"]}
    except Exception:
        return {"iptm": None, "ligand_iptm": None, "confidence_score": None,
                "complex_pde": None, "protein_iptm": None}


def mpae_london(pred_dir: Path, uid: str, n_lig_atoms: int) -> tuple:
    """Return (mpae_min, mpae_5pct): min and 5th-percentile of cross-chain PAE."""
    p = pred_dir / f"pae_{uid}_model_0.npz"
    if not p.exists() or n_lig_atoms <= 0:
        return (None, None)
    try:
        d = np.load(p)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
        if pae.ndim != 2:
            return (None, None)
        N = pae.shape[0]
        lig_lo = N - n_lig_atoms
        if lig_lo <= 0:
            return (None, None)
        cross = pae[:lig_lo, lig_lo:]
        if cross.size == 0:
            return (None, None)
        return (float(np.min(cross)), float(np.percentile(cross, 5)))
    except Exception:
        return (None, None)


def parse_cif(cif_path: Path):
    """Return (prot_atoms, lig_atoms) list-of-dicts."""
    try:
        st = gemmi.read_structure(str(cif_path))
        try:
            st.setup_entities()
        except Exception:
            pass
    except Exception:
        return [], []
    prot, lig = [], []
    for model in st:
        for chain in model:
            target = prot if chain.name == "A" else lig if chain.name == "B" else None
            if target is None:
                continue
            for res in chain:
                for atom in res:
                    target.append({
                        "resname": res.name, "resi": res.seqid.num,
                        "name": atom.name, "element": atom.element.name,
                        "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float),
                    })
        break
    return prot, lig


def d_SG(prot, lig, cb_name: str) -> float | None:
    if not cb_name:
        return None
    sg = next((p["pos"] for p in prot if p["resi"] == CYS_RESI and p["name"] == "SG"), None)
    cb = next((l["pos"] for l in lig if l["name"] == cb_name), None)
    if sg is None or cb is None:
        return None
    return float(np.linalg.norm(sg - cb))


def burgi_dunitz(prot, lig, cb_name: str) -> float | None:
    if not cb_name:
        return None
    sg = next((p["pos"] for p in prot if p["resi"] == CYS_RESI and p["name"] == "SG"), None)
    cb_at = next((l for l in lig if l["name"] == cb_name), None)
    if sg is None or cb_at is None:
        return None
    # Find nearest C neighbor to Cβ (excluding Cβ itself)
    candidates = [l for l in lig if l["element"] == "C" and l["name"] != cb_name
                  and 1.0 <= float(np.linalg.norm(l["pos"] - cb_at["pos"])) <= 1.9]
    if not candidates:
        return None
    ca = min(candidates, key=lambda l: float(np.linalg.norm(l["pos"] - cb_at["pos"])))
    v1 = sg - cb_at["pos"]
    v2 = ca["pos"] - cb_at["pos"]
    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    cos_a = max(-1.0, min(1.0, cos_a))
    return abs(float(np.degrees(np.arccos(cos_a))) - BURGI_DUNITZ_DEG)


def count_h_bonds(prot, lig) -> int:
    n = 0
    for l in lig:
        if l["element"] not in ("N", "O"):
            continue
        for p in prot:
            if p["element"] not in ("N", "O"):
                continue
            if p["resi"] == CYS_RESI and p["name"] == "SG":
                continue
            d = float(np.linalg.norm(l["pos"] - p["pos"]))
            if HB_MIN_A <= d <= HB_MAX_A:
                n += 1
                break
    return n


def count_vdw_contacts(prot, lig) -> int:
    n = 0
    for l in lig:
        if l["element"] != "C":
            continue
        for p in prot:
            if p["element"] != "C":
                continue
            d = float(np.linalg.norm(l["pos"] - p["pos"]))
            if VDW_MIN_A <= d <= VDW_MAX_A:
                n += 1
                break
    return n


def pocket_occupancy(prot, lig) -> float | None:
    if not lig or not prot:
        return None
    pocket_atoms = [p for p in prot if p["resi"] in ATP_POCKET_RESIS]
    if not pocket_atoms:
        return None
    n_close = 0
    for l in lig:
        for p in pocket_atoms:
            if float(np.linalg.norm(l["pos"] - p["pos"])) <= 4.0:
                n_close += 1
                break
    return 100.0 * n_close / len(lig)


def ligand_props(smi: str, mol1_fp):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {"valid": False, "mw": None, "logp": None, "qed": None,
                "has_acryl": False, "tc_mol1": None}
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    return {
        "valid": True,
        "mw": Descriptors.MolWt(m),
        "logp": Descriptors.MolLogP(m),
        "qed": QED.qed(m),
        "has_acryl": len(m.GetSubstructMatches(ACRYL_SMARTS)) > 0,
        "tc_mol1": DataStructs.TanimotoSimilarity(fp, mol1_fp),
    }


def process_one(uid: str, method: str, smi: str, wh_atom: str, pred_dir_str: str,
                mol1_fp_bits: str) -> dict:
    """Worker function: computes all metrics for one mol."""
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect
    pred_dir = Path(pred_dir_str)
    mol1_fp = ExplicitBitVect(2048)
    mol1_fp.FromBase64(mol1_fp_bits)

    out = {"uid": uid, "method": method, "smiles": smi, "warhead_atom": wh_atom}

    # Confidence JSON
    conf = load_confidence(pred_dir, uid)
    out.update(conf)

    # CIF pose
    cif_path = pred_dir / f"{uid}_model_0.cif"
    prot, lig = parse_cif(cif_path)
    n_lig_atoms = len(lig)
    out["n_lig_atoms"] = n_lig_atoms

    # London mPAE (needs n_lig_atoms)
    mp_min, mp_5pct = mpae_london(pred_dir, uid, n_lig_atoms)
    out["mpae_min"] = mp_min
    out["mpae_5pct"] = mp_5pct

    # Covalent geometry
    out["d_SG"] = d_SG(prot, lig, wh_atom)
    out["burgi_dunitz_dev"] = burgi_dunitz(prot, lig, wh_atom)
    out["geom_ok"] = (out["d_SG"] is not None and out["d_SG"] <= 5.0)

    # Contacts + pocket
    out["n_h_bonds"] = count_h_bonds(prot, lig) if lig else 0
    out["n_vdw_contacts"] = count_vdw_contacts(prot, lig) if lig else 0
    out["pocket_occupancy_pct"] = pocket_occupancy(prot, lig)

    # Ligand-only
    out.update(ligand_props(smi, mol1_fp))
    return out


def main():
    print(f"[{pd.Timestamp.now():%H:%M:%S}] Building uid → dir index...")
    idx = build_uid_to_dir()
    print(f"  Found {len(idx)} cofold dirs")

    manifest = pd.read_csv(MANIFEST)
    print(f"  Manifest has {len(manifest)} entries")

    # Prepare Mol1 fingerprint (serialized for workers)
    mol1 = Chem.MolFromSmiles(MOL1_SMILES)
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    mol1_fp_bits = mol1_fp.ToBase64()

    # Build worker task list
    tasks = []
    for _, r in manifest.iterrows():
        uid = r["uid"]
        pred_dir = idx.get(uid)
        if pred_dir is None:
            continue
        tasks.append((uid, r["method"], r["smiles"], r["warhead_atom_name"],
                      str(pred_dir), mol1_fp_bits))
    print(f"  Running metrics on {len(tasks)} mols across 8 workers...")

    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_one, *t): t[0] for t in tasks}
        done = 0
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as e:
                rows.append({"uid": futs[f], "error": str(e)})
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(tasks)} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[{pd.Timestamp.now():%H:%M:%S}] Per-mol CSV → {OUT_CSV} ({len(df)} rows)")

    # Per-method aggregation
    metric_cols = ["iptm", "ligand_iptm", "confidence_score", "complex_pde", "protein_iptm",
                   "mpae_min", "mpae_5pct", "d_SG", "burgi_dunitz_dev",
                   "geom_ok", "n_h_bonds", "n_vdw_contacts", "pocket_occupancy_pct",
                   "mw", "logp", "qed", "has_acryl", "tc_mol1"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    summary = df.groupby("method")[metric_cols].agg(["count", "median", "mean", "std"])
    summary.to_csv(SUMMARY_CSV)
    print(f"[{pd.Timestamp.now():%H:%M:%S}] Per-method summary → {SUMMARY_CSV}")

    # Print compact median table
    print("\n=== 4-way median comparison ===")
    med = df.groupby("method")[metric_cols].median()
    print(med.T.to_string(float_format=lambda x: f"{x:.3f}"))

    # Pairwise KS tests on key metrics
    key_metrics = ["iptm", "ligand_iptm", "mpae_min", "d_SG", "burgi_dunitz_dev",
                   "n_vdw_contacts", "pocket_occupancy_pct", "tc_mol1"]
    key_metrics = [c for c in key_metrics if c in df.columns]
    methods = sorted(df["method"].dropna().unique())
    ks_rows = []
    for metric in key_metrics:
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                x1 = df.loc[df["method"] == m1, metric].dropna().values
                x2 = df.loc[df["method"] == m2, metric].dropna().values
                if len(x1) < 5 or len(x2) < 5:
                    continue
                stat, p = stats.ks_2samp(x1, x2)
                ks_rows.append({"metric": metric, "method_a": m1, "method_b": m2,
                                "n_a": len(x1), "n_b": len(x2),
                                "ks_stat": stat, "p_value": p,
                                "median_a": float(np.median(x1)),
                                "median_b": float(np.median(x2))})
    ks_df = pd.DataFrame(ks_rows)
    ks_df.to_csv(KS_CSV, index=False)
    print(f"\n[{pd.Timestamp.now():%H:%M:%S}] KS tests → {KS_CSV}")

    # Print significant KS results
    print("\n=== Significant KS pairs (p<0.05) ===")
    sig = ks_df[ks_df["p_value"] < 0.05].copy()
    sig["Δ_median"] = sig["median_b"] - sig["median_a"]
    print(sig[["metric", "method_a", "method_b", "median_a", "median_b",
               "Δ_median", "p_value"]].to_string(index=False,
                                                  float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
