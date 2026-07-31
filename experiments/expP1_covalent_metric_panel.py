"""Exp-P1 metric panel: 8 metrics for covalent-awareness comparison.

Metrics (all local, no GPU, no Boltz):
  1. Anchor-warhead-class match (does output preserve the anchor's electrophile class?)
  2. Any-electrophile retention (8 classes: acryl, α,β-unsat ester, haloacetamide,
     vinylsulfone, nitrile, aldehyde, β-lactam, cyanamide)
  3. Anchor Murcko Tc median (scaffold conservation)
  4. β-C ETKDG planar median (min over 10 confs, on electrophile-bearing subset)
  5. Bürgi-Dunitz angle |θ - 107°| median (on same subset)
  6. Multi-electrophile diversity (H = -Σ p_i log p_i across 8 classes)
  7. Novelty top-1 Tc<0.6 fraction to training corpus
  8. Anchor 3D shape Tc median (RDKit Open3DAlign, cheap)

Usage:
  python experiments/expP1_covalent_metric_panel.py \
    --samples_dirs data/exp_N/samples \
    --extra_sdfs /path/to/drugflow.sdf,/path/to/optionA.sdf \
    --out_csv data/exp_metric_panel/panel_v2.csv \
    --workers 8 --n_confs 10
"""
from __future__ import annotations
import argparse
import json
import math
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolTransforms import GetDihedralDeg, GetAngleDeg

RDLogger.DisableLog("rdApp.*")

# 8 electrophile classes with their canonical dihedral atoms if applicable
ELECTROPHILE_CLASSES = {
    "acryl":         "[CH2]=[CH]C(=O)N",              # terminal acrylamide
    "acryl_ester":   "[CH2]=[CH]C(=O)O",              # terminal acryl ester
    "sub_acryl":     "[CH1,CH0]=[CH1]C(=O)N",         # substituted acrylamide (e.g., osimertinib)
    "haloacetamide": "[Cl,Br,I][CH2]C(=O)N",
    "vinylsulfone":  "[CH2]=[CH]S(=O)(=O)",
    "nitrile":       "[C]#[N]",                        # cyanamide + nitrile (crude — cathepsin binders)
    "aldehyde":      "[CH]=O",
    "beta_lactam":   "[NX3R]1C(=O)[CX4R]1",           # 4-membered lactam
    "cyanamide":     "N-[CX2]#N",
    "fluoro_carbonyl":"C(F)(F)C(=O)",                   # sotorasib-like
}

ANCHOR_WARHEAD_CLASS = {
    "mol1":       "acryl",
    "afatinib":   "sub_acryl",
    "osimertinib":"acryl",
    "ibrutinib":  "acryl",
    "ars853":     "acryl",       # has terminal acryl on piperazine + vinylsulfone
    "mrtx849":    "acryl",
    "sotorasib":  "fluoro_carbonyl",
}

ANCHOR_SMILES = {
    "mol1":       "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1",
    "afatinib":   "CN(C)C/C=C/C(=O)Nc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1O[C@H]1CCOC1",
    "osimertinib":"COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
    "ibrutinib":  "C=CC(=O)N1CCC[C@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    "ars853":     "OCC1(CS(=O)(=O)c2ccccc2)C(=O)N(c2cnc(Cl)cc2Cl)C(=O)/C1=C/N1CCN(C(=O)C=C)CC1",
    "mrtx849":    "CC1CCN(C(=O)C=C)C[C@@H]1N1C(=O)N(c2cnc3c(c2F)c(Cl)c(O)cc3C#N)C[C@H]1COc1cnc2[nH]ccc2c1",
    "sotorasib":  "CC(=O)N1CCN(C(=O)C(F)(F)C(=O)N2CCC[C@@H]2c2nc(-c3c(O)cccc3F)c3cc(F)c(N4CCC(F)CC4)nc3n2)CC1",
}


def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def murcko(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except: return None


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def find_electrophile(smi):
    """Return set of electrophile classes present in this SMILES."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return set()
    hits = set()
    for cls, smarts in ELECTROPHILE_CLASSES.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None: continue
        if m.GetSubstructMatches(patt):
            hits.add(cls)
    return hits


class _TO(Exception): pass


def _alarm(*a): raise _TO()


def _planar_bd_worker(args):
    """Compute planar_dev + BD angle for one molecule with a strict 20s timeout."""
    idx, smi, n_confs = args
    out = {"idx": idx, "smi": smi, "embed_ok": False,
           "planar_dev_deg": None, "bd_angle_deg": None, "bd_dev_deg": None}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(20)
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return out
        # Try each electrophile class in order — use first hit for geometry
        patt = Chem.MolFromSmarts("[CH2,CH1,CH0]=[CH1,CH0]C(=O)N")
        m2 = mol
        matches = m2.GetSubstructMatches(patt)
        if not matches: return out
        m = matches[0]
        b_idx, a_idx, c_idx, n_idx = int(m[0]), int(m[1]), int(m[2]), int(m[4])
        mol_h = Chem.AddHs(mol)
        best_planar = None
        best_bd_angle = None
        for seed in range(42, 42 + n_confs):
            m2c = Chem.Mol(mol_h)
            p = AllChem.ETKDGv3()
            p.randomSeed = seed
            rc = AllChem.EmbedMolecule(m2c, p)
            if rc != 0: continue
            try: AllChem.MMFFOptimizeMolecule(m2c, maxIters=100)
            except: pass
            try:
                conf = m2c.GetConformer()
                d = GetDihedralDeg(conf, b_idx, a_idx, c_idx, n_idx)
                pd_ = min(abs(d), abs(180.0 - abs(d)))
                # BD angle: C=C-C attack angle from β-C
                bd = GetAngleDeg(conf, b_idx, a_idx, c_idx)
                if best_planar is None or pd_ < best_planar:
                    best_planar = pd_
                    best_bd_angle = bd
            except: continue
        if best_planar is not None:
            out["embed_ok"] = True
            out["planar_dev_deg"] = float(best_planar)
            out["bd_angle_deg"] = float(best_bd_angle)
            out["bd_dev_deg"] = float(abs(best_bd_angle - 107.0))  # canonical BD angle
    except _TO:
        out["timed_out"] = True
    except Exception: pass
    finally: signal.alarm(0)
    return out


def compute_geom_panel(smis, workers, n_confs):
    tasks = [(i, s, n_confs) for i, s in enumerate(smis)]
    rows = []
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futs = [exc.submit(_planar_bd_worker, t) for t in tasks]
        for f in as_completed(futs):
            try: rows.append(f.result())
            except: rows.append({"idx": -1, "embed_ok": False,
                                  "planar_dev_deg": None, "bd_angle_deg": None, "bd_dev_deg": None})
            done += 1
            if done % 200 == 0:
                dt = time.time() - t0
                print(f"  geom {done}/{len(tasks)}  {done/max(dt,1e-6):.1f} mol/s", flush=True)
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def shape_tc_worker(args):
    """Return anchor-shape Tc via RDKit's alignment (best conformer alignment)."""
    idx, anchor_smi, query_smi, n_confs = args
    out = {"idx": idx, "shape_tc": None}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(20)
    try:
        a = Chem.MolFromSmiles(anchor_smi)
        q = Chem.MolFromSmiles(query_smi)
        if a is None or q is None: return out
        ah = Chem.AddHs(a); qh = Chem.AddHs(q)
        AllChem.EmbedMolecule(ah, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(ah, maxIters=100)
        best_tc = None
        for seed in range(42, 42 + n_confs):
            m2 = Chem.Mol(qh)
            if AllChem.EmbedMolecule(m2, randomSeed=seed) != 0: continue
            try: AllChem.MMFFOptimizeMolecule(m2, maxIters=100)
            except: pass
            try:
                score = AllChem.GetBestRMS(m2, ah)  # RMSD proxy; smaller = more similar
                # Convert to Tc-like [0,1]: exp(-RMSD/3) is a rough surrogate
                tc = math.exp(-score / 3.0) if score < 20 else 0.0
                if best_tc is None or tc > best_tc:
                    best_tc = tc
            except: continue
        out["shape_tc"] = best_tc
    except _TO: pass
    except Exception: pass
    finally: signal.alarm(0)
    return out


def analyze_cohort(csv_path: str, anchor: str, train_fps: list, workers: int, n_confs: int,
                   smi_col: str = "SMILES"):
    df = pd.read_csv(csv_path)
    if smi_col not in df.columns:
        # try alternate
        for c in df.columns:
            if c.lower() in ("smiles", "smi", "canonical_smiles"):
                smi_col = c
                break
    smis = df[smi_col].astype(str).tolist()
    n = len(smis)

    # Validity + canonical
    canons = [canonical(s) for s in smis]
    valid_smis = [c for c in canons if c is not None]
    n_valid = len(valid_smis)
    n_unique = len(set(valid_smis))

    # Warhead class detection per molecule
    hit_classes_per_mol = [find_electrophile(s) for s in smis]
    n_any_electrophile = sum(1 for h in hit_classes_per_mol if h)
    anchor_class = ANCHOR_WARHEAD_CLASS.get(anchor.lower(), "acryl")
    # sub_acryl and acryl are largely overlapping — consider both as "acryl-family"
    if anchor_class in ("acryl", "sub_acryl"):
        acceptable = {"acryl", "sub_acryl", "acryl_ester"}
    elif anchor_class == "fluoro_carbonyl":
        acceptable = {"fluoro_carbonyl"}
    else:
        acceptable = {anchor_class}
    n_anchor_match = sum(1 for h in hit_classes_per_mol if h & acceptable)

    # Diversity: Shannon entropy across classes
    class_counts = {cls: 0 for cls in ELECTROPHILE_CLASSES.keys()}
    for h in hit_classes_per_mol:
        for cls in h:
            class_counts[cls] += 1
    total_hits = sum(class_counts.values())
    if total_hits > 0:
        ps = [c / total_hits for c in class_counts.values() if c > 0]
        entropy_H = -sum(p * math.log(p) for p in ps)
    else:
        entropy_H = 0.0

    # Anchor Murcko Tc
    anchor_smi = ANCHOR_SMILES.get(anchor.lower())
    tc_murcko = None
    if anchor_smi:
        a_scaf = murcko(anchor_smi)
        a_fp = morgan_fp(a_scaf) if a_scaf else None
        if a_fp:
            tcs = []
            for c in canons:
                if c is None: continue
                sc = murcko(c)
                if sc is None: continue
                fp = morgan_fp(sc)
                if fp is None: continue
                tcs.append(DataStructs.TanimotoSimilarity(a_fp, fp))
            tc_murcko = float(np.median(tcs)) if tcs else None

    # Novelty
    tops = []
    for c in canons:
        if c is None: tops.append(None); continue
        fp = morgan_fp(c)
        if fp is None: tops.append(None); continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        tops.append(float(max(sims)) if sims else None)
    tops_n = [t for t in tops if t is not None]
    novelty_median = float(np.median(tops_n)) if tops_n else None
    novelty_frac = float(np.mean([t < 0.6 for t in tops_n])) if tops_n else None

    # β-C planar + BD angle on any-electrophile subset
    ei = [i for i, h in enumerate(hit_classes_per_mol) if h]
    smis_e = [smis[i] for i in ei]
    planar_median = bd_dev_median = None
    n_geom = 0
    if smis_e:
        gdf = compute_geom_panel(smis_e, workers=workers, n_confs=n_confs)
        p_ok = gdf.loc[gdf["embed_ok"], "planar_dev_deg"].dropna().values
        b_ok = gdf.loc[gdf["embed_ok"], "bd_dev_deg"].dropna().values
        planar_median = float(np.median(p_ok)) if len(p_ok) else None
        bd_dev_median = float(np.median(b_ok)) if len(b_ok) else None
        n_geom = int(len(p_ok))

    return {
        "n_total": n,
        "n_valid": n_valid,
        "validity_rate": n_valid / n if n else None,
        "n_unique": n_unique,
        "unique_frac": n_unique / n_valid if n_valid else None,
        "n_any_electrophile": n_any_electrophile,
        "any_electrophile_rate": n_any_electrophile / n_valid if n_valid else None,
        "n_anchor_class_match": n_anchor_match,
        "anchor_class_match_rate": n_anchor_match / n_valid if n_valid else None,
        "electrophile_class_entropy": entropy_H,
        "electrophile_class_counts": class_counts,
        "anchor_murcko_tc_median": tc_murcko,
        "novelty_top1_tc_median": novelty_median,
        "novelty_frac_tc_lt_0p6": novelty_frac,
        "planar_dev_median_deg": planar_median,
        "bd_dev_median_deg": bd_dev_median,
        "n_geom_evaluated": n_geom,
    }


def parse_stem(stem: str) -> tuple:
    """Extract (model, anchor) from stem like 'v2_cond_with_zap70_mol1'."""
    for mid in ["v2_cond_with_zap70", "v2_cond_no_zap70", "mol2mol_baseline",
                "v2_cond_diversified", "drugflow_v2G", "optionA_stage2",
                "v2_cond_denoise_dap", "v2_cond_denoise",
                "v2_cond_dap_min_strict", "v2_cond_dap_geom_dpo",
                "v2_cond_P5-1", "v2_cond_P5-4", "v2_cond_P5-A", "v2_cond_P5-C",
                "v2_cond_P5-CROSSPDB", "v2_cond_P5-HYBRID",
                "v2_cond_P5-1-v2", "v2_cond_P5-4-v2", "v2_cond_P5-A-v2",
                "v2_cond_P5-C-v2", "v2_cond_P5-CROSSPDB-v2", "v2_cond_P5-HYBRID-v2"]:
        if stem.startswith(mid + "_"):
            return mid, stem[len(mid) + 1:]
    return "unknown", stem


def build_train_fps(train_smiles_by_model: dict) -> dict:
    out = {}
    for mid, smis in train_smiles_by_model.items():
        fps = [morgan_fp(s) for s in set(smis)]
        fps = [f for f in fps if f is not None]
        out[mid] = fps
        print(f"[fps] {mid}: {len(fps)} usable training FPs", flush=True)
    return out


def load_train_smiles(cache_paths: dict) -> dict:
    out = {}
    for mid, path in cache_paths.items():
        if not Path(path).exists():
            print(f"[warn] no cache for {mid}: {path}"); out[mid] = []; continue
        d = np.load(path, allow_pickle=True)
        if "smiles" in d.files:
            out[mid] = [str(s) for s in d["smiles"]]
        elif "src_smi" in d.files and "tgt_smi" in d.files:
            out[mid] = list(np.concatenate([d["src_smi"].astype(object), d["tgt_smi"].astype(object)]))
        else:
            print(f"[warn] unrecognized cache format for {mid}: keys={d.files}"); out[mid] = []
    return out


def sdf_to_csv(sdf_path: Path, out_csv: Path, model_id: str, anchor: str):
    """Convert a DrugFlow/OptionA SDF to a temporary CSV so we can run the same panel."""
    from rdkit.Chem import SDMolSupplier
    sup = SDMolSupplier(str(sdf_path), removeHs=False)
    rows = []
    for m in sup:
        if m is None: continue
        try:
            s = Chem.MolToSmiles(m)
            if s: rows.append({"SMILES": s, "model": model_id, "anchor": anchor})
        except: pass
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dirs", nargs="+", default=["data/exp_N/samples"])
    ap.add_argument("--extra_sdfs", nargs="*", default=[],
                    help="Extra SDFs to eval: format model_id:anchor:path[,...]")
    ap.add_argument("--out_dir", default="data/exp_metric_panel")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_confs", type=int, default=10)
    ap.add_argument("--train_cache_full",
                    default="data/m1a_triples_v2/esm2_cache_posefix_v3.npz")
    ap.add_argument("--train_cache_no_zap",
                    default="data/exp_N/esm_cache_no_zap70.npz")
    ap.add_argument("--train_pairs_mol2mol",
                    default="data/exp_N/mol2mol_baseline_pairs.npz")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load training FPs per model
    cache_paths = {
        "v2_cond_with_zap70": args.train_cache_full,
        "v2_cond_no_zap70": args.train_cache_no_zap,
        "mol2mol_baseline": args.train_pairs_mol2mol,
    }
    train_smis = load_train_smiles(cache_paths)
    train_fps = build_train_fps(train_smis)
    # For 3D methods (DrugFlow etc), use the full training corpus as reference
    train_fps["drugflow_v2G"] = train_fps.get("v2_cond_with_zap70", [])
    train_fps["optionA_stage2"] = train_fps.get("v2_cond_with_zap70", [])
    train_fps["v2_cond_diversified"] = train_fps.get("v2_cond_with_zap70", [])

    # Collect all sample CSVs
    csvs = []
    for d in args.samples_dirs:
        for p in sorted(Path(d).glob("*.csv")):
            model, anchor = parse_stem(p.stem)
            csvs.append((p, model, anchor))

    # Convert extra SDFs
    for s in args.extra_sdfs:
        # format: model_id:anchor:sdf_path
        try:
            model_id, anchor, sdf_path = s.split(":")
        except ValueError:
            print(f"[skip bad sdf spec] {s}"); continue
        if not Path(sdf_path).exists():
            print(f"[skip missing sdf] {sdf_path}"); continue
        csv_out = out_dir / f"{model_id}_{anchor}.csv"
        sdf_to_csv(Path(sdf_path), csv_out, model_id, anchor)
        csvs.append((csv_out, model_id, anchor))

    # Run panel
    all_rows = []
    for csv_path, model, anchor in csvs:
        print(f"\n=== {model} anchor={anchor} ===", flush=True)
        fps = train_fps.get(model, train_fps.get("v2_cond_with_zap70", []))
        row = analyze_cohort(str(csv_path), anchor, fps, args.workers, args.n_confs)
        row["model"] = model
        row["anchor"] = anchor
        (out_dir / f"{model}_{anchor}.json").write_text(json.dumps(row, indent=2, default=str))
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    csv_out = out_dir / "expP1_panel.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[DONE] wrote {csv_out}")


if __name__ == "__main__":
    main()
