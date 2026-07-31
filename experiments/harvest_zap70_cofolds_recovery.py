"""ZAP70 acrylamide Boltz cofold RECOVERY harvest.

Adds three previously-skipped sources to the earlier harvest at
`data/paper_pair_training/zap70_cofold_harvest/zap70_acryl_mpae.csv`:

  Source 1  split_a  (884 dirs, PAE already in-tree, but SMILES source
                     was unresolvable → now joined to
                     experiments/boltz_inputs/survivors_1887__zap70_cys346/*.yaml).
  Source 2  split_b  (884 dirs, no PAE in-tree; 119 of them have PAE
                     inside boltz_b_scores.tgz — extracted to
                     data/boltz_results/tgz_extracted/b_pae/). The
                     remaining 765 use complex_ipde substitute.
  Source 3  m1a_v2_vs_covft/boltz_out/ (903 predictions, ZAP70 Cys346,
                     PAE fully present, SMILES from manifest.csv).

Output:
  data/paper_pair_training/zap70_cofold_harvest/zap70_acryl_mpae_extended.csv
  data/paper_pair_training/zap70_cofold_harvest/recovery_report.md

Recipe (identical to the earlier harvest — reused from harvest_zap70_cofolds.py):
  * Acrylamide-on-largest-fragment SMARTS filter
  * mpae_warhead_cys = 0.5 * (PAE[Cys346_tok, wh_tok] + PAE[wh_tok, Cys346_tok])
  * Pose geometry: d_SG, Bürgi-Dunitz, planar dihedral
  * Deduplicate by canonical SMILES (keep first, but here KEEP the union
    of the original 5,693 plus the new rows — old rows win in dedup)
"""
from __future__ import annotations
import json, os, sys, math, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = ROOT / "data/paper_pair_training/zap70_cofold_harvest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_CSV = OUT_DIR / "zap70_acryl_mpae.csv"
EXTENDED_CSV = OUT_DIR / "zap70_acryl_mpae_extended.csv"
RECOVERY_REPORT = OUT_DIR / "recovery_report.md"

CYS_RESI = 346
ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2;X3]=[CH;X3]C(=O)[N]")

_WH_SMARTS_LIST = [
    ("acrylamide",    Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")),
    ("acrylate",      Chem.MolFromSmarts("[CH2]=[CH]C(=O)O")),
    ("vinyl_sulfone", Chem.MolFromSmarts("[CH2]=[CH]S(=O)(=O)")),
    ("haloacetamide", Chem.MolFromSmarts("[Cl,Br,I][CH2]C(=O)N")),
    ("propiolamide",  Chem.MolFromSmarts("C#CC(=O)N")),
]


def derive_warhead_atom_name(smi: str) -> str | None:
    if not smi: return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m_h = AllChem.AddHs(m)
    can = list(AllChem.CanonicalRankAtoms(m_h))
    for _, patt in _WH_SMARTS_LIST:
        if patt is None: continue
        matches = m_h.GetSubstructMatches(patt)
        if matches:
            return f"C{can[matches[0][0]] + 1}"
    return None


def parse_yaml_smi_wh(yaml_path: Path) -> dict | None:
    """Extract SMILES + warhead atom name from a simple Boltz yaml (avoid pyyaml)."""
    if not yaml_path.exists(): return None
    txt = yaml_path.read_text()
    m_smi = re.search(r"smiles:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", txt, re.MULTILINE)
    m_bond = re.search(r"atom2:\s*\[B,\s*1,\s*([A-Z]\d+)\]", txt)
    if not (m_smi and m_bond): return None
    return {"smiles": m_smi.group(1).strip(), "warhead_atom": m_bond.group(1)}


def parse_cif(cif_path: Path):
    st = gemmi.read_structure(str(cif_path))
    A_res = []; A_sg = None; A_ca = None; A_cb = None; B_atoms = []
    for m in st:
        for ch in m:
            if ch.name == "A":
                for r in ch:
                    A_res.append((r.seqid.num, r.name))
                    if r.seqid.num == CYS_RESI and r.name == "CYS":
                        for a in r:
                            if a.name == "SG":  A_sg = np.array([a.pos.x, a.pos.y, a.pos.z])
                            elif a.name == "CA": A_ca = np.array([a.pos.x, a.pos.y, a.pos.z])
                            elif a.name == "CB": A_cb = np.array([a.pos.x, a.pos.y, a.pos.z])
            elif ch.name == "B":
                for r in ch:
                    for a in r:
                        B_atoms.append({
                            "name": a.name, "element": a.element.name,
                            "pos": np.array([a.pos.x, a.pos.y, a.pos.z]),
                        })
        break
    return A_res, A_sg, A_ca, A_cb, B_atoms


def find_by_name(atoms, n):
    for a in atoms:
        if a["name"] == n: return a
    return None


def find_alpha_carbon(atoms, cb_name):
    cb = find_by_name(atoms, cb_name)
    if cb is None: return None
    best = None
    for a in atoms:
        if a["element"] != "C" or a["name"] == cb_name: continue
        d = float(np.linalg.norm(a["pos"] - cb["pos"]))
        if 1.0 <= d <= 1.9:
            if best is None or d < best[0]: best = (d, a)
    return best[1] if best else None


def find_carbonyl(atoms, cb_name, alpha_c_name):
    if alpha_c_name is None: return None
    ca = find_by_name(atoms, alpha_c_name)
    if ca is None: return None
    for c in atoms:
        if c["element"] != "C" or c["name"] in (cb_name, alpha_c_name): continue
        d = float(np.linalg.norm(c["pos"] - ca["pos"]))
        if not (1.0 <= d <= 1.9): continue
        for o in atoms:
            if o["element"] != "O": continue
            do = float(np.linalg.norm(o["pos"] - c["pos"]))
            if 1.15 <= do <= 1.35:
                return c
    return None


def dihedral(p1, p2, p3, p4):
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    x = float(np.dot(n1, n2)); y = float(np.dot(m1, n2))
    return math.degrees(math.atan2(y, x))


def pose_metrics(A_sg, A_ca, A_cb, B_atoms, wh):
    out = {"d_b_nuc_angstrom": None, "bd_angle_deg": None, "phi_planar_deg": None}
    if wh is None or A_sg is None: return out
    cb = find_by_name(B_atoms, wh)
    if cb is None: return out
    out["d_b_nuc_angstrom"] = float(np.linalg.norm(A_sg - cb["pos"]))
    ca = find_alpha_carbon(B_atoms, wh)
    if ca is None: return out
    v1 = A_sg - cb["pos"]; v2 = ca["pos"] - cb["pos"]
    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    cos_a = max(-1.0, min(1.0, cos_a))
    out["bd_angle_deg"] = float(np.degrees(math.acos(cos_a)))
    cg = find_carbonyl(B_atoms, wh, ca["name"])
    if cg is not None:
        out["phi_planar_deg"] = dihedral(A_sg, cb["pos"], ca["pos"], cg["pos"])
    return out


def compute_mpae(pae_npz: Path, n_res_A, wh, B_atoms):
    if pae_npz is None or not pae_npz.exists(): return None
    try:
        d = np.load(pae_npz)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception:
        return None
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]: return None
    N = pae.shape[0]; n_lig = len(B_atoms)
    if N != n_res_A + n_lig: return None
    i_cys = CYS_RESI - 1
    lig_pos = None
    for i, a in enumerate(B_atoms):
        if a["name"] == wh:
            lig_pos = i; break
    if lig_pos is None: return None
    i_wh = n_res_A + lig_pos
    return 0.5 * (float(pae[i_cys, i_wh]) + float(pae[i_wh, i_cys]))


def read_conf(json_path: Path) -> dict:
    if not json_path.exists(): return {}
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return {}
    return {
        "complex_iptm": d.get("iptm") or d.get("complex_iptm"),
        "ligand_iptm": d.get("ligand_iptm"),
        "complex_plddt": d.get("complex_plddt"),
        "complex_ipde": d.get("complex_ipde") or d.get("complex_pde"),
    }


def is_acrylamide_on_largest(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags: return False
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    return largest.HasSubstructMatch(ACRYLAMIDE_SMARTS)


def canonical_smiles(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return Chem.MolToSmiles(m, canonical=True)


# ------------------------------------------------------------
# Task enumeration for the three new sources
# ------------------------------------------------------------

SURVIVORS_YAMLS = ROOT / "experiments/boltz_inputs/survivors_1887__zap70_cys346"


def load_survivors_idx() -> dict[str, dict]:
    """stem (e.g. '000033_Tier_3_v3_Mol2Mol_warhea_33') → {smiles, warhead_atom}."""
    idx = {}
    for yp in SURVIVORS_YAMLS.glob("*.yaml"):
        info = parse_yaml_smi_wh(yp)
        if info is None: continue
        idx[yp.stem] = info
    return idx


def enumerate_split_a(survivors_idx) -> list[tuple]:
    """Split-A: pae is present in-tree; join SMILES via survivors_1887 yamls.
    NOTE: split_a stores PDB, not CIF — parse_cif also handles PDB via gemmi."""
    base = ROOT / "data/boltz_results/split_a"
    tasks = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("boltz_results_"): continue
        name = entry.name.replace("boltz_results_", "")
        pred = entry / "predictions" / name
        # Prefer CIF; fall back to PDB (split_a uses PDB)
        cif = pred / f"{name}_model_0.cif"
        pdb = pred / f"{name}_model_0.pdb"
        struct = cif if cif.exists() else (pdb if pdb.exists() else None)
        pae = pred / f"pae_{name}_model_0.npz"
        conf = pred / f"confidence_{name}_model_0.json"
        if struct is None or not conf.exists(): continue
        meta = survivors_idx.get(name)
        if meta is None: continue
        tasks.append((
            "split_a", name, str(struct),
            str(pae) if pae.exists() else None,
            str(conf), meta["smiles"], meta["warhead_atom"], name,
        ))
    return tasks


def enumerate_split_b(survivors_idx) -> list[tuple]:
    """Split-B: no in-tree pae; 119 have pae in tgz_extracted/b_pae/.
    NOTE: split_b stores PDB, not CIF."""
    base = ROOT / "data/boltz_results/split_b"
    tgz_base = ROOT / "data/boltz_results/tgz_extracted/b_pae"
    tasks = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("boltz_results_"): continue
        name = entry.name.replace("boltz_results_", "")
        pred = entry / "predictions" / name
        cif = pred / f"{name}_model_0.cif"
        pdb = pred / f"{name}_model_0.pdb"
        struct = cif if cif.exists() else (pdb if pdb.exists() else None)
        conf = pred / f"confidence_{name}_model_0.json"
        if struct is None or not conf.exists(): continue
        # PAE from tgz — same relative path
        pae_tgz = tgz_base / f"boltz_results_{name}" / "predictions" / name / f"pae_{name}_model_0.npz"
        meta = survivors_idx.get(name)
        if meta is None: continue
        tasks.append((
            "split_b", name, str(struct),
            str(pae_tgz) if pae_tgz.exists() else None,
            str(conf), meta["smiles"], meta["warhead_atom"], name,
        ))
    return tasks


def enumerate_m1a(manifest_df: pd.DataFrame) -> list[tuple]:
    """m1a_v2_vs_covft/boltz_out/boltz_results_yamls/predictions/<mol_id>/…"""
    base = ROOT / "data/m1a_v2_vs_covft/boltz_out/boltz_results_yamls/predictions"
    # manifest already has SMILES + warhead_atom_name
    lookup = {}
    for _, r in manifest_df.iterrows():
        mid = r["mol_id"]
        smi = r["smiles"]
        wh = r["warhead_atom_name"]
        if not isinstance(mid, str) or not isinstance(smi, str): continue
        lookup[mid] = {"smiles": smi, "warhead_atom": wh if isinstance(wh, str) else None}
    tasks = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir(): continue
        name = entry.name
        cif = entry / f"{name}_model_0.cif"
        pae = entry / f"pae_{name}_model_0.npz"
        conf = entry / f"confidence_{name}_model_0.json"
        if not cif.exists() or not conf.exists(): continue
        meta = lookup.get(name)
        if meta is None: continue
        wh = meta["warhead_atom"]
        if wh is None:
            # Derive
            wh = derive_warhead_atom_name(meta["smiles"])
        if wh is None: continue
        tasks.append((
            "m1a_v2_vs_covft", name, str(cif),
            str(pae) if pae.exists() else None,
            str(conf), meta["smiles"], wh, name,
        ))
    return tasks


# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------

def process(args):
    source_dir, name, cif, pae, conf_path, smi, wh, row_id = args
    try:
        A_res, A_sg, A_ca, A_cb, B_atoms = parse_cif(Path(cif))
        n_A = len(A_res)
        pose = pose_metrics(A_sg, A_ca, A_cb, B_atoms, wh)
        conf = read_conf(Path(conf_path))
        mpae = None; mpae_source = None
        if pae:
            mpae = compute_mpae(Path(pae), n_A, wh, B_atoms)
            if mpae is not None:
                # Distinguish by source
                if source_dir == "split_a":
                    mpae_source = "pae_tensor_split_a_recovered"
                elif source_dir == "split_b":
                    mpae_source = "pae_tensor_split_b_recovered"
                elif source_dir == "m1a_v2_vs_covft":
                    mpae_source = "pae_tensor_m1a_v2_vs_covft"
                else:
                    mpae_source = "pae_tensor"
        if mpae is None:
            if conf.get("complex_ipde") is not None:
                mpae = float(conf["complex_ipde"])
                mpae_source = "complex_ipde_substitute"
            elif conf.get("ligand_iptm") is not None:
                mpae = 1.0 - float(conf["ligand_iptm"])
                mpae_source = "one_minus_ligand_iptm_substitute"
            else:
                mpae_source = "none"
        canon = canonical_smiles(smi)
        return {
            "row_id": row_id, "source_dir": source_dir, "name": name,
            "smiles": smi, "canonical_smiles": canon,
            "warhead_atom": wh, "mpae_warhead_cys": mpae, "mpae_source": mpae_source,
            "complex_iptm": conf.get("complex_iptm"),
            "ligand_iptm": conf.get("ligand_iptm"),
            "complex_plddt": conf.get("complex_plddt"),
            "complex_ipde": conf.get("complex_ipde"),
            **pose, "error": None,
        }
    except Exception as e:
        return {
            "row_id": row_id, "source_dir": source_dir, "name": name,
            "smiles": smi, "canonical_smiles": None, "warhead_atom": wh,
            "mpae_warhead_cys": None, "mpae_source": "error",
            "complex_iptm": None, "ligand_iptm": None, "complex_plddt": None,
            "complex_ipde": None, "d_b_nuc_angstrom": None,
            "bd_angle_deg": None, "phi_planar_deg": None,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("[1/6] Loading survivors_1887 yaml index…", flush=True)
    survivors_idx = load_survivors_idx()
    print(f"       {len(survivors_idx)} yamls parsed", flush=True)

    print("[2/6] Loading m1a_v2_vs_covft manifest…", flush=True)
    m1a_manifest = pd.read_csv(ROOT / "data/m1a_v2_vs_covft/manifest.csv")
    print(f"       manifest rows: {len(m1a_manifest)}", flush=True)

    print("[3/6] Enumerating tasks…", flush=True)
    tasks_a = enumerate_split_a(survivors_idx)
    tasks_b = enumerate_split_b(survivors_idx)
    tasks_m = enumerate_m1a(m1a_manifest)
    print(f"       split_a: {len(tasks_a)}  split_b: {len(tasks_b)}  m1a: {len(tasks_m)}", flush=True)

    per_src_attempted = {
        "split_a": len(tasks_a),
        "split_b": len(tasks_b),
        "m1a_v2_vs_covft": len(tasks_m),
    }

    all_tasks = tasks_a + tasks_b + tasks_m

    print("[4/6] Acrylamide-on-largest-fragment filter…", flush=True)
    kept = []; smi_ok = {}
    for t in all_tasks:
        smi = t[5]
        if smi in smi_ok:
            if smi_ok[smi]: kept.append(t)
            continue
        ok = is_acrylamide_on_largest(smi)
        smi_ok[smi] = ok
        if ok: kept.append(t)
    print(f"       {len(kept)} / {len(all_tasks)} pass acrylamide filter", flush=True)

    per_src_acryl = {}
    for t in kept:
        per_src_acryl[t[0]] = per_src_acryl.get(t[0], 0) + 1

    print("[5/6] Processing cofolds in parallel…", flush=True)
    results = []
    n_workers = max(1, min(8, os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(process, args) for args in kept]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r is not None: results.append(r)
            if (i + 1) % 250 == 0:
                print(f"       {i+1}/{len(kept)}", flush=True)

    df_new = pd.DataFrame(results)
    print(f"       {len(df_new)} rows harvested (raw new)", flush=True)

    # ---- Combine with existing ----
    print("[6/6] Combining with existing 5,693 rows + dedup…", flush=True)
    df_old = pd.read_csv(EXISTING_CSV, low_memory=False)
    # Align columns
    all_cols = ["row_id", "source_dir", "name", "smiles", "canonical_smiles",
                "warhead_atom", "mpae_warhead_cys", "mpae_source",
                "complex_iptm", "ligand_iptm", "complex_plddt", "complex_ipde",
                "d_b_nuc_angstrom", "bd_angle_deg", "phi_planar_deg", "error"]
    for c in all_cols:
        if c not in df_old.columns: df_old[c] = None
        if c not in df_new.columns: df_new[c] = None

    df_old = df_old[all_cols]
    df_new = df_new[all_cols]

    combined = pd.concat([df_old, df_new], ignore_index=True)
    combined = combined.dropna(subset=["canonical_smiles"])
    n_before_dedup = len(combined)
    # OLD rows come first — keep first preserves them
    combined = combined.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
    n_after_dedup = len(combined)

    combined.to_csv(EXTENDED_CSV, index=False)
    print(f"       wrote {EXTENDED_CSV} ({n_after_dedup} rows, {n_before_dedup - n_after_dedup} dupes)", flush=True)

    # ---- Per-source recovery stats ----
    # Which of the new rows survived dedup?
    old_canon = set(df_old["canonical_smiles"].dropna().unique())
    df_new_ok = df_new.dropna(subset=["canonical_smiles"])
    new_kept = df_new_ok[~df_new_ok["canonical_smiles"].isin(old_canon)]
    # Dedup within new_kept as well (older new-row wins)
    new_kept = new_kept.drop_duplicates("canonical_smiles", keep="first")
    per_src_recovered = new_kept["source_dir"].value_counts().to_dict()
    per_src_failed = {src: per_src_attempted[src] - per_src_acryl.get(src, 0)
                      for src in per_src_attempted}
    per_src_dupe_of_old = {}
    per_src_dupe_of_new = {}
    for src in per_src_attempted:
        src_new = df_new_ok[df_new_ok["source_dir"] == src]
        dupe_of_old = int(src_new["canonical_smiles"].isin(old_canon).sum())
        per_src_dupe_of_old[src] = dupe_of_old

    write_report(df_old, df_new, combined, new_kept,
                 per_src_attempted, per_src_acryl,
                 per_src_recovered, per_src_dupe_of_old)

    # summary
    print()
    print("=== TL;DR ===")
    print(f"OLD rows: {len(df_old)}")
    print(f"NEW rows kept (after dedup): {len(new_kept)}")
    print(f"COMBINED total: {n_after_dedup}")
    print(f"Per-source new-kept: {per_src_recovered}")


def write_report(df_old, df_new, combined, new_kept,
                 per_src_attempted, per_src_acryl,
                 per_src_recovered, per_src_dupe_of_old):
    L = ["# ZAP70 Cofold Recovery Report", ""]
    L.append(f"Old total: **{len(df_old)}**   New rows kept: **{len(new_kept)}**   "
             f"Extended total: **{len(combined)}**")
    L.append("")

    # Per-source rundown
    L.append("## Per-source rundown")
    L.append("")
    L.append("| source | attempted | acryl-passed | dupe-of-old | new-kept | failed |")
    L.append("|--------|-----------|--------------|-------------|----------|--------|")
    for src in per_src_attempted:
        att = per_src_attempted[src]
        ac = per_src_acryl.get(src, 0)
        dp = per_src_dupe_of_old.get(src, 0)
        rk = per_src_recovered.get(src, 0)
        fl = att - ac
        L.append(f"| {src} | {att} | {ac} | {dp} | {rk} | {fl} |")
    L.append("")

    # mpae_source distribution overall
    L.append("## mpae_source distribution in extended CSV")
    L.append("")
    ms = combined["mpae_source"].value_counts()
    L.append("| mpae_source | count |")
    L.append("|-------------|-------|")
    for k, v in ms.items():
        L.append(f"| {k} | {int(v)} |")
    L.append("")

    # Distribution of NEW real-PAE rows per source vs original F4
    L.append("## Distribution comparison of `mpae_warhead_cys`")
    L.append("")
    L.append("| subset | n | median | mean | std | IQR | min | max |")
    L.append("|--------|---|--------|------|-----|-----|-----|-----|")

    def stat_row(label, s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            L.append(f"| {label} | 0 | - | - | - | - | - | - |")
            return
        L.append(f"| {label} | {len(s)} | {s.median():.3f} | {s.mean():.3f} | {s.std():.3f} | "
                 f"{s.quantile(0.75) - s.quantile(0.25):.3f} | {s.min():.3f} | {s.max():.3f} |")

    # Original F4 pae_tensor rows
    old_real = df_old[df_old["mpae_source"] == "pae_tensor"]
    stat_row("OLD F4 pae_tensor", old_real["mpae_warhead_cys"])
    # Original complex_ipde
    old_ipde = df_old[df_old["mpae_source"] == "complex_ipde_substitute"]
    stat_row("OLD complex_ipde_substitute", old_ipde["mpae_warhead_cys"])
    # New real PAE per-source
    for src_label, ms_key in [
        ("NEW split_a pae recovered", "pae_tensor_split_a_recovered"),
        ("NEW split_b pae recovered", "pae_tensor_split_b_recovered"),
        ("NEW m1a_v2_vs_covft pae",   "pae_tensor_m1a_v2_vs_covft"),
    ]:
        sub = new_kept[new_kept["mpae_source"] == ms_key]
        stat_row(src_label, sub["mpae_warhead_cys"])
    new_sub = new_kept[new_kept["mpae_source"] == "complex_ipde_substitute"]
    stat_row("NEW complex_ipde_substitute", new_sub["mpae_warhead_cys"])
    L.append("")

    # Recommendation
    L.append("## Interpretation")
    L.append("")
    old_real_med = old_real["mpae_warhead_cys"].median() if len(old_real) else float("nan")
    L.append(f"- OLD F4 real-PAE median: {old_real_med:.3f} (std={old_real['mpae_warhead_cys'].std():.3f}, n={len(old_real)})")
    for src_label, ms_key in [
        ("NEW split_a", "pae_tensor_split_a_recovered"),
        ("NEW split_b", "pae_tensor_split_b_recovered"),
        ("NEW m1a_v2_vs_covft", "pae_tensor_m1a_v2_vs_covft"),
    ]:
        sub = new_kept[new_kept["mpae_source"] == ms_key]
        if len(sub):
            m = sub["mpae_warhead_cys"].median()
            s = sub["mpae_warhead_cys"].std()
            delta = (m - old_real_med) / max(old_real["mpae_warhead_cys"].std(), 1e-6)
            L.append(f"- {src_label} median: {m:.3f} (std={s:.3f}, n={len(sub)}) — "
                     f"{'ON-SCALE' if abs(delta) < 1.0 else 'OFF-SCALE'} vs OLD F4 "
                     f"({delta:+.2f} σ_old shift)")
    L.append("")
    L.append("### Recommendation")
    L.append("")
    n_real_total = int((combined["mpae_source"].str.startswith("pae_tensor")).sum())
    L.append(f"- Total rows with a real PAE tensor in extended CSV: **{n_real_total}**")
    if n_real_total >= 4800:
        L.append(f"- If per-source medians are within ~1 σ of the OLD F4 median, the "
                 f"real-PAE subset (n={n_real_total}) is homogeneous enough for direct training.")
        L.append(f"- Otherwise recommend per-source normalization (subtract source median).")
    else:
        L.append(f"- Real-PAE subset < 4,800 — may need to also use the substitute rows.")

    RECOVERY_REPORT.write_text("\n".join(L))
    print(f"       wrote {RECOVERY_REPORT}", flush=True)


if __name__ == "__main__":
    main()
