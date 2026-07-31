"""ZAP70 acrylamide Boltz cofold harvest.

Scans all local Boltz cofold sources for ZAP70 acrylamide predictions,
extracts mpae_warhead_cys (PAE at [Cys346 residue token, warhead-Cβ atom token]),
plus pose geometry (d_SG, Bürgi-Dunitz, planar dihedral) and confidence scalars.

Sources scanned:
  data/boltz_f4_results/ai-gpu-a100{,-b,-c}/            (F4 first pull)
  data/boltz_f4_extra_results/ai-gpu-a100{,-d}/         (F4 extras)
  data/boltz_results/cohort_3597_full/from_{a..h}/      (Tier-3 v2/v3, no PAE)

Output: data/paper_pair_training/zap70_cofold_harvest/
  zap70_acryl_mpae.csv
  harvest_summary.json
  variability_report.md
"""
from __future__ import annotations
import json, os, sys, math, gc, re
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

CYS_RESI = 346  # ZAP70 reactive cys
# Acrylamide chemistry: vinyl-amide (any N substitution). F4 warheads are tertiary amides
# like C=CC(=O)N1Cc2cccc(...)c2C1, cohort_3597 warheads are secondary amides
# (C=CC(=O)Nc1cccc(...)c1F). Both are ZAP70-relevant Michael acceptors.
ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2;X3]=[CH;X3]C(=O)[N]")

# For warhead-atom derivation when manifest lacks it (F4 extras).
# Uses same canonical-rank logic as gen_covalid_boltz_yamls.py.
_WH_SMARTS_LIST = [
    ("acrylamide",    Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")),
    ("acrylate",      Chem.MolFromSmarts("[CH2]=[CH]C(=O)O")),
    ("vinyl_sulfone", Chem.MolFromSmarts("[CH2]=[CH]S(=O)(=O)")),
    ("haloacetamide", Chem.MolFromSmarts("[Cl,Br,I][CH2]C(=O)N")),
    ("propiolamide",  Chem.MolFromSmarts("C#CC(=O)N")),
]


def derive_warhead_atom_name(smi: str) -> str | None:
    """Boltz atom name for the warhead Cβ, matching gen_covalid_boltz_yamls canonical-rank logic."""
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

# ------------------------------------------------------------
# SMILES + warhead-atom source resolution
# ------------------------------------------------------------

def load_f4_manifests() -> dict[str, dict]:
    """Return dict: (source_dir_key, row_name) → {'smiles', 'warhead_atom', 'row_id'}."""
    idx = {}
    # F4 first-pull manifests (row00001 → row00001)
    for path, key in [
        (ROOT / "data/boltz_f4_results/ai-gpu-a100/manifest.csv",   "f4_a100"),
        (ROOT / "data/boltz_f4_results/ai-gpu-a100-b/manifest.csv", "f4_a100b"),
        (ROOT / "data/boltz_f4_results/ai-gpu-a100-c/manifest.csv", "f4_a100c"),
    ]:
        if not path.exists(): continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            idx[(key, r["name"])] = {
                "smiles": r["smiles"],
                "warhead_atom": r["warhead_atom"],
                "row_id": r["row_id"],
            }
    # F4 extras: use tier4_scored/F4_boltz_extra_v2.csv (row_id=EXTRA_00000, warhead atom derived)
    extras_csv = ROOT / "data/tier4_scored/F4_boltz_extra_v2.csv"
    if extras_csv.exists():
        df = pd.read_csv(extras_csv, low_memory=False)
        for _, r in df.iterrows():
            rid = r.get("row_id")
            smi = r.get("smiles")
            if not isinstance(rid, str) or not rid.startswith("EXTRA_"): continue
            if not isinstance(smi, str): continue
            wh = derive_warhead_atom_name(smi)
            # register under both keys (f4x_a100 and f4x_a100d — enumerator resolves to actual dir)
            for k in ("f4x_a100", "f4x_a100d"):
                idx[(k, rid)] = {"smiles": smi, "warhead_atom": wh, "row_id": rid}
    return idx


def parse_cohort_yaml(yaml_path: Path) -> dict | None:
    """Grab SMILES + warhead atom name from a cohort_3597 yaml (simple regex — avoid pyyaml dep)."""
    if not yaml_path.exists(): return None
    txt = yaml_path.read_text()
    m_smi = re.search(r"smiles:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", txt, re.MULTILINE)
    m_bond = re.search(r"atom2:\s*\[B,\s*1,\s*([A-Z]\d+)\]", txt)
    if not (m_smi and m_bond): return None
    return {"smiles": m_smi.group(1).strip(), "warhead_atom": m_bond.group(1)}


def load_cohort3597_yaml_index() -> dict[str, dict]:
    """stem (e.g. '598309') → {smiles, warhead_atom}."""
    idx = {}
    for chunk_dir in sorted((ROOT / "data/boltz_results/cohort_3597_full/yamls").iterdir()):
        if not chunk_dir.is_dir(): continue
        for yp in chunk_dir.glob("*.yaml"):
            info = parse_cohort_yaml(yp)
            if info is None: continue
            idx[yp.stem] = info
    return idx

# ------------------------------------------------------------
# Cofold parsing + pose geometry
# ------------------------------------------------------------

def parse_cif(cif_path: Path):
    """Return (chain_A_resis[list of (resi,resname)], chain_B_atoms[list of (name,element,pos)],
    Cys346 SG pos or None, protein N-token count)."""
    st = gemmi.read_structure(str(cif_path))
    A_res = []      # (resi, resname)
    A_sg_pos = None
    A_ca_346 = None
    A_cb_346 = None
    B_atoms = []    # (name, element, pos)
    for m in st:
        for ch in m:
            if ch.name == "A":
                for r in ch:
                    A_res.append((r.seqid.num, r.name))
                    if r.seqid.num == CYS_RESI and r.name == "CYS":
                        for a in r:
                            if a.name == "SG":
                                A_sg_pos = np.array([a.pos.x, a.pos.y, a.pos.z])
                            elif a.name == "CA":
                                A_ca_346 = np.array([a.pos.x, a.pos.y, a.pos.z])
                            elif a.name == "CB":
                                A_cb_346 = np.array([a.pos.x, a.pos.y, a.pos.z])
            elif ch.name == "B":
                for r in ch:
                    for a in r:
                        B_atoms.append({
                            "name": a.name,
                            "element": a.element.name,
                            "pos": np.array([a.pos.x, a.pos.y, a.pos.z]),
                        })
        break
    return A_res, A_sg_pos, A_ca_346, A_cb_346, B_atoms


def find_atom_by_name(atoms: list[dict], name: str):
    for a in atoms:
        if a["name"] == name: return a
    return None


def find_alpha_carbon(atoms: list[dict], cb_name: str):
    cb = find_atom_by_name(atoms, cb_name)
    if cb is None: return None
    best = None
    for a in atoms:
        if a["element"] != "C" or a["name"] == cb_name: continue
        d = float(np.linalg.norm(a["pos"] - cb["pos"]))
        if 1.0 <= d <= 1.9:
            if best is None or d < best[0]:
                best = (d, a)
    return best[1] if best else None


def find_carbonyl_carbon(atoms: list[dict], cb_name: str, alpha_c_name: str | None):
    """Find the C=O carbon (Cγ) — the C bonded to alpha_C that's also bonded to O.
    We approximate: search atoms of C element within 1.0..1.9 of alpha_C
    that have an O within 1.15..1.35 (C=O)."""
    if alpha_c_name is None: return None
    ca = find_atom_by_name(atoms, alpha_c_name)
    if ca is None: return None
    for c in atoms:
        if c["element"] != "C" or c["name"] in (cb_name, alpha_c_name): continue
        d = float(np.linalg.norm(c["pos"] - ca["pos"]))
        if not (1.0 <= d <= 1.9): continue
        # any O within 1.15..1.35 of this C?
        for o in atoms:
            if o["element"] != "O": continue
            do = float(np.linalg.norm(o["pos"] - c["pos"]))
            if 1.15 <= do <= 1.35:
                return c
    return None


def dihedral(p1, p2, p3, p4):
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2)+1e-12))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return math.degrees(math.atan2(y, x))


def compute_pose_metrics(A_sg, A_ca_346, A_cb_346, B_atoms, warhead_atom_name):
    """Return dict of d_SG, BD angle deviation, phi_planar (S–Cβ–Cα–C=O dihedral)."""
    out = {"d_b_nuc_angstrom": None, "bd_angle_deg": None, "phi_planar_deg": None}
    if warhead_atom_name is None or A_sg is None: return out
    cb = find_atom_by_name(B_atoms, warhead_atom_name)
    if cb is None: return out
    out["d_b_nuc_angstrom"] = float(np.linalg.norm(A_sg - cb["pos"]))

    ca = find_alpha_carbon(B_atoms, warhead_atom_name)
    if ca is None: return out
    # Bürgi–Dunitz angle (SG, Cβ, Cα) — vertex at Cβ
    v1 = A_sg - cb["pos"]; v2 = ca["pos"] - cb["pos"]
    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    cos_a = max(-1.0, min(1.0, cos_a))
    bd_angle_deg = float(np.degrees(math.acos(cos_a)))
    out["bd_angle_deg"] = bd_angle_deg

    # phi_planar: dihedral S–Cβ–Cα–Cγ  (Cγ = carbonyl carbon)
    cg = find_carbonyl_carbon(B_atoms, warhead_atom_name, ca["name"])
    if cg is not None:
        out["phi_planar_deg"] = dihedral(A_sg, cb["pos"], ca["pos"], cg["pos"])
    return out


def compute_mpae_warhead_cys(pae_npz_path: Path, n_res_A: int, wh_atom_name: str, B_atoms: list[dict]):
    """Boltz token layout: [chain-A residues... chain-B atoms...].
    mpae_warhead_cys = mean(PAE[Cys346_tok, wh_tok], PAE[wh_tok, Cys346_tok])."""
    if not pae_npz_path.exists(): return None
    try:
        d = np.load(pae_npz_path)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception:
        return None
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]: return None
    N = pae.shape[0]
    n_lig = len(B_atoms)
    if N != n_res_A + n_lig: return None
    # Cys346 = res index (num-1 since numbering starts at 1)
    i_cys = CYS_RESI - 1
    # ligand token = n_res_A + position of wh atom in B_atoms list
    lig_pos = None
    for i, a in enumerate(B_atoms):
        if a["name"] == wh_atom_name:
            lig_pos = i; break
    if lig_pos is None: return None
    i_wh = n_res_A + lig_pos
    v_cw = float(pae[i_cys, i_wh])
    v_wc = float(pae[i_wh, i_cys])
    return 0.5 * (v_cw + v_wc)


def read_confidence(json_path: Path) -> dict:
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


# ------------------------------------------------------------
# Acrylamide filter (SMILES)
# ------------------------------------------------------------

def is_acrylamide_on_largest(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    # largest fragment
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags: return False
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    return largest.HasSubstructMatch(ACRYLAMIDE_SMARTS)


def canonical_smiles(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return Chem.MolToSmiles(m, canonical=True)


# ------------------------------------------------------------
# Per-cofold worker
# ------------------------------------------------------------

def process_row(args) -> dict | None:
    source_dir, name, cif_path, pae_path, conf_path, smiles, warhead_atom, row_id = args
    try:
        A_res, A_sg, A_ca346, A_cb346, B_atoms = parse_cif(Path(cif_path))
        n_res_A = len(A_res)
        pose = compute_pose_metrics(A_sg, A_ca346, A_cb346, B_atoms, warhead_atom)
        conf = read_confidence(Path(conf_path))
        mpae = None; mpae_source = None
        if pae_path:
            mpae = compute_mpae_warhead_cys(Path(pae_path), n_res_A, warhead_atom, B_atoms)
            if mpae is not None:
                mpae_source = "pae_tensor"
        if mpae is None:
            # substitute: use complex_ipde as scalar proxy (correlates with token-token PAE quality)
            # else ligand_iptm complement
            if conf.get("complex_ipde") is not None:
                mpae = float(conf["complex_ipde"])
                mpae_source = "complex_ipde_substitute"
            elif conf.get("ligand_iptm") is not None:
                mpae = 1.0 - float(conf["ligand_iptm"])
                mpae_source = "one_minus_ligand_iptm_substitute"
            else:
                mpae_source = "none"
        canon = canonical_smiles(smiles)
        return {
            "row_id": row_id,
            "source_dir": source_dir,
            "name": name,
            "smiles": smiles,
            "canonical_smiles": canon,
            "warhead_atom": warhead_atom,
            "mpae_warhead_cys": mpae,
            "mpae_source": mpae_source,
            "complex_iptm": conf.get("complex_iptm"),
            "ligand_iptm": conf.get("ligand_iptm"),
            "complex_plddt": conf.get("complex_plddt"),
            "complex_ipde": conf.get("complex_ipde"),
            **pose,
            "error": None,
        }
    except Exception as e:
        return {
            "row_id": row_id, "source_dir": source_dir, "name": name,
            "smiles": smiles, "canonical_smiles": None,
            "warhead_atom": warhead_atom,
            "mpae_warhead_cys": None, "mpae_source": "error",
            "complex_iptm": None, "ligand_iptm": None, "complex_plddt": None, "complex_ipde": None,
            "d_b_nuc_angstrom": None, "bd_angle_deg": None, "phi_planar_deg": None,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


# ------------------------------------------------------------
# Task enumeration
# ------------------------------------------------------------

def enumerate_f4(f4_manifest_idx) -> list[tuple]:
    """Yield task tuples for F4 sources (F4 first pull + extras)."""
    tasks = []
    src_map = [
        ("f4_a100",   ROOT / "data/boltz_f4_results/ai-gpu-a100"),
        ("f4_a100b",  ROOT / "data/boltz_f4_results/ai-gpu-a100-b"),
        ("f4_a100c",  ROOT / "data/boltz_f4_results/ai-gpu-a100-c"),
        ("f4x_a100",  ROOT / "data/boltz_f4_extra_results/ai-gpu-a100"),
        ("f4x_a100d", ROOT / "data/boltz_f4_extra_results/ai-gpu-a100-d"),
    ]
    for key, base in src_map:
        if not base.exists(): continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("boltz_results_"): continue
            name = entry.name.replace("boltz_results_", "")
            pred = entry / "predictions" / name
            cif = pred / f"{name}_model_0.cif"
            pae = pred / f"pae_{name}_model_0.npz"
            conf = pred / f"confidence_{name}_model_0.json"
            if not cif.exists() or not conf.exists(): continue
            # Lookup key: F4 first pull uses (key, "row00001") from manifest;
            # F4 extras use (key, "EXTRA_00000") from F4_boltz_extra_v2.csv.
            meta = f4_manifest_idx.get((key, name))
            if meta is None: continue
            if not meta.get("warhead_atom"):
                # Skip cofolds where we couldn't derive the warhead atom
                continue
            tasks.append((
                key, name, str(cif), str(pae) if pae.exists() else None, str(conf),
                meta["smiles"], meta["warhead_atom"], meta["row_id"],
            ))
    return tasks


def enumerate_cohort3597(yaml_idx) -> list[tuple]:
    tasks = []
    base = ROOT / "data/boltz_results/cohort_3597_full"
    for sub in sorted(base.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("from_"): continue
        key = f"cohort3597_{sub.name}"
        for entry in sorted(sub.iterdir()):
            if not entry.is_dir(): continue
            name = entry.name  # stem is the row_id
            cif = entry / f"{name}_model_0.cif"
            conf = entry / f"confidence_{name}_model_0.json"
            if not cif.exists() or not conf.exists(): continue
            meta = yaml_idx.get(name)
            if meta is None: continue
            tasks.append((
                key, name, str(cif), None, str(conf),
                meta["smiles"], meta["warhead_atom"], name,
            ))
    return tasks


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("[1/6] Loading F4 manifests...", flush=True)
    f4_manifest_idx = load_f4_manifests()
    print(f"       {len(f4_manifest_idx)} manifest rows across F4 sources", flush=True)

    print("[2/6] Loading cohort_3597 YAML index (~3.6k files)...", flush=True)
    yaml_idx = load_cohort3597_yaml_index()
    print(f"       {len(yaml_idx)} yamls parsed", flush=True)

    print("[3/6] Enumerating cofold tasks...", flush=True)
    tasks = enumerate_f4(f4_manifest_idx) + enumerate_cohort3597(yaml_idx)
    print(f"       {len(tasks)} cofold tasks", flush=True)

    per_source_counts = {}
    for t in tasks:
        per_source_counts[t[0]] = per_source_counts.get(t[0], 0) + 1
    print(f"       per source: {per_source_counts}", flush=True)

    print("[4/6] Acrylamide-on-largest-fragment filter...", flush=True)
    kept = []
    smi_seen_bad = set()
    smi_seen_good = set()
    for t in tasks:
        smi = t[5]
        if smi in smi_seen_bad: continue
        if smi in smi_seen_good:
            kept.append(t); continue
        if is_acrylamide_on_largest(smi):
            smi_seen_good.add(smi); kept.append(t)
        else:
            smi_seen_bad.add(smi)
    print(f"       {len(kept)} / {len(tasks)} tasks pass acrylamide filter", flush=True)

    print("[5/6] Processing cofolds in parallel...", flush=True)
    results = []
    n_workers = max(1, min(8, os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(process_row, args) for args in kept]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r is not None: results.append(r)
            if (i + 1) % 250 == 0:
                print(f"       {i+1}/{len(kept)}", flush=True)

    df = pd.DataFrame(results)
    print(f"       {len(df)} rows harvested (raw)", flush=True)

    # Dedupe by canonical SMILES — keep the row with lowest mpae_warhead_cys
    n_before = len(df)
    df_ok = df.dropna(subset=["canonical_smiles"]).copy()
    df_ok = df_ok.sort_values("mpae_warhead_cys", na_position="last")
    df_dedup = df_ok.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    n_dupes = n_before - len(df_dedup)

    csv_out = OUT_DIR / "zap70_acryl_mpae.csv"
    df_dedup.to_csv(csv_out, index=False)
    print(f"       wrote {csv_out} ({len(df_dedup)} rows, {n_dupes} dupes dropped)", flush=True)

    # ---- Summary JSON ----
    per_source_scanned = per_source_counts
    per_source_acryl = {}
    for t in kept:
        per_source_acryl[t[0]] = per_source_acryl.get(t[0], 0) + 1
    mpae_source_counts = df_dedup["mpae_source"].value_counts().to_dict()
    summary = {
        "per_source_scanned": per_source_scanned,
        "per_source_acryl_kept": per_source_acryl,
        "n_processed_raw": int(n_before),
        "n_dedup_by_canonical_smiles": int(len(df_dedup)),
        "n_dupes_dropped": int(n_dupes),
        "mpae_source_breakdown": mpae_source_counts,
        "n_with_pae_tensor": int((df_dedup["mpae_source"] == "pae_tensor").sum()),
    }
    (OUT_DIR / "harvest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[6/6] Wrote harvest_summary.json", flush=True)

    # ---- Variability report ----
    report = write_variability_report(df_dedup)
    (OUT_DIR / "variability_report.md").write_text(report)
    print(f"       wrote variability_report.md", flush=True)

    # console tl;dr
    print()
    print("=== TL;DR ===")
    print(f"N labeled: {len(df_dedup)}")
    if (df_dedup["mpae_source"] == "pae_tensor").any():
        real = df_dedup[df_dedup["mpae_source"] == "pae_tensor"]["mpae_warhead_cys"]
        print(f"mpae_warhead_cys (pae_tensor only, N={len(real)}): median={real.median():.3f}  "
              f"mean={real.mean():.3f}  std={real.std():.3f}  IQR=[{real.quantile(0.25):.3f}, {real.quantile(0.75):.3f}]")


def write_variability_report(df: pd.DataFrame) -> str:
    lines = ["# ZAP70 Acrylamide Boltz Cofold Harvest — Variability Report", ""]
    lines.append(f"Total labeled cofolds (dedup by canonical SMILES): **{len(df)}**")
    lines.append("")

    # Sub-split with real PAE
    real = df[df["mpae_source"] == "pae_tensor"].copy()
    subst = df[df["mpae_source"] != "pae_tensor"].copy()
    lines.append(f"- With real PAE tensor (`mpae_warhead_cys` from pae_*.npz): **{len(real)}**")
    lines.append(f"- With substitute metric (complex_ipde etc.): **{len(subst)}**")
    lines.append("")

    def stats_block(v: pd.Series, label: str):
        b = [f"## {label} (n={len(v)})"]
        if len(v) == 0: return b
        b.append(f"- median = {v.median():.3f}")
        b.append(f"- mean   = {v.mean():.3f}")
        b.append(f"- std    = {v.std():.3f}")
        b.append(f"- min    = {v.min():.3f}")
        b.append(f"- Q25    = {v.quantile(0.25):.3f}")
        b.append(f"- Q75    = {v.quantile(0.75):.3f}")
        b.append(f"- max    = {v.max():.3f}")
        b.append(f"- IQR    = {v.quantile(0.75) - v.quantile(0.25):.3f}")
        # histogram (10 bins)
        counts, edges = np.histogram(v.dropna().values, bins=10)
        b.append("")
        b.append("### Histogram (10 bins)")
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            b.append(f"  [{lo:6.3f}, {hi:6.3f}) : {int(c):5d}")
        return b

    if len(real):
        lines += stats_block(real["mpae_warhead_cys"], "mpae_warhead_cys distribution (pae_tensor)")
        lines.append("")

    # Room for improvement scalar
    if len(real) >= 20:
        q1 = real["mpae_warhead_cys"].quantile(0.25)
        q3 = real["mpae_warhead_cys"].quantile(0.75)
        best_q = real[real["mpae_warhead_cys"] <= q1]["mpae_warhead_cys"]
        worst_q = real[real["mpae_warhead_cys"] >= q3]["mpae_warhead_cys"]
        if len(best_q) and len(worst_q) and best_q.median() > 0:
            ratio = worst_q.median() / best_q.median()
            lines.append(f"## Room-for-improvement scalar")
            lines.append(f"- median(worst quartile) / median(best quartile) = "
                         f"{worst_q.median():.3f} / {best_q.median():.3f} = **{ratio:.2f}x**")
            lines.append("")

    # Top 25 / Bottom 25
    if len(real) >= 25:
        top = real.nsmallest(25, "mpae_warhead_cys")[["row_id", "source_dir", "canonical_smiles", "mpae_warhead_cys", "complex_iptm"]]
        bot = real.nlargest(25, "mpae_warhead_cys")[["row_id", "source_dir", "canonical_smiles", "mpae_warhead_cys", "complex_iptm"]]
        lines.append("## Top-25 BEST cofolds (lowest mpae_warhead_cys)")
        lines.append("")
        lines.append("| rank | row_id | source | mpae | complex_iptm | smiles |")
        lines.append("|------|--------|--------|------|--------------|--------|")
        for i, (_, r) in enumerate(top.iterrows(), 1):
            lines.append(f"| {i} | `{r['row_id']}` | {r['source_dir']} | {r['mpae_warhead_cys']:.3f} | {r['complex_iptm']:.3f} | `{r['canonical_smiles']}` |")
        lines.append("")
        lines.append("## Bottom-25 WORST cofolds (highest mpae_warhead_cys)")
        lines.append("")
        lines.append("| rank | row_id | source | mpae | complex_iptm | smiles |")
        lines.append("|------|--------|--------|------|--------------|--------|")
        for i, (_, r) in enumerate(bot.iterrows(), 1):
            lines.append(f"| {i} | `{r['row_id']}` | {r['source_dir']} | {r['mpae_warhead_cys']:.3f} | {r['complex_iptm']:.3f} | `{r['canonical_smiles']}` |")
        lines.append("")

    # Correlations
    if len(real) >= 20:
        lines.append("## Correlations of `mpae_warhead_cys` (pae_tensor subset)")
        lines.append("")
        lines.append("| feature | Pearson r | Spearman ρ | n |")
        lines.append("|---------|-----------|-----------|---|")
        for col in ["d_b_nuc_angstrom", "bd_angle_deg", "phi_planar_deg",
                    "ligand_iptm", "complex_iptm", "complex_plddt", "complex_ipde"]:
            sub = real.dropna(subset=[col, "mpae_warhead_cys"])
            if len(sub) < 5:
                lines.append(f"| {col} | – | – | {len(sub)} |")
                continue
            pr = sub["mpae_warhead_cys"].corr(sub[col], method="pearson")
            sp = sub["mpae_warhead_cys"].corr(sub[col], method="spearman")
            lines.append(f"| {col} | {pr:+.3f} | {sp:+.3f} | {len(sub)} |")
        lines.append("")

    # Guardrail flag
    lines.append("## Guardrail check")
    if len(df) < 500:
        lines.append(f"- **FLAG**: total labeled ZAP70-acryl cofolds = {len(df)} < 500 — may not match earlier '3K+' figure.")
    elif len(df) > 3000:
        lines.append(f"- **GO for retrain**: {len(df)} labeled cofolds is enough for a serious training run.")
    else:
        lines.append(f"- Marginal: {len(df)} labeled cofolds — usable but not huge.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
