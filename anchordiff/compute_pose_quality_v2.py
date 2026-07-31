"""Compute defensible pose-quality metrics for ZAP70 Cys346 cofolds.

2026-05-24 update: the top1000 manifest has `warhead_atom_name: null` for 100%
of entries (not 9% as earlier estimates suggested). We RE-DERIVE the warhead
atom name from SMILES using the same canonical-rank logic as
`experiments/gen_covalid_boltz_yamls.py:56` (boltz_atom_name_for_warhead).
Without this, d_SG and Bürgi-Dunitz can't be computed for ANY of the 997 mols.


Replaces the rejected metrics from `compute_contacts_occupancy.py` per
scientific-analyst review:
  - n_stabilizing_contacts: composite was double-counting d_SG, π-π broken — REJECTED
  - pocket_occupancy_pct (convex hull): noise, dropped
  - geom_ok (distance-only boolean): replaced with Bürgi-Dunitz angle deviation
  - d_SG: removed "nearest C fallback" so failed warhead detection is honest

New metrics (per scientific-analyst recommendation):
  1. d_SG (Å, continuous) — None if warhead_atom_name unset (no fallback)
  2. burgi_dunitz_dev_deg — |angle(SG, Cβ, Cα) − 107°|
  3. n_h_bonds — only H-bonds, no composite count
  4. atp_pocket_fraction — frac of ligand heavies within 4 Å of curated ATP-pocket residues
  5. ligand_plddt_mean — mean Boltz pLDDT over ligand atoms (from plddt_*.npz)
  6. hinge_hbond — boolean: any H-bond to hinge backbone Met414.N / Glu415.{N,O} / Met416.{N,O}

Curated ZAP70 ATP-pocket residues (from 4K2R within 5 Å of ANP, see derivation):
  P-loop:    344-350 (LEU,GLY,CYS,GLY,ASN,PHE,GLY)
  β3:        352 VAL, 367 ALA
  cat-Lys:   369 LYS
  αC:        399 VAL
  hinge:     414-418 (MET,GLU,MET,ALA,GLY) + 420 GLY + 421 PRO + 424 LYS
  cat-loop:  461 ASN, 465 ARG, 466 ASN, 468 LEU
  DFG:       479 ASP, 482 LEU

Hinge subset for hinge_hbond: 414 MET, 415 GLU, 416 MET (backbone N + O).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Soft warhead detection — multiple electrophile classes (not just acrylamide).
# Per user 2026-05-24: softened from exact acrylamide to "any covalent-warhead-like
# substructure". For warhead atom-name derivation we still use SMARTS-matched Cβ.
WARHEAD_SMARTS = [
    ("acrylamide",    "[CH2]=[CH]C(=O)N"),
    ("acrylate",      "[CH2]=[CH]C(=O)O"),
    ("vinyl_sulfone", "[CH2]=[CH]S(=O)(=O)"),
    ("haloacetamide", "[Cl,Br,I][CH2]C(=O)N"),
    ("propiolamide",  "C#CC(=O)N"),
]
_WARHEAD_SMARTS_MOLS = [(n, Chem.MolFromSmarts(s)) for n, s in WARHEAD_SMARTS]


def derive_warhead_atom_name(smi: str) -> tuple[str | None, str | None]:
    """Return (warhead_class, boltz_atom_name) for the Cβ (Michael-attacked carbon).
    Matches the canonical-rank logic in experiments/gen_covalid_boltz_yamls.py."""
    mol = AllChem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    for wh_class, sm in _WARHEAD_SMARTS_MOLS:
        if sm is None:
            continue
        matches = mol_h.GetSubstructMatches(sm)
        if matches:
            term_ch2_idx = matches[0][0]  # first atom of SMARTS = Cβ
            return wh_class, f"C{can[term_ch2_idx] + 1}"
    return None, None

PRED_DIR = PROJECT_ROOT / "data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions"
MANIFEST = PROJECT_ROOT / "data/boltz_poses/top1000_manifest__zap70_cys346.json"
OUT_CSV = PROJECT_ROOT / "data/boltz_poses/pose_quality_v2.csv"

CYS_RESI = 346
HINGE_RESIS = {414, 415, 416}  # for hinge H-bond detection

# ZAP70 ATP-pocket residues derived from 4K2R ANP-binding site (≤5 Å)
ATP_POCKET_RESIS = {344, 345, 346, 347, 348, 349, 350, 352, 367, 369, 399,
                    414, 415, 416, 417, 418, 420, 421, 424,
                    461, 465, 466, 468, 479, 482}

# Bürgi-Dunitz expected angle for Michael addition (sp3-like nucleophile attack)
BURGI_DUNITZ_DEG = 107.0

# H-bond geometry thresholds (distance only — directionality not enforced at this stage)
HB_MIN_A, HB_MAX_A = 2.5, 3.5


def parse_cofold(cif_path: Path):
    """Return (prot_atoms, lig_atoms) lists of dicts with chain/resname/resi/name/element/pos."""
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    prot, lig = [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    rec = {
                        "chain": chain.name, "resname": res.name, "resi": res.seqid.num,
                        "name": atom.name, "element": atom.element.name,
                        "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                    }
                    if chain.name == "A":
                        prot.append(rec)
                    elif chain.name == "B":
                        lig.append(rec)
        break
    return prot, lig


def d_SG_honest(prot, lig, warhead_atom_name: str | None) -> float | None:
    """Cys-Sγ to LIGAND ATOM NAMED in manifest. Returns None if warhead_atom_name is None.
    NO nearest-C fallback."""
    if warhead_atom_name is None or (isinstance(warhead_atom_name, float) and np.isnan(warhead_atom_name)):
        return None
    sg = next((p["pos"] for p in prot
               if p["resi"] == CYS_RESI and p["resname"] == "CYS" and p["name"] == "SG"), None)
    if sg is None:
        return None
    cb = next((l["pos"] for l in lig if l["name"] == warhead_atom_name), None)
    if cb is None:
        return None
    return float(np.linalg.norm(sg - cb))


def find_warhead_alpha_carbon(lig, warhead_atom_name: str | None):
    """Find the ligand carbon adjacent to the warhead Cβ (= Cα in S-Cβ-Cα).
    Heuristic: closest other carbon to Cβ within 1.0-1.9 Å (typical C-C bond length).

    Widened from 1.2-1.8 to 1.0-1.9 on 2026-06-07 because the Boltz cofold for
    row_id=553242 has C35-C34 at 1.162 Å, just outside the original cutoff,
    causing burgi_dunitz_dev_deg to NaN-fail. Real C-C bonds rarely exceed 1.6 Å
    or fall below 1.20; 1.0-1.9 catches Boltz-pose numerical slop without
    accepting non-bonded contacts."""
    if warhead_atom_name is None:
        return None
    cb = next((l for l in lig if l["name"] == warhead_atom_name), None)
    if cb is None:
        return None
    candidates = []
    for l in lig:
        if l["element"] != "C" or l["name"] == warhead_atom_name:
            continue
        d = float(np.linalg.norm(l["pos"] - cb["pos"]))
        if 1.0 <= d <= 1.9:
            candidates.append((d, l))
    if not candidates:
        return None
    # Pick the closest (should be just one; sp2 Cα of the acrylamide)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def burgi_dunitz_dev(prot, lig, warhead_atom_name: str | None) -> float | None:
    """|angle(SG, Cβ, Cα) − 107°|. None if any atom missing."""
    if warhead_atom_name is None:
        return None
    sg = next((p["pos"] for p in prot
               if p["resi"] == CYS_RESI and p["resname"] == "CYS" and p["name"] == "SG"), None)
    cb = next((l["pos"] for l in lig if l["name"] == warhead_atom_name), None)
    if sg is None or cb is None:
        return None
    ca_atom = find_warhead_alpha_carbon(lig, warhead_atom_name)
    if ca_atom is None:
        return None
    ca = ca_atom["pos"]
    # angle at vertex Cβ between (Sγ→Cβ) and (Cα→Cβ)
    v1 = sg - cb
    v2 = ca - cb
    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    cos_a = max(-1.0, min(1.0, cos_a))
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    return abs(angle_deg - BURGI_DUNITZ_DEG)


def n_h_bonds(prot, lig) -> tuple[int, list[str]]:
    """Count ligand N/O ↔ protein N/O at HB_MIN..HB_MAX Å. Excludes Cys346.SG."""
    hbs = []
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
                hbs.append(f"{p['resname']}{p['resi']}.{p['name']}-lig.{l['name']} d={d:.2f}")
    return len(hbs), hbs


def hinge_hbond_present(prot, lig) -> bool:
    """True if any H-bond to hinge backbone N/O of residues 414, 415, 416.
    Backbone names: N (amide donor), O (carbonyl acceptor)."""
    for l in lig:
        if l["element"] not in ("N", "O"):
            continue
        for p in prot:
            if p["resi"] not in HINGE_RESIS:
                continue
            if p["name"] not in ("N", "O"):
                continue
            d = float(np.linalg.norm(l["pos"] - p["pos"]))
            if HB_MIN_A <= d <= HB_MAX_A:
                return True
    return False


def atp_pocket_fraction(prot, lig, cutoff_A: float = 4.0) -> float:
    """Fraction of ligand heavy atoms within `cutoff` of any ATP-pocket residue heavy atom."""
    pocket_atoms = [p["pos"] for p in prot
                    if p["resi"] in ATP_POCKET_RESIS and p["element"] != "H"]
    lig_heavies = [l["pos"] for l in lig if l["element"] != "H"]
    if not lig_heavies or not pocket_atoms:
        return 0.0
    pocket_arr = np.array(pocket_atoms)
    n_in = 0
    for lh in lig_heavies:
        d_min = float(np.min(np.linalg.norm(pocket_arr - lh, axis=1)))
        if d_min <= cutoff_A:
            n_in += 1
    return n_in / len(lig_heavies)


def ligand_plddt_mean(plddt_npz_path: Path, n_protein: int, n_ligand: int) -> float | None:
    """Mean of Boltz per-atom pLDDT over the ligand atoms (last n_ligand entries)."""
    if not plddt_npz_path.exists():
        return None
    try:
        d = np.load(plddt_npz_path)
        arr = d[list(d.keys())[0]]
        if arr.ndim != 1:
            return None
        # Boltz writes protein atoms first then ligand. We assume n_protein + n_ligand == len(arr).
        if len(arr) < n_ligand:
            return None
        lig_plddt = arr[-n_ligand:]
        return float(np.mean(lig_plddt))
    except Exception:
        return None


def main():
    print(f"Loading manifest: {MANIFEST}")
    m = json.loads(MANIFEST.read_text())
    rows_in = list(m.values())
    print(f"  {len(rows_in)} entries")

    out_rows = []
    n_ok = 0
    n_warhead_null = 0
    n_warhead_derived = 0
    for i, entry in enumerate(rows_in):
        name = entry["yaml_name"]
        cif = PRED_DIR / name / f"{name}_model_0.cif"
        plddt_npz = PRED_DIR / name / f"plddt_{name}_model_0.npz"
        warhead_atom = entry.get("warhead_atom_name")
        # 2026-05-24 fix: manifest has 100% null. Re-derive from SMILES.
        warhead_class = None
        if warhead_atom is None and entry.get("smiles"):
            wh_class, wh_atom = derive_warhead_atom_name(entry["smiles"])
            if wh_atom is not None:
                warhead_atom = wh_atom
                warhead_class = wh_class
                n_warhead_derived += 1
        if warhead_atom is None:
            n_warhead_null += 1
        if not cif.exists():
            out_rows.append({"yaml_name": name, "row_id": entry["row_id"],
                             "warhead_atom_name": warhead_atom,
                             "d_SG": None, "burgi_dunitz_dev_deg": None,
                             "n_h_bonds": None, "atp_pocket_fraction": None,
                             "ligand_plddt_mean": None, "hinge_hbond": None,
                             "error": "no_cif"})
            continue
        try:
            prot, lig = parse_cofold(cif)
            d_sg = d_SG_honest(prot, lig, warhead_atom)
            bd = burgi_dunitz_dev(prot, lig, warhead_atom)
            nhb, _ = n_h_bonds(prot, lig)
            apf = atp_pocket_fraction(prot, lig, cutoff_A=4.0)
            n_lig = len([l for l in lig if l["element"] != "H"])
            n_prot = len([p for p in prot if p["element"] != "H"])
            # Boltz pLDDT npz has all atoms (incl. H typically); use total len for safe slicing
            # but we only need the last `n_ligand_all` (including any H in ligand chain).
            n_lig_total = sum(1 for l in lig)
            lp = ligand_plddt_mean(plddt_npz, n_prot, n_lig_total)
            hh = hinge_hbond_present(prot, lig)
            out_rows.append({
                "yaml_name": name, "row_id": entry["row_id"],
                "warhead_atom_name": warhead_atom,
                "d_SG": d_sg,
                "burgi_dunitz_dev_deg": bd,
                "n_h_bonds": nhb,
                "atp_pocket_fraction": apf,
                "ligand_plddt_mean": lp,
                "hinge_hbond": hh,
                "error": None,
            })
            n_ok += 1
        except Exception as e:
            out_rows.append({"yaml_name": name, "row_id": entry["row_id"],
                             "warhead_atom_name": warhead_atom,
                             "d_SG": None, "burgi_dunitz_dev_deg": None,
                             "n_h_bonds": None, "atp_pocket_fraction": None,
                             "ligand_plddt_mean": None, "hinge_hbond": None,
                             "error": f"{type(e).__name__}: {str(e)[:200]}"})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1:4d}/{len(rows_in)}]  ok={n_ok}  warhead_null={n_warhead_null}")

    df = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows, {n_ok} successful, {n_warhead_null} null warhead)")
    print()
    print("Distribution summary:")
    if df["d_SG"].notna().any():
        print(f"  d_SG median: {df['d_SG'].median():.2f} Å  (n={df['d_SG'].notna().sum()})")
    if df["burgi_dunitz_dev_deg"].notna().any():
        print(f"  burgi_dunitz_dev median: {df['burgi_dunitz_dev_deg'].median():.1f}°")
    if df["n_h_bonds"].notna().any():
        print(f"  n_h_bonds median: {df['n_h_bonds'].median():.0f}")
    if df["atp_pocket_fraction"].notna().any():
        print(f"  atp_pocket_fraction median: {df['atp_pocket_fraction'].median():.2%}")
    if df["ligand_plddt_mean"].notna().any():
        print(f"  ligand_plddt_mean median: {df['ligand_plddt_mean'].median():.2f}")
    if df["hinge_hbond"].notna().any():
        print(f"  hinge_hbond present: {df['hinge_hbond'].sum()}/{df['hinge_hbond'].notna().sum()}")


if __name__ == "__main__":
    main()
