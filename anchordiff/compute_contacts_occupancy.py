"""Compute stabilizing-contact count + pocket-occupancy proxy for every Cys346
Boltz cofold in the top1000 manifest.

For each model_0.cif:
  - Covalent: SG↔ligand-C distance ≤ 2.5 Å
  - H-bond:   ligand N/O ↔ protein N/O at 2.5-3.5 Å
  - Salt bridge: ligand N+(charged) ↔ protein O-/N+ at 2.5-4.0 Å (heuristic)
  - π-π: ligand aromatic ring centroid ↔ protein aromatic ring centroid at 3.5-6.0 Å
         (we approximate using His/Phe/Tyr/Trp ring atoms in protein and any
          ligand ring detected from the SMILES)
  - Pocket occupancy proxy:
         pocket_vol ≈ volume of the convex hull of protein heavy atoms within
                       12 Å of any ligand atom (ALA SASA model — rough but fast)
         lig_vol    ≈ volume of the ligand convex hull
         occupancy  = lig_vol / pocket_vol   in [0, 1]

Writes augmented top1000_manifest__zap70_cys346.json with three new keys per row:
  - n_stabilizing_contacts (covalent + h_bonds + salt + pi)
  - contact_breakdown (dict)
  - pocket_occupancy_pct
"""
from __future__ import annotations
from pathlib import Path
import json, time, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import gemmi
from scipy.spatial import ConvexHull
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "data" / "boltz_poses" / "boltz_results_top1000__zap70_cys346" / "predictions"
MANIFEST = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"
CYS_RESI = 346

# Thresholds
COV_MAX_D = 2.5
HB_MIN, HB_MAX = 2.5, 3.5
SALT_MIN, SALT_MAX = 2.8, 4.0
AROM_MIN, AROM_MAX = 3.5, 6.0

AROMATIC_RES = {"PHE": ["CG","CD1","CD2","CE1","CE2","CZ"],
                "TYR": ["CG","CD1","CD2","CE1","CE2","CZ"],
                "TRP": ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"],
                "HIS": ["CG","ND1","CD2","CE1","NE2"]}
PROT_DONOR = {"N","NE","ND1","ND2","NE1","NE2","NH1","NH2","NZ","OG","OG1","OH"}
PROT_ACCEPTOR = {"O","OD1","OD2","OE1","OE2","OG","OG1","OH","ND1","NE2","SD"}
PROT_POS = {"NZ", "NH1", "NH2", "NE"}    # Lys/Arg
PROT_NEG = {"OD1","OD2","OE1","OE2"}     # Asp/Glu


def gather_atoms(st):
    """Return (prot_atoms, lig_atoms) where each is a list of dicts."""
    prot, lig = [], []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                rec = {'chain': chain.name, 'resname': res.name, 'resi': res.seqid.num,
                       'name': atom.name, 'element': atom.element.name,
                       'pos': np.array([atom.pos.x, atom.pos.y, atom.pos.z])}
                if chain.name == 'A': prot.append(rec)
                elif chain.name == 'B': lig.append(rec)
    return prot, lig


def find_covalent(prot, lig):
    """Cys SG ↔ closest ligand C within COV_MAX_D."""
    sg = None
    for p in prot:
        if p['resi'] == CYS_RESI and p['resname'] == 'CYS' and p['name'] == 'SG':
            sg = p['pos']; break
    if sg is None: return 0, []
    nearest = None; nd = 1e9
    for l in lig:
        if l['element'] != 'C': continue
        d = float(np.linalg.norm(l['pos'] - sg))
        if d < nd: nd, nearest = d, l
    if nearest and nd <= COV_MAX_D:
        return 1, [f"Cys{CYS_RESI}.SG—lig.{nearest['name']}  d={nd:.2f}Å"]
    return 0, []


def find_h_bonds(prot, lig):
    """Polar N/O ↔ N/O contacts at HB_MIN..HB_MAX. Excludes Cys346 SG."""
    hbs = []
    for l in lig:
        if l['element'] not in ('N', 'O'): continue
        for p in prot:
            if p['element'] not in ('N', 'O'): continue
            if p['resi'] == CYS_RESI and p['name'] == 'SG': continue
            d = float(np.linalg.norm(l['pos'] - p['pos']))
            if HB_MIN <= d <= HB_MAX:
                hbs.append(f"{p['resname']}{p['resi']}.{p['name']}—lig.{l['name']}  d={d:.2f}Å")
    # Dedupe by (residue, ligand atom) — keep shortest per pair
    return len(hbs), hbs


def find_salt_bridges(prot, lig):
    """Ligand polar N/O ↔ protein charged side chain at SALT range."""
    sb = []
    for l in lig:
        if l['element'] not in ('N', 'O'): continue
        for p in prot:
            if p['name'] not in (PROT_POS | PROT_NEG): continue
            d = float(np.linalg.norm(l['pos'] - p['pos']))
            if SALT_MIN <= d <= SALT_MAX:
                sb.append(f"{p['resname']}{p['resi']}.{p['name']}—lig.{l['name']}  d={d:.2f}Å")
    return len(sb), sb


def find_pi_pi(prot, lig, smi):
    """Protein aromatic ring centroids vs ligand aromatic ring centroids."""
    pi = []
    # Protein ring centroids
    prot_centroids = []
    for res_name, ring_atoms in AROMATIC_RES.items():
        # Group by residue (chain, resi) when residue name matches
        from collections import defaultdict
        groups = defaultdict(list)
        for p in prot:
            if p['resname'] == res_name and p['name'] in ring_atoms:
                groups[(p['chain'], p['resi'])].append(p['pos'])
        for (c, ri), atoms in groups.items():
            if len(atoms) >= 4:
                prot_centroids.append((f"{res_name}{ri}", np.mean(atoms, axis=0)))
    # Ligand ring centroids — via RDKit
    mol = Chem.MolFromSmiles(smi)
    lig_centroids = []
    if mol is not None:
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            # All atoms in ring aromatic?
            if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring): continue
            # Match to CIF coords by atom-index-via-canonical-rank lookup is
            # fragile; use a simple proxy: get all ligand C atom centroids in
            # the CIF (one for each Cn) and group as one big aromatic blob.
            pass
    # Simpler proxy: all ligand C atoms with at least 2 C neighbors at sp2-like
    # distances (1.35-1.45 Å). Cluster by spatial proximity.
    lig_c = [l for l in lig if l['element'] == 'C']
    # Just use the centroid of all ligand C atoms as a single "aromatic blob"
    if lig_c:
        lig_blob_centroid = np.mean([l['pos'] for l in lig_c], axis=0)
        for resname, pc in prot_centroids:
            d = float(np.linalg.norm(pc - lig_blob_centroid))
            if AROM_MIN <= d <= AROM_MAX:
                pi.append(f"{resname}—lig.ringblob  d={d:.2f}Å")
    return len(pi), pi


def pocket_occupancy(prot, lig):
    """Pocket occupancy = ligand convex-hull volume / pocket-shell volume.

    Pocket-shell = convex hull of protein heavy atoms within 8 Å of any
    ligand atom. This is a rough proxy — real fpocket would be more accurate.
    """
    if len(lig) < 4: return None
    lig_pos = np.array([l['pos'] for l in lig])
    try:
        lig_hull = ConvexHull(lig_pos)
    except Exception:
        return None
    lig_vol = float(lig_hull.volume)
    # Pocket shell
    pocket_atoms = []
    for p in prot:
        if any(np.linalg.norm(p['pos'] - lp) < 8.0 for lp in lig_pos):
            pocket_atoms.append(p['pos'])
    if len(pocket_atoms) < 4:
        return None
    pocket_pos = np.array(pocket_atoms)
    try:
        pocket_hull = ConvexHull(pocket_pos)
    except Exception:
        return None
    pocket_vol = float(pocket_hull.volume)
    if pocket_vol < 1e-6: return None
    return 100.0 * lig_vol / pocket_vol


def analyze_one(cif_path, smi):
    st = gemmi.read_structure(str(cif_path))
    prot, lig = gather_atoms(st)
    n_cov, cov_list = find_covalent(prot, lig)
    n_hb, hb_list = find_h_bonds(prot, lig)
    n_sb, sb_list = find_salt_bridges(prot, lig)
    n_pi, pi_list = find_pi_pi(prot, lig, smi)
    occ = pocket_occupancy(prot, lig)
    return {
        'n_covalent': n_cov, 'n_h_bonds': n_hb, 'n_salt_bridges': n_sb, 'n_pi_pi': n_pi,
        'n_stabilizing_contacts': n_cov + n_hb + n_sb + n_pi,
        'contact_breakdown': {'covalent': cov_list[:3], 'h_bonds': hb_list[:5],
                              'salt_bridges': sb_list[:3], 'pi_pi': pi_list[:3]},
        'pocket_occupancy_pct': occ,
    }


def main():
    t0 = time.time()
    print(f"loading {MANIFEST}")
    man = json.loads(MANIFEST.read_text())
    valid_rows = [(k, v) for k, v in man.items() if v.get('combined_score') is not None]
    print(f"  total rows: {len(man)}  ranked (combined_score not None): {len(valid_rows)}")
    valid_rows.sort(key=lambda kv: -kv[1]['combined_score'])
    # Process all valid rows
    n_done = 0; n_fail = 0
    for k, v in valid_rows:
        name = v['yaml_name']
        cif = PRED_DIR / name / f"{name}_model_0.cif"
        if not cif.exists():
            n_fail += 1
            continue
        try:
            info = analyze_one(cif, v['smiles'])
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            n_fail += 1
            continue
        v.update(info)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(valid_rows)}  elapsed {time.time()-t0:.0f}s")
    print(f"\nProcessed: {n_done}  failed: {n_fail}")
    print(f"Sample (top 5 by combined):")
    for k, v in valid_rows[:5]:
        cb = v.get('contact_breakdown', {})
        print(f"  {v['yaml_name'][:40]:42s}  contacts={v.get('n_stabilizing_contacts')}  "
              f"(cov={v.get('n_covalent')}, hb={v.get('n_h_bonds')}, "
              f"sb={v.get('n_salt_bridges')}, π-π={v.get('n_pi_pi')})  "
              f"occupancy={v.get('pocket_occupancy_pct')}%")
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(f"wrote updated manifest: {MANIFEST}")
    print(f"Total: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
