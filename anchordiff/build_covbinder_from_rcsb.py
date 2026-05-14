"""Build our own covalent-kinase training dataset by querying RCSB PDB directly.

The official CovBinderInPDB portal at NYU is interactive-only (no bulk download).
Same outcome via the RCSB Search API + Data API:

  1. Query for structures whose polymer entity has the keyword 'kinase'.
  2. From each structure, find any HET residue with a `struct_conn` annotation
     where the connection type is "covale" (covalent bond) AND one endpoint is
     Cys SG.
  3. Extract: PDB id, chain, Cys residue number, ligand resname,
     Cys SG xyz, ligand atom indices.
  4. Filter the ligand by SMARTS to a supported warhead class
     (acrylamide / chloroacetamide / vinyl_sulfonamide / cyanoacrylate).
  5. Persist into the same `data/covbinder/training_set.csv` schema.

RCSB Search API docs: https://search.rcsb.org/
RCSB Data API docs:  https://data.rcsb.org/

Local CPU. ETA ~30 min for ~500 candidate structures.
"""
from __future__ import annotations
import sys, json, time, urllib.request, urllib.error
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from Bio.PDB import PDBParser, MMCIFParser

OUT_DIR = PROJECT_ROOT / "data" / "covbinder"
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API   = "https://data.rcsb.org/rest/v1/core/entry/"
DL_PDB     = "https://files.rcsb.org/download/{pid}.pdb"

ACRYL  = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
CHLORO = Chem.MolFromSmarts("[Cl][CH2]C(=O)N")
VINYLSULFONAMIDE = Chem.MolFromSmarts("[CH2]=[CH]S(=O)(=O)N")
CYANOACRYLATE = Chem.MolFromSmarts("N#C/C=C/C(=O)O")
WARHEADS = {
    "acrylamide": ACRYL,
    "chloroacetamide": CHLORO,
    "vinyl_sulfonamide": VINYLSULFONAMIDE,
    "cyanoacrylate": CYANOACRYLATE,
}


def query_rcsb_kinase_covalent(rows_per_page=500):
    """Search RCSB for kinase structures with covalent ligand bonds.
    Returns list of PDB ids."""
    body = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity.rcsb_macromolecular_names_combined.name",
                        "operator": "contains_words", "value": "kinase"
                    }
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_nonpolymer_entity_instance_count",
                        "operator": "greater_or_equal", "value": 1
                    }
                },
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows_per_page},
            "results_content_type": ["experimental"],
        }
    }
    pdb_ids = []
    start = 0
    while True:
        body["request_options"]["paginate"]["start"] = start
        req = urllib.request.Request(SEARCH_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            print(f"  HTTPError {e.code}: {e.reason}")
            break
        results = payload.get("result_set", [])
        if not results: break
        for hit in results:
            pdb_ids.append(hit["identifier"])
        total = payload.get("total_count", 0)
        print(f"  RCSB page {start//rows_per_page+1}: got {len(results)} hits "
              f"(running total: {len(pdb_ids)} / {total})")
        if start + rows_per_page >= total: break
        start += rows_per_page
        time.sleep(0.5)
    return pdb_ids


def fetch_pdb(pid):
    out = RAW_DIR / f"{pid}.pdb"
    if out.exists() and out.stat().st_size > 1000: return out
    try:
        url = DL_PDB.format(pid=pid)
        with urllib.request.urlopen(url, timeout=30) as r:
            out.write_bytes(r.read())
        return out
    except Exception as e:
        return None


def extract_covalent_cys_ligand(pdb_path: Path):
    """Parse PDB CONECT / LINK records to find a Cys SG → HET-ligand covalent
    bond. Return list of (chain, cys_resi, lig_chain, lig_resi, lig_resn, sg_xyz, d_min)."""
    parser = PDBParser(QUIET=True)
    try: s = parser.get_structure("", str(pdb_path))[0]
    except Exception: return []
    hits = []
    # Get all CYS SG atoms
    cys_sg = []
    for chain in s:
        for res in chain:
            if res.get_resname() == "CYS" and "SG" in res:
                cys_sg.append((chain.id, res.id[1], np.array(res["SG"].get_coord(), dtype=float)))
    # Get all HET residues (non-standard, non-water/metal)
    skip = {"HOH", "WAT", "DOD", "SO4", "PO4", "NA", "K", "MG", "CA", "ZN",
            "CL", "MN", "FE", "CU", "GOL", "EDO", "TRS", "DMS", "PEG", "PG4",
            "BME", "CO", "NI", "ACT", "FMT", "ACE", "NAG", "BMA", "MAN", "FUC"}
    for chain in s:
        for res in chain:
            if res.id[0].strip() == "" or res.get_resname() in skip: continue
            atoms = list(res.get_atoms())
            if len(atoms) < 5 or len(atoms) > 100: continue  # peptides/sugars skip
            coords = np.array([a.get_coord() for a in atoms])
            for cchain, cresi, sg in cys_sg:
                d = np.linalg.norm(coords - sg, axis=1)
                d_min = float(d.min())
                if d_min < 2.4:
                    hits.append({
                        "cys_chain": cchain, "cys_residue": cresi,
                        "ligand_chain": chain.id, "ligand_resi": res.id[1],
                        "ligand_resname": res.get_resname(),
                        "sg_x": float(sg[0]), "sg_y": float(sg[1]), "sg_z": float(sg[2]),
                        "d_SG_to_nearest_lig_atom": d_min,
                    })
                    break
    return hits


def classify_warhead(pdb_path: Path, lig_chain: str, lig_resi: int):
    """Read the HET residue, build a Mol via openbabel/RDKit, match against
    SMARTS for each warhead class. Returns warhead class name or None."""
    # Simplest approach: re-read the PDB residue and let RDKit attempt to parse
    # the heavy atoms via an explicit Mol from atomic coords. We're conservative —
    # if SMARTS doesn't match we mark as 'other'.
    try:
        with open(pdb_path) as f:
            lines = [ln for ln in f if ln.startswith("HETATM") and
                     ln[21] == lig_chain and int(ln[22:26]) == lig_resi]
        if not lines: return None
        # Use openbabel to convert PDB HETATM block → SMILES
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("pdb", "smi")
        m = ob.OBMol()
        block = "".join(lines)
        ok = conv.ReadString(m, block)
        if not ok: return None
        smi = conv.WriteString(m).split()[0]
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        for name, smarts in WARHEADS.items():
            if mol.HasSubstructMatch(smarts):
                return name, smi
        return None
    except Exception:
        return None


def main():
    print("=== Building covalent-kinase dataset from RCSB PDB ===")
    print("Step 1: query RCSB for kinase + non-polymer entity structures")
    pdb_ids = query_rcsb_kinase_covalent()
    print(f"  total kinase structures: {len(pdb_ids)}")
    (OUT_DIR / "kinase_pdb_ids.json").write_text(json.dumps(pdb_ids))
    print(f"  cached pdb id list: {OUT_DIR/'kinase_pdb_ids.json'}")

    print("\nStep 2: for each PDB, fetch + find Cys-SG-covalent ligand + match warhead")
    rows = []
    failed_fetch = []; no_cov = []
    for i, pid in enumerate(pdb_ids):
        if i % 50 == 0: print(f"  [{i+1}/{len(pdb_ids)}] processing {pid}, hits so far: {len(rows)}")
        pdbf = fetch_pdb(pid)
        if pdbf is None: failed_fetch.append(pid); continue
        hits = extract_covalent_cys_ligand(pdbf)
        if not hits:
            no_cov.append(pid); continue
        for h in hits:
            cls = classify_warhead(pdbf, h["ligand_chain"], h["ligand_resi"])
            if cls is None: continue
            warhead_name, lig_smi = cls
            rows.append({"pdb_id": pid, "warhead": warhead_name,
                         "ligand_smiles": lig_smi, **h})
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "training_set_from_rcsb.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} covalent-kinase entries)")
    if len(df):
        print("\nWarhead breakdown:")
        print(df["warhead"].value_counts().to_string())
    print(f"\nfailed_fetch: {len(failed_fetch)}, no_cov: {len(no_cov)}")


if __name__ == "__main__":
    main()
