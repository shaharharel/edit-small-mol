#!/usr/bin/env python3
"""Generate Boltz-2 YAML inputs for the COValid (London JACS 2026) benchmark.

For each of 10 cysteine sites, produce 1 YAML per protomer (active + decoy).
Total = 874 actives + 37919 decoys × 1 YAML each.

For TONIGHT we sample TOP-N per side:
  TOP_ACTIVE = 30 / target  (or all if fewer)
  TOP_DECOY  = 70 / target  (~600 cofolds total)

Outputs:
  experiments/boltz_inputs/covalid_top100/<target>/<ligand>_<idx>.yaml
  experiments/boltz_inputs/covalid_top100/manifest.csv

YAML format (Boltz-2.2.1):
  version: 1
  sequences:
    - protein: {id: A, sequence: <uniprot fasta>}
    - ligand:  {id: B, smiles: '<smi>'}
  constraints:
    - bond:
        atom1: [A, <cys_resi>, SG]
        atom2: [B, 1, <boltz_atom_name>]
"""
from __future__ import annotations
import sys, json, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
COVALID_STRUCT = PROJECT_ROOT / "data" / "covalid" / "ja5c22222_si_002.xlsx"
COVALID_PROTS = PROJECT_ROOT / "data" / "covalid" / "ja5c22222_si_004.xlsx"
OUT_ROOT = PROJECT_ROOT / "experiments" / "boltz_inputs" / "covalid_top100"
OUT_MANIFEST = OUT_ROOT / "manifest.csv"

TOP_ACTIVE = 30
TOP_DECOY = 70

ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
ACRYLATE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)O")


def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Fetch the canonical sequence from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    lines = r.text.splitlines()
    return ''.join(line for line in lines if not line.startswith('>'))


def boltz_atom_name_for_warhead(smi: str, smarts_list=(ACRYLAMIDE_SMARTS, ACRYLATE_SMARTS)):
    """Return Boltz atom name for the warhead β-CH2 (Michael target)."""
    mol = AllChem.MolFromSmiles(smi)
    if mol is None: return None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    for smarts in smarts_list:
        matches = mol_h.GetSubstructMatches(smarts)
        if matches:
            term_ch2_idx = matches[0][0]
            return f"C{can[term_ch2_idx] + 1}"
    return None


def make_yaml(seq: str, smi: str, cys_resi: int, warhead_atom: str) -> str:
    return f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
  - ligand:
      id: B
      smiles: '{smi}'
constraints:
  - bond:
      atom1: [A, {cys_resi}, SG]
      atom2: [B, 1, {warhead_atom}]
"""


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Loading COValid structure metadata + protomers …")
    struct_df = pd.read_excel(COVALID_STRUCT, sheet_name='Covalent_PDB_structures')
    # Map target → (uniprot, cys_resi, ligand_info)
    target_meta = {}
    for _, r in struct_df.iterrows():
        t = r['protein_target']
        # Normalize 'KRAS G12C' to 'KRAS'
        t_norm = t.replace(' G12C', '')
        # For FGFR4 with two Cys, encode as FGFR4_477 / FGFR4_552
        key = f"{t_norm}_{r['Cys_index']}" if t_norm == 'FGFR4' else t_norm
        target_meta[key] = {
            'uniprot': r['uniprot_ID'],
            'cys_resi': int(r['Cys_index']),
            'pdb_id': r['PDB_id'],
            'ligand_info': r['ligand_info'],
        }
    print(f"  10 covalent targets: {list(target_meta.keys())}")

    # Fetch sequences
    print("Fetching UniProt sequences …")
    for key, meta in target_meta.items():
        seq = fetch_uniprot_sequence(meta['uniprot'])
        meta['sequence'] = seq
        print(f"  {key} ({meta['uniprot']}) len={len(seq)} cys_resi={meta['cys_resi']}")
        if seq[meta['cys_resi'] - 1] != 'C':
            print(f"    WARNING: residue {meta['cys_resi']} is '{seq[meta['cys_resi']-1]}' not C — likely numbering mismatch")
        time.sleep(0.2)  # be nice to UniProt

    # Load protomers
    print("\nLoading protomers …")
    xl = pd.ExcelFile(COVALID_PROTS)
    manifest = []
    for key, meta in target_meta.items():
        # Map our key to the sheet name in si_004
        sheet_target = key if key.startswith('FGFR4') else key
        actives_sheet = f"{sheet_target}_actives"
        decoys_sheet = f"{sheet_target}_decoys"
        if actives_sheet not in xl.sheet_names:
            print(f"  WARNING: no sheet {actives_sheet}; skipping")
            continue
        actives = pd.read_excel(COVALID_PROTS, sheet_name=actives_sheet)
        decoys = pd.read_excel(COVALID_PROTS, sheet_name=decoys_sheet)
        # SMILES column normalization
        sm_col_a = next(c for c in actives.columns if 'smiles' in c.lower())
        sm_col_d = next(c for c in decoys.columns if 'smiles' in c.lower())
        # Sample top-N from each
        actives_sub = actives.head(TOP_ACTIVE)
        decoys_sub = decoys.head(TOP_DECOY)
        print(f"  {key}: {len(actives_sub)} actives + {len(decoys_sub)} decoys (from {len(actives)}/{len(decoys)} available)")
        # Build YAMLs
        target_dir = OUT_ROOT / key
        target_dir.mkdir(parents=True, exist_ok=True)
        for label, sub, sm_col in [('act', actives_sub, sm_col_a), ('dec', decoys_sub, sm_col_d)]:
            for i, row in sub.iterrows():
                smi = row[sm_col]
                atom_name = boltz_atom_name_for_warhead(smi)
                if atom_name is None:
                    continue  # no warhead detected (rare for COValid)
                name = f"{key}_{label}_{int(row.get('protomer_ind', i)):05d}"
                yaml_text = make_yaml(meta['sequence'], smi, meta['cys_resi'], atom_name)
                (target_dir / f"{name}.yaml").write_text(yaml_text)
                manifest.append({
                    'name': name,
                    'target': key,
                    'label': label,
                    'is_active': int(label == 'act'),
                    'smiles': smi,
                    'uniprot': meta['uniprot'],
                    'cys_resi': meta['cys_resi'],
                    'warhead_atom_name': atom_name,
                    'protomer_ind': int(row.get('protomer_ind', i)),
                })
    man = pd.DataFrame(manifest)
    man.to_csv(OUT_MANIFEST, index=False)
    print(f"\nwrote {len(man)} YAMLs across {man['target'].nunique()} targets")
    print(f"  actives: {(man.is_active == 1).sum()}")
    print(f"  decoys:  {(man.is_active == 0).sum()}")
    print(f"manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
