"""
Stream C — pocket prep for DiffSBDD inference.

Output structure for each target:
  anchordiff/pockets/<TARGET>/
    receptor.pdb                    — protein only
    ref_ligand.sdf                  — reference ligand (for pocket definition)
    fixed_atoms.txt                 — list of warhead atom names to anchor
    notes.md                        — notes

NOTE: Uses anchordiff/config.py for the cysteine target so we never drift.
"""
import os
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from anchordiff.config import ZAP70_CYS346, BTK_CYS481, assert_target

OUT = PROJECT_ROOT / "anchordiff" / "pockets"
OUT.mkdir(parents=True, exist_ok=True)


def write_target(name, pdb_src, ref_smiles, target_cys_resi, warhead_atom_names_in_sdf, notes):
    d = OUT / name
    d.mkdir(exist_ok=True)
    # Receptor copy (could strip waters/ions, but DiffSBDD handles that)
    if Path(pdb_src).exists():
        os.system(f"cp '{pdb_src}' '{d}/receptor.pdb'")
    else:
        print(f"[WARN] {pdb_src} missing for {name}")

    # Reference ligand SDF — embed via RDKit
    mol = Chem.MolFromSmiles(ref_smiles)
    if mol is None:
        print(f"[ERR] could not parse SMILES for {name}")
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    sdf_path = d / "ref_ligand.sdf"
    with Chem.SDWriter(str(sdf_path)) as w:
        w.write(mol)

    # Compute Boltz-style atom names (capital element symbol + canonical_rank+1)
    mol_h = AllChem.AddHs(mol)
    can_ranks = list(AllChem.CanonicalRankAtoms(mol_h))
    pat = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
    matches = mol_h.GetSubstructMatches(pat)
    if matches:
        # the first 4 atoms of the match are the warhead: CH2, CH, C(=O), N
        warhead_indices = list(matches[0])
        boltz_names = []
        for idx in warhead_indices:
            atom = mol_h.GetAtomWithIdx(idx)
            sym = atom.GetSymbol().upper()
            rank = can_ranks[idx]
            boltz_names.append(f"{sym}{rank+1}")
    else:
        boltz_names = []
        warhead_indices = []

    (d / "fixed_atoms.txt").write_text(
        f"# Boltz atom-name convention (Symbol + canonical_rank+1) for the warhead\n"
        f"# atoms in order: C_β (terminal CH2 = Michael acceptor), C_α, C_carb, N_amide\n"
        f"# To use with DiffSBDD inpaint: --fix_atoms {' '.join(boltz_names)}\n"
        + "\n".join(boltz_names) + "\n"
    )
    (d / "notes.md").write_text(
        f"# {name}\n\n"
        f"- ref SMILES: `{ref_smiles}`\n"
        f"- target cysteine: residue {target_cys_resi}\n"
        f"- warhead Boltz atom names (in order C_β, C_α, C_carb, N_amide):\n"
        f"  - {boltz_names}\n"
        f"- warhead atom indices (RDKit, 0-based, into the H-added mol):\n"
        f"  - {warhead_indices}\n\n"
        f"## Notes\n{notes}\n"
    )
    print(f"  {name}: receptor.pdb ✓ ref_ligand.sdf ✓ warhead atoms {boltz_names}")


# ── ZAP70 Cys346 (P-loop) ────────────────────────────────────────────────
write_target(
    name=ZAP70_CYS346.name,
    pdb_src=str(PROJECT_ROOT / ZAP70_CYS346.pdb_path),
    ref_smiles="C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1",  # Mol 1
    target_cys_resi=ZAP70_CYS346.cys_residue,
    warhead_atom_names_in_sdf=None,
    notes=ZAP70_CYS346.notes,
)

# ── BTK Cys481 ────────────────────────────────────────────────────────────
write_target(
    name=BTK_CYS481.name,
    pdb_src=str(PROJECT_ROOT / BTK_CYS481.pdb_path),
    ref_smiles="C=CC(=O)N1CCC[C@@H](C1)n1nc(c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",  # ibrutinib
    target_cys_resi=BTK_CYS481.cys_residue,
    warhead_atom_names_in_sdf=None,
    notes=BTK_CYS481.notes,
)

print(f"\nWrote pocket prep to: {OUT}")
