"""Extract the 5-atom acrylamide warhead from each cohort's ref_ligand.sdf
into a tiny warhead-only SDF, used as `--fix_atoms warhead.sdf` for DiffSBDD
inpaint.py.

Atom order (matches notes.md & boltz_atom_name(...) prep step):
  0: C_β (terminal CH2 of CH2=CH-)
  1: C_α (the =CH-)
  2: C_carbonyl
  3: O carbonyl
  4: N amide
"""
from __future__ import annotations
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POCKETS = PROJECT_ROOT / "anchordiff" / "pockets"


def extract_warhead(ref_sdf: Path, out_sdf: Path):
    suppl = Chem.SDMolSupplier(str(ref_sdf), sanitize=False, removeHs=False)
    mol = suppl[0]
    if mol is None:
        raise RuntimeError(f"failed to parse {ref_sdf}")

    # Per pocket prep convention, warhead atoms are indices [0,1,2,3,4]:
    # CH2 = CH - C(=O) - N
    keep = [0, 1, 2, 3, 4]
    elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in keep]
    expected = ["C", "C", "C", "O", "N"]
    assert elems == expected, f"warhead head atoms {elems} != {expected} in {ref_sdf}"

    rwm = Chem.RWMol()
    old_to_new = {}
    for new_i, old_i in enumerate(keep):
        a = mol.GetAtomWithIdx(old_i)
        new_a = Chem.Atom(a.GetSymbol())
        new_a.SetNoImplicit(True)
        rwm.AddAtom(new_a)
        old_to_new[old_i] = new_i

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in old_to_new and j in old_to_new:
            rwm.AddBond(old_to_new[i], old_to_new[j], b.GetBondType())

    new_mol = rwm.GetMol()
    conf = Chem.Conformer(len(keep))
    src_conf = mol.GetConformer()
    for new_i, old_i in enumerate(keep):
        p = src_conf.GetAtomPosition(old_i)
        conf.SetAtomPosition(new_i, p)
    new_mol.AddConformer(conf)

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    w.write(new_mol)
    w.close()
    print(f"  wrote {out_sdf} ({new_mol.GetNumAtoms()} atoms, {new_mol.GetNumBonds()} bonds)")


def main():
    for tgt in ["zap70_cys346", "btk_cys481"]:
        ref = POCKETS / tgt / "ref_ligand.sdf"
        out = POCKETS / tgt / "warhead.sdf"
        print(f"=== {tgt} ===")
        print(f"  source: {ref}")
        extract_warhead(ref, out)


if __name__ == "__main__":
    main()
