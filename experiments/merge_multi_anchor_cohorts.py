"""Merge the per-chassis SDF cohorts produced by run_lingo3dmol_multi_anchor_ensemble.py
into a single ENSEMBLE SDF, deduped by canonical SMILES of the largest fragment.

Inputs (defaults relative to repo root):
  data/lingo3dmol_L2_scaffold_C5/samples_T10.sdf       (chassis A)
  data/lingo3dmol_multi_anchor/B/samples_T10.sdf
  data/lingo3dmol_multi_anchor/C/samples_T10.sdf
  data/lingo3dmol_multi_anchor/D/samples_T10.sdf

Output:
  data/lingo3dmol_multi_anchor/ENSEMBLE/samples.sdf
  data/lingo3dmol_multi_anchor/ENSEMBLE/dedup_summary.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUTS = [
    ("A_isoindoline_C5",            PROJECT_ROOT / "data/lingo3dmol_L2_scaffold_C5/samples_T10.sdf"),
    ("B_isoindoline_amide_piperidine", PROJECT_ROOT / "data/lingo3dmol_multi_anchor/B/samples_T10.sdf"),
    ("C_THIQ_aryl",                  PROJECT_ROOT / "data/lingo3dmol_multi_anchor/C/samples_T10.sdf"),
    ("D_azaindoline_C5",             PROJECT_ROOT / "data/lingo3dmol_multi_anchor/D/samples_T10.sdf"),
]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/lingo3dmol_multi_anchor/ENSEMBLE/samples.sdf"


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    if mol is None:
        return None
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    if not frags:
        return None
    return max(frags, key=lambda f: f.GetNumHeavyAtoms())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="Optional list of CHASSIS_ID=PATH pairs.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    inputs = []
    if args.inputs:
        for kv in args.inputs:
            k, v = kv.split("=", 1)
            inputs.append((k, Path(v)))
    else:
        inputs = DEFAULT_INPUTS

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_smiles: set[str] = set()
    writer = Chem.SDWriter(str(output))
    per_chassis_counts: dict[str, int] = {}
    per_chassis_added: dict[str, int] = {}
    total_input = 0
    total_unique = 0

    for chassis_id, path in inputs:
        if not Path(path).exists():
            print(f"  ! missing input: {chassis_id} -> {path} (skipping)")
            per_chassis_counts[chassis_id] = 0
            per_chassis_added[chassis_id] = 0
            continue
        if Path(path).stat().st_size == 0:
            print(f"  ! empty input: {chassis_id} -> {path} (skipping)")
            per_chassis_counts[chassis_id] = 0
            per_chassis_added[chassis_id] = 0
            continue
        suppl = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        n_in_this = 0
        n_added_this = 0
        for mol in suppl:
            if mol is None:
                continue
            n_in_this += 1
            total_input += 1
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            largest = _largest_fragment(mol)
            if largest is None:
                continue
            try:
                canon = Chem.MolToSmiles(largest)
            except Exception:
                continue
            if canon in seen_smiles:
                continue
            seen_smiles.add(canon)
            try:
                mol.SetProp("chassis_id", chassis_id)
                mol.SetProp("canonical_smiles", canon)
                if not mol.HasProp("_Name") or not mol.GetProp("_Name"):
                    mol.SetProp("_Name", f"{chassis_id}_{n_added_this}")
                writer.write(mol)
                n_added_this += 1
                total_unique += 1
            except Exception as e:
                print(f"  ! write failure ({chassis_id}): {e}")
        per_chassis_counts[chassis_id] = n_in_this
        per_chassis_added[chassis_id] = n_added_this
        print(f"  + {chassis_id}: {n_in_this} input -> {n_added_this} unique after dedup")
    writer.close()

    summary = {
        "output": str(output),
        "per_chassis_input_counts": per_chassis_counts,
        "per_chassis_unique_added": per_chassis_added,
        "total_input": total_input,
        "total_unique_merged": total_unique,
    }
    sum_path = output.parent / "dedup_summary.json"
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote merged SDF: {output} ({total_unique} unique mols from {total_input} input)")
    print(f"Wrote summary:    {sum_path}")


if __name__ == "__main__":
    main()
