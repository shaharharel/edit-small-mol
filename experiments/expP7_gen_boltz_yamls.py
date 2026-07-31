"""Exp-P7: generate Boltz-2 YAMLs for 4 methods × 300 unique valid mols on ZAP70.

Reads existing sample CSVs + top-up CSVs, filters to 300 unique valid mols per
method, generates one YAML per mol constraining acrylamide β-CH2 → Cys346 SG.

Output: experiments/boltz_inputs_p7/<method>/<uid>.yaml + manifest.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_ROOT = ROOT / "experiments/boltz_inputs_p7"

ZAP70_SEQ = (
    "MPDPAAHLPFFYGSISRAEAEEHLKLAGMADGLFLLRQCLRSLGGYVLSLVHDVRFHHFPIERQLNGTYAI"
    "AGGKAHCGPAELCEFYSRDPDGLPCNLRKPCNRPSGLEPQPGVFDCLRDAMVRDYVRQTWKLEGEALEQAI"
    "ISQAPQVEKLIATTAHERMPWYHSSLTREEAERKLYSGAQTDGKFLLRPRKEQGTYALSLIYGKTVYHYLI"
    "SQDKAGKYCIPEGTKFDTLWQLVEYLKLKADGLIYCLKEACPNSSASNASGAAAPTLPAHPSTLTHPQRRI"
    "DTLNSDGYTPEPARITSPDKPRPMPMDTSVYESPYSDPEELKDKKLFLKRDNLLIADIELGCGNFGSVRQG"
    "VYRMRKKQIDVAIKVLKQGTEKADTEEMMREAQIMHQLDNPYIVRLIGVCQAEALMLVMEMAGGGPLHKFL"
    "VGKREEIPVSNVAELLHQVSMGMKYLEEKNFVHRDLAARNVLLVNRHYAKISDFGLSKALGADDSYYTARS"
    "AGKWPLKWYAPECINFRKFSSRSDVWSYGVTMWEALSYGQKPYKKMKGPEVMAFIEQGKRMECPPECPPEL"
    "YALMSDCWIYKWEDRPDFLTVEQRMRACYYSLASKVEGPPGSTQKAEAACA"
)
TARGET_CYS = 346
assert ZAP70_SEQ[TARGET_CYS - 1] == "C"
ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def read_smis(path):
    p = Path(path)
    if not p.exists(): return []
    if p.suffix == '.smi':
        return [l.strip() for l in open(p) if l.strip()]
    return pd.read_csv(p)['SMILES'].astype(str).tolist()


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def acryl_atom_name(smi: str):
    """Returns canonical atom name for β-CH2 in acryl warhead, or None."""
    mol = AllChem.MolFromSmiles(smi)
    if mol is None: return None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    matches = mol_h.GetSubstructMatches(ACRYL)
    if not matches: return None
    return f"C{can[matches[0][0]] + 1}"


def yaml_for(smi: str, atom_name: str) -> str:
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {ZAP70_SEQ}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{smi}'\n"
        "constraints:\n"
        "  - bond:\n"
        f"      atom1: [A, {TARGET_CYS}, SG]\n"
        f"      atom2: [B, 1, {atom_name}]\n"
    )


def collect_unique_valid(paths, target_n=300):
    """Merge, canonicalize, dedupe → return first target_n unique valid SMILES."""
    seen = set()
    out = []
    for p in paths:
        for s in read_smis(p):
            c = canon(s)
            if c is None: continue
            if c in seen: continue
            seen.add(c)
            out.append(c)
            if len(out) >= target_n:
                return out
    return out  # may be < target_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_n", type=int, default=300)
    args = ap.parse_args()

    # 4 methods with source paths (in priority order — top-up preferred as it has more mols)
    methods = {
        "mol2mol_baseline": [
            ROOT / "data/exp_P5/covvina/mol2mol_baseline_input.smi",
        ],
        "P5-1v2_base": [
            ROOT / "data/exp_P6/samples_topup/P5-1v2_base_topup.csv",
            ROOT / "data/exp_P6/samples/P5-1v2_baseline_mol1_temp1.csv",
        ],
        "P6-POSENOISE_s03": [
            ROOT / "data/exp_P6/samples_topup/P6-POSENOISE_topup.csv",
            ROOT / "data/exp_P6/samples/P6-POSENOISE_mol1.csv",
        ],
        "RL_GEOMEAN": [
            ROOT / "data/exp_P6/samples_topup/RL_GEOMEAN_topup.csv",
            ROOT / "data/exp_P6/samples/RL_GEOMEAN_from_P51v2_mol1.csv",
        ],
    }

    master_manifest = []
    for method, paths in methods.items():
        smis = collect_unique_valid(paths, target_n=args.target_n * 2)  # over-collect
        out_dir = OUT_ROOT / method
        out_dir.mkdir(parents=True, exist_ok=True)
        wrote = skipped = 0
        for smi in smis:
            atom = acryl_atom_name(smi)
            if atom is None:
                skipped += 1
                continue
            uid = f"{method}_{wrote:04d}"
            (out_dir / f"{uid}.yaml").write_text(yaml_for(smi, atom))
            master_manifest.append({"uid": uid, "method": method,
                                    "smiles": smi, "warhead_atom_name": atom})
            wrote += 1
            if wrote >= args.target_n: break
        print(f"[{method}] wrote={wrote} skipped_no_acryl={skipped} pool_size={len(smis)}",
              flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(master_manifest).to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"[done] manifest → {OUT_ROOT/'manifest.csv'} ({len(master_manifest)} yamls)")


if __name__ == "__main__":
    main()
