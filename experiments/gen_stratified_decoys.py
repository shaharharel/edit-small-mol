"""Generate stratified decoy YAMLs for arbitrary COValid targets.

Generalization of `gen_bmx_stratified_decoys.py` — replaces si_004's
`.head(100)` (which concentrates on 2-3 parents because rows are sorted)
with a sample that spans ALL parent actives.

For each requested target:
  - Read <target>_decoys sheet from si_004
  - Stratify: ceil(N_DECOYS / n_parents) per parent, then trim to N_DECOYS
  - Write YAMLs to OUT_ROOT/<target>/<target>_dec_strat_*.yaml
  - Append rows to OUT_ROOT/manifest.csv

Usage:
  python gen_stratified_decoys.py BTK EGFR JAK3
  python gen_stratified_decoys.py --n-decoys 100 BTK EGFR JAK3
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
SI2 = PROJECT_ROOT / "data/covalid/ja5c22222_si_002.xlsx"
SI4 = PROJECT_ROOT / "data/covalid/ja5c22222_si_004.xlsx"
OUT_ROOT = Path("/tmp/stratified_yamls")
SEED = 42

ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
ACRYLATE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)O")


def boltz_atom_name_for_warhead(smi: str) -> str | None:
    mol = AllChem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    for smarts in (ACRYLAMIDE_SMARTS, ACRYLATE_SMARTS):
        matches = mol_h.GetSubstructMatches(smarts)
        if matches:
            return f"C{can[matches[0][0]] + 1}"
    return None


def fetch_uniprot_sequence(uniprot_id: str) -> str:
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", timeout=15)
    r.raise_for_status()
    return "".join(line for line in r.text.splitlines() if not line.startswith(">"))


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


def stratified_sample(decoys: pd.DataFrame, n_decoys: int, seed: int) -> pd.DataFrame:
    """Sample n_decoys rows spanning all parent actives as evenly as possible."""
    rng = np.random.default_rng(seed)
    parents = list(decoys["active_compound_name"].unique())
    rng.shuffle(parents)
    per_parent = int(np.ceil(n_decoys / len(parents)))
    pieces = []
    for p in parents:
        g = decoys[decoys["active_compound_name"] == p]
        k = min(per_parent, len(g))
        pieces.append(g.sample(k, random_state=int(rng.integers(0, 2**31))))
    pool = pd.concat(pieces, ignore_index=True)
    if len(pool) > n_decoys:
        pool = pool.sample(n_decoys, random_state=seed).reset_index(drop=True)
    return pool


def generate_for_target(target: str, n_decoys: int, struct_df: pd.DataFrame,
                        si4_path: Path, out_root: Path) -> list[dict]:
    """Generate stratified YAMLs for one target. Returns manifest rows."""
    sheet_name = f"{target}_decoys"
    si4_xl = pd.ExcelFile(si4_path)
    if sheet_name not in si4_xl.sheet_names:
        print(f"[{target}] NO sheet {sheet_name} — skipping")
        return []

    # Find structure row(s). FGFR4 has two; require exact match for now
    matches = struct_df[struct_df["protein_target"].str.replace(" G12C", "", regex=False) == target]
    if len(matches) == 0:
        print(f"[{target}] not in si_002 structure sheet — skipping")
        return []
    if len(matches) > 1:
        print(f"[{target}] multiple structures ({len(matches)}); using first (cys {matches.iloc[0]['Cys_index']})")
    s = matches.iloc[0]
    cys_resi = int(s["Cys_index"])
    uniprot = s["uniprot_ID"]
    seq = fetch_uniprot_sequence(uniprot)
    if seq[cys_resi - 1] != "C":
        print(f"[{target}] WARN: residue {cys_resi} = '{seq[cys_resi-1]}', not C — numbering mismatch likely")

    decoys = pd.read_excel(si4_path, sheet_name=sheet_name)
    n_parents = decoys["active_compound_name"].nunique()
    print(f"[{target}] uniprot={uniprot} cys={cys_resi}  decoys_total={len(decoys)} parents={n_parents}")

    sampled = stratified_sample(decoys, n_decoys, SEED)
    print(f"[{target}] sampled {len(sampled)} decoys across {sampled['active_compound_name'].nunique()} parents")

    tdir = out_root / target
    tdir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_skip = 0
    for _, row in sampled.iterrows():
        smi = row["protomer_smiles"]
        atom = boltz_atom_name_for_warhead(smi)
        if atom is None:
            n_skip += 1
            continue
        prot_ind = int(row["protomer_ind"])
        name = f"{target}_dec_strat_{prot_ind:05d}"
        (tdir / f"{name}.yaml").write_text(make_yaml(seq, smi, cys_resi, atom))
        manifest_rows.append({
            "name": name, "target": target, "label": "dec", "is_active": 0,
            "smiles": smi, "uniprot": uniprot, "cys_resi": cys_resi,
            "warhead_atom_name": atom, "protomer_ind": prot_ind,
            "parent_active": row["active_compound_name"],
        })
    print(f"[{target}] wrote {len(manifest_rows)} YAMLs, skipped {n_skip} (no warhead match)")
    return manifest_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="Target names matching si_004 sheet prefixes (e.g. BTK EGFR JAK3)")
    ap.add_argument("--n-decoys", type=int, default=100)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    struct_df = pd.read_excel(SI2, sheet_name="Covalent_PDB_structures")

    all_rows = []
    for tgt in args.targets:
        all_rows.extend(generate_for_target(tgt, args.n_decoys, struct_df, SI4, args.out_root))

    df = pd.DataFrame(all_rows)
    manifest_path = args.out_root / "manifest.csv"
    if manifest_path.exists():
        prev = pd.read_csv(manifest_path)
        df = pd.concat([prev[~prev["target"].isin(args.targets)], df], ignore_index=True)
    df.to_csv(manifest_path, index=False)
    print(f"\nTotal {len(all_rows)} new rows; manifest: {manifest_path}")
    print(df.groupby("target").size().to_string())


if __name__ == "__main__":
    main()
