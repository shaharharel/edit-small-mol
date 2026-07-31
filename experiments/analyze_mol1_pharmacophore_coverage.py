"""Count Mol1-Murcko / THIQ-acryl / both across all tier4 cohorts (old + new).

Generates:
- Per-cohort + aggregate stats
- 3 PNGs highlighting the substructures on Mol1
"""
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
MURCKO_SMARTS = Chem.MolFromSmarts("O=C(Nc1cncn1)c1cccc2c1CNC2")
THIQ_ACRYL_SMARTS = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
ACRYL_SMARTS = Chem.MolFromSmarts("C=CC(=O)N")

# Cohort groupings
OLD_COHORTS = [
    ("exp2", "Tier-4 EXP2 (Mol2Mol Covalent FT)"),
    ("exp2_v2_rl_v2", "Tier-4 EXP2 v2 RL v2"),
    ("exp6", "Tier-4 EXP6 (Mol2Mol Warhead-tokens)"),
    ("exp6_v3", "Tier-4 EXP6 v3 (warhead-tokens, no RL)"),
    ("exp6_v4", "Tier-4 EXP6 v4 (RL anchor cluster)"),
    ("exp6_v5", "Tier-4 EXP6 v5 (Mol1 anchor RL)"),
]
NEW_COHORTS = [
    "mol1RL_v5_seed_mol1_only",
    "thiq_rl_mol1only",
    "thiq_rl_zap70",
    "thiq_rl_kinase",
    "thiq_rl_exp2_zap70",
    "thiq_rl_exp2_kinase",
    "murcko_rl_zap70",
    "murcko_rl_kinase",
    "murcko_rl_exp2_zap70",
    "murcko_rl_exp2_kinase",
]


def annotate_df(df):
    murcko, thiq, acryl = [], [], []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            murcko.append(False); thiq.append(False); acryl.append(False); continue
        murcko.append(m.HasSubstructMatch(MURCKO_SMARTS))
        thiq.append(m.HasSubstructMatch(THIQ_ACRYL_SMARTS))
        acryl.append(m.HasSubstructMatch(ACRYL_SMARTS))
    df["m_murcko"] = murcko
    df["m_thiq"] = thiq
    df["m_acryl"] = acryl
    return df


def summarize_cohort(path, label):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    n = len(df)
    df = annotate_df(df)
    return {
        "label": label,
        "n": n,
        "n_murcko": int(df["m_murcko"].sum()),
        "n_thiq":   int(df["m_thiq"].sum()),
        "n_acryl":  int(df["m_acryl"].sum()),
        "n_murcko_and_acryl": int((df["m_murcko"] & df["m_acryl"]).sum()),
        "n_murcko_and_thiq":  int((df["m_murcko"] & df["m_thiq"]).sum()),
        "n_thiq_and_acryl":   int((df["m_thiq"] & df["m_acryl"]).sum()),
        "n_all_three":        int((df["m_murcko"] & df["m_thiq"] & df["m_acryl"]).sum()),
    }


def render_highlight(label, smarts, out_path, color):
    mol = Chem.MolFromSmiles(MOL1_SMI)
    AllChem.Compute2DCoords(mol)
    matches = mol.GetSubstructMatches(smarts)
    atoms = set()
    bonds = set()
    for match in matches:
        atoms.update(match)
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in match and a2 in match:
                bonds.add(bond.GetIdx())
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
    opts = drawer.drawOptions()
    opts.legendFontSize = 24
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atoms),
        highlightBonds=list(bonds),
        highlightAtomColors={a: color for a in atoms},
        highlightBondColors={b: color for b in bonds},
        legend=label,
    )
    drawer.FinishDrawing()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    drawer.WriteDrawingText(str(out_path))
    print(f"  PNG: {out_path}")


def main():
    base = Path("data/tier4_scored")
    rows = []
    print(">>> Analyzing OLD cohorts (no precomputed cols, must annotate)")
    for tag, label in OLD_COHORTS:
        r = summarize_cohort(base / f"{tag}_scored.csv", label)
        if r:
            rows.append({"cohort": tag, **r, "type": "OLD"})
            print(f"  {tag}: N={r['n']:>6}  murcko={r['n_murcko']:>5}  thiq={r['n_thiq']:>5}  acryl={r['n_acryl']:>5}  m+a={r['n_murcko_and_acryl']:>4}")
    print(">>> Analyzing NEW cohorts")
    for tag in NEW_COHORTS:
        r = summarize_cohort(base / f"{tag}_scored.csv", tag)
        if r:
            rows.append({"cohort": tag, **r, "type": "NEW"})
            print(f"  {tag}: N={r['n']:>6}  murcko={r['n_murcko']:>5}  thiq={r['n_thiq']:>5}  acryl={r['n_acryl']:>5}  m+a={r['n_murcko_and_acryl']:>4}")

    df = pd.DataFrame(rows)
    df.to_csv("data/tier4_scored/mol1_pharmacophore_coverage.csv", index=False)

    # AGGREGATE
    tot = df.sum(numeric_only=True)
    n_all = int(tot["n"])
    print()
    print("=" * 70)
    print(f"AGGREGATE across {len(df)} cohorts, N={n_all:,} molecules")
    print("=" * 70)
    print(f"  Mol1 Murcko (lenient SMARTS):     {int(tot['n_murcko']):>7,}  ({100*tot['n_murcko']/n_all:5.2f}%)")
    print(f"  THIQ-acrylamide pharmacophore:    {int(tot['n_thiq']):>7,}  ({100*tot['n_thiq']/n_all:5.2f}%)")
    print(f"  Acrylamide warhead (any):         {int(tot['n_acryl']):>7,}  ({100*tot['n_acryl']/n_all:5.2f}%)")
    print(f"  Murcko AND acrylamide:            {int(tot['n_murcko_and_acryl']):>7,}  ({100*tot['n_murcko_and_acryl']/n_all:5.2f}%)")
    print(f"  Murcko AND THIQ (subsumes acryl): {int(tot['n_murcko_and_thiq']):>7,}  ({100*tot['n_murcko_and_thiq']/n_all:5.2f}%)")
    print(f"  THIQ AND acryl:                   {int(tot['n_thiq_and_acryl']):>7,}  ({100*tot['n_thiq_and_acryl']/n_all:5.2f}%)")
    print(f"  All three (Murcko + THIQ + acryl):{int(tot['n_all_three']):>7,}  ({100*tot['n_all_three']/n_all:5.2f}%)")

    # OLD vs NEW split
    print()
    for typ in ["OLD", "NEW"]:
        sub = df[df["type"] == typ].sum(numeric_only=True)
        n_sub = int(sub["n"])
        print(f"  [{typ}] N={n_sub:,}  Murcko={int(sub['n_murcko']):>6,} ({100*sub['n_murcko']/n_sub:5.2f}%)  THIQ={int(sub['n_thiq']):>6,} ({100*sub['n_thiq']/n_sub:5.2f}%)  M+A={int(sub['n_murcko_and_acryl']):>6,} ({100*sub['n_murcko_and_acryl']/n_sub:5.2f}%)")

    # PNGs
    print()
    print(">>> Rendering Mol1 with substructure highlights")
    render_highlight("Mol1 Murcko (THIQ-amide-imidazole)", MURCKO_SMARTS,
                     "results/paper_evaluation/mol1_murcko_highlight.png", (0.6, 0.85, 1.0))
    render_highlight("THIQ-acrylamide pharmacophore", THIQ_ACRYL_SMARTS,
                     "results/paper_evaluation/mol1_thiq_acryl_highlight.png", (1.0, 0.7, 0.7))
    render_highlight("Acrylamide warhead", ACRYL_SMARTS,
                     "results/paper_evaluation/mol1_acryl_highlight.png", (1.0, 0.85, 0.5))


if __name__ == "__main__":
    main()
