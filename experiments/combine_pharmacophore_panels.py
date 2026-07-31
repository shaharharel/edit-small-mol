"""Combine 4 substructure views of Mol1 into a side-by-side figure.

Panel 1: Mol1 Murcko (blue)
Panel 2: THIQ-acrylamide pharmacophore (red)
Panel 3: Acrylamide warhead only (orange)
Panel 4: Murcko AND Acrylamide together (overlay: blue + orange)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
MURCKO = Chem.MolFromSmarts("O=C(Nc1cncn1)c1cccc2c1CNC2")
THIQ_ACRYL = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
ACRYL = Chem.MolFromSmarts("C=CC(=O)N")

PANELS = [
    ("Mol1 Murcko (blue)\n26.7% total / 40.1% NEW", [(MURCKO, (0.6, 0.85, 1.0))], "mol1_murcko_highlight.png"),
    ("THIQ-acrylamide pharmacophore (red)\n65.0% total / 96.4% NEW", [(THIQ_ACRYL, (1.0, 0.7, 0.7))], "mol1_thiq_acryl_highlight.png"),
    ("Acrylamide warhead alone (orange)\n99.0% total / ~100% NEW", [(ACRYL, (1.0, 0.85, 0.5))], "mol1_acryl_highlight.png"),
    ("Murcko AND Acrylamide (blue + orange overlay)\n26.2% total / 39.4% NEW", [(MURCKO, (0.6, 0.85, 1.0)), (ACRYL, (1.0, 0.85, 0.5))], "mol1_murcko_AND_acryl_highlight.png"),
]


def render(label, patterns, out_path):
    mol = Chem.MolFromSmiles(MOL1_SMI)
    AllChem.Compute2DCoords(mol)
    atoms, bonds, atom_colors, bond_colors = set(), set(), {}, {}
    for patt, color in patterns:
        for match in mol.GetSubstructMatches(patt):
            for a in match:
                if a not in atom_colors:
                    atom_colors[a] = color
                atoms.add(a)
            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in match and a2 in match:
                    if bond.GetIdx() not in bond_colors:
                        bond_colors[bond.GetIdx()] = color
                    bonds.add(bond.GetIdx())
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
    opts = drawer.drawOptions()
    opts.legendFontSize = 22
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atoms),
        highlightBonds=list(bonds),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        legend="",
    )
    drawer.FinishDrawing()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    drawer.WriteDrawingText(str(out_path))


# Render fresh
out_dir = Path("results/paper_evaluation")
for label, patterns, fname in PANELS:
    render(label, patterns, out_dir / fname)

# Stitch
imgs = [Image.open(out_dir / fname) for _, _, fname in PANELS]
labels = [label for label, _, _ in PANELS]
w, h = imgs[0].size
gap = 24
title_h = 60
caption_h = 80
total_w = w * 4 + gap * 5
total_h = h + title_h + caption_h
canvas = Image.new("RGB", (total_w, total_h), "white")
draw = ImageDraw.Draw(canvas)
try:
    title_font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 28)
    cap_font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 18)
except Exception:
    title_font = cap_font = ImageFont.load_default()
draw.text((total_w // 2 - 360, 14), "Mol1 substructure groups — counted across all 16 cohorts (636,684 mols)", fill="black", font=title_font)
for i, (img, label) in enumerate(zip(imgs, labels)):
    x = gap + i * (w + gap)
    canvas.paste(img, (x, title_h))
    lines = label.split("\n")
    for j, line in enumerate(lines):
        draw.text((x + 20, title_h + h + 8 + j * 24), line, fill="black", font=cap_font)
out_path = out_dir / "mol1_pharmacophore_3groups.png"
canvas.save(out_path)
print(f"Saved: {out_path}")
