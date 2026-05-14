# zap70

- ref SMILES: `C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1`
- target cysteine: residue 560
- warhead Boltz atom names (in order C_β, C_α, C_carb, N_amide):
  - ['C26', 'C27', 'C25', 'O22', 'N38']
- warhead atom indices (RDKit, 0-based, into the H-added mol):
  - [0, 1, 2, 3, 4]

## Notes
PDB 4K2R is ZAP70 kinase domain bound to imatinib. We use Mol 1 as the reference ligand (defines the binding pocket via 4K2R's natural ligand pose; DiffSBDD will then sample new ligands in residues within 8 Å of the ref ligand). Cys560 is in the activation-loop area accessible from the ATP pocket.
