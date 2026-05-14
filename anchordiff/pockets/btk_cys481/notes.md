# btk_cys481

- ref SMILES: `C=CC(=O)N1CCC[C@@H](C1)n1nc(c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21`
- target cysteine: residue 481
- warhead Boltz atom names (in order C_β, C_α, C_carb, N_amide):
  - ['C31', 'C32', 'C30', 'O25', 'N51']
- warhead atom indices (RDKit, 0-based, into the H-added mol):
  - [0, 1, 2, 3, 4]

## Notes
Canonical TK-family front-pocket cysteine. Hit by all FDA-approved BTK covalent inhibitors (ibrutinib, acalabrutinib, zanubrutinib). Used in Shamir et al. JACS 2026 (London lab) for AF3 covalent virtual screening (YS1, IC50 30 nM, kinact/Ki 916 M⁻¹s⁻¹).
