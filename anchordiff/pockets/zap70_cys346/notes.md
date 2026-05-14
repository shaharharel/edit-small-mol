# zap70_cys346

- ref SMILES: `C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1`
- target cysteine: residue 346
- warhead Boltz atom names (in order C_β, C_α, C_carb, N_amide):
  - ['C26', 'C27', 'C25', 'O22', 'N38']
- warhead atom indices (RDKit, 0-based, into the H-added mol):
  - [0, 1, 2, 3, 4]

## Notes
Cys346 sits in the glycine-rich P-loop motif (GxGxxG: residues 344-350 = LGCGNFG) at the entrance of the ATP pocket. This is the literature-validated ZAP70 covalent target — see Shi et al. JMC 2021 (PMID 33845236, RDN009 / compound 18, IC50 60 nM, X-ray confirmed) and Wang et al. JMC 2023 (PMID 37594408, optimized analogs for psoriasis).
