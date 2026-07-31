#!/usr/bin/env python3
"""Extract triples from CovBinderInPDB (7,375 records, 3,555 PDBs, 2,189 binders).

Re-uses build_triples_v2's worker logic by mapping CovBinderInPDB columns to
the CovInDB schema:

  CovBinderInPDB              -> CovInDB (build_triples_v2)
  ----------------------------------------------------------
  pdb_id                      -> PDB
  chain_id                    -> Resi_chain
  res_num                     -> Resi_posi
  full_residue_name (mapped)  -> Resi_name (3-letter)
  binder_chain_id             -> Ligand_chain
  binder_num                  -> Ligand_position
  binder_id_in_adduct         -> Ligand_name
  binder_smiles               -> SMILES
  warhead_name (mapped)       -> Warhead

Maps PDBs from BOTH covbinder_inpdb/PDB/ (newly downloaded 593) and
covbinder/raw_covindb2/PDB/ (existing 3,445), trying both locations.
"""
from __future__ import annotations
import os, sys, json, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).parent))
from build_triples_v2 import _worker_safe

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
META_CSV = PROJECT_ROOT / "data/covbinder_inpdb/CovBinderInPDB_2022Q4_AllRecords.csv"
PDB_DIRS = [
    PROJECT_ROOT / "data/covbinder_inpdb/PDB",
    PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB",
]
OUT = PROJECT_ROOT / "data/m1a_triples_v2/covbinder_inpdb_triples.parquet"

RESIDUE_MAP = {
    "Cysteine": "CYS", "Serine": "SER", "Threonine": "THR", "Lysine": "LYS",
    "Histidine": "HIS", "Glutamic Acid": "GLU", "Aspartic Acid": "ASP",
    "Tyrosine": "TYR", "Methionine": "MET", "Asparagine": "ASN",
    "Glutamine": "GLN", "Arginine": "ARG", "Tryptophan": "TRP",
    "Glycine": "GLY", "Proline": "PRO", "Alanine": "ALA", "Valine": "VAL",
    "Leucine": "LEU", "Isoleucine": "ILE", "Phenylalanine": "PHE",
    "Selenocysteine": "SEC",
}

# Normalize CovBinderInPDB warhead names -> our vocab
WARHEAD_REMAP = {
    "Ketone": "ketone",
    "Aldehyde": "aldehyde",
    "Boronic_Acid": "boronic_acid",
    "Acrylamide": "acrylamide",
    "β-lactam": "beta_lactam",
    "Epoxide": "epoxide",
    "α-halomethyl_Ketone": "halomethyl_ketone",
    "Haloacetamide": "haloacetamide",
    "Vinyl_Sulfone": "vinyl_sulfone",
    "Acrylate": "acrylate",
    "Nitrile": "nitrile",
    "α,β-unsaturated_Carbonyl": "michael_acceptor",
    "Imidazolidinone": "imidazolidinone",
    "Alkyl_Halide": "halohydrocarbon",
    "α-ketoamide": "alpha_ketoamide",
    "Disulfide": "disulfide",
    "β-lactone": "beta_lactone",
    "Ester": "ester",
    "Halophosphonate": "phosphonate",
    "Unclassified": "unclassified",
    "Vinyl_Ester": "vinyl_ester",
    "Lactone": "lactone",
    "Carbamate": "carbamate",
    "Halophosphate": "phosphate",
    "Phosphonate": "phosphonate",
    "Phosphate": "phosphate",
    "Sulfonate": "sulfonate",
    "Sulfonyl_Fluoride": "sulfonyl_fluoride",
    "Sulfonyl_Chloride": "sulfonyl_chloride",
    "Sulfonamide": "sulfonamide",
    "Sulfonic_Acid": "sulfonic_acid",
    "Vinyl_Nitrile": "vinyl_nitrile",
    "α-Phenoxymethyl_Ketone": "phenoxymethyl_ketone",
    "Hemiketal": "hemiketal",
    "Hemiacetal": "hemiacetal",
    "Isocyanate": "isocyanate",
    "Isothiocyanate": "isothiocyanate",
    "Cyanamide": "cyanamide",
    "Aziridine": "aziridine",
    "γ-lactam": "gamma_lactam",
    "Diazomethyl_Ketone": "diazomethyl_ketone",
    "Carbonate": "carbonate",
    "Acid_Anhydride": "acid_anhydride",
    "Methylsulfonyl_Tetrazole": "sulfonyl_tetrazole",
    "1,3-Dioxol-2-one": "dioxolone",
    "Tropone": "tropone",
    "Azide": "azide",
    "Urea": "urea_carbonyl",
    "Sulfamoyl_Fluoride": "sulfamoyl_fluoride",
    "γ-lactone": "gamma_lactone",
    "Sulfoxide": "sulfoxide",
    "α-acyloxymethyl_Ketone": "acyloxymethyl_ketone",
    "Sultam": "sultam",
    "α-Tosyloxymethyl_Ketone": "tosyloxymethyl_ketone",
    "Vinyl_Heterocycle": "vinyl_heterocycle",
    "Diazomethyl_Carbonyl": "diazomethyl_carbonyl",
    "Vinyl_Pyridinium": "vinyl_pyridinium",
    "Salicylate_Ester": "ester",
    "Phosphoramide": "phosphoramide",
    "Phosphoramidate": "phosphoramide",
    "Halofluorosulfate": "fluorosulfate",
    "Halophenol": "halophenol",
    "Aryl_Halide": "aryl_halide",
}


def find_pdb(pdb_id: str) -> Path | None:
    pdb_id_lc = pdb_id.lower()
    for d in PDB_DIRS:
        p = d / f"{pdb_id_lc}.pdb"
        if p.exists() and p.stat().st_size > 0:
            return p
        p_upper = d / f"{pdb_id.upper()}.pdb"
        if p_upper.exists() and p_upper.stat().st_size > 0:
            return p_upper
    return None


def main():
    df = pd.read_csv(META_CSV)
    print(f"CovBinderInPDB: {len(df)} records, {df['pdb_id'].nunique()} PDBs, "
           f"{df['binder_id'].nunique()} binders", flush=True)

    # Build work items
    work = []
    skipped_pdb = 0
    skipped_smi = 0
    skipped_res = 0
    for _, row in df.iterrows():
        pdb_path = find_pdb(str(row["pdb_id"]))
        if pdb_path is None:
            skipped_pdb += 1
            continue
        smi = row.get("binder_smiles")
        if not isinstance(smi, str) or len(smi) < 3:
            skipped_smi += 1
            continue
        # Map to CovInDB schema
        nuc_aa = RESIDUE_MAP.get(row.get("full_residue_name"))
        if nuc_aa is None:
            skipped_res += 1
            continue
        warhead_raw = row.get("warhead_name")
        # Use REMAP if present, else fall back to lowercased identity
        # (build_triples_v2 will accept any string)
        mapped = WARHEAD_REMAP.get(warhead_raw,
                                     str(warhead_raw).lower().replace(" ", "_"))
        rd = {
            "PDB": str(row["pdb_id"]),
            "SMILES": smi,
            "Warhead": warhead_raw,  # raw name preserved; will be in WARHEAD_NORM check
            "Resi_chain": str(row.get("chain_id", "A")),
            "Resi_posi": row.get("res_num"),
            "Resi_name": nuc_aa,
            "Ligand_chain": str(row.get("binder_chain_id", row.get("chain_id", "A"))),
            "Ligand_position": row.get("binder_num"),
            "Ligand_name": str(row.get("binder_id_in_adduct", "")).upper(),
            "Protein_name": str(row.get("unp_accessionid", "")),
            "_source": "covbinder_inpdb",
            # Override warhead class for SMARTS class hinting; build_triples_v2
            # will pass this through WARHEAD_NORM. We patch by injecting
            # the mapped name into WARHEAD_NORM via the per-row class hint:
            # build_triples_v2 reads Warhead -> normalizes. Provide normalized as
            # Warhead so it falls into the WARHEAD_NORM identity branch.
            # Use mapped name directly:
        }
        # Hack: store mapped warhead directly so WARHEAD_NORM identity passes
        rd["Warhead"] = mapped
        # But the SMARTS warhead_class hint expects the normalized form already
        # WARHEAD_NORM in build_triples_v2 only knows CovInDB names; so by setting
        # Warhead to our mapped name, the WARHEAD_NORM.get() will fall back
        # which produces e.g. "boronic_acid" -> stays "boronic_acid"
        work.append((str(pdb_path), rd))

    print(f"Built {len(work)} work items "
           f"(skipped {skipped_pdb} missing PDBs, {skipped_smi} bad SMILES, "
           f"{skipped_res} unknown residues)", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(_worker_safe, w) for w in work]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                rows.append(r)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(work)}  kept={len(rows)}", flush=True)

    print(f"FINAL kept={len(rows)} / {len(work)} "
           f"({100.0*len(rows)/max(1,len(work)):.1f}%)", flush=True)

    if not rows:
        print("FATAL: no triples", flush=True); sys.exit(1)

    out_df = pd.DataFrame(rows)
    out_df["pocket_residues"] = out_df["pocket_residues"].apply(json.dumps)
    out_df["warhead_pose_6d"] = out_df["warhead_pose_6d"].apply(lambda x: json.dumps(list(x)))
    out_df["nucleophile_xyz"] = out_df["nucleophile_xyz"].apply(lambda x: json.dumps(list(x)))
    out_df["warhead_b_xyz"] = out_df["warhead_b_xyz"].apply(lambda x: json.dumps(list(x)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)
    print(f"Wrote {len(out_df)} triples to {OUT}", flush=True)

    print()
    print("=== Warhead class distribution ===")
    print(out_df["warhead_class"].value_counts().head(20).to_string())
    print()
    print("=== Nucleophile distribution ===")
    print(out_df["nucleophile_resname"].value_counts().to_string())
    print()
    sizes = out_df["pocket_residues"].apply(lambda s: len(json.loads(s)))
    print(f"Pocket sizes: min={sizes.min()} median={int(sizes.median())} "
           f"mean={sizes.mean():.1f} max={sizes.max()}")
    print(f"BD angle median={out_df['bd_angle_deg'].median():.1f} deg")


if __name__ == "__main__":
    main()
