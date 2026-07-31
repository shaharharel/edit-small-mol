"""Compute tox structural alerts + k_inact proxy on F4_boltz_full.csv.

Adds columns:
  tox_alerts_count : int     — number of structural alerts hit (extended Brenk + panel-flagged)
  tox_alert_names  : str     — comma-separated alert names, e.g. "anilinopyridine,vinyl_aniline"
  k_inact_proxy    : float   — covalent productivity score (thiolate fraction × geometry factor)

Tox alerts (~30 SMARTS) cover the panel-flagged liabilities Brenk missed:
  aniline class, anilinopyridine, vinyl-aniline, ortho-F-aniline, peroxide,
  cyclic sulfamide, acyl hydrazide, vinyl ether / dihydrofuran, enamine, enol,
  strained-ring + nitrile (CN release), N-F bond, alpha-halo carbonyl,
  isocyanate, triazole/tetrazole scaffold drift, sulfonyl chloride, etc.

k_inact_proxy = thiolate_fraction × geometry_factor
  thiolate_fraction = 1 / (1 + 10^(pKa_Cys346 - 7.4))    # PROPKA-driven
  geometry_factor   = exp( -((d_SG - 1.8)² + (burgi_dunitz_dev_deg/10)²) )

LUMO factor is omitted (all 58 share an unsubstituted primary acrylamide → LUMO is
constant across cohort; one-time DFT calc would just rescale all values equally).

Writes back to data/tier4_scored/F4_boltz_full.csv in place.
"""
import math
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ── Extended tox alerts (panel-flagged liabilities + Brenk gaps) ──
# Each entry: (alert_name, SMARTS, severity)
# severity: "major" = single hit is a kill signal; "minor" = warning.
ALERTS = [
    # Aromatic amines (Ames/Bioactivation)
    ("primary_aniline",         "[NX3H2]c1ccccc1",                  "major"),
    ("secondary_aniline",       "[NX3H1]([#6;!H0])c1ccccc1",        "major"),
    ("ortho_F_aniline",         "Nc1cccc(F)c1",                     "major"),
    ("anilinopyridine_2",       "Nc1ccccn1",                        "major"),
    ("anilinopyridine_3",       "Nc1cccnc1",                        "major"),
    ("anilinopyridine_4",       "Nc1ccncc1",                        "major"),
    ("aniline_pyridine_link",   "c1ccc(N[#6]c2ccncc2)cc1",          "major"),
    ("aniline_pyrimidine",      "Nc1ncccc1[NX3]",                   "major"),
    # Vinyl-aniline (Michael + aryl amine)
    ("vinyl_aniline",           "C=Cc1ccc(N)cc1",                   "major"),
    ("vinyl_aniline_alt",       "C=Cc1cccc(N)c1",                   "major"),
    # Peroxides / endoperoxides
    ("peroxide",                "[OX2][OX2]",                       "major"),
    # Acyl hydrazides (INH hepatotox class)
    ("acyl_hydrazide",          "[CX3](=O)[NX3H1][NX3H2]",          "major"),
    ("acyl_hydrazide_subst",    "[CX3](=O)[NX3H1][NX3H1]",          "major"),
    # Cyclic sulfamides (Stevens-Johnson class)
    ("cyclic_sulfamide",        "[NX3R][SX4](=O)(=O)[NX3R]",        "major"),
    # Vinyl ethers / dihydrofuran (CYP epoxidation)
    ("vinyl_ether_aliphatic",   "[#6]=[#6][OX2][#6]",               "major"),
    ("dihydrofuran",            "C1=CCO[CH2]1",                     "major"),
    # Enamine (iminium → DNA alkylation, MPTP-class)
    ("enamine",                 "[NX3]([!#1])([!#1])C=C",           "major"),
    ("tetrahydropyridine",      "[NX3R]1[CH2][CH2]C=C[CH2]1",       "major"),
    # Enol / vinyl-alcohol (→ enone tautomer) — REMOVED 2026-06-13: marginal in practice, not a tox kill signal
    # ("enol",                  "[OX2H][CX3]=[CX3]",                "minor"),
    # Strained-ring + nitrile (cyanohydrin reversal → HCN release)
    ("cyclopropyl_CN",          "[#6;R3](C#N)",                     "major"),
    ("cyclobutyl_CN",           "[#6;R4](C#N)",                     "major"),
    ("alpha_dicyano",           "[CX4](C#N)(C#N)",                  "major"),
    # N-F bond (rare; alkylator-grade)
    ("N_F_bond",                "[NX3][F]",                         "major"),
    # Alpha-halo carbonyl (alkylator)
    ("alpha_halo_carbonyl",     "[F,Cl,Br,I][CX4][CX3](=O)",        "major"),
    # Isocyanate / thiocyanate
    ("isocyanate",              "[NX2]=[CX2]=[OX1]",                "major"),
    ("thiocyanate",             "[#6][SX2][CX2]#[NX1]",             "major"),
    # Sulfonyl chloride
    ("sulfonyl_chloride",       "[SX4](=O)(=O)[Cl]",                "major"),
    # 2026-06-13: tetrazole_core / triazole_1_2_4 REMOVED — they are FDA-approved drug scaffolds
    # (losartan, valsartan, fluconazole, ribavirin) and NOT tox alerts. They were originally
    # included as cohort-QA flags for chemotype drift, but that belongs in a separate metric,
    # not the tox-alerts count.
    # ("tetrazole_core",        "c1nnnn1",                          "minor"),
    # ("triazole_1_2_4",        "c1ncnn1",                          "minor"),
    # Michael acceptors beyond the warhead (acrylamide is OK; flag a SECOND one)
    # Already counted: simple acrylamide (the warhead) is everywhere
    ("nitroalkene",             "[CX3]=[CX3][NX3](=O)=O",           "major"),
    # Second Michael acceptor: α,β-unsat C(=O)-X where X is NOT the amide nitrogen (excludes warhead).
    # Only flag enones (X=C) / α,β-unsat aldehydes (X=H) / α,β-unsat esters (X=O).
    ("second_michael_enone",    "[CX3]=[CX3][CX3](=O)[#6;!$([NX3])]","major"),
    ("second_michael_ester",    "[CX3]=[CX3][CX3](=O)[OX2][#6]",     "major"),
    # Aromatic nitro (Ames+)
    ("aromatic_nitro",          "[c][NX3](=O)=O",                   "major"),
    # Reactive epoxide
    ("epoxide",                 "C1OC1",                            "major"),
    # 2026-06-13: primary_amine_strained REMOVED — soft P-gp/PK flag, not a real tox kill signal.
    # ("primary_amine_strained",  "[NX3H2][CX4R3,CX4R4]",           "minor"),
    # Imidoyl chloride
    ("imidoyl_chloride",        "[CX3](=[NX2])[Cl]",                "major"),
]


def count_alerts(smi: str):
    """Return (count_int, names_csv) of tox alerts hit, or (None, None) on parse failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    hits = []
    for name, smarts, _sev in ALERTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and m.HasSubstructMatch(patt):
            hits.append(name)
    return len(hits), (",".join(hits) if hits else "")


def k_inact_proxy(pKa_Cys: float, d_SG: float, bd_dev: float):
    """Covalent productivity = thiolate_fraction × geometry_factor.

    pKa_Cys  : PROPKA pKa of Cys346 in this pose
    d_SG     : warhead Cβ → Cys-Sγ distance (Å). Productive ≈ 1.8.
    bd_dev   : Bürgi-Dunitz angle deviation from 107° (deg). Productive ≈ 0.
    """
    if any(v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
           for v in (pKa_Cys, d_SG)):
        return None
    if bd_dev is None or (isinstance(bd_dev, float) and (math.isnan(bd_dev) or math.isinf(bd_dev))):
        bd_dev = 30.0  # missing angle → assume mild penalty
    try:
        thiolate = 1.0 / (1.0 + 10 ** (float(pKa_Cys) - 7.4))
        d_term = (float(d_SG) - 1.8) ** 2
        bd_term = (float(bd_dev) / 10.0) ** 2
        geom = math.exp(-(d_term + bd_term))
        return thiolate * geom
    except (TypeError, ValueError, OverflowError):
        return None


def main():
    csv = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/tier4_scored/F4_boltz_full.csv")
    df = pd.read_csv(csv)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols")

    # tox alerts
    counts, names = [], []
    for smi in df["smiles"].tolist():
        c, n = count_alerts(smi)
        counts.append(c)
        names.append(n)
    df["tox_alerts_count"] = counts
    df["tox_alert_names"] = names

    # k_inact_proxy
    pKaC = df.get("pKa_Cys346")
    dSG = df.get("d_SG")
    bd = df.get("burgi_dunitz_dev_deg")
    kp = [k_inact_proxy(p, d, b) for p, d, b in zip(pKaC, dSG, bd)]
    df["k_inact_proxy"] = kp

    df.to_csv(csv, index=False)
    print(f"\nSaved 3 new columns to {csv.name}")

    # Distribution summaries
    print(f"\ntox_alerts_count distribution (n={df['tox_alerts_count'].notna().sum()}):")
    print(df["tox_alerts_count"].value_counts().sort_index().to_string())
    print(f"\nMolecules with ≥1 alert: {(df['tox_alerts_count'].fillna(0) >= 1).sum():,} "
          f"({100*(df['tox_alerts_count'].fillna(0) >= 1).sum()/len(df):.1f}%)")

    print(f"\nMost frequent alerts:")
    from collections import Counter
    c = Counter()
    for s in df["tox_alert_names"].dropna():
        if s:
            for tag in s.split(","):
                c[tag] += 1
    for tag, n in c.most_common(15):
        print(f"  {tag:30s} {n:5d}")

    print(f"\nk_inact_proxy distribution (n={df['k_inact_proxy'].notna().sum()}):")
    print(df["k_inact_proxy"].describe().to_string())


if __name__ == "__main__":
    main()
