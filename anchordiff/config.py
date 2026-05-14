"""
ANCHORDIFF GLOBAL CONFIG — single source of truth for target / cysteine identity.

Every script that generates or consumes pose data MUST import from here and
assert that the cysteine target it's working with matches CURRENT_TARGET.

Reasoning: an early QA pass (May 2026) found that the project had drifted
between Cys560 (incorrectly chosen) and Cys346 (the literature-validated
ZAP70 covalent target per PMID 33845236, PMID 37594408). Mixing data from
different cysteine targets silently produces meaningless aggregates. This
config + the assertion pattern below prevents that.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CovalentTarget:
    """Specifies which protein, which cysteine, and which PDB structure
    to use for covalent docking experiments."""
    name: str                  # short name, e.g., "zap70_cys346"
    uniprot_id: str            # e.g., "P43403"
    cys_residue: int           # 1-indexed UniProt position
    pdb_id: str                # PDB ID for the holo structure
    pdb_path: str              # local path to the PDB file
    notes: str                 # human-readable notes
    sg_atom_name: str = "SG"   # Boltz/PDB convention for sulfur atom


# ── Authoritative target definitions ────────────────────────────────────────

ZAP70_CYS346 = CovalentTarget(
    name="zap70_cys346",
    uniprot_id="P43403",
    cys_residue=346,
    pdb_id="4K2R",
    pdb_path="data/docking_500/4K2R.pdb",
    notes=(
        "Cys346 sits in the glycine-rich P-loop motif (GxGxxG: residues "
        "344-350 = LGCGNFG) at the entrance of the ATP pocket. This is the "
        "literature-validated ZAP70 covalent target — see Shi et al. JMC 2021 "
        "(PMID 33845236, RDN009 / compound 18, IC50 60 nM, X-ray confirmed) "
        "and Wang et al. JMC 2023 (PMID 37594408, optimized analogs for psoriasis)."
    ),
)

# Deprecated — kept for diff/audit purposes only
ZAP70_CYS560_DEPRECATED = CovalentTarget(
    name="zap70_cys560_DEPRECATED",
    uniprot_id="P43403",
    cys_residue=560,
    pdb_id="4K2R",
    pdb_path="data/docking_500/4K2R.pdb",
    notes=(
        "DEPRECATED. Cys560 sits in the C-lobe back of the kinase, not "
        "structurally accessible to ATP-pocket binders. No published evidence "
        "of small-molecule covalent reach to it. Initial pipeline used this "
        "by mistake; data tagged with this target should NOT be aggregated "
        "with Cys346 data."
    ),
)

EGFR_CYS797 = CovalentTarget(
    name="egfr_cys797",
    uniprot_id="P00533",
    cys_residue=797,
    pdb_id="4ZAU",
    pdb_path="anchordiff/pockets/egfr_cys797/receptor.pdb",
    notes=(
        "EGFR kinase Cys797 — front-pocket cysteine targeted by all 3rd-gen "
        "EGFR covalent inhibitors (afatinib, osimertinib). 4ZAU = osimertinib "
        "co-crystal. Canonical example for multi-target portability of the "
        "covalent design framework."
    ),
)

KRAS_G12C = CovalentTarget(
    name="kras_g12c",
    uniprot_id="P01116",
    cys_residue=12,  # 12 is a mutation site, not a wild-type Cys
    pdb_id="6OIM",
    pdb_path="anchordiff/pockets/kras_g12c/receptor.pdb",
    notes=(
        "KRAS-G12C mutant — gain-of-function Cys at position 12. 6OIM = "
        "MRTX849 (adagrasib) co-crystal. Hit by sotorasib + adagrasib (both "
        "FDA-approved). Different geometry from kinase-front cysteines."
    ),
)

BTK_CYS481 = CovalentTarget(
    name="btk_cys481",
    uniprot_id="Q06187",
    cys_residue=481,
    pdb_id="5P9J",
    pdb_path="data/btk_pocket/5P9J.pdb",
    notes=(
        "Canonical TK-family front-pocket cysteine. Hit by all FDA-approved "
        "BTK covalent inhibitors (ibrutinib, acalabrutinib, zanubrutinib). "
        "Used in Shamir et al. JACS 2026 (London lab) for AF3 covalent "
        "virtual screening (YS1, IC50 30 nM, kinact/Ki 916 M⁻¹s⁻¹)."
    ),
)


# ── The currently active target ────────────────────────────────────────────
# CHANGE THIS LINE to switch the entire pipeline. Every script must
# assert against this value, see assert_target() below.
CURRENT_TARGET = ZAP70_CYS346
SECONDARY_TARGET = BTK_CYS481  # for multi-target validation


def assert_target(expected: CovalentTarget, *, allow_secondary: bool = False) -> None:
    """Call this at the top of any script that generates or consumes pose data
    to fail fast if the target has drifted. Pass `allow_secondary=True` for
    multi-target scripts like the BTK validation runs."""
    if expected.name == CURRENT_TARGET.name:
        return
    if allow_secondary and expected.name == SECONDARY_TARGET.name:
        return
    raise RuntimeError(
        f"Target mismatch! Script expects {expected.name} but anchordiff/config.py "
        f"says CURRENT_TARGET={CURRENT_TARGET.name}. Either change CURRENT_TARGET "
        f"or update the script. NEVER mix data across targets."
    )


def tag_filename(stem: str, target: CovalentTarget = None) -> str:
    """Return a filename tagged with the target identity, so files on disk
    are unambiguous. e.g., tag_filename("manifest.json") =
    "manifest__zap70_cys346.json".
    """
    target = target or CURRENT_TARGET
    return f"{stem}__{target.name}"


if __name__ == "__main__":
    print(f"CURRENT_TARGET = {CURRENT_TARGET.name}")
    print(f"  protein = {CURRENT_TARGET.uniprot_id}")
    print(f"  cysteine = Cys{CURRENT_TARGET.cys_residue}")
    print(f"  pdb = {CURRENT_TARGET.pdb_id} ({CURRENT_TARGET.pdb_path})")
    print(f"\nSECONDARY_TARGET = {SECONDARY_TARGET.name}")
    print(f"  cysteine = Cys{SECONDARY_TARGET.cys_residue}")
    print(f"\nDeprecated (do NOT use): {ZAP70_CYS560_DEPRECATED.name}")
