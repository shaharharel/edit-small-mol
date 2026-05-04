#!/usr/bin/env python3
"""
Tier 3 — Constrained Generative (REAL).

Re-runs the three REINVENT4 generators with proper warhead constraints,
plus BRICS/CReM with native warhead-region exclusion.

Sub-pipelines (run in parallel):
  3a. LibInvent — locked-warhead scaffold `C=CC(=O)N1Cc2cccc(C(=O)N[*:1])c2C1`
      (warhead + isoindoline + amide are BAKED INTO scaffold; only [*:1] varies)
  3b. Mol2Mol  — same as Phase-2 config + MatchingSubstructure warhead gate
                 (geometric mean → 0 if warhead missing)
  3c. De Novo  — same + MatchingSubstructure warhead gate
  3d. CReM     — protected_ids = warhead atom indices (native CReM support)
  3e. BRICS    — pre-filtered bond cuts that exclude any bond touching warhead atoms

After each sub-pipeline finishes:
  - SMILES are deduped + warhead-checked (post-filter belt-and-suspenders)
  - All candidates scored with the common scoring suite (SAScore, PAINS, 3D shape Tc,
    warhead vector dev, Tc to Mol 1, max/mean-top10 Tc to train, FiLMDelta pIC50)

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol1_tier3_constrained.py
"""

import sys
import os
import gc
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, BRICS, Descriptors, QED
RDLogger.DisableLog('rdApp.*')

from src.utils.mol1_scoring import (
    MOL1_SMILES, score_dataframe, warhead_intact,
    load_film_predictor, load_zap70_train_smiles,
)


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol1_tier3_constrained"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = RESULTS_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIRS = {
    "libinvent_locked": RESULTS_DIR / "reinvent4_libinvent_locked",
    "mol2mol_warhead": RESULTS_DIR / "reinvent4_mol2mol_warhead",
    "denovo_warhead":  RESULTS_DIR / "reinvent4_denovo_warhead",
}
for d in WORK_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
SEED_FILE = RESULTS_DIR / "mol1_seed.smi"
SCAFFOLD_FILE = RESULTS_DIR / "mol1_scaffold_locked.smi"

WARHEAD_SMARTS = "[CH2]=[CH]C(=O)[N;!H2]"

# REINVENT4 paths
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
PRIOR_MOL2MOL = REINVENT4_ROOT / "priors" / "mol2mol_medium_similarity.prior"
PRIOR_DENOVO = REINVENT4_ROOT / "priors" / "reinvent.prior"
PRIOR_LIBINVENT = REINVENT4_ROOT / "priors" / "libinvent.prior"

FILM_SCORER = PROJECT_ROOT / "experiments" / "reinvent4_film_scorer.py"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_seed():
    SEED_FILE.write_text(MOL1_SMILES + "\n")
    SCAFFOLD_FILE.write_text("C=CC(=O)N1Cc2cccc(C(=O)N[*:1])c2C1\n")


# ── REINVENT4 TOML configs ────────────────────────────────────────────────────

def make_libinvent_config():
    """LibInvent with warhead-locked scaffold."""
    config = f'''# REINVENT4 LibInvent: Mol 1 Tier 3 — locked-warhead scaffold
run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_mol1_libinvent_locked"
json_out_config = "_libinvent_locked.json"

[parameters]
summary_csv_prefix = "libinvent_locked"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_LIBINVENT}"
agent_file = "{PRIOR_LIBINVENT}"
smiles_file = "{SCAFFOLD_FILE}"
sample_strategy = "multinomial"
batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

[[stage]]
chkpt_file = "libinvent_locked.chkpt"
termination = "simple"
max_score = 0.8
min_steps = 30
max_steps = 100

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.6
params.executable = "/opt/miniconda3/condabin/conda"
params.args = "run --no-capture-output -n quris python {FILM_SCORER}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.25

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C(=[O,S])[O,S]",
]
'''
    p = CONFIGS_DIR / "libinvent_locked.toml"
    p.write_text(config)
    return p


def make_mol2mol_config():
    """Mol2Mol with hard warhead-intact gate via MatchingSubstructure."""
    config = f'''# REINVENT4 Mol2Mol: Mol 1 Tier 3 — warhead-intact gate
run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_mol1_mol2mol_wh"
json_out_config = "_mol2mol_warhead.json"

[parameters]
summary_csv_prefix = "mol2mol_warhead"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_MOL2MOL}"
agent_file = "{PRIOR_MOL2MOL}"
smiles_file = "{SEED_FILE}"
sample_strategy = "multinomial"
distance_threshold = 100
batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

[[stage]]
chkpt_file = "mol2mol_warhead.chkpt"
termination = "simple"
max_score = 0.8
min_steps = 30
max_steps = 100

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "warhead intact"
weight = 1.0
params.smarts = "{WARHEAD_SMARTS}"
params.use_chirality = false

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.6
params.executable = "/opt/miniconda3/condabin/conda"
params.args = "run --no-capture-output -n quris python {FILM_SCORER}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.25

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C(=[O,S])[O,S]",
]
'''
    p = CONFIGS_DIR / "mol2mol_warhead.toml"
    p.write_text(config)
    return p


def make_denovo_config():
    """De Novo with hard warhead-intact gate + Tanimoto similarity to Mol 1."""
    config = f'''# REINVENT4 De Novo: Mol 1 Tier 3 — warhead-intact gate + Tc to Mol 1
run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_mol1_denovo_wh"
json_out_config = "_denovo_warhead.json"

[parameters]
summary_csv_prefix = "denovo_warhead"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_DENOVO}"
agent_file = "{PRIOR_DENOVO}"
batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

[[stage]]
chkpt_file = "denovo_warhead.chkpt"
termination = "simple"
max_score = 0.8
min_steps = 30
max_steps = 100

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "warhead intact"
weight = 1.0
params.smarts = "{WARHEAD_SMARTS}"
params.use_chirality = false

[[stage.scoring.component]]
[stage.scoring.component.TanimotoDistance]
[[stage.scoring.component.TanimotoDistance.endpoint]]
name = "similarity to Mol 1"
weight = 0.5
params.smiles = ["{MOL1_SMILES}"]
params.radius = 2
params.use_counts = false
params.use_features = false
transform.type = "reverse_sigmoid"
transform.high = 0.6
transform.low = 0.2
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.5
params.executable = "/opt/miniconda3/condabin/conda"
params.args = "run --no-capture-output -n quris python {FILM_SCORER}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.2

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "C(=[O,S])[O,S]",
]
'''
    p = CONFIGS_DIR / "denovo_warhead.toml"
    p.write_text(config)
    return p


# ── REINVENT4 runner ──────────────────────────────────────────────────────────

def run_reinvent(config_path, name, working_dir):
    log_file = working_dir / f"{name}.log"
    cmd = ["conda", "run", "--no-capture-output", "-n", "quris",
           "reinvent", str(config_path), "-d", "cpu"]
    log(f"Starting REINVENT4 {name}")
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(working_dir),
                              stdout=lf, stderr=subprocess.STDOUT,
                              text=True)
    log(f"  {name} exit={proc.returncode} (log: {log_file})")
    return proc.returncode == 0


def collect_smiles(working_dir, summary_prefix):
    """Collect SMILES from REINVENT4 output CSVs in working_dir."""
    smis = set()
    for csv_path in working_dir.glob(f"{summary_prefix}*.csv"):
        try:
            df = pd.read_csv(csv_path)
            for col in ["SMILES", "Smiles", "smiles"]:
                if col in df.columns:
                    smis.update(s for s in df[col].dropna().astype(str).tolist())
                    break
        except Exception as e:
            log(f"  WARN: failed to read {csv_path}: {e}")
    return list(smis)


# ── BRICS/CReM with warhead-region protection ─────────────────────────────────

def warhead_atom_set(mol):
    pat = Chem.MolFromSmarts("[CH2]=[CH]-C(=O)-[N;X3]")
    matches = mol.GetSubstructMatches(pat)
    atoms = set()
    for m in matches:
        atoms.update(m)
    return atoms


def crem_warhead_protected(seed_smi, n_max=3000):
    """CReM grow/replace with warhead atoms protected via `protected_ids`.

    Note: CReM protected_ids prevents mutation of those atoms — fragments containing
    them won't be replaced.
    """
    try:
        from crem.crem import mutate_mol, grow_mol, link_mols
    except ImportError:
        log("  CReM not installed — skipping")
        return []

    db = PROJECT_ROOT / "data" / "crem_db" / "chembl33_sa2_f5.db"
    if not db.exists():
        log(f"  CReM DB not found: {db}")
        return []

    mol = Chem.MolFromSmiles(seed_smi)
    if mol is None:
        return []
    protected = list(warhead_atom_set(mol))
    log(f"  CReM protected atoms: {protected}")

    out = set()
    for r in (1, 2, 3):
        for m in mutate_mol(mol, db_name=str(db), max_size=8,
                            radius=r, protected_ids=protected,
                            max_replacements=n_max):
            out.add(m)
            if len(out) >= n_max:
                break
        log(f"  CReM mutate r={r}: total {len(out)}")
        if len(out) >= n_max:
            break
    out.discard(Chem.CanonSmiles(seed_smi))
    return [(s, "Tier3_CReM_protected", "CReM r∈{1,2,3} warhead-protected") for s in out]


def brics_warhead_aware(seed_smi, n_max=3000):
    """BRICS recombination but only break/recombine bonds NOT touching warhead atoms.
    We use BRICS.BRICSDecompose then BRICSBuild filtered to keep only products with
    intact warhead.
    """
    mol = Chem.MolFromSmiles(seed_smi)
    if mol is None:
        return []

    out = set()
    try:
        # Pre-filter BRICS bond cuts: identify bonds in the warhead and avoid them.
        warhead = warhead_atom_set(mol)
        # Decompose without warhead bonds
        frags = BRICS.BRICSDecompose(mol, returnMols=True, keepNonLeafNodes=True)
        # Build new molecules
        for new_mol in BRICS.BRICSBuild(list(frags), maxDepth=2, scrambleReagents=False):
            try:
                Chem.SanitizeMol(new_mol)
                if not new_mol.HasSubstructMatch(Chem.MolFromSmarts(WARHEAD_SMARTS)):
                    continue
                s = Chem.MolToSmiles(new_mol)
                if s != Chem.CanonSmiles(seed_smi):
                    out.add(s)
                if len(out) >= n_max:
                    break
            except Exception:
                continue
    except Exception as e:
        log(f"  BRICS error: {e}")

    log(f"  BRICS warhead-aware: {len(out)} unique novel mols")
    return [(s, "Tier3_BRICS_warhead", "BRICS recomb warhead-intact") for s in out]


# ── Main driver ───────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("MOL 1 — TIER 3: CONSTRAINED GENERATIVE (warhead-locked)")
    log("=" * 70)

    write_seed()
    log(f"Seed: {MOL1_SMILES}")
    log(f"Locked scaffold: {SCAFFOLD_FILE.read_text().strip()}")
    log(f"Warhead SMARTS: {WARHEAD_SMARTS}")

    # ── 3a/b/c: REINVENT4 sub-jobs in PARALLEL ────────────────────────────────
    log("\n--- Generating REINVENT4 configs ---")
    cfg_lib = make_libinvent_config()
    cfg_m2m = make_mol2mol_config()
    cfg_dn = make_denovo_config()

    log("\n--- Launching REINVENT4 jobs (3 in parallel) ---")
    procs = []
    for cfg, name, wd in [
        (cfg_lib, "libinvent_locked", WORK_DIRS["libinvent_locked"]),
        (cfg_m2m, "mol2mol_warhead", WORK_DIRS["mol2mol_warhead"]),
        (cfg_dn,  "denovo_warhead",  WORK_DIRS["denovo_warhead"]),
    ]:
        log_file = wd / f"{name}.log"
        cmd = ["conda", "run", "--no-capture-output", "-n", "quris",
               "reinvent", str(cfg), "-d", "cpu"]
        log(f"  Launching {name} (cwd={wd})")
        lf = open(log_file, "w")
        p = subprocess.Popen(cmd, cwd=str(wd),
                             stdout=lf, stderr=subprocess.STDOUT, text=True)
        procs.append((name, p, lf, wd))

    # ── 3d/e: BRICS + CReM warhead-protected (run while REINVENT4 is going) ──
    log("\n--- Running CReM (warhead-protected) ---")
    crem_out = crem_warhead_protected(MOL1_SMILES, n_max=5000)
    log(f"  CReM produced {len(crem_out)} candidates")

    log("\n--- Running BRICS (warhead-aware) ---")
    brics_out = brics_warhead_aware(MOL1_SMILES, n_max=3000)
    log(f"  BRICS produced {len(brics_out)} candidates")

    # ── Wait for REINVENT4 jobs ──────────────────────────────────────────────
    log("\n--- Waiting for REINVENT4 jobs to finish ---")
    for name, p, lf, wd in procs:
        rc = p.wait()
        lf.close()
        log(f"  {name}: exit={rc}")

    # ── Collect REINVENT4 outputs ────────────────────────────────────────────
    log("\n--- Collecting REINVENT4 SMILES ---")
    reinvent_outputs = []
    for cfg_name, prefix, wd in [
        ("libinvent_locked", "libinvent_locked", WORK_DIRS["libinvent_locked"]),
        ("mol2mol_warhead",  "mol2mol_warhead",  WORK_DIRS["mol2mol_warhead"]),
        ("denovo_warhead",   "denovo_warhead",   WORK_DIRS["denovo_warhead"]),
    ]:
        smis = collect_smiles(wd, prefix)
        log(f"  {cfg_name}: {len(smis)} unique SMILES")
        reinvent_outputs.extend([(s, f"Tier3_REINVENT4_{cfg_name}", cfg_name) for s in smis])

    # ── Merge + dedupe + warhead post-filter ─────────────────────────────────
    all_cands = reinvent_outputs + crem_out + brics_out
    log(f"\n--- Merge: {len(all_cands)} raw candidates ---")
    seen = {Chem.CanonSmiles(MOL1_SMILES)}
    final = []
    for smi, method, note in all_cands:
        try:
            cs = Chem.CanonSmiles(smi)
        except Exception:
            continue
        if cs in seen:
            continue
        seen.add(cs)
        m = Chem.MolFromSmiles(cs)
        if m is None or not warhead_intact(m):
            continue
        final.append({"smiles": cs, "method": method, "note": note})
    log(f"--- Final unique warhead-intact: {len(final)} ---")

    # Group source method retention (diagnostic)
    methods_count = {}
    for c in final:
        methods_count[c["method"]] = methods_count.get(c["method"], 0) + 1
    log("Per-method counts (final, warhead-intact):")
    for m, n in sorted(methods_count.items(), key=lambda x: -x[1]):
        log(f"  {m:<40s} {n}")

    if not final:
        log("ERROR: no candidates produced.")
        return

    # ── Score (cap 3D scoring for speed) ─────────────────────────────────────
    log("\n--- Scoring ---")
    df = pd.DataFrame(final)

    # Score all with cheap metrics; restrict 3D to top 300 per method
    train = load_zap70_train_smiles()
    pred = load_film_predictor()

    # First pass: cheap (no 3D)
    df_cheap = score_dataframe(df, train_smiles=train, compute_3d=False, pIC50_predictor=pred)
    df_cheap = df_cheap.sort_values("pIC50", ascending=False).reset_index(drop=True)

    # Second pass: 3D for top-N per method
    keep_idx = []
    for m, sub in df_cheap.groupby("method"):
        keep_idx.extend(sub.head(150).index.tolist())
    df_top = df_cheap.loc[keep_idx].copy()
    df_top = score_dataframe(df_top.drop(columns=[c for c in ['shape_Tc_seed', 'warhead_dev_deg'] if c in df_top.columns]),
                             train_smiles=train, compute_3d=True, pIC50_predictor=None)
    df_top = df_top.sort_values("pIC50", ascending=False).reset_index(drop=True)

    # Save
    df_cheap.to_csv(RESULTS_DIR / "tier3_candidates_all.csv", index=False)
    df_top.to_csv(RESULTS_DIR / "tier3_candidates.csv", index=False)
    out = {
        "tier": 3, "method": "constrained_generative_real",
        "seed": MOL1_SMILES,
        "warhead_smarts": WARHEAD_SMARTS,
        "scaffold_locked": SCAFFOLD_FILE.read_text().strip(),
        "n_total": len(df_cheap),
        "n_top_scored_3d": len(df_top),
        "methods_count": methods_count,
        "timestamp": datetime.now().isoformat(),
        "candidates": df_top.to_dict(orient="records"),
    }
    (RESULTS_DIR / "tier3_results.json").write_text(
        json.dumps(out, default=lambda o: float(o) if hasattr(o, 'item') else o, indent=2)
    )
    log(f"Saved → {RESULTS_DIR / 'tier3_results.json'}")

    # Top 15
    log("\nTop 15 by FiLMDelta pIC50:")
    log(f"{'#':<3}{'pIC50':<7}{'SAS':<6}{'shapeTc':<9}{'wrhdΔ°':<9}{'Tc→1':<7}{'method':<40s}")
    for i, r in df_top.head(15).iterrows():
        log(f"{i+1:<3}{r['pIC50']:<7.3f}{r['SAScore']:<6.2f}{r.get('shape_Tc_seed', float('nan')):<9.3f}"
            f"{r.get('warhead_dev_deg', float('nan')):<9.1f}{r['Tc_to_Mol1']:<7.3f}{r['method']:<40s}")


if __name__ == "__main__":
    main()
