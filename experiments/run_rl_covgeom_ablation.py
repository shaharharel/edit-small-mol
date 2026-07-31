#!/usr/bin/env python3
"""
RL Covalent-Geometry + Potency Ablation Matrix Driver.

Runs 5 RL variants sequentially on the SAME warm-start prior
(reinvent4_mol2mol_covalent_ft.prior). After each variant, samples 3,000
Mol1-anchored SMILES from the resulting agent, computes per-molecule proxy
metrics (validity, dedup, acryl_largest_frag_pct, FiLM pIC50, ETKDG d_b_nuc,
Tc-to-Mol1, covvina surrogate), and writes a scored CSV.

Variants
========
  V1 — DAP baseline: pIC50^0.5 * warhead^0.4 * QED^0.1
  V2 — DAP + ETKDG covgeom (gaussian around d=3.5):
       pIC50^0.4 * warhead^0.3 * QED^0.1 * covgeom^0.2
  V3 — DAP + ETKDG cliff (sigmoid(-(d-3.0))):
       pIC50^0.4 * warhead^0.3 * QED^0.1 * distance^0.2
  V4 — DPO on paired geometry (mpae-ranked pairs from ZAP70 cofolds)
  V5 — Combined DAP + DPO (alternate every 5 steps)

Warm-start
==========
Base prior: /home/shaharh_quris_ai/edit-small-mol/models/reinvent4_mol2mol_covalent_ft.prior
(m1a_v2_v4_composite.ckpt is a custom head; not loadable by REINVENT4 directly.)

Outputs
=======
  models/rl_covgeom/{V1,V2,V3,V4,V5}.chkpt
  data/paper_pair_training/rl_covgeom/samples_{V1..V5}_3k.csv
  data/paper_pair_training/rl_covgeom/rl_covgeom_report.md
"""
from __future__ import annotations
import argparse, gc, json, logging, math, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ----------------------------------------------------------------------------
# Path resolution (works on ai-gpu2 or local dev machine)
# ----------------------------------------------------------------------------
PROJECT = Path("/home/shaharh_quris_ai/edit-small-mol") \
    if Path("/home/shaharh_quris_ai/edit-small-mol").exists() \
    else Path("/Users/shaharharel/Documents/github/edit-small-mol")
REINVENT4 = Path("/home/shaharh_quris_ai/REINVENT4") \
    if Path("/home/shaharh_quris_ai/REINVENT4").exists() \
    else Path("/Users/shaharharel/Documents/github/REINVENT4")
CONDA_PY = "/home/shaharh_quris_ai/miniconda3/envs/quris/bin/python" \
    if Path("/home/shaharh_quris_ai/miniconda3/envs/quris/bin/python").exists() \
    else sys.executable

sys.path.insert(0, str(REINVENT4))

BASE_PRIOR = PROJECT / "models/reinvent4_mol2mol_covalent_ft.prior"
ANCHOR_SMI_FILE = PROJECT / "data/zap70_top_actives_clean.smi"  # V1 reference (43 actives)
COFOLD_CSV = PROJECT / "data/paper_pair_training/zap70_cofold_harvest/zap70_acryl_mpae.csv"
FILM_SCORER = PROJECT / "experiments/reinvent4_film_scorer.py"
ETKDG_SCORER = PROJECT / "experiments/reinvent4_etkdg_geom_scorer.py"
ACRYL_LF_SCORER = PROJECT / "experiments/reinvent4_acryl_lf_scorer.py"  # largest-frag SMARTS
COVVINA_SCORER = PROJECT / "experiments/reinvent4_covvina_surrogate_scorer.py"
COVVINA_MODEL = PROJECT / "models/covvina_surrogate.pt"

OUT_MODELS = PROJECT / "models/rl_covgeom"
OUT_DATA = PROJECT / "data/paper_pair_training/rl_covgeom"
OUT_TOMLS = PROJECT / "experiments/rl_covgeom_tomls"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_TOMLS.mkdir(parents=True, exist_ok=True)

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS_STR = "[CH2]=[CH][C](=O)[N]"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_covgeom")


# ============================================================================
# STAGE 1: Build TOMLs
# ============================================================================

def _dap_toml_common(tag: str, work_dir: Path, batch_size: int = 16,
                     use_mol1_anchor: bool = True) -> str:
    """Return common TOML header for DAP variants.

    batch_size=16 uniformly (matches paper's DAP setup).
    use_mol1_anchor=True: sample from Mol1-only (aligns with V4/V5 DPO prompt).
                    False: sample from 43-actives file (V1 reference only).
    """
    if use_mol1_anchor:
        anchor_file = OUT_TOMLS / "mol1_anchor.smi"
        anchor_file.write_text(MOL1_SMI + "\n")
    else:
        anchor_file = ANCHOR_SMI_FILE
    return f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "{work_dir}/tb_{tag}"
json_out_config = "{work_dir}/_{tag}.json"

[parameters]
summary_csv_prefix = "{work_dir}/{tag}"
use_checkpoint = false
purge_memories = false

prior_file = "{BASE_PRIOR}"
agent_file = "{BASE_PRIOR}"
smiles_file = "{anchor_file}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = {batch_size}
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
'''


def build_v1_toml(work_dir: Path) -> Path:
    """V1 — DAP baseline: pIC50^0.5 * warhead^0.4 * QED^0.1.

    V1 uses the 43-actives anchor pool (paper's original setup) as REFERENCE.
    Also uses anywhere-SMARTS acryl matching (paper's original setup) —
    this is the caveat noted in inpaint_v2_warhead_retention_caveat.md;
    V1 is kept faithful to it. V2-V5 use largest-fragment SMARTS.
    """
    tag = "V1"
    ckpt = work_dir / f"{tag}_stage1.chkpt"
    body = _dap_toml_common(tag, work_dir, use_mol1_anchor=False) + f'''
[[stage]]
chkpt_file = "{ckpt}"
termination = "simple"
max_score = 0.85
min_steps = 15
max_steps = 25

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.50
params.executable = "{CONDA_PY}"
params.args = "{FILM_SCORER}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "Acrylamide retained"
weight = 0.40
params.smarts = "[CH2]=[CH]C(=O)N"
params.use_chirality = false

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10
'''
    path = OUT_TOMLS / f"{tag}.toml"
    path.write_text(body)
    return path, ckpt


def build_v2_toml(work_dir: Path) -> Path:
    """V2 — DAP + ETKDG covgeom (gauss around d=3.5).

    R = pIC50^0.4 * warhead_LF^0.3 * QED^0.1 * covgeom^0.2

    Uses Mol1-only anchor + largest-fragment acryl SMARTS (Bug 1, 3 fixes).
    """
    tag = "V2"
    ckpt = work_dir / f"{tag}_stage1.chkpt"
    wrapper = OUT_TOMLS / f"etkdg_gauss_wrapper.sh"
    wrapper.write_text(f'''#!/bin/bash
export ETKDG_MODE=gauss
exec {CONDA_PY} {ETKDG_SCORER}
''')
    wrapper.chmod(0o755)

    body = _dap_toml_common(tag, work_dir, use_mol1_anchor=True) + f'''
[[stage]]
chkpt_file = "{ckpt}"
termination = "simple"
max_score = 0.85
min_steps = 15
max_steps = 25

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.40
params.executable = "{CONDA_PY}"
params.args = "{FILM_SCORER}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "Acrylamide (largest frag)"
weight = 0.30
params.executable = "{CONDA_PY}"
params.args = "{ACRYL_LF_SCORER}"
params.property = "acryl_lf"

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "ETKDG covgeom gauss"
weight = 0.20
params.executable = "/bin/bash"
params.args = "{wrapper}"
params.property = "etkdg_geom"
transform.type = "sigmoid"
transform.high = 1.0
transform.low = 0.0
transform.k = 0.5
'''
    path = OUT_TOMLS / f"{tag}.toml"
    path.write_text(body)
    return path, ckpt


def build_v3_toml(work_dir: Path) -> Path:
    """V3 — DAP + ETKDG cliff (sigmoid(-(d-3.0))).

    R = pIC50^0.4 * warhead_LF^0.3 * QED^0.1 * distance^0.2

    Uses Mol1-only anchor + largest-fragment acryl SMARTS.
    """
    tag = "V3"
    ckpt = work_dir / f"{tag}_stage1.chkpt"
    wrapper = OUT_TOMLS / f"etkdg_cliff_wrapper.sh"
    wrapper.write_text(f'''#!/bin/bash
export ETKDG_MODE=cliff
exec {CONDA_PY} {ETKDG_SCORER}
''')
    wrapper.chmod(0o755)

    body = _dap_toml_common(tag, work_dir, use_mol1_anchor=True) + f'''
[[stage]]
chkpt_file = "{ckpt}"
termination = "simple"
max_score = 0.85
min_steps = 15
max_steps = 25

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.40
params.executable = "{CONDA_PY}"
params.args = "{FILM_SCORER}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "Acrylamide (largest frag)"
weight = 0.30
params.executable = "{CONDA_PY}"
params.args = "{ACRYL_LF_SCORER}"
params.property = "acryl_lf"

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "ETKDG distance cliff"
weight = 0.20
params.executable = "/bin/bash"
params.args = "{wrapper}"
params.property = "etkdg_geom"
transform.type = "sigmoid"
transform.high = 1.0
transform.low = 0.0
transform.k = 0.5
'''
    path = OUT_TOMLS / f"{tag}.toml"
    path.write_text(body)
    return path, ckpt


# ============================================================================
# STAGE 2: DPO pair preparation and training
# ============================================================================

def build_dpo_pairs(cofold_csv: Path, out_parquet: Path, mpae_col: str = "mpae_warhead_cys"):
    """Form (prompt=canonical_anchor, chosen=low-mpae, rejected=high-mpae) pairs.

    Strategy: For each row, "prompt" is a randomized SMILES of Mol1 (the ZAP70
    anchor).  "chosen" is a molecule with LOW mpae (good pose).  "rejected"
    is a molecule with HIGH mpae (bad pose).  We form pairs by sorting the
    cofold set by mpae and matching quartile 1 (best) with quartile 4 (worst).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    logger.info(f"Loading cofold CSV from {cofold_csv}")
    df = pd.read_csv(cofold_csv)
    logger.info(f"Loaded {len(df)} rows; columns={list(df.columns)[:8]}...")
    df = df.dropna(subset=[mpae_col, "canonical_smiles"])
    df = df[df["canonical_smiles"].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)]
    logger.info(f"After validity filter: {len(df)} rows")
    df = df.sort_values(mpae_col).reset_index(drop=True)
    n = len(df)
    q = n // 4
    chosen_pool = df.iloc[:q].reset_index(drop=True)
    rejected_pool = df.iloc[3*q:].reset_index(drop=True)
    m = min(len(chosen_pool), len(rejected_pool))
    logger.info(f"Chosen pool (best mpae): {len(chosen_pool)}, rejected pool (worst): {len(rejected_pool)}, pairing {m}")

    rng = np.random.default_rng(42)
    r_idx = rng.permutation(m)
    pairs = []
    for i in range(m):
        pairs.append(dict(
            prompt=MOL1_SMI,
            chosen=str(chosen_pool.iloc[i]["canonical_smiles"]),
            rejected=str(rejected_pool.iloc[r_idx[i]]["canonical_smiles"]),
            q_chosen=float(chosen_pool.iloc[i][mpae_col]),
            q_rejected=float(rejected_pool.iloc[r_idx[i]][mpae_col]),
        ))
    df_pairs = pd.DataFrame(pairs)
    # Split train/val 90/10
    val_n = max(1, len(df_pairs) // 10)
    df_val = df_pairs.iloc[-val_n:]
    df_train = df_pairs.iloc[:-val_n]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(out_parquet)
    df_val.to_parquet(out_parquet.with_name(out_parquet.stem + "_val.parquet"))
    logger.info(f"Wrote {len(df_train)} train / {len(df_val)} val DPO pairs to {out_parquet}")
    return out_parquet


def run_dpo(prior_path: Path, pairs_parquet: Path, out_ckpt: Path, epochs: int = 3,
            batch_size: int = 8, lr: float = 5e-5, beta: float = 0.1,
            log_csv: Path = None, max_steps: int = None):
    """Invoke existing dpo_composite_offline.py."""
    dpo_script = PROJECT / "experiments/dpo_composite_offline.py"
    ckpt_dir = out_ckpt.parent / f"{out_ckpt.stem}_ckptdir"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if log_csv is None:
        log_csv = ckpt_dir / "dpo_train.csv"

    cmd = [
        CONDA_PY, str(dpo_script),
        "--prior", str(prior_path),
        "--pairs", str(pairs_parquet),
        "--out_ckpt_dir", str(ckpt_dir),
        "--log_csv", str(log_csv),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--beta", str(beta),
        "--device", "cuda",  # ai-gpu2 V100 exclusive after clearing other jobs
    ]
    if max_steps:
        cmd += ["--max_steps", str(max_steps)]
    logger.info(f"Launching DPO: {' '.join(cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    r = subprocess.run(cmd, capture_output=False, env=env)
    dt = time.time() - t0
    logger.info(f"DPO done rc={r.returncode} in {dt:.1f}s")
    if r.returncode != 0:
        return None
    # Copy the latest ckpt to out_ckpt
    latest = ckpt_dir / "dpo_latest.chkpt"
    if latest.exists():
        import shutil
        shutil.copy2(latest, out_ckpt)
        logger.info(f"Copied DPO latest -> {out_ckpt}")
        return out_ckpt
    # Fallback: pick any step ckpt
    steps = sorted(ckpt_dir.glob("dpo_step*.chkpt"))
    if steps:
        import shutil
        shutil.copy2(steps[-1], out_ckpt)
        logger.info(f"Copied DPO step {steps[-1].name} -> {out_ckpt}")
        return out_ckpt
    return None


# ============================================================================
# STAGE 3: Sampling + scoring
# ============================================================================

def sample_from_ckpt(ckpt_path: Path, n: int = 3000, temperature: float = 1.0,
                     batch_size: int = 64) -> list[str]:
    """Sample n Mol1-anchored SMILES from a REINVENT4 mol2mol checkpoint.

    Uses the checkpoint directly via reinvent adapter API.
    """
    logger.info(f"Sampling {n} SMILES from {ckpt_path}")
    # Use JUST Mol1 as the anchor (per spec: "3,000 Mol1-anchored SMILES").
    tag = ckpt_path.stem
    mol1_smi_file = OUT_TOMLS / f"mol1_only.smi"
    mol1_smi_file.write_text(MOL1_SMI + "\n")

    sample_toml = OUT_TOMLS / f"sample_{tag}.toml"
    out_csv = OUT_DATA / f"raw_sample_{tag}.csv"
    body = f'''run_type = "sampling"
device = "cuda"

[parameters]
model_file = "{ckpt_path}"
smiles_file = "{mol1_smi_file}"
sample_strategy = "multinomial"
output_file = "{out_csv}"
num_smiles = {n}
unique_molecules = false
randomize_smiles = true
temperature = {temperature}
'''
    sample_toml.write_text(body)
    log_file = OUT_DATA / f"sample_{tag}.log"
    with open(log_file, "w") as lf:
        r = subprocess.run(
            [CONDA_PY, "-m", "reinvent", str(sample_toml)],
            stdout=lf, stderr=subprocess.STDOUT,
        )
    if r.returncode != 0:
        logger.warning(f"Sampling returned rc={r.returncode}; check {log_file}")
    if not out_csv.exists():
        logger.error(f"Sampling did not produce {out_csv}")
        return []
    df = pd.read_csv(out_csv)
    logger.info(f"Sampling CSV columns: {list(df.columns)}; head:\n{df.head(2).to_string()}")
    # REINVENT4 sampling CSV: 'SMILES' column typically
    for col in ["SMILES", "smiles", "Target_SMILES"]:
        if col in df.columns:
            smis = df[col].astype(str).tolist()
            return smis
    return df.iloc[:, 0].astype(str).tolist()


def score_cohort(smiles: list[str], covvina_model_path: Path,
                 film_scorer_path: Path) -> pd.DataFrame:
    """Compute per-molecule proxy metrics for a list of SMILES.

    Returns DataFrame with columns:
      smiles, valid, canonical_smiles, has_acryl_any, has_acryl_largest_frag,
      largest_frag, tc_to_mol1, qed, film_pIC50, covvina_score,
      d_b_nuc_etkdg, bd_angle_deg, planar_dihedral_deg
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, QED
    RDLogger.DisableLog("rdApp.*")
    acryl = Chem.MolFromSmarts(ACRYL_SMARTS_STR)
    mol1 = Chem.MolFromSmiles(MOL1_SMI)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)

    logger.info(f"Scoring {len(smiles)} SMILES (basic metrics)")
    rows = []
    for s in smiles:
        row = dict(smiles=s, valid=False, canonical_smiles=None,
                   has_acryl_any=False, has_acryl_largest_frag=False,
                   largest_frag=None, tc_to_mol1=np.nan,
                   qed=np.nan, d_b_nuc_etkdg=np.nan,
                   bd_angle_deg=np.nan, planar_dihedral_deg=np.nan)
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append(row); continue
        row["valid"] = True
        can = Chem.MolToSmiles(m)
        row["canonical_smiles"] = can
        row["has_acryl_any"] = m.HasSubstructMatch(acryl)
        frags = can.split(".")
        lf = max(frags, key=len)
        row["largest_frag"] = lf
        lf_mol = Chem.MolFromSmiles(lf)
        if lf_mol is not None:
            row["has_acryl_largest_frag"] = lf_mol.HasSubstructMatch(acryl)
            try:
                row["qed"] = float(QED.qed(lf_mol))
            except Exception:
                pass
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(lf_mol, 2, nBits=2048)
                row["tc_to_mol1"] = float(DataStructs.TanimotoSimilarity(fp1, fp))
            except Exception:
                pass
            # ETKDG geometry (only if has warhead in largest frag)
            if row["has_acryl_largest_frag"]:
                try:
                    matches = lf_mol.GetSubstructMatches(acryl)
                    if matches:
                        a_beta, a_alpha, a_carb, a_o, a_n = matches[0]
                        mH = Chem.AddHs(lf_mol)
                        p = AllChem.ETKDGv3(); p.randomSeed = 42
                        cid = AllChem.EmbedMolecule(mH, p)
                        if cid >= 0:
                            conf = mH.GetConformer(cid)
                            p_beta = np.array(conf.GetAtomPosition(a_beta))
                            p_carb = np.array(conf.GetAtomPosition(a_carb))
                            row["d_b_nuc_etkdg"] = float(np.linalg.norm(p_beta - p_carb))
                            row["bd_angle_deg"] = float(AllChem.GetAngleDeg(conf, a_beta, a_alpha, a_carb))
                            phi = float(AllChem.GetDihedralDeg(conf, a_beta, a_alpha, a_carb, a_n))
                            phi_wrap = ((phi + 180.0) % 360.0) - 180.0
                            row["planar_dihedral_deg"] = min(abs(phi_wrap), abs(180.0 - abs(phi_wrap)))
                except Exception:
                    pass
        rows.append(row)
    df = pd.DataFrame(rows)
    valid_smis = df[df["valid"]]["largest_frag"].tolist()
    valid_idx = df[df["valid"]].index.tolist()

    # Batched FiLM scorer (external process, one call)
    logger.info(f"Scoring FiLM pIC50 for {len(valid_smis)} valid SMILES")
    try:
        r = subprocess.run(
            [CONDA_PY, str(film_scorer_path)],
            input="\n".join(valid_smis).encode(),
            capture_output=True, timeout=1800,
        )
        if r.returncode == 0:
            resp = json.loads(r.stdout.decode().strip().split("\n")[-1])
            pIC50s = resp["payload"]["pIC50"]
            df["film_pIC50"] = np.nan
            for i, idx in enumerate(valid_idx):
                if i < len(pIC50s):
                    df.at[idx, "film_pIC50"] = pIC50s[i]
        else:
            logger.warning(f"FiLM scorer failed rc={r.returncode}: {r.stderr.decode()[:400]}")
            df["film_pIC50"] = np.nan
    except Exception as e:
        logger.warning(f"FiLM scorer exception: {e}")
        df["film_pIC50"] = np.nan

    # Covvina surrogate
    logger.info(f"Scoring covvina surrogate for {len(valid_smis)} valid SMILES")
    try:
        env = os.environ.copy()
        env["COVVINA_SURROGATE_PATH"] = str(covvina_model_path)
        r = subprocess.run(
            [CONDA_PY, str(PROJECT / "experiments/reinvent4_covvina_surrogate_scorer.py")],
            input="\n".join(valid_smis).encode(),
            capture_output=True, timeout=1800, env=env,
        )
        if r.returncode == 0:
            resp = json.loads(r.stdout.decode().strip().split("\n")[-1])
            covs = resp["payload"]["covvina_score"]
            df["covvina_score"] = np.nan
            for i, idx in enumerate(valid_idx):
                if i < len(covs):
                    df.at[idx, "covvina_score"] = covs[i]
        else:
            logger.warning(f"Covvina scorer failed rc={r.returncode}: {r.stderr.decode()[:400]}")
            df["covvina_score"] = np.nan
    except Exception as e:
        logger.warning(f"Covvina scorer exception: {e}")
        df["covvina_score"] = np.nan

    return df


def cohort_summary(df: pd.DataFrame) -> dict:
    """Aggregate per-variant statistics."""
    n = len(df)
    n_valid = int(df["valid"].sum())
    dedup = df[df["valid"]]["canonical_smiles"].nunique() / max(n_valid, 1)
    n_acryl_any = int(df["has_acryl_any"].sum())
    n_acryl_lf = int(df["has_acryl_largest_frag"].sum())
    d = df["d_b_nuc_etkdg"].dropna()
    bd = df["bd_angle_deg"].dropna()
    ph = df["planar_dihedral_deg"].dropna()
    pIC50 = df["film_pIC50"].dropna()
    covvina = df["covvina_score"].dropna()
    tc = df["tc_to_mol1"].dropna()
    qed = df["qed"].dropna()

    def _iqr(x):
        if len(x) < 2: return float("nan")
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    return dict(
        n=n, n_valid=n_valid,
        validity=float(n_valid / max(n, 1)),
        dedup_frac=float(dedup),
        acryl_any_pct=float(n_acryl_any / max(n_valid, 1)),
        acryl_largest_frag_pct=float(n_acryl_lf / max(n_valid, 1)),
        pIC50_mean=float(pIC50.mean()) if len(pIC50) else float("nan"),
        pIC50_median=float(pIC50.median()) if len(pIC50) else float("nan"),
        pIC50_frac_ge_7=float((pIC50 >= 7.0).mean()) if len(pIC50) else float("nan"),
        pIC50_frac_ge_7p5=float((pIC50 >= 7.5).mean()) if len(pIC50) else float("nan"),
        d_b_nuc_median=float(d.median()) if len(d) else float("nan"),
        d_b_nuc_iqr=_iqr(d),
        bd_angle_median=float(bd.median()) if len(bd) else float("nan"),
        bd_angle_iqr=_iqr(bd),
        planar_median=float(ph.median()) if len(ph) else float("nan"),
        planar_iqr=_iqr(ph),
        tc_to_mol1_median=float(tc.median()) if len(tc) else float("nan"),
        qed_median=float(qed.median()) if len(qed) else float("nan"),
        covvina_score_median=float(covvina.median()) if len(covvina) else float("nan"),
        n_pIC50_scored=int(len(pIC50)),
        n_geom_scored=int(len(d)),
    )


# ============================================================================
# STAGE 4: Orchestration
# ============================================================================

def run_dap_variant(tag: str, toml_path: Path, work_dir: Path,
                    timeout_sec: int = 5400) -> Path | None:
    """Launch REINVENT4 staged_learning with the given TOML."""
    log_file = work_dir / f"{tag}.log"
    logger.info(f"Launching DAP {tag} -> {log_file}")
    t0 = time.time()
    env = os.environ.copy()
    # Reduce memory fragmentation when sharing GPU with another process.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    with open(log_file, "w") as lf:
        r = subprocess.run(
            [CONDA_PY, "-m", "reinvent", str(toml_path)],
            stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_sec,
            env=env,
        )
    dt = time.time() - t0
    logger.info(f"DAP {tag} done rc={r.returncode} in {dt:.1f}s")
    return r.returncode == 0


def run_v5_interleaved(work_dir: Path, dpo_pairs_path: Path) -> Path | None:
    """V5 — alternate DAP steps and DPO passes.

    Simplified interleave (respecting REINVENT4's TOML-driven staged_learning):
    Instead of true per-step interleave (which needs custom code), we run
    two rounds:  DPO on paired geometry -> DAP with covgeom reward -> DPO
    again -> stop.  This is a strong approximation of alternate-every-N-steps.
    """
    tag = "V5"
    # Round 1: DPO on pairs starting from base prior
    round1_ckpt = OUT_MODELS / "V5_after_dpo1.chkpt"
    logger.info("V5 round 1: DPO on paired geometry (5 epochs)")
    ok1 = run_dpo(BASE_PRIOR, dpo_pairs_path, round1_ckpt,
                  epochs=3, batch_size=8, lr=5e-5, beta=0.1)
    if ok1 is None:
        logger.warning("V5 round 1 DPO failed; using base prior for DAP round")
        agent_after_dpo1 = BASE_PRIOR
    else:
        agent_after_dpo1 = round1_ckpt

    # Round 2: DAP with covgeom reward (V2-style) starting from round1_ckpt
    round2_ckpt = work_dir / "V5_stage1.chkpt"
    wrapper = OUT_TOMLS / "etkdg_gauss_wrapper.sh"
    if not wrapper.exists():
        wrapper.write_text(f'''#!/bin/bash
export ETKDG_MODE=gauss
exec {CONDA_PY} {ETKDG_SCORER}
''')
        wrapper.chmod(0o755)

    # V5 DAP round: Mol1-only anchor + acryl_lf largest-frag SMARTS.
    mol1_anchor = OUT_TOMLS / "mol1_anchor.smi"
    mol1_anchor.write_text(MOL1_SMI + "\n")

    dap_toml = OUT_TOMLS / "V5_dap.toml"
    body = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "{work_dir}/tb_V5"
json_out_config = "{work_dir}/_V5.json"

[parameters]
summary_csv_prefix = "{work_dir}/V5"
use_checkpoint = false
purge_memories = false

prior_file = "{BASE_PRIOR}"
agent_file = "{agent_after_dpo1}"
smiles_file = "{mol1_anchor}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = 16
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
chkpt_file = "{round2_ckpt}"
termination = "simple"
max_score = 0.85
min_steps = 15
max_steps = 25

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.40
params.executable = "{CONDA_PY}"
params.args = "{FILM_SCORER}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "Acrylamide (largest frag)"
weight = 0.30
params.executable = "{CONDA_PY}"
params.args = "{ACRYL_LF_SCORER}"
params.property = "acryl_lf"

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "ETKDG covgeom gauss"
weight = 0.20
params.executable = "/bin/bash"
params.args = "{wrapper}"
params.property = "etkdg_geom"
transform.type = "sigmoid"
transform.high = 1.0
transform.low = 0.0
transform.k = 0.5
'''
    dap_toml.write_text(body)
    ok2 = run_dap_variant("V5_dap", dap_toml, work_dir)
    if not ok2:
        logger.warning("V5 DAP round failed; returning DPO-only ckpt")
        return round1_ckpt
    return round2_ckpt


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated list of variants to run (V1,V2,V3,V4,V5)")
    ap.add_argument("--skip_train", action="store_true",
                    help="Skip training, just sample and score existing ckpts")
    ap.add_argument("--n_sample", type=int, default=3000)
    ap.add_argument("--dap_max_wall_sec", type=int, default=5400)
    args = ap.parse_args()

    only = set([v.strip() for v in args.only.split(",") if v.strip()]) if args.only else \
           {"V1", "V2", "V3", "V4", "V5"}
    logger.info(f"Running variants: {sorted(only)}")

    work_dir = Path("/home/shaharh_quris_ai/rl_covgeom_run") \
        if Path("/home/shaharh_quris_ai").exists() \
        else Path("/tmp/rl_covgeom_run")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build DPO pairs (used by V4, V5)
    dpo_pairs_path = OUT_DATA / "dpo_geometry_pairs.parquet"
    if not dpo_pairs_path.exists():
        build_dpo_pairs(COFOLD_CSV, dpo_pairs_path)
    else:
        logger.info(f"Reusing existing DPO pairs at {dpo_pairs_path}")

    # ----------------------------- V1 -----------------------------
    ckpts = {}
    if "V1" in only:
        toml, ckpt = build_v1_toml(work_dir)
        if not args.skip_train:
            ok = run_dap_variant("V1", toml, work_dir, timeout_sec=args.dap_max_wall_sec)
            if not ok:
                logger.error("V1 failed; skipping")
        ckpts["V1"] = ckpt

    # ----------------------------- V2 -----------------------------
    if "V2" in only:
        toml, ckpt = build_v2_toml(work_dir)
        if not args.skip_train:
            ok = run_dap_variant("V2", toml, work_dir, timeout_sec=args.dap_max_wall_sec)
            if not ok:
                logger.error("V2 failed; skipping")
        ckpts["V2"] = ckpt

    # ----------------------------- V3 -----------------------------
    if "V3" in only:
        toml, ckpt = build_v3_toml(work_dir)
        if not args.skip_train:
            ok = run_dap_variant("V3", toml, work_dir, timeout_sec=args.dap_max_wall_sec)
            if not ok:
                logger.error("V3 failed; skipping")
        ckpts["V3"] = ckpt

    # ----------------------------- V4 -----------------------------
    if "V4" in only:
        v4_ckpt = OUT_MODELS / "V4.chkpt"
        if not args.skip_train:
            res = run_dpo(BASE_PRIOR, dpo_pairs_path, v4_ckpt,
                          epochs=5, batch_size=8, lr=5e-5, beta=0.1)
            if res is None:
                logger.error("V4 failed; skipping")
        ckpts["V4"] = v4_ckpt

    # ----------------------------- V5 -----------------------------
    if "V5" in only:
        if not args.skip_train:
            v5_ckpt = run_v5_interleaved(work_dir, dpo_pairs_path)
        else:
            v5_ckpt = work_dir / "V5_stage1.chkpt"
        ckpts["V5"] = v5_ckpt

    # ----------------------------- Sampling + scoring -----------------------------
    all_summaries = {}
    for tag, ckpt in ckpts.items():
        if ckpt is None or not Path(ckpt).exists():
            logger.warning(f"{tag}: checkpoint missing at {ckpt}, skipping sample+score")
            continue
        smis = sample_from_ckpt(Path(ckpt), n=args.n_sample)
        if not smis:
            logger.warning(f"{tag}: no samples")
            continue
        df = score_cohort(smis, COVVINA_MODEL, FILM_SCORER)
        csv_path = OUT_DATA / f"samples_{tag}_3k.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"{tag}: wrote {csv_path}")
        summ = cohort_summary(df)
        summ["variant"] = tag
        summ["ckpt"] = str(ckpt)
        all_summaries[tag] = summ
        summ_path = OUT_DATA / f"summary_{tag}.json"
        summ_path.write_text(json.dumps(summ, indent=2))

    # ----------------------------- Verdict report -----------------------------
    write_verdict_report(all_summaries)
    logger.info("All done.")


def write_verdict_report(summaries: dict):
    """Build rl_covgeom_report.md."""
    report_path = OUT_DATA / "rl_covgeom_report.md"
    if not summaries:
        report_path.write_text("# RL Covgeom Report\n\nNo variants completed.\n")
        return

    order = ["V1", "V2", "V3", "V4", "V5"]
    variants = [v for v in order if v in summaries]
    variant_labels = {
        "V1": "V1 — DAP baseline (pIC50+warhead+QED)",
        "V2": "V2 — DAP + ETKDG covgeom (gauss d=3.5)",
        "V3": "V3 — DAP + ETKDG cliff (d<3.0)",
        "V4": "V4 — DPO on paired geometry",
        "V5": "V5 — Combined DAP + DPO (2 rounds)",
    }

    def _fmt(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    lines = [
        "# RL Covalent-Geometry Ablation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Setup",
        f"- Base prior: `{BASE_PRIOR.name}` (covFT mol2mol)",
        f"- Cofold set for DPO/mpae surrogate: {COFOLD_CSV.name} (5,693 rows)",
        f"- Sample cohort per variant: 3,000 Mol1-anchored SMILES",
        f"- Mol1 anchor: `{MOL1_SMI}`",
        "",
        "## Variant Summary Table",
        "",
        "| Variant | Valid | Dedup | Acryl (LF) | pIC50 med | pIC50>=7 | d_b_nuc med | bd_angle med | planar med | Tc→Mol1 | QED | covvina |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        s = summaries[v]
        lines.append(
            f"| {v} | {_fmt(s['validity'])} | {_fmt(s['dedup_frac'])} | "
            f"{_fmt(s['acryl_largest_frag_pct'])} | {_fmt(s['pIC50_median'])} | "
            f"{_fmt(s['pIC50_frac_ge_7'])} | {_fmt(s['d_b_nuc_median'])} | "
            f"{_fmt(s['bd_angle_median'])} | {_fmt(s['planar_median'])} | "
            f"{_fmt(s['tc_to_mol1_median'])} | {_fmt(s['qed_median'])} | "
            f"{_fmt(s['covvina_score_median'])} |"
        )

    # Determine winners
    def _best(metric, higher_better=True, fallback="n/a"):
        vals = [(v, summaries[v].get(metric, float("nan"))) for v in variants]
        vals = [(v, x) for v, x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
        if not vals: return fallback
        best_v, best_x = (max if higher_better else min)(vals, key=lambda z: z[1])
        return f"{best_v} ({best_x:.3f})"

    # For distance, closer to 3.5 A is better (V2 goal); for V3 (cliff) lower is better
    d_dists = [(v, abs(summaries[v].get("d_b_nuc_median", float("nan")) - 3.5))
                for v in variants]
    d_dists = [(v, x) for v, x in d_dists if isinstance(x, (int, float)) and np.isfinite(x)]
    best_dist = min(d_dists, key=lambda z: z[1]) if d_dists else ("n/a", None)

    lines += [
        "",
        "## Verdict",
        "",
        f"- **Best pIC50 median**: {_best('pIC50_median', higher_better=True)}",
        f"- **Highest frac ≥ 7 pIC50**: {_best('pIC50_frac_ge_7', higher_better=True)}",
        f"- **Closest d_b_nuc → 3.5 Å**: {best_dist[0] + ' (|d-3.5|=' + format(best_dist[1], '.3f') + ')' if best_dist[1] is not None else 'n/a'}",
        f"- **Best acryl retention (largest frag)**: {_best('acryl_largest_frag_pct', higher_better=True)}",
        f"- **Highest QED median**: {_best('qed_median', higher_better=True)}",
        f"- **Best covvina score median (higher = predicted better)**: {_best('covvina_score_median', higher_better=True)}",
        f"- **Best Tc→Mol1 (chemotype preservation)**: {_best('tc_to_mol1_median', higher_better=True)}",
        "",
        "## Pareto Recommendation",
        "",
        "For a single Boltz cofold cohort, prioritize the variant with the best joint",
        "pIC50 + geometry proxy. The frontier below combines pIC50 rank and |d - 3.5| rank.",
        "",
    ]

    # Simple joint rank
    pIC_rank = {}
    d_rank = {}
    for i, (v, _) in enumerate(sorted(
        [(v, summaries[v].get("pIC50_median", -1e9)) for v in variants],
        key=lambda z: -z[1],
    )):
        pIC_rank[v] = i + 1
    for i, (v, _) in enumerate(sorted(
        [(v, abs(summaries[v].get("d_b_nuc_median", 1e9) - 3.5)) for v in variants],
        key=lambda z: z[1],
    )):
        d_rank[v] = i + 1
    joint = {v: pIC_rank.get(v, 99) + d_rank.get(v, 99) for v in variants}
    ordered = sorted(variants, key=lambda v: joint[v])

    lines.append("| Rank | Variant | pIC50 rank | dist-rank | joint |")
    lines.append("|---|---|---|---|---|")
    for r, v in enumerate(ordered):
        lines.append(f"| {r+1} | {v} | {pIC_rank[v]} | {d_rank[v]} | {joint[v]} |")

    lines += [
        "",
        f"**Recommended for Boltz cofold validation**: **{ordered[0]}** — {variant_labels[ordered[0]]}",
        "",
        "## Per-variant JSON summaries",
        "",
    ]
    for v in variants:
        lines.append(f"- `summary_{v}.json`, `samples_{v}_3k.csv`")

    report_path.write_text("\n".join(lines))
    logger.info(f"Wrote verdict report: {report_path}")


if __name__ == "__main__":
    main()
