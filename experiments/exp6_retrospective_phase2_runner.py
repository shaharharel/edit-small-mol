#!/usr/bin/env python3
"""
EXP6 Retrospective LO — Phase 2 remote runner (executes on a100-b).

For each (target, strategy, iter) triple:
  - Write a target-aware mol2mol RL TOML using exp6_retrospective_phase2_scorer.py
    as the FiLMDelta+DirectAbs ensemble component, plus MatchingSubstructure
    on the target-specific warhead and QED.
  - Run REINVENT4 staged_learning.
  - Sample 5,500 from the resulting checkpoint.
  - Score the cohort with the same scorer + warhead match + Tc-to-drug (diagnostic).
  - Promote top-50 = 25 by predicted pIC50 + 25 by Murcko-unique scaffolds
    (all filtered for Tc>=0.3 to original anchor).
  - Use promoted_50 + original anchors as seeds for the next iter.

Saves all checkpoints + cohort CSVs under data/exp6_retrospective/<target>/.

At end, writes data/exp6_retrospective/_PHASE2_DONE so the local driver can stop the VM.
"""
from __future__ import annotations
import os
import sys
import json
import time
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")
os.environ.setdefault("RDK_DEPRECATION_WARNING", "off")

import numpy as np
import pandas as pd
import torch

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

HOME = Path(os.environ.get("HOME", "/home/shaharh_quris_ai"))
PROJ = HOME / "edit-small-mol"
DATA_ROOT = PROJ / "data" / "exp6_retrospective"
SCORER = PROJ / "experiments" / "exp6_retrospective_phase2_scorer.py"
PYTHON = HOME / "miniconda3" / "envs" / "quris" / "bin" / "python"
REINVENT = HOME / "miniconda3" / "envs" / "quris" / "bin" / "reinvent"
PRIOR = PROJ / "models" / "reinvent4_mol2mol_warhead_tokens_v2.prior"
DONE_FLAG = DATA_ROOT / "_PHASE2_DONE"

TARGETS = ["egfr_t790m", "btk", "kras_g12c"]
STRATEGIES = ["single_anchor", "anchor_pool"]
ITER_STEPS = {1: 50, 2: 25, 3: 25}
SAMPLE_N = 5500
TC_PROMOTE = 0.3
N_PROMOTE_TOTAL = 50
TIMEOUT_RL = 3600   # 60 min
TIMEOUT_SAMPLE = 1800  # 30 min

# Anchors per target (from Phase 1).
ANCHOR_INFO = {
    "egfr_t790m": {
        "anchor_smiles": "C=CC(=O)Nc1cccc(Oc2nc(Nc3ccc(N4CCN(C)CC4)cc3OC)ncc2Cl)c1",
        "drug_smiles": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    },
    "btk": {
        "anchor_smiles": "C=CC(=O)N1CCC(n2nc(-c3ccc(Oc4ccccn4)cc3)c3c(N)ncnc32)CC1",
        "drug_smiles": "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    },
    "kras_g12c": {
        "anchor_smiles": "C=CC(=O)N1CCN(c2ncnc3c(F)c(-c4c(O)cccc4F)c(Cl)cc23)CC1",
        "drug_smiles": "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C(C)C)c3nc(-c4c(O)cccc4F)c(F)cc23)[C@@H](C)C1",
    },
}


def log(msg):
    print(time.strftime("[%FT%TZ] [runner] ") + msg, flush=True)


def smi_to_fp(smi: str, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tc(smi_a: str, smi_b: str) -> float:
    fa, fb = smi_to_fp(smi_a), smi_to_fp(smi_b)
    if fa is None or fb is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def murcko(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return ""


def write_seed_smi(path: Path, smiles_list: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in smiles_list:
            if s:
                f.write(s.strip() + "\n")


def write_rl_toml(toml_path: Path, run_dir: Path, target_dir: Path,
                  seed_smi: Path, warhead_smarts_generic: str,
                  warhead_smarts_strict: str, n_steps: int,
                  tag: str):
    """Write a staged_learning TOML with the ensemble scorer + warhead + QED."""
    chkpt = run_dir / f"{tag}.chkpt"
    # We pass EXP6_TARGET_DIR via params.env (REINVENT4 supports env in ExternalProcess).
    # Fallback: prefix the executable with /usr/bin/env -i ... (not used here).
    toml = f"""run_type = "staged_learning"
device = "cuda"
tb_logdir = "{run_dir / 'tb'}"
json_out_config = "{run_dir / '_config.json'}"

[parameters]
summary_csv_prefix = "{tag}"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR}"
agent_file = "{PRIOR}"
smiles_file = "{seed_smi}"
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
chkpt_file = "{chkpt}"
termination = "simple"
max_score = 0.8
min_steps = {max(5, n_steps - 5)}
max_steps = {n_steps}

[stage.scoring]
type = "geometric_mean"

# --- target-specific FiLMDelta + DirectAbs ensemble ---
[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "Ensemble pIC50"
weight = 0.50
params.executable = "{PYTHON}"
params.args = "{SCORER}"
params.property = "pIC50"
# NOTE: EXP6_TARGET_DIR is exported in the parent shell before reinvent runs
# (see exp6_retrospective_phase2_runner.py run_reinvent env_extra) — REINVENT4
# ExternalProcess inherits the parent env and passes it to the scorer subprocess.
transform.type = "sigmoid"
transform.high = 8.0
transform.low = 6.0
transform.k = 0.5

# --- warhead retention (target-specific, generic acrylamide) ---
[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "Warhead retained"
weight = 0.30
params.smarts = "{warhead_smarts_generic}"
params.use_chirality = false

# --- QED ---
[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.20
"""
    toml_path.write_text(toml)


def write_sampling_toml(toml_path: Path, chkpt: Path, seed_smi: Path, out_csv: Path, n: int):
    toml = f"""run_type = "sampling"
device = "cuda"
json_out_config = "{toml_path.parent / '_sampling.json'}"

[parameters]
model_file = "{chkpt}"
smiles_file = "{seed_smi}"
sample_strategy = "multinomial"
temperature = 1.0
output_file = "{out_csv}"
num_smiles = {n}
unique_molecules = true
randomize_smiles = true
"""
    toml_path.write_text(toml)


def run_reinvent(toml: Path, label: str, env_extra: dict, timeout: int) -> bool:
    env = os.environ.copy()
    env.update(env_extra)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log(f"REINVENT [{label}] toml={toml.name} timeout={timeout}s")
    try:
        proc = subprocess.run(
            [str(REINVENT), str(toml), "-d", "cuda"],
            cwd=str(toml.parent),
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        rc = proc.returncode
        if rc != 0:
            log(f"  rc={rc} stderr_tail:\n{proc.stderr[-1200:]}")
            return False
        log(f"  rc=0 OK")
        return True
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s")
        return False


def score_cohort(smiles_list: List[str], target_dir: Path) -> List[float]:
    """Pipe SMILES through the same exp6 scorer and parse JSON output."""
    env = os.environ.copy()
    env["EXP6_TARGET_DIR"] = str(target_dir)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [str(PYTHON), str(SCORER)],
        input="\n".join(smiles_list),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        log(f"  scorer rc={proc.returncode} stderr_tail:\n{proc.stderr[-600:]}")
        return [0.0] * len(smiles_list)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return payload["payload"]["pIC50"]
    except Exception as e:
        log(f"  scorer JSON parse failed: {e} stdout_tail:\n{proc.stdout[-400:]}")
        return [0.0] * len(smiles_list)


def promote(cohort_df: pd.DataFrame, anchor_smiles: str, warhead_smarts: str,
            n_total: int = 50, tc_min: float = 0.3) -> pd.DataFrame:
    """top-25 by pred + top-25 by Murcko diversity, all filtered by Tc>=anchor."""
    df = cohort_df.copy()
    # warhead match
    pat = Chem.MolFromSmarts(warhead_smarts)
    keep = []
    for s in df["SMILES"]:
        m = Chem.MolFromSmiles(s)
        keep.append(bool(m and pat is not None and m.HasSubstructMatch(pat)))
    df["warhead_match"] = keep
    df = df[df["warhead_match"] & (df["tc_to_anchor"] >= tc_min)]
    if df.empty:
        return df
    df = df.sort_values("pred_pIC50", ascending=False).reset_index(drop=True)
    n_pred = n_total // 2
    top_pred = df.head(n_pred)
    # diversity selection — Murcko-unique not already in top_pred
    seen_scaffolds = set(top_pred["murcko"].tolist())
    div_pool = df[~df.index.isin(top_pred.index)]
    div = []
    for _, row in div_pool.iterrows():
        if row["murcko"] and row["murcko"] not in seen_scaffolds:
            div.append(row)
            seen_scaffolds.add(row["murcko"])
            if len(div) >= (n_total - n_pred):
                break
    div_df = pd.DataFrame(div) if div else pd.DataFrame(columns=df.columns)
    promoted = pd.concat([top_pred, div_df], ignore_index=True).drop_duplicates(subset=["SMILES"])
    return promoted.head(n_total)


def parse_sampling_csv(p: Path) -> List[str]:
    if not p.exists():
        return []
    df = pd.read_csv(p)
    # REINVENT4 sampling outputs columns like "SMILES" or "smiles"
    col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else df.columns[0])
    return [s for s in df[col].astype(str).tolist() if s and s.lower() != "nan"]


def run_target(target: str):
    target_dir = DATA_ROOT / target
    if not target_dir.exists():
        log(f"SKIP {target} — dir missing")
        return
    warhead = json.loads((target_dir / "warhead_smarts.json").read_text())
    anchor_smi = ANCHOR_INFO[target]["anchor_smiles"]
    drug_smi = ANCHOR_INFO[target]["drug_smiles"]
    anchor_pool_df = pd.read_csv(target_dir / "anchor_pool_strategy_b.csv")
    pool_smis = anchor_pool_df["smiles"].astype(str).tolist()

    for strategy in STRATEGIES:
        log(f"=== {target} / {strategy} ===")
        if strategy == "single_anchor":
            seeds_orig = [anchor_smi]
        else:
            seeds_orig = pool_smis[:100]

        prev_seeds = list(seeds_orig)
        for it in (1, 2, 3):
            tag = f"{target}_{strategy}_iter{it}"
            run_dir = target_dir / f"iter{it}_{strategy}"
            run_dir.mkdir(parents=True, exist_ok=True)
            seed_smi = run_dir / "seeds.smi"
            write_seed_smi(seed_smi, prev_seeds)

            toml = run_dir / f"{tag}_rl.toml"
            write_rl_toml(toml, run_dir, target_dir, seed_smi,
                          warhead["smarts_generic"], warhead["smarts_strict"],
                          n_steps=ITER_STEPS[it], tag=tag)

            ok = run_reinvent(toml, f"{tag} RL", env_extra={"EXP6_TARGET_DIR": str(target_dir)},
                              timeout=TIMEOUT_RL)
            if not ok:
                log(f"  RL failed for {tag} — continuing to sampling with raw prior")
                # fall back: use prior as model_file
                chkpt_path = PRIOR
            else:
                chkpt_path = run_dir / f"{tag}.chkpt"
                if not chkpt_path.exists():
                    # REINVENT4 may write multiple checkpoints; pick latest
                    cands = sorted(run_dir.glob(f"*.chkpt"))
                    chkpt_path = cands[-1] if cands else PRIOR

            sample_toml = run_dir / f"{tag}_sample.toml"
            sample_csv = run_dir / f"{tag}_sample.csv"
            write_sampling_toml(sample_toml, chkpt_path, seed_smi, sample_csv, SAMPLE_N)
            ok = run_reinvent(sample_toml, f"{tag} sampling", env_extra={}, timeout=TIMEOUT_SAMPLE)
            if not ok or not sample_csv.exists():
                log(f"  sampling failed for {tag}")
                continue

            sampled = parse_sampling_csv(sample_csv)
            if not sampled:
                log(f"  no SMILES sampled for {tag}")
                continue
            log(f"  sampled n={len(sampled)} unique, scoring...")
            preds = score_cohort(sampled, target_dir)
            cohort = pd.DataFrame({"SMILES": sampled, "pred_pIC50": preds})
            cohort["tc_to_anchor"] = [tc(s, anchor_smi) for s in cohort["SMILES"]]
            cohort["tc_to_drug"] = [tc(s, drug_smi) for s in cohort["SMILES"]]
            cohort["murcko"] = [murcko(s) for s in cohort["SMILES"]]

            cohort_path = target_dir / f"iter{it}_{strategy}_cohort.csv"
            cohort.to_csv(cohort_path, index=False)
            log(f"  wrote {cohort_path.name} n={len(cohort)} "
                f"max pred={cohort['pred_pIC50'].max():.2f} "
                f"max Tc-drug={cohort['tc_to_drug'].max():.3f}")

            # promote
            promoted = promote(cohort, anchor_smi, warhead["smarts_generic"],
                               n_total=N_PROMOTE_TOTAL, tc_min=TC_PROMOTE)
            (target_dir / f"iter{it}_{strategy}_promoted.csv").write_text(promoted.to_csv(index=False))
            # prev seeds for next iter = promoted + original anchors
            prev_seeds = promoted["SMILES"].tolist() + seeds_orig
            log(f"  promoted n={len(promoted)} for next iter")


def main():
    log("Phase 2 runner starting")
    log(f"GPU available: {torch.cuda.is_available()}")
    if not REINVENT.exists():
        log(f"FATAL: reinvent not found at {REINVENT}")
        sys.exit(1)

    for target in TARGETS:
        try:
            run_target(target)
        except Exception as e:
            log(f"FATAL exception in {target}: {e}")
            import traceback
            traceback.print_exc()

    DONE_FLAG.touch()
    log("All targets done. DONE flag written.")


if __name__ == "__main__":
    main()
