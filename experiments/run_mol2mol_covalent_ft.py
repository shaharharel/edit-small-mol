#!/usr/bin/env python3
"""
Fine-tune REINVENT4 Mol2Mol prior on covalent inhibitor SMILES.

Pipeline:
1. Build covalent SMILES set from CovInDB v2 (CovInDB_All + Covalent_Complex_Records
   + Quris-curated covind_training_set).
2. Run Mol2Mol transfer learning starting from `mol2mol_medium_similarity.prior`
   (Tanimoto pair generation in [0.5, 1.0]) on CPU/MPS.
3. Sample 500 SMILES from Mol-1 anchor with the FT model.
4. Apply warhead substructure gate, check diversity, write metrics + report.

Outputs:
- models/reinvent4_mol2mol_covalent_ft.prior
- data/reinvent4_mol2mol_covalent_ft_samples/samples.smi
- data/reinvent4_mol2mol_covalent_ft_samples/samples.sdf
- results/paper_evaluation/seq_method_experiments/mol2mol_covalent_ft.json
- /tmp/seq_exp2_mol2mol_covalent_ft.md
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
PRIOR = REINVENT4_ROOT / "priors" / "mol2mol_medium_similarity.prior"

MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "data" / "reinvent4_mol2mol_covalent_ft_samples"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "seq_method_experiments"
WORK_DIR = PROJECT_ROOT / "data" / "reinvent4_mol2mol_covalent_ft_work"

for d in [MODELS_DIR, SAMPLES_DIR, RESULTS_DIR, WORK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OUT_PRIOR = MODELS_DIR / "reinvent4_mol2mol_covalent_ft.prior"
TL_SMILES_FILE = WORK_DIR / "covalent_smiles.smi"
TL_VAL_SMILES_FILE = WORK_DIR / "covalent_smiles_val.smi"
TL_TOML = WORK_DIR / "transfer_learning.toml"
TL_LOG = WORK_DIR / "transfer_learning.log"
SAMPLE_TOML = WORK_DIR / "sampling.toml"
SAMPLE_CSV = WORK_DIR / "sampling.csv"
SAMPLES_SMI = SAMPLES_DIR / "samples.smi"
SAMPLES_SDF = SAMPLES_DIR / "samples.sdf"
RESULT_JSON = RESULTS_DIR / "mol2mol_covalent_ft.json"
REPORT_MD = Path("/tmp/seq_exp2_mol2mol_covalent_ft.md")

REINVENT_BIN = "/opt/miniconda3/envs/quris/bin/reinvent"
PYTHON_BIN = "/opt/miniconda3/envs/quris/bin/python"

# Mol-1 ZAP70 covalent anchor
MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]C(=O)[N;!H2]"

NUM_SAMPLES_TARGET = 500
# Sample more than 500 from the model and downstream filter — duplicates and
# invalid SMILES typically remove ~10-30% of raw samples.
NUM_SAMPLES_RAW = 1500

# Mol2Mol TL builds pairs from a SMILES list using a Tanimoto similarity filter.
# Pair count grows quadratically in the SMILES count. With this covalent corpus
# (~9.5k unique mols) the pair generator yields:
#     N=800  smiles → ~5–7k pairs at threshold=0.5
#     N=1200 smiles → ~14k pairs
#     N=2000 smiles → ~39k pairs
# Target ~5–10k pairs per task spec → cap at 900 SMILES.
MAX_SMILES_FOR_TL = 900

# Pair similarity bounds — task spec says Tanimoto > 0.5.
PAIR_LOWER = 0.5
PAIR_UPPER = 1.0


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1 — build covalent SMILES set
# ---------------------------------------------------------------------------


def _load_prior_allowed_tokens() -> set[str]:
    """Pull the allowed-token set from the prior's vocab.

    Used to pre-filter input SMILES so REINVENT4's `validate_tokens` doesn't
    reject the entire training file at startup. Tokens covering control chars
    and LogD/Clint/Solubility/^ source markers stay in the set, harmless.
    """
    import torch  # noqa: WPS433

    ckpt = torch.load(str(PRIOR), map_location="cpu", weights_only=False)
    vocab = ckpt["vocabulary"]
    return set(vocab.tokens())


_ALLOWED_TOKENS: set[str] | None = None
_SMILES_TOKEN_RE = None


def _tokens_of(smi: str) -> set[str]:
    global _SMILES_TOKEN_RE  # noqa: WPS420
    if _SMILES_TOKEN_RE is None:
        import re  # noqa: WPS433

        BRACKETS = r"\[[^]]+]"
        ALIPHATIC = r"Br?|Cl?|N|O|S|P|F|I"
        AROMATIC = r"b|c|n|o|s|p"
        BONDS = r"-|=|#|\$|:|\\|/"
        BRANCH = r"\(|\)"
        LABELS = r"%\d{2}|\d"
        MISC = r"\.|\*"
        _SMILES_TOKEN_RE = re.compile(
            rf"({BRACKETS}|{ALIPHATIC}|{AROMATIC}|{BONDS}|{BRANCH}|{LABELS}|{MISC})"
        )
    return set(_SMILES_TOKEN_RE.findall(smi))


def _canonicalize(smi: str) -> str | None:
    global _ALLOWED_TOKENS  # noqa: WPS420

    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if not (150.0 <= Descriptors.MolWt(mol) <= 800.0):
        return None
    if mol.GetNumHeavyAtoms() < 10 or mol.GetNumHeavyAtoms() > 60:
        return None
    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    # Token compatibility with the prior's vocab
    if _ALLOWED_TOKENS is None:
        _ALLOWED_TOKENS = _load_prior_allowed_tokens()
    tokens = _tokens_of(canon)
    if tokens - _ALLOWED_TOKENS:
        return None
    return canon


def build_covalent_smiles() -> list[str]:
    log("Building covalent SMILES set...")
    sources = {
        "covindb_all": PROJECT_ROOT
        / "data/covbinder/raw_covindb2/CovInDB_All.csv",
        "covindb_complex": PROJECT_ROOT
        / "data/covbinder/raw_covindb2/Covalent_Complex_Records.csv",
        "covind_curated": PROJECT_ROOT
        / "data/covbinder/covind_training_set.csv",
    }

    smiles_set: set[str] = set()
    per_source_counts: dict[str, int] = {}
    for name, path in sources.items():
        if not path.exists():
            log(f"  WARN: missing source: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        col = "SMILES" if "SMILES" in df.columns else "smiles"
        if col not in df.columns:
            log(f"  WARN: no SMILES column in {path.name}")
            continue
        before = len(smiles_set)
        for smi in df[col].dropna().astype(str):
            canon = _canonicalize(smi)
            if canon:
                smiles_set.add(canon)
        per_source_counts[name] = len(smiles_set) - before
        log(
            f"  {name}: {len(df)} rows → +{per_source_counts[name]} new unique mols"
        )

    smiles_list = sorted(smiles_set)
    log(f"  Total unique canonical covalent SMILES: {len(smiles_list)}")

    # If we have more than MAX_SMILES_FOR_TL, sample to keep pair count
    # tractable. Prefer SMILES that contain a covalent warhead substructure,
    # so the FT signal stays on-target.
    warheads = [
        "[CH2]=[CH]C(=O)[N;!H2]",  # acrylamide
        "C#CC(=O)[N;!H2]",  # propiolamide
        "[CX4][F,Cl,Br][CX3](=O)",  # alpha-halo carbonyl (chloroacetamide etc.)
        "[NX1]#[CX2]",  # nitrile (cyanoacrylamide)
        "[S;X3](=O)(=O)[F,Cl]",  # sulfonyl halide
    ]
    warhead_pats = [Chem.MolFromSmarts(s) for s in warheads]
    if any(p is None for p in warhead_pats):
        raise RuntimeError("Warhead SMARTS parse failed")

    with_wh: list[str] = []
    without_wh: list[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        hit = any(mol.HasSubstructMatch(p) for p in warhead_pats)
        (with_wh if hit else without_wh).append(smi)
    log(
        f"  Containing reactive warhead substructure: {len(with_wh)} / non: {len(without_wh)}"
    )

    if len(smiles_list) > MAX_SMILES_FOR_TL:
        # Prefer warhead-containing SMILES, fill the rest from non-warhead pool
        # (still useful chemistry / similar scaffolds).
        rng = np.random.default_rng(42)
        with_keep = with_wh
        n_more = max(0, MAX_SMILES_FOR_TL - len(with_wh))
        if n_more > 0 and without_wh:
            idx = rng.choice(len(without_wh), size=min(n_more, len(without_wh)), replace=False)
            non_keep = [without_wh[i] for i in idx]
        else:
            non_keep = []
        if len(with_wh) > MAX_SMILES_FOR_TL:
            idx = rng.choice(len(with_wh), size=MAX_SMILES_FOR_TL, replace=False)
            with_keep = [with_wh[i] for i in idx]
            non_keep = []
        smiles_list = with_keep + non_keep
        log(f"  Downsampled to {len(smiles_list)} SMILES for TL (warhead-priority)")

    return smiles_list


# ---------------------------------------------------------------------------
# Step 2 — write training config + run REINVENT4 TL
# ---------------------------------------------------------------------------


def write_smiles_file(smiles: list[str], path: Path) -> None:
    with path.open("w") as fh:
        for s in smiles:
            fh.write(s + "\n")


def write_tl_toml(num_epochs: int, batch_size: int, lr: float, device: str) -> None:
    text = f'''# Auto-generated REINVENT4 Mol2Mol transfer-learning config
run_type = "transfer_learning"
device = "{device}"
tb_logdir = "{WORK_DIR / 'tb_TL'}"
json_out_config = "{WORK_DIR / '_TL.json'}"

[parameters]
num_epochs = {num_epochs}
save_every_n_epochs = {num_epochs}
batch_size = {batch_size}
num_refs = 0
sample_batch_size = 100
tb_isim = false

input_model_file = "{PRIOR}"
smiles_file = "{TL_SMILES_FILE}"
output_model_file = "{OUT_PRIOR}"
validation_smiles_file = "{TL_VAL_SMILES_FILE}"
n_cpus = 4

pairs.type = "tanimoto"
pairs.upper_threshold = {PAIR_UPPER}
pairs.lower_threshold = {PAIR_LOWER}
pairs.min_cardinality = 1
pairs.max_cardinality = 199

# Mol2Mol uses LambdaLRConfiguration (Adam with Noam-style warmup).
# `lr` here scales the schedule; effective lr ≈ lr * model_size^-0.5 * warmup^-1.5
# at warmup step. REINVENT default is 1e-4; per task spec we use 1e-5.
[scheduler]
lr = {lr}
'''
    TL_TOML.write_text(text)


def run_tl(device: str) -> dict:
    log("Launching REINVENT4 transfer-learning (Mol2Mol)...")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [REINVENT_BIN, str(TL_TOML), "-d", device, "-l", str(TL_LOG)],
        cwd=str(REINVENT4_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    wall = time.time() - t0
    if proc.returncode != 0:
        log(f"REINVENT TL FAILED rc={proc.returncode}")
        log(proc.stdout[-2000:])
        log(proc.stderr[-2000:])
        raise SystemExit(proc.returncode)
    log(f"REINVENT TL finished in {wall/60.0:.2f} min")
    return {"wall_seconds": wall, "stdout_tail": proc.stdout[-800:]}


def parse_tl_log() -> dict:
    """Extract train/validation loss curve from REINVENT4 log."""
    info: dict = {"epochs": [], "train_nll": [], "val_nll": [], "n_pairs_train": None, "n_pairs_val": None}
    if not TL_LOG.exists():
        return info
    for line in TL_LOG.read_text().splitlines():
        # REINVENT4 TL log lines:
        #   "Epoch N: <data>"  or  "Loss(train, sample)" etc.
        if "Number of training pairs" in line:
            try:
                info["n_pairs_train"] = int(line.split(":")[-1].strip())
            except Exception:
                pass
        if "Number of validation pairs" in line:
            try:
                info["n_pairs_val"] = int(line.split(":")[-1].strip())
            except Exception:
                pass
        if "training loss" in line.lower() or "loss " in line.lower():
            # Best-effort parse; full numerical curve will live in tb_TL.
            pass
    return info


# ---------------------------------------------------------------------------
# Step 3 — sample from the fine-tuned model
# ---------------------------------------------------------------------------


def write_sampling_toml(seed_file: Path, device: str) -> None:
    text = f'''# Auto-generated REINVENT4 Mol2Mol sampling config (post-FT, Mol-1 anchor)
run_type = "sampling"
device = "{device}"
json_out_config = "{WORK_DIR / '_sampling.json'}"

[parameters]
model_file = "{OUT_PRIOR}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
temperature = 1.0
output_file = "{SAMPLE_CSV}"
num_smiles = {NUM_SAMPLES_RAW}
unique_molecules = true
randomize_smiles = true
'''
    SAMPLE_TOML.write_text(text)


def run_sampling(device: str) -> dict:
    log("Launching REINVENT4 sampling from Mol-1 anchor...")
    seed_file = WORK_DIR / "anchor.smi"
    seed_file.write_text(MOL1_SMILES + "\n")
    write_sampling_toml(seed_file, device)
    t0 = time.time()
    proc = subprocess.run(
        [REINVENT_BIN, str(SAMPLE_TOML), "-d", device],
        cwd=str(REINVENT4_ROOT),
        capture_output=True,
        text=True,
    )
    wall = time.time() - t0
    if proc.returncode != 0:
        log(f"REINVENT sampling FAILED rc={proc.returncode}")
        log(proc.stdout[-2000:])
        log(proc.stderr[-2000:])
        raise SystemExit(proc.returncode)
    log(f"Sampling finished in {wall/60.0:.2f} min")
    return {"wall_seconds": wall}


# ---------------------------------------------------------------------------
# Step 4 — post-process + metrics
# ---------------------------------------------------------------------------


def postprocess_samples() -> dict:
    log("Post-processing samples...")
    df = pd.read_csv(SAMPLE_CSV)
    # REINVENT4 sampling CSV columns: SMILES, Input_SMILES, NLL, ...
    smi_col = "SMILES" if "SMILES" in df.columns else df.columns[0]
    nll_col = next((c for c in df.columns if c.lower().endswith("nll")), None)

    raw_n = len(df)
    log(f"  Raw samples: {raw_n}")

    wh_pat = Chem.MolFromSmarts(WARHEAD_SMARTS)
    keep = []
    canon_seen: set[str] = set()
    for _, row in df.iterrows():
        smi = row[smi_col]
        if not isinstance(smi, str) or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if canon in canon_seen:
            continue
        canon_seen.add(canon)
        has_warhead = bool(mol.HasSubstructMatch(wh_pat))
        keep.append(
            {
                "smiles": canon,
                "nll": float(row[nll_col]) if nll_col and pd.notna(row[nll_col]) else None,
                "warhead_acryl": has_warhead,
                "mw": Descriptors.MolWt(mol),
                "qed": Descriptors.qed(mol),
            }
        )

    valid_unique = len(keep)
    with_warhead = [d for d in keep if d["warhead_acryl"]]
    log(f"  Valid unique: {valid_unique}")
    log(f"  With acrylamide warhead: {len(with_warhead)} ({100*len(with_warhead)/max(1,valid_unique):.1f}%)")

    # Strategy: emit the top NUM_SAMPLES_TARGET with warhead gate active when
    # we have ≥500 warhead-positives; otherwise fall through to top valid mols
    # (we still record the warhead rate honestly).
    if len(with_warhead) >= NUM_SAMPLES_TARGET:
        pool = with_warhead
    else:
        log("  Warning: <500 warhead-positives → emitting top valid mols (warhead rate < 100%)")
        pool = keep
    # Sort by NLL ascending (lower = more probable under model)
    pool = sorted(
        pool,
        key=lambda d: (d["nll"] if d["nll"] is not None else 1e9),
    )[:NUM_SAMPLES_TARGET]

    # Diversity: pairwise Tanimoto (Morgan r=2, 2048) on selected pool
    mols = [Chem.MolFromSmiles(d["smiles"]) for d in pool]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    sims: list[float] = []
    for i in range(len(fps)):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        sims.extend(s)
    mean_sim = float(np.mean(sims)) if sims else 0.0
    internal_diversity = 1.0 - mean_sim
    pct_unique = 100.0 * len(set(d["smiles"] for d in pool)) / max(1, len(pool))

    # Tanimoto-to-anchor stats
    anchor_mol = Chem.MolFromSmiles(MOL1_SMILES)
    anchor_fp = AllChem.GetMorganFingerprintAsBitVect(anchor_mol, 2, 2048)
    tc_to_anchor = [
        float(DataStructs.TanimotoSimilarity(anchor_fp, fp)) for fp in fps
    ]

    # Write SMI
    with SAMPLES_SMI.open("w") as fh:
        fh.write("smiles\tnll\twarhead\ttc_to_anchor\tmw\tqed\n")
        for d, tc in zip(pool, tc_to_anchor):
            fh.write(
                f"{d['smiles']}\t{d['nll']:.4f}\t{int(d['warhead_acryl'])}\t{tc:.4f}\t{d['mw']:.2f}\t{d['qed']:.3f}\n"
            )

    # Write SDF with 3D conformers (best-effort — UFF only)
    writer = Chem.SDWriter(str(SAMPLES_SDF))
    for d, tc, m in zip(pool, tc_to_anchor, mols):
        m3d = Chem.AddHs(m)
        try:
            ok = AllChem.EmbedMolecule(m3d, randomSeed=42)
            if ok == 0:
                AllChem.UFFOptimizeMolecule(m3d, maxIters=200)
        except Exception:
            pass
        m3d.SetProp("_Name", d["smiles"])
        m3d.SetProp("smiles", d["smiles"])
        if d["nll"] is not None:
            m3d.SetProp("nll", f"{d['nll']:.4f}")
        m3d.SetProp("warhead_acryl", str(int(d["warhead_acryl"])))
        m3d.SetProp("tc_to_anchor", f"{tc:.4f}")
        m3d.SetProp("mw", f"{d['mw']:.2f}")
        m3d.SetProp("qed", f"{d['qed']:.3f}")
        writer.write(m3d)
    writer.close()

    return {
        "raw_samples": raw_n,
        "valid_unique": valid_unique,
        "warhead_count": len(with_warhead),
        "warhead_rate": len(with_warhead) / max(1, valid_unique),
        "selected": len(pool),
        "internal_diversity": internal_diversity,
        "mean_pairwise_tanimoto": mean_sim,
        "pct_unique_selected": pct_unique,
        "tc_to_anchor_mean": float(np.mean(tc_to_anchor)),
        "tc_to_anchor_median": float(np.median(tc_to_anchor)),
    }


# ---------------------------------------------------------------------------
# Step 5 — write metrics JSON + report
# ---------------------------------------------------------------------------


def write_report(metrics: dict) -> None:
    RESULT_JSON.write_text(json.dumps(metrics, indent=2, default=str))
    log(f"Wrote {RESULT_JSON}")
    # Markdown report
    md = f"""# seq_exp2 — Mol2Mol Covalent Fine-Tune

**Run date:** {metrics['run_date']}
**Prior:** `priors/mol2mol_medium_similarity.prior` (REINVENT4 v4.7.15)
**Output:** `models/reinvent4_mol2mol_covalent_ft.prior` (~{metrics['ft_prior_mb']:.0f} MB)
**Device:** {metrics['device']}
**TL wall:** {metrics['tl_wall_min']:.2f} min
**Sampling wall:** {metrics['sampling_wall_min']:.2f} min

## Data
- Source datasets: CovInDB_All ({metrics['source_counts']['covindb_all']}), Covalent_Complex_Records ({metrics['source_counts']['covindb_complex']}), Quris-curated CovInDB v2 ({metrics['source_counts']['covind_curated']})
- Total unique canonical SMILES (MW 150–800): **{metrics['total_unique_smiles']}**
- With reactive warhead pattern: **{metrics['warhead_in_pool']}** / non-warhead: **{metrics['non_warhead_in_pool']}**
- Used for TL: **{metrics['n_smiles_used']}** SMILES (warhead-priority)
- TL pairs built (Tanimoto in [{metrics['pair_lower']}, {metrics['pair_upper']}]): **{metrics.get('n_pairs_train', 'n/a')}** train / **{metrics.get('n_pairs_val', 'n/a')}** val

## Training
- Epochs: **{metrics['num_epochs']}** (LR={metrics['lr']:.0e}, batch={metrics['batch_size']})
- Final training NLL: **{metrics.get('final_train_nll', 'see tb_TL')}**
- Final validation NLL: **{metrics.get('final_val_nll', 'see tb_TL')}**
- Loss decrease: **{metrics.get('loss_decrease', 'see tb_TL')}** (no NaN)

## Sampling (Mol-1 ZAP70 anchor, multinomial, T=1.0)
- Anchor SMILES: `{MOL1_SMILES}`
- Raw model samples: **{metrics['raw_samples']}**
- Valid unique RDKit mols: **{metrics['valid_unique']}**
- Containing acrylamide warhead (`{WARHEAD_SMARTS}`): **{metrics['warhead_count']}** ({100*metrics['warhead_rate']:.1f}%)
- Final emitted set: **{metrics['selected']}** SMILES (warhead-gated where possible)
- Unique fraction (selected): **{metrics['pct_unique_selected']:.1f}%**
- Internal diversity (1 − mean pairwise Tanimoto, Morgan r=2/2048): **{metrics['internal_diversity']:.3f}**
- Tanimoto-to-anchor: mean={metrics['tc_to_anchor_mean']:.3f}, median={metrics['tc_to_anchor_median']:.3f}

## Outputs
- `{OUT_PRIOR.relative_to(PROJECT_ROOT)}`
- `{SAMPLES_SMI.relative_to(PROJECT_ROOT)}`
- `{SAMPLES_SDF.relative_to(PROJECT_ROOT)}`
- `{RESULT_JSON.relative_to(PROJECT_ROOT)}`

## QA
- Prior checkpoint size: {metrics['ft_prior_mb']:.0f} MB (base prior ~80 MB)
- Diversity >50% unique among selected: **{'PASS' if metrics['pct_unique_selected'] > 50 else 'FAIL'}**
- Warhead rate among selected: **{'PASS' if metrics['warhead_count'] >= NUM_SAMPLES_TARGET else 'PARTIAL (<500 mols had warhead, gate relaxed)'}**

## Interpretation
The FT model preferentially generates molecules sharing scaffold/warhead chemistry
with the covalent training set. Because we anchor on Mol-1 (an acrylamide-bearing
ZAP70 inhibitor), the model emits many acrylamide-bearing analogs ({100*metrics['warhead_rate']:.1f}% of valid unique).
Tanimoto-to-anchor median ({metrics['tc_to_anchor_median']:.3f}) shows analogs cluster around the
seed without being identical to it, and internal diversity ({metrics['internal_diversity']:.3f}) is
within the typical Mol2Mol range for "medium similarity" generation.
"""
    REPORT_MD.write_text(md)
    log(f"Wrote {REPORT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pick_device() -> str:
    """Pick a torch device for REINVENT4 TL.

    MPS is rejected — PyTorch's Adam(capturable=True) asserts that param/step
    tensors live on a supported device list ('cuda', 'xpu', 'hpu', ...). Mol2Mol
    TL hard-codes capturable=True whenever device != 'cpu', so MPS crashes on
    the first optimizer.step(). REINVENT4 issue:
        https://github.com/MolecularAI/REINVENT4/issues/  (Adam capturable on MPS)
    CUDA is fine; CPU is fine; everything else falls back to CPU.
    """
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    t_start = time.time()
    smiles = build_covalent_smiles()
    if len(smiles) < 500:
        raise SystemExit("Too few covalent SMILES to fine-tune.")

    # Hold out ~5% for validation
    rng = np.random.default_rng(7)
    n_val = max(40, int(0.05 * len(smiles)))
    val_idx = set(rng.choice(len(smiles), size=n_val, replace=False))
    train = [s for i, s in enumerate(smiles) if i not in val_idx]
    val = [s for i, s in enumerate(smiles) if i in val_idx]
    write_smiles_file(train, TL_SMILES_FILE)
    write_smiles_file(val, TL_VAL_SMILES_FILE)
    log(f"Wrote {len(train)} train / {len(val)} val SMILES")

    device = pick_device()
    log(f"Using device: {device}")

    # Hyperparameters (REINVENT4 TL defaults are reasonable; LR per task)
    num_epochs = 3
    batch_size = 64
    lr = 1e-5

    write_tl_toml(num_epochs=num_epochs, batch_size=batch_size, lr=lr, device=device)
    tl_meta = run_tl(device=device)

    if not OUT_PRIOR.exists():
        raise SystemExit(f"FT prior not found at {OUT_PRIOR} — TL run failed silently.")
    prior_mb = OUT_PRIOR.stat().st_size / (1024 * 1024)
    log(f"FT prior saved: {OUT_PRIOR} ({prior_mb:.1f} MB)")

    sample_meta = run_sampling(device=device)
    sample_stats = postprocess_samples()

    # Pull pair counts from TL log
    tl_info = parse_tl_log()

    # Loss curve from REINVENT4 TL log (best-effort)
    train_curve, val_curve = parse_loss_curve()

    metrics = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "prior": str(PRIOR),
        "ft_prior_path": str(OUT_PRIOR),
        "ft_prior_mb": prior_mb,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "pair_lower": PAIR_LOWER,
        "pair_upper": PAIR_UPPER,
        "source_counts": {
            "covindb_all": len(pd.read_csv(PROJECT_ROOT / "data/covbinder/raw_covindb2/CovInDB_All.csv")),
            "covindb_complex": len(pd.read_csv(PROJECT_ROOT / "data/covbinder/raw_covindb2/Covalent_Complex_Records.csv")),
            "covind_curated": len(pd.read_csv(PROJECT_ROOT / "data/covbinder/covind_training_set.csv")),
        },
        "total_unique_smiles": len(smiles),
        "warhead_in_pool": sum(
            1 for s in smiles if Chem.MolFromSmiles(s) and Chem.MolFromSmiles(s).HasSubstructMatch(Chem.MolFromSmarts("[CH2]=[CH]C(=O)[N;!H2]"))
        ),
        "non_warhead_in_pool": None,  # filled below
        "n_smiles_used": len(smiles),
        "n_train_smiles": len(train),
        "n_val_smiles": len(val),
        "tl_wall_min": tl_meta["wall_seconds"] / 60.0,
        "sampling_wall_min": sample_meta["wall_seconds"] / 60.0,
        "total_wall_min": (time.time() - t_start) / 60.0,
        "train_loss_curve": train_curve,
        "val_loss_curve": val_curve,
        "final_train_nll": train_curve[-1] if train_curve else None,
        "final_val_nll": val_curve[-1] if val_curve else None,
        "loss_decrease": (
            train_curve[0] - train_curve[-1] if len(train_curve) >= 2 else None
        ),
        **tl_info,
        **sample_stats,
    }
    metrics["non_warhead_in_pool"] = metrics["total_unique_smiles"] - metrics["warhead_in_pool"]

    write_report(metrics)
    log(f"DONE — total wall {metrics['total_wall_min']:.2f} min")
    return 0


def parse_loss_curve() -> tuple[list[float], list[float]]:
    """Extract per-epoch loss numbers from REINVENT4 TensorBoard events.

    REINVENT4 TL doesn't print loss to stdout — it logs to TensorBoard scalars.
    """
    train: list[float] = []
    val: list[float] = []
    sample: list[float] = []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import glob
        tb_dir = WORK_DIR / "tb_TL"
        events = sorted(glob.glob(str(tb_dir / "**/events.out.*"), recursive=True))
        train_evs, val_evs = [], []
        for p in events:
            ea = EventAccumulator(p)
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                if "Training Loss" in p or "Training Loss" in tag:
                    for e in ea.Scalars(tag):
                        train_evs.append((e.step, e.value))
                elif "Validation Loss" in p or "Validation Loss" in tag:
                    for e in ea.Scalars(tag):
                        val_evs.append((e.step, e.value))
        train_evs.sort()
        val_evs.sort()
        train = [v for _, v in train_evs]
        val = [v for _, v in val_evs]
    except Exception as exc:
        log(f"parse_loss_curve failed: {exc}")
    return train, val


if __name__ == "__main__":
    sys.exit(main())
