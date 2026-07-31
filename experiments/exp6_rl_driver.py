"""EXP6 retrospective LO — Phase 2 RL driver.

Loops over (target × strategy × iteration × scorer) cells. For each cell:
  1. Build seed SMILES file (A=anchor only, B=100 anchors; iter2/3 promotes prev cohort)
  2. Render REINVENT4 staged_learning DAP TOML pointing at the right REST server
  3. Run `reinvent <toml>`
  4. Sample N=5500 from the resulting checkpoint into cohort CSV
  5. Compute cell metrics (warhead retention, mean pIC50, max Tc-to-drug, n_scaffolds)

Promotion logic (iter > 1):
  filter cohort for Tc>=0.3 to the original anchor SMILES;
  take top-25 by predicted pIC50 + 25 Murcko-unique diverse;
  union with the original anchor pool -> seed file for next iter.

Designed to be RESUMABLE — checks for cohort CSV existence and skips finished cells.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = "/home/shaharh_quris_ai/edit-small-mol"
EXP6_ROOT = f"{PROJECT_ROOT}/data/exp6_retrospective"
RL_OUT_ROOT = f"{EXP6_ROOT}/_rl"
TOML_OUT_ROOT = f"{EXP6_ROOT}/_tomls"
PRIOR_FILE = f"{PROJECT_ROOT}/models/reinvent4_mol2mol_covalent_ft.prior"
SRC_ROOT = PROJECT_ROOT

PORTS = {
    ("egfr_t790m", "film"): 8088,
    ("btk",         "film"): 8089,
    ("kras_g12c",   "film"): 8090,
    ("egfr_t790m", "dabs"): 8091,
    ("btk",         "dabs"): 8092,
    ("kras_g12c",   "dabs"): 8093,
}

def _load_anchor_drug():
    out_anchor, out_drug = {}, {}
    for tgt in ("egfr_t790m", "btk", "kras_g12c"):
        p = Path(EXP6_ROOT) / tgt / "prep_summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            out_anchor[tgt] = d["anchor"]["smiles"]
            out_drug[tgt] = d["drug"]["smiles"]
    return out_anchor, out_drug


ANCHOR_SMILES, DRUG_SMILES = _load_anchor_drug()


def _fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _murcko(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc else None
    except Exception:
        return None


def _tc(fp_a, fp_b):
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def render_toml(target: str, strategy: str, scorer: str, iter_n: int,
                seed_file: str, chkpt_file: str, csv_prefix: str,
                tb_dir: str, batch_size: int = 32, sigma: int = 128,
                lr: float = 1e-4, n_steps: int = 50) -> str:
    """Write a REINVENT4 staged_learning DAP TOML and return its path."""
    port = PORTS[(target, scorer)]
    cell_id = f"{target}_{strategy}_iter{iter_n}_{scorer}"
    toml = f"""# EXP6 retrospective LO — {cell_id}
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "{tb_dir}"
json_out_config = "{TOML_OUT_ROOT}/{cell_id}.json"

[parameters]
summary_csv_prefix = "{csv_prefix}"
use_checkpoint = false
purge_memories = true

prior_file = "{PRIOR_FILE}"
agent_file = "{PRIOR_FILE}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = {batch_size}
unique_sequences = false
randomize_smiles = true
tb_isim = false

[learning_strategy]
type = "dap"
sigma = {sigma}
rate = {lr}

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4
penalty_multiplier = 0.5

[[stage]]
chkpt_file = "{chkpt_file}"
termination = "simple"
max_score = 1.0
max_steps = {n_steps}
min_steps = {n_steps}

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.REST]
[[stage.scoring.component.REST.endpoint]]
name = "exp6_{scorer}_composite"
weight = 1.0
params.server_url = "http://127.0.0.1"
params.server_port = {port}
params.server_endpoint = "score"
params.predictor_id = "composite"
params.predictor_version = "exp6_v1"
"""
    Path(TOML_OUT_ROOT).mkdir(parents=True, exist_ok=True)
    toml_path = f"{TOML_OUT_ROOT}/{cell_id}.toml"
    with open(toml_path, "w") as f:
        f.write(toml)
    return toml_path


def render_sample_toml(model_file: str, out_csv: str, smiles_seed: str,
                       num_smiles: int = 5500, cell_id: str = "cell") -> str:
    """Write a REINVENT4 sampling TOML.

    NB: REINVENT4 Mol2Mol sampling interprets `num_smiles` as PER input seed,
    so for a 100-seed pool with num_smiles=55 we get ~5500 total samples.
    We adjust num_smiles dynamically based on the seed file size.
    """
    # count seeds (lines in smiles_file)
    try:
        with open(smiles_seed) as f:
            n_seeds = sum(1 for line in f if line.strip())
    except Exception:
        n_seeds = 1
    n_seeds = max(1, n_seeds)
    per_seed = max(1, int(round(num_smiles / n_seeds)))

    name = f"{cell_id}_sample"
    toml = f"""run_type = "sampling"
device = "cuda:0"
json_out_config = "{TOML_OUT_ROOT}/{name}.json"

[parameters]
model_file = "{model_file}"
output_file = "{out_csv}"
smiles_file = "{smiles_seed}"
sample_strategy = "multinomial"
num_smiles = {per_seed}
unique_molecules = false
randomize_smiles = true
"""
    Path(TOML_OUT_ROOT).mkdir(parents=True, exist_ok=True)
    toml_path = f"{TOML_OUT_ROOT}/{name}.toml"
    with open(toml_path, "w") as f:
        f.write(toml)
    return toml_path


_UNSUPPORTED_TOKEN_RE = None


def _is_supported_smi(smi: str) -> bool:
    """Reject SMILES with tokens the covalent_ft mol2mol prior cannot encode.
    Empirical: the prior's vocab has only C, N, O, S, F, Cl, Br, I, *, plus
    a few brackets. Anything outside is a hard ValueError in REINVENT4.
    """
    # The prior allows ONLY these element tokens (case-sensitive for organic
    # aromatic). Lowercase 'p' (aromatic P) and bare 'P' both rejected.
    import re
    s = str(smi)
    if not s:
        return False
    # bare-letter atoms (outside brackets)
    if re.search(r"(?<![A-Za-z])(?:P|p|B)(?![a-z])", s):
        # 'P' as bare atom or 'p' aromatic — not in vocab (Br is fine, 'B'
        # bare = boron). Phosphorus + boron both unsupported.
        return False
    # bracket-token atoms outside the allowed set
    allowed_bracket = {
        "O", "C", "N", "S", "I", "F", "Cl", "Br",
        "C@H", "C@@H", "C@", "C@@", "N@", "N@@",
        "nH", "NH", "OH",
        "n+", "N+", "N@+", "N@@+",
        "O-", "O", "C-", "n-", "o", "s", "n",
        "*", "18F", "19F", "11c", "11C", "11CH3", "76Br", "123I",
    }
    # The Mol2Mol covalent-FT prior accepts only [N+], [N@+], [N@@+], [O-],
    # [nH], [n+], [C@H], [C@@H], [C@], [C@@], [O], [18F], [19F], [76Br],
    # [123I], [11c], [11C], [11CH3], plus naked tokens above.
    allowed_brackets = {
        "N+", "N@+", "N@@+", "O-", "nH", "n+",
        "C@H", "C@@H", "C@", "C@@",
        "O", "*",
        "18F", "19F", "76Br", "123I", "11c", "11C", "11CH3",
        "S@", "S@@",
        # explicitly allow these (vocab includes them per error message)
    }
    # Conservative: keep only the safest subset above. Drop anything else
    # including charged carbons, nitriles like [C-]#[N+], non-standard
    # isotopes, all metal-like atoms.
    for m in re.findall(r"\[([^\]]+)\]", s):
        if m in allowed_brackets:
            continue
        return False
    return True


def _filter_seeds(smis):
    return [s for s in smis if s and _is_supported_smi(s)]


def _strategy_b_pool_path(target_dir: str) -> str:
    """Prefer the vocab-filtered pool if it exists, else fall back to the raw one."""
    filt = f"{target_dir}/anchor_pool_strategy_b_filtered.csv"
    if Path(filt).exists():
        return filt
    return f"{target_dir}/anchor_pool_strategy_b.csv"


def build_iter1_seed(target: str, strategy: str, out_path: str):
    """Iteration 1 seed:
      A: just anchor SMILES (1 line)
      B: 100-anchor pool (filtered for mol2mol vocab)"""
    target_dir = f"{EXP6_ROOT}/{target}"
    if strategy == "A":
        smi = ANCHOR_SMILES[target]
        smis = [smi]
    else:
        df = pd.read_csv(_strategy_b_pool_path(target_dir))
        smis = df["smiles"].dropna().astype(str).tolist()
    smis = _filter_seeds(smis)
    with open(out_path, "w") as f:
        for s in smis:
            f.write(s + "\n")


def build_promoted_seed(target: str, strategy: str, prev_cohort_csv: str,
                        out_path: str, n_top: int = 25, n_div: int = 25,
                        tc_min: float = 0.3):
    """Iter 2/3 seed: original anchors + 25 top-pIC50 + 25 Murcko-diverse
    (all promoted candidates must satisfy Tc>=0.3 vs original anchor)."""
    target_dir = f"{EXP6_ROOT}/{target}"
    anchor_smi = ANCHOR_SMILES[target]
    anchor_fp = _fp(anchor_smi)

    # base = strategy A anchor only or B pool
    if strategy == "A":
        base = [anchor_smi]
    else:
        base = pd.read_csv(_strategy_b_pool_path(target_dir))["smiles"].dropna().astype(str).tolist()

    if not Path(prev_cohort_csv).exists():
        # fall back to base only (still apply token filter)
        base = _filter_seeds(base)
        with open(out_path, "w") as f:
            for s in base:
                f.write(s + "\n")
        return

    df = pd.read_csv(prev_cohort_csv)
    if "smiles" not in df.columns or len(df) == 0:
        base = _filter_seeds(base)
        with open(out_path, "w") as f:
            for s in base:
                f.write(s + "\n")
        return

    # filter for Tc>=tc_min vs anchor
    df["fp"] = df["smiles"].apply(_fp)
    df["tc_anchor"] = df["fp"].apply(lambda fp: _tc(fp, anchor_fp))
    df_keep = df[df["tc_anchor"] >= tc_min].copy()
    if "predicted_pIC50" not in df_keep.columns:
        df_keep["predicted_pIC50"] = 0.0
    # top-25 by pIC50
    df_top = df_keep.sort_values("predicted_pIC50", ascending=False).head(n_top)
    # then Murcko-unique 25 outside top set
    df_rest = df_keep[~df_keep.index.isin(df_top.index)].copy()
    df_rest["scaffold"] = df_rest["smiles"].apply(_murcko)
    df_rest = df_rest.dropna(subset=["scaffold"]).drop_duplicates(subset=["scaffold"]).head(n_div)
    promoted = pd.concat([df_top, df_rest], ignore_index=True)["smiles"].dropna().astype(str).tolist()

    final = list(dict.fromkeys(base + promoted))  # dedupe, preserve order
    final = _filter_seeds(final)
    with open(out_path, "w") as f:
        for s in final:
            f.write(s + "\n")


def parse_sample_csv(sample_csv: str) -> pd.DataFrame:
    df = pd.read_csv(sample_csv)
    # REINVENT4 sampling CSV has columns: Input_SMILES, Output_SMILES, ...
    # different versions may differ — try both common conventions.
    smi_col = None
    for cand in ("SMILES", "Output_SMILES", "smiles"):
        if cand in df.columns:
            smi_col = cand
            break
    if smi_col is None:
        smi_col = df.columns[0]
    df = df.rename(columns={smi_col: "smiles"})
    return df[["smiles"]].dropna()


def score_cohort(cohort_csv: str, scorer_port: int) -> pd.DataFrame:
    """Hit the REST server for predicted pIC50 (predictor_id=pic50_raw)."""
    import requests
    df = pd.read_csv(cohort_csv)
    smis = df["smiles"].astype(str).tolist()
    out_scores = []
    batch = 200
    for i in range(0, len(smis), batch):
        chunk = smis[i:i + batch]
        body = [{"input_string": s, "query_id": str(j)} for j, s in enumerate(chunk)]
        try:
            r = requests.post(
                f"http://127.0.0.1:{scorer_port}/score",
                params={"predictor_id": "pic50_raw"},
                json=body, timeout=120,
            )
            r.raise_for_status()
            succ = r.json()["output"]["successes_list"]
            out_scores.extend(float(s["output_value"]) for s in succ)
        except Exception as e:
            print(f"[exp6-rl] score_cohort error batch {i}: {e}", flush=True)
            out_scores.extend([0.0] * len(chunk))
    df["predicted_pIC50"] = out_scores[:len(df)]
    df.to_csv(cohort_csv, index=False)
    return df


def compute_cell_metrics(cohort_csv: str, target: str) -> dict:
    df = pd.read_csv(cohort_csv)
    anchor_fp = _fp(ANCHOR_SMILES[target])
    drug_fp = _fp(DRUG_SMILES[target])

    # warhead retention against the target-specific strict SMARTS
    with open(f"{EXP6_ROOT}/{target}/warhead_smarts.json") as f:
        spec = json.load(f)
    strict = Chem.MolFromSmarts(spec.get("smarts_strict") or spec["smarts_generic"])
    generic = Chem.MolFromSmarts(spec["smarts_generic"])

    n_total = 0
    n_valid = 0
    n_match_strict = 0
    n_match_generic = 0
    tc_drug_max = 0.0
    tc_drug_top10 = []
    scaffolds = set()
    pics = []
    for _i, row in df.iterrows():
        smi = str(row["smiles"])
        n_total += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_valid += 1
        try:
            if strict is not None and mol.HasSubstructMatch(strict):
                n_match_strict += 1
            if generic is not None and mol.HasSubstructMatch(generic):
                n_match_generic += 1
        except Exception:
            pass
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            if sc is not None:
                scaffolds.add(Chem.MolToSmiles(sc))
        except Exception:
            pass
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        tc = float(DataStructs.TanimotoSimilarity(fp, drug_fp)) if drug_fp is not None else 0.0
        if tc > tc_drug_max:
            tc_drug_max = tc
        tc_drug_top10.append(tc)
        if "predicted_pIC50" in row and not pd.isna(row["predicted_pIC50"]):
            pics.append(float(row["predicted_pIC50"]))

    tc_drug_top10 = sorted(tc_drug_top10, reverse=True)[:10]
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_frac": (n_valid / n_total) if n_total else 0.0,
        "warhead_strict_frac": (n_match_strict / n_valid) if n_valid else 0.0,
        "warhead_generic_frac": (n_match_generic / n_valid) if n_valid else 0.0,
        "predicted_pIC50_mean": float(np.mean(pics)) if pics else 0.0,
        "predicted_pIC50_max": float(np.max(pics)) if pics else 0.0,
        "tc_drug_max": tc_drug_max,
        "tc_drug_top10_mean": float(np.mean(tc_drug_top10)) if tc_drug_top10 else 0.0,
        "tc_drug_ge_0.5_count": int(sum(1 for t in tc_drug_top10 if t >= 0.5)),
        "n_unique_scaffolds": len(scaffolds),
    }


def run_cell(target: str, strategy: str, scorer: str, iter_n: int,
             prev_cohort_csv: str | None, n_steps: int,
             batch_size: int = 32, time_budget_sec: int = 3600) -> dict:
    cell_id = f"{target}_{strategy}_iter{iter_n}_{scorer}"
    cell_dir = f"{EXP6_ROOT}/_rl/{cell_id}"
    Path(cell_dir).mkdir(parents=True, exist_ok=True)

    cohort_csv = f"{EXP6_ROOT}/{target}/iter{iter_n}_{strategy}_{scorer}_cohort.csv"
    chkpt_file = f"{cell_dir}/agent.prior"
    seed_file = f"{cell_dir}/seed.smi"
    csv_prefix = f"{cell_dir}/learn"
    tb_dir = f"{cell_dir}/tb"
    sample_csv = f"{cell_dir}/sampled.csv"

    # resume if cohort exists
    if Path(cohort_csv).exists():
        try:
            existing = pd.read_csv(cohort_csv)
            if "predicted_pIC50" in existing.columns and len(existing) >= 100:
                print(f"[exp6-rl] SKIP {cell_id} (cohort already exists, n={len(existing)})", flush=True)
                m = compute_cell_metrics(cohort_csv, target)
                m.update({"cell_id": cell_id, "status": "resumed"})
                return m
        except Exception:
            pass

    # 1. build seed
    if iter_n == 1:
        build_iter1_seed(target, strategy, seed_file)
    else:
        build_promoted_seed(target, strategy, prev_cohort_csv or "", seed_file)

    # 2. render learn TOML
    toml_path = render_toml(target, strategy, scorer, iter_n,
                            seed_file, chkpt_file, csv_prefix, tb_dir,
                            batch_size=batch_size, n_steps=n_steps)
    print(f"[exp6-rl] {cell_id} START  (steps={n_steps}, bs={batch_size})", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        ["reinvent", toml_path],
        cwd=PROJECT_ROOT,
        timeout=time_budget_sec,
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    log_path = f"{cell_dir}/run.log"
    with open(log_path, "w") as f:
        f.write(f"# return_code={proc.returncode} dt={dt:.1f}s\n")
        f.write("== STDOUT ==\n")
        f.write(proc.stdout or "")
        f.write("\n== STDERR ==\n")
        f.write(proc.stderr or "")
    if proc.returncode != 0:
        # try OOM fallback to bs=8
        if batch_size > 8 and "out of memory" in (proc.stderr or "").lower():
            print(f"[exp6-rl] {cell_id} OOM -> retry bs=8", flush=True)
            return run_cell(target, strategy, scorer, iter_n,
                             prev_cohort_csv, n_steps, batch_size=8,
                             time_budget_sec=time_budget_sec)
        print(f"[exp6-rl] {cell_id} FAILED (rc={proc.returncode})", flush=True)
        return {"cell_id": cell_id, "status": "fail", "dt": dt,
                 "stderr_tail": (proc.stderr or "")[-500:]}

    if not Path(chkpt_file).exists():
        # checkpoint may have been written under a different name
        cands = list(Path(cell_dir).glob("*.prior")) + list(Path(cell_dir).glob("*.chkpt"))
        if cands:
            chkpt_file = str(cands[0])
        else:
            print(f"[exp6-rl] {cell_id} NO CHKPT FOUND", flush=True)
            return {"cell_id": cell_id, "status": "no_chkpt", "dt": dt}

    # 3. sample (5500 total per spec, distributed across seed pool)
    print(f"[exp6-rl] {cell_id} SAMPLE 5500 from {chkpt_file}", flush=True)
    sample_toml = render_sample_toml(chkpt_file, sample_csv, seed_file, num_smiles=5500,
                                      cell_id=cell_id)
    proc2 = subprocess.run(
        ["reinvent", sample_toml],
        cwd=PROJECT_ROOT,
        timeout=1800,
        capture_output=True,
        text=True,
    )
    with open(log_path, "a") as f:
        f.write("\n== SAMPLE STDOUT ==\n")
        f.write(proc2.stdout or "")
        f.write("\n== SAMPLE STDERR ==\n")
        f.write(proc2.stderr or "")
    if proc2.returncode != 0:
        print(f"[exp6-rl] {cell_id} SAMPLE FAILED rc={proc2.returncode}", flush=True)
        return {"cell_id": cell_id, "status": "sample_fail", "dt": dt}

    if not Path(sample_csv).exists():
        print(f"[exp6-rl] {cell_id} SAMPLE missing csv", flush=True)
        return {"cell_id": cell_id, "status": "no_sample", "dt": dt}

    # 4. cohort + score
    cohort_df = parse_sample_csv(sample_csv)
    cohort_df.to_csv(cohort_csv, index=False)
    score_cohort(cohort_csv, PORTS[(target, scorer)])

    metrics = compute_cell_metrics(cohort_csv, target)
    metrics.update({"cell_id": cell_id, "status": "ok", "dt": dt, "n_steps": n_steps})
    print(f"[exp6-rl] {cell_id} DONE  warhead_strict={metrics['warhead_strict_frac']:.3f}  "
          f"mean_pIC50={metrics['predicted_pIC50_mean']:.2f}  tc_drug_max={metrics['tc_drug_max']:.3f}  "
          f"dt={dt:.1f}s", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["egfr_t790m", "btk", "kras_g12c"])
    ap.add_argument("--strategies", nargs="+", default=["A", "B"])
    ap.add_argument("--iters", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--scorers", nargs="+", default=["film", "dabs"])
    ap.add_argument("--time_budget_h", type=float, default=10.0)
    ap.add_argument("--out_json", default=f"{EXP6_ROOT}/_rl/rl_results.json")
    args = ap.parse_args()

    Path(RL_OUT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(TOML_OUT_ROOT).mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()
    time_budget = args.time_budget_h * 3600.0

    # load any existing results so we can resume
    out_p = Path(args.out_json)
    if out_p.exists():
        try:
            results = json.loads(out_p.read_text())
            print(f"[exp6-rl] Resuming with {len(results)} existing results", flush=True)
        except Exception:
            results = []
    done_cells = {r.get("cell_id") for r in results if r.get("status") in ("ok", "resumed")}

    for target in args.targets:
        for strategy in args.strategies:
            for scorer in args.scorers:
                prev_cohort = None
                for iter_n in args.iters:
                    if time.time() - t_start > time_budget:
                        print(f"[exp6-rl] TIME BUDGET EXCEEDED ({args.time_budget_h}h)", flush=True)
                        Path(args.out_json).write_text(json.dumps(results, indent=2))
                        return
                    cell_id = f"{target}_{strategy}_iter{iter_n}_{scorer}"
                    if cell_id in done_cells:
                        # use existing cohort for promotion
                        prev_cohort = f"{EXP6_ROOT}/{target}/iter{iter_n}_{strategy}_{scorer}_cohort.csv"
                        continue
                    # n_steps trimmed for multi-seed cells to fit ~4h budget
                    # (each multi-seed step iterates through all seeds, so 25 steps
                    # x 50 seeds ~ 12 min at bs=8). Trim to 12 for iter2/3.
                    if iter_n == 1 and strategy == "A":
                        n_steps = 50
                    elif iter_n == 1:
                        n_steps = 25
                    else:
                        n_steps = 12
                    # bs sizing: A-iter1 (1 seed) fits at 32; everything else use bs=8.
                    if strategy == "A" and iter_n == 1:
                        bs = 32
                    else:
                        bs = 8
                    m = run_cell(target, strategy, scorer, iter_n, prev_cohort,
                                 n_steps=n_steps, batch_size=bs,
                                 time_budget_sec=int(time_budget - (time.time() - t_start)))
                    results.append(m)
                    Path(args.out_json).write_text(json.dumps(results, indent=2))
                    if m.get("status") in ("ok", "resumed"):
                        prev_cohort = f"{EXP6_ROOT}/{target}/iter{iter_n}_{strategy}_{scorer}_cohort.csv"
                    else:
                        # don't promote from failed cell — abort the chain
                        break

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"[exp6-rl] ALL DONE. {len(results)} cells. results -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
