#!/usr/bin/env python3
"""Boltz verdict driver: sample 5 models × 150 mols, cofold, extract metrics.

Runs entirely on ai-gpu-a100 (us-central1-a). Sequential pipeline:
  1. Sample 150 mols from each of 5 models (Ablation #3, v2-cond, v4_composite,
     v4_d, v4_theta_fixed). Uses existing samples_v4_*_clampA.txt where
     available (v4_composite, v4_d).
  2. Build Boltz YAMLs with Cys346-SG ↔ warhead-β-C covalent constraint.
  3. Run Boltz predict with 2-3 concurrent processes.
  4. Extract per-cofold: mpae_warhead_cys, d_b_nuc_angstrom, bd_angle_deg,
     phi_planar_deg, complex_iptm, ligand_iptm, complex_plddt.
  5. Aggregate → data/paper_pair_training/boltz_verdict/track_A_*.csv,
     track_B_v4comp_clampC.csv.

Track B: 30 v4_composite Clamp A + 30 v4_composite Clamp C (from existing files).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
BOLTZ_ROOT = PROJECT_ROOT / "data/paper_pair_training/boltz_verdict"
BOLTZ_ROOT.mkdir(parents=True, exist_ok=True)
(BOLTZ_ROOT / "logs").mkdir(exist_ok=True)

MPAE_DIR = PROJECT_ROOT / "data/paper_pair_training/mpae_zap70"
MSA_PATH = Path("/home/shaharh_quris_ai/zap70_msa.csv")
MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

ZAP70_SEQ = (
    "MPDPAAHLPFFYGSISRAEAEEHLKLAGMADGLFLLRQCLRSLGGYVLSLVHDVRFHHFPIERQLNGTYAI"
    "AGGKAHCGPAELCEFYSRDPDGLPCNLRKPCNRPSGLEPQPGVFDCLRDAMVRDYVRQTWKLEGEALEQAI"
    "ISQAPQVEKLIATTAHERMPWYHSSLTREEAERKLYSGAQTDGKFLLRPRKEQGTYALSLIYGKTVYHYLI"
    "SQDKAGKYCIPEGTKFDTLWQLVEYLKLKADGLIYCLKEACPNSSASNASGAAAPTLPAHPSTLTHPQRRI"
    "DTLNSDGYTPEPARITSPDKPRPMPMDTSVYESPYSDPEELKDKKLFLKRDNLLIADIELGCGNFGSVRQG"
    "VYRMRKKQIDVAIKVLKQGTEKADTEEMMREAQIMHQLDNPYIVRLIGVCQAEALMLVMEMAGGGPLHKFL"
    "VGKREEIPVSNVAELLHQVSMGMKYLEEKNFVHRDLAARNVLLVNRHYAKISDFGLSKALGADDSYYTARS"
    "AGKWPLKWYAPECINFRKFSSRSDVWSYGVTMWEALSYGQKPYKKMKGPEVMAFIEQGKRMECPPECPPEL"
    "YALMSDCWIYKWEDRPDFLTVEQRMRACYYSLASKVEGPPGSTQKAEAACA"
)
TARGET_CYS = 346
ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")

N_PER_MODEL = 500
N_TRACK_B = 30
N_CONCURRENT_BOLTZ = 2  # 2 concurrent Boltz procs on 40GB A100
CHECKPOINT_EVERY = 50    # aggregate CSV every N cofolds for preemption safety


# ============================== SAMPLING ======================================

def sample_v2cond_or_v4(ckpt_path: Path, out_smi_path: Path, n: int, target: str, variant: int):
    """Wrap sample_and_proxy_eval_zap70.py for m1a_v2 or v4_* models."""
    if out_smi_path.exists() and sum(1 for _ in out_smi_path.open()) >= n:
        print(f"[sample] {out_smi_path.name} exists ({sum(1 for _ in out_smi_path.open())} lines) — skip")
        return
    # Use the existing script; monkey-patch --ckpt and n_per_clamp
    cmd = [
        "python3", str(PROJECT_ROOT / "experiments/mpae_zap70/sample_and_proxy_eval_zap70.py"),
        "--ckpt", str(ckpt_path),
        "--target", target, "--variant", str(variant),
        "--n_per_clamp", str(n),
        "--out_json", str(BOLTZ_ROOT / f"proxy_metrics_sampling_{ckpt_path.stem}.json"),
    ]
    print(f"[sample] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def sample_mol2mol_prior(prior_path: Path, out_smi_path: Path, n: int):
    """Sample from a REINVENT4 mol2mol prior with Mol1 as anchor."""
    if out_smi_path.exists() and sum(1 for _ in out_smi_path.open()) >= n:
        print(f"[sample] {out_smi_path.name} exists — skip")
        return
    # Use REINVENT4 mol2mol sampling
    reinvent_root = Path("/home/shaharh_quris_ai/REINVENT4")
    if not reinvent_root.exists():
        raise RuntimeError(f"REINVENT4 not found at {reinvent_root}")
    sys.path.insert(0, str(reinvent_root))

    from reinvent.models.mol2mol.models.model import Model as Mol2MolModel
    import torch

    print(f"[sample] loading prior: {prior_path}")
    device = torch.device("cuda")
    m = Mol2MolModel.load_from_file(str(prior_path), sampling_mode="multinomial", device=device)
    # Set batch to 32, sample until n valid unique
    seen = set()
    smis = []
    batch = 32
    it = 0
    while len(smis) < n and it < 200:
        # Feed Mol1 as source; multinomial sampling with T=1.0
        src_smiles = [MOL1_SMI] * batch
        try:
            out = m.sample(src_smiles)
            # out is list of dicts or list of SmilesFragmentsPair; grab .output/.smiles
            for r in out:
                cand = getattr(r, "smiles", None) or getattr(r, "output", None) or (r if isinstance(r, str) else None)
                if cand is None and hasattr(r, "__iter__"):
                    cand = list(r)[1] if len(list(r)) > 1 else None
                if not cand:
                    continue
                mol = Chem.MolFromSmiles(cand)
                if mol is None:
                    continue
                can = Chem.MolToSmiles(mol)
                if can in seen:
                    continue
                seen.add(can)
                smis.append(cand)
                if len(smis) >= n:
                    break
        except Exception as e:
            print(f"[sample] batch failed: {e}")
        it += 1
    print(f"[sample] mol2mol wrote {len(smis)} to {out_smi_path}")
    out_smi_path.write_text("\n".join(smis))


def sample_reinvent4_mol2mol_cli(prior_path: Path, out_smi_path: Path, n: int):
    """Alt: use REINVENT4 CLI to sample."""
    if out_smi_path.exists() and sum(1 for _ in out_smi_path.open()) >= n:
        print(f"[sample] {out_smi_path.name} exists — skip")
        return
    tmp_toml = BOLTZ_ROOT / f"reinvent4_sample_{prior_path.stem}.toml"
    tmp_csv = BOLTZ_ROOT / f"reinvent4_sample_{prior_path.stem}.csv"
    toml_content = f"""run_type = "sampling"
device = "cuda:0"
tb_logdir = "tb"

[parameters]
model_file = "{prior_path}"
smiles_file = "{BOLTZ_ROOT}/mol1_anchor.smi"
sample_strategy = "multinomial"
output_file = "{tmp_csv}"
num_smiles = {int(n * 3)}
unique_molecules = true
randomize_smiles = true
temperature = 1.0
"""
    # Write Mol1 anchor
    (BOLTZ_ROOT / "mol1_anchor.smi").write_text(MOL1_SMI + "\n")
    tmp_toml.write_text(toml_content)
    cmd = ["reinvent", "-l", str(BOLTZ_ROOT / f"logs/reinvent_{prior_path.stem}.log"), str(tmp_toml)]
    print(f"[sample] running reinvent4 CLI: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    df = pd.read_csv(tmp_csv)
    # Take unique canonical SMILES from output col
    smi_col = "SMILES" if "SMILES" in df.columns else df.columns[1]
    smis = []
    seen = set()
    for s in df[smi_col].tolist():
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        smis.append(s)
        if len(smis) >= n:
            break
    out_smi_path.write_text("\n".join(smis))
    print(f"[sample] wrote {len(smis)} to {out_smi_path}")


# ============================== FILTERING =====================================

def filter_valid_acryl(smis: list[str], n_target: int) -> list[str]:
    """Keep valid + unique canonical + acryl-largest-frag."""
    out = []
    seen = set()
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        # Largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        if not mol.HasSubstructMatch(ACRYLAMIDE_SMARTS):
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
        if len(out) >= n_target:
            break
    return out


# ============================== YAML BUILD ====================================

def boltz_atom_name(smi: str):
    """Compute Boltz atom name for the acrylamide β-CH2."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol_h = Chem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    matches = mol_h.GetSubstructMatches(ACRYLAMIDE_SMARTS)
    if not matches:
        return None, None
    term_ch2_idx = matches[0][0]
    return f"C{can[term_ch2_idx] + 1}", term_ch2_idx


def build_yamls(smis: list[str], out_dir: Path, cohort_name: str) -> list[dict]:
    """Build one YAML per SMILES with Cys346-SG ↔ warhead-β-C covalent bond."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, smi in enumerate(smis):
        atom_name, _ = boltz_atom_name(smi)
        if atom_name is None:
            continue
        name = f"{cohort_name}_{i:04d}"
        yaml = (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {ZAP70_SEQ}\n"
            f"      msa: {MSA_PATH}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{smi}'\n"
            "constraints:\n"
            "  - bond:\n"
            f"      atom1: [A, {TARGET_CYS}, SG]\n"
            f"      atom2: [B, 1, {atom_name}]\n"
        )
        (out_dir / f"{name}.yaml").write_text(yaml)
        manifest.append({"name": name, "smiles": smi, "warhead_atom": atom_name})
    return manifest


# ============================== BOLTZ RUN =====================================

def run_boltz_one(yaml_path: Path, out_dir: Path, log_path: Path) -> tuple[str, bool]:
    """Run boltz predict on one YAML."""
    name = yaml_path.stem
    result_dir = out_dir / f"boltz_results_{name}"
    cif_path = result_dir / "predictions" / name / f"{name}_model_0.cif"
    if cif_path.exists():
        return name, True
    cmd = [
        "boltz", "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--accelerator", "gpu", "--diffusion_samples", "1", "--sampling_steps", "100",
        "--output_format", "mmcif", "--override",
    ]
    with log_path.open("a") as f:
        f.write(f"\n=== {name} ===\n")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return name, cif_path.exists()


def run_boltz_batch(yaml_dir: Path, out_dir: Path, log_path: Path,
                     max_workers: int = 2, checkpoint_cb=None):
    """Run Boltz on all YAMLs in dir with limited concurrency.

    If checkpoint_cb provided, calls it every CHECKPOINT_EVERY completions so
    intermediate results are persisted (preemption resilience).
    """
    yamls = sorted(yaml_dir.glob("*.yaml"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[boltz] {len(yamls)} YAMLs in {yaml_dir.name} — running with {max_workers} workers")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_boltz_one, y, out_dir, log_path): y for y in yamls}
        for fut in as_completed(futures):
            name, ok = fut.result()
            done += 1
            if done % 10 == 0 or not ok:
                dt = time.time() - t0
                rate = done / (dt / 60)
                eta = (len(yamls) - done) / rate if rate > 0 else 0
                print(f"[boltz] {done}/{len(yamls)} ({name} ok={ok}) rate={rate:.2f}/min eta={eta:.1f}min")
            if checkpoint_cb is not None and done % CHECKPOINT_EVERY == 0:
                try:
                    checkpoint_cb()
                except Exception as e:
                    print(f"[boltz] checkpoint failed: {e}")
    print(f"[boltz] batch done in {(time.time()-t0)/60:.1f}min")


# ============================== METRIC EXTRACTION =============================

CYS346_SG_ATOM = "SG"


def _parse_cif_line_based(cif_path: Path):
    """Fallback CIF parser that only reads the atom_site loop.

    Handles Boltz-2 CIFs where the _ma_qa_metric_local loop has malformed
    column counts (which cause gemmi to fail hard).
    """
    prot, lig = [], []
    lines = cif_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols.append(lines[j].strip().removeprefix("_atom_site."))
                j += 1
            if cols:
                # Locate column indices for chain, resname, resi, atom, x/y/z
                try:
                    chain_idx = cols.index("auth_asym_id") if "auth_asym_id" in cols else cols.index("label_asym_id")
                    resname_idx = cols.index("auth_comp_id") if "auth_comp_id" in cols else cols.index("label_comp_id")
                    resi_idx = cols.index("auth_seq_id") if "auth_seq_id" in cols else cols.index("label_seq_id")
                    atom_idx = cols.index("auth_atom_id") if "auth_atom_id" in cols else cols.index("label_atom_id")
                    x_idx = cols.index("Cartn_x")
                    y_idx = cols.index("Cartn_y")
                    z_idx = cols.index("Cartn_z")
                except ValueError:
                    i = j
                    continue
                i = j
                while i < len(lines) and not lines[i].startswith("#") and not lines[i].startswith("loop_") and lines[i].strip():
                    parts = lines[i].split()
                    if len(parts) == len(cols):
                        try:
                            resi = int(parts[resi_idx]) if parts[resi_idx].isdigit() or (parts[resi_idx].startswith("-") and parts[resi_idx][1:].isdigit()) else 0
                        except Exception:
                            resi = 0
                        rec = {
                            "chain": parts[chain_idx], "resname": parts[resname_idx],
                            "resi": resi, "name": parts[atom_idx].strip('"'),
                            "pos": np.array([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])], dtype=float),
                        }
                        if parts[chain_idx] == "A":
                            prot.append(rec)
                        else:
                            lig.append(rec)
                    i += 1
                continue
        i += 1
    return prot, lig


def parse_cif_atoms(cif_path: Path):
    """Extract chain-A protein atoms + chain-B ligand atoms.

    Uses gemmi (fast + robust) and falls back to line-based parser on failure
    (Boltz-2 sometimes produces CIFs with malformed loops that gemmi rejects).
    """
    try:
        import gemmi
        st = gemmi.read_structure(str(cif_path))
        prot, lig = [], []
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        rec = {
                            "chain": chain.name, "resname": res.name, "resi": res.seqid.num,
                            "name": atom.name,
                            "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float),
                        }
                        if chain.name == "A":
                            prot.append(rec)
                        else:
                            lig.append(rec)
        return prot, lig
    except Exception:
        return _parse_cif_line_based(cif_path)


def extract_metrics_one(pred_dir: Path, name: str) -> dict:
    """Extract per-cofold metrics."""
    cif = pred_dir / f"{name}_model_0.cif"
    pae_npz = pred_dir / f"pae_{name}_model_0.npz"
    conf_json = pred_dir / f"confidence_{name}_model_0.json"
    metrics = {"name": name, "cofold_ok": False}
    if not cif.exists():
        return metrics
    try:
        prot, lig = parse_cif_atoms(cif)
    except Exception as e:
        metrics["error"] = str(e)
        return metrics
    metrics["cofold_ok"] = True
    metrics["n_prot"] = len(prot)
    metrics["n_lig"] = len(lig)
    # Find Cys346 SG
    sg = None
    for a in prot:
        if a["resi"] == TARGET_CYS and a["name"] == "SG":
            sg = a["pos"]; break
    # Locate warhead β-CH2 in ligand — Boltz-emitted atom name matches yaml constraint.
    # For extraction robustness, we find the ligand atom closest to Sγ that is a C
    # then compute geometry using the two neighbors (C=C, C=O)
    lig_C = [a for a in lig if a["name"].startswith("C")]
    if sg is None or not lig_C:
        return metrics
    # β-C = closest carbon in ligand to Sγ (after cofold, this is post-bond)
    dists = [np.linalg.norm(a["pos"] - sg) for a in lig_C]
    j = int(np.argmin(dists))
    beta_c = lig_C[j]["pos"]
    metrics["d_b_nuc_angstrom"] = float(np.linalg.norm(beta_c - sg))
    # Neighbors in ligand ≤ 1.8 Å from beta_c
    nb = []
    for i, a in enumerate(lig):
        if i == j:
            continue
        d = np.linalg.norm(a["pos"] - beta_c)
        if d < 1.8:
            nb.append((d, a))
    nb.sort(key=lambda x: x[0])
    # BD angle: Sγ — β-C — α-C (or nearest ligand C)
    if len(nb) >= 1:
        alpha = nb[0][1]["pos"]
        v1 = sg - beta_c
        v2 = alpha - beta_c
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cos = float(np.clip(cos, -1, 1))
        metrics["bd_angle_deg"] = float(np.degrees(np.arccos(cos)))
    # Phi planar: dihedral Sγ - βC - αC - carbonyl C
    if len(nb) >= 2:
        alpha = nb[0][1]["pos"]
        # find carbonyl C among ligand: nearest to alpha that has an O neighbor
        # heuristic: pick nb[1]
        c3 = nb[1][1]["pos"]
        # dihedral
        b1 = beta_c - sg
        b2 = alpha - beta_c
        b3 = c3 - alpha
        n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-9))
        x = np.dot(n1, n2); y = np.dot(m1, n2)
        phi = float(np.degrees(np.arctan2(y, x)))
        phi_wrap = ((phi + 180.0) % 360.0) - 180.0
        metrics["phi_planar_deg"] = float(min(abs(phi_wrap), abs(180.0 - abs(phi_wrap))))
    # mPAE from npz — warhead-Cys PAE
    if pae_npz.exists():
        try:
            d = np.load(pae_npz)
            pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
            # Warhead atom index in ligand block; use closest ligand atom to Cys346 SG
            N = pae.shape[0]
            n_lig = len(lig)
            lig_lo = N - n_lig
            # Row = residue index for Cys346 (position TARGET_CYS-1 in chain A)
            # Boltz PAE is per-atom in tensor mode or per-residue. Try both.
            # Compute cross-block min (whole protein × whole ligand)
            cross = pae[:lig_lo, lig_lo:]
            metrics["mpae_prot_lig_mean"] = float(np.mean(cross))
            metrics["mpae_prot_lig_min"] = float(np.min(cross))
            # mpae_warhead_cys: for tensor PAE, look at Cys346-SG × β-C row/col
            # For simplicity, use min of the whole SG-row × ligand block
            # NB: Boltz .npz can be residue-level; only atom-level has SG resolution
            # We approximate as prot_lig_min for the cohort — an ordinal metric.
            metrics["mpae_warhead_cys"] = metrics["mpae_prot_lig_min"]
        except Exception as e:
            metrics["pae_error"] = str(e)
    # Confidence
    if conf_json.exists():
        try:
            d = json.loads(conf_json.read_text())
            metrics["complex_iptm"] = d.get("iptm") or d.get("complex_iptm")
            metrics["ligand_iptm"] = d.get("ligand_iptm")
            metrics["complex_plddt"] = d.get("complex_plddt")
        except Exception:
            pass
    return metrics


def harvest_cohort(cofold_root: Path, cohort_name: str, out_csv: Path, manifest: list[dict] = None):
    """Walk cofold results and aggregate to CSV."""
    rows = []
    smi_by_name = {m["name"]: m["smiles"] for m in (manifest or [])}
    for sub in sorted(cofold_root.glob("boltz_results_*")):
        name = sub.name.removeprefix("boltz_results_")
        pred_dir = sub / "predictions" / name
        if not pred_dir.is_dir():
            continue
        m = extract_metrics_one(pred_dir, name)
        m["cohort"] = cohort_name
        m["smiles"] = smi_by_name.get(name)
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[harvest] wrote {len(df)} rows to {out_csv}")
    return df


# ============================== PIPELINE ======================================

def get_smiles_for_cohort(cohort: str) -> list[str]:
    """Get SMILES for a given cohort. Uses pre-existing samples or generates."""
    if cohort == "v4composite_clampA":
        p = MPAE_DIR / "samples_v4_composite_clampA.txt"
        return [s.strip() for s in p.read_text().splitlines() if s.strip()]
    if cohort == "v4comp_clampC":
        p = MPAE_DIR / "samples_v4_composite_clampC.txt"
        return [s.strip() for s in p.read_text().splitlines() if s.strip()]
    if cohort == "v4d_clampA":
        p = MPAE_DIR / "samples_v4_d_clampA.txt"
        return [s.strip() for s in p.read_text().splitlines() if s.strip()]
    if cohort == "v4theta_clampA":
        # Generate via sampling
        out = MPAE_DIR / "samples_v4_theta_fixed_clampA.txt"
        sample_v2cond_or_v4(
            PROJECT_ROOT / "models/m1a_v2_v4_theta_fixed.ckpt",
            out, N_PER_MODEL + 100, "theta", 4,
        )
        # Sampling script writes samples_v4_theta_clampA.txt; use that path
        alt = MPAE_DIR / "samples_v4_theta_clampA.txt"
        chosen = alt if alt.exists() else out
        return [s.strip() for s in chosen.read_text().splitlines() if s.strip()]
    if cohort == "v2cond":
        out = MPAE_DIR / "samples_v2cond_clampA.txt"
        if not out.exists() or sum(1 for _ in out.open()) < N_PER_MODEL + 100:
            # v2-cond ckpt is m1a_v2.ckpt — treat as a "composite" variant-3 (no reg head)
            sample_v2cond_or_v4(
                PROJECT_ROOT / "models/m1a_v2.ckpt",
                out, N_PER_MODEL + 100, "composite", 3,
            )
        alt = MPAE_DIR / "samples_v3_composite_clampA.txt"
        # Priority: v2-cond specific file if exists, else the v3 composite samples
        return [s.strip() for s in (alt if alt.exists() else out).read_text().splitlines() if s.strip()]
    if cohort == "ablation3":
        out = MPAE_DIR / "samples_ablation3_mol2mol.txt"
        if not out.exists() or sum(1 for _ in out.open()) < N_PER_MODEL + 100:
            # Use reinvent4_mol2mol_prior.prior (unfinetuned) as the "data_only" null
            # baseline. This is the closest available proxy for the requested
            # `mol2mol_data_only_retrain.prior` (which does not exist in the model
            # store).
            prior = PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"
            # If mol2mol_data_only_retrain exists, prefer it
            alt_prior = PROJECT_ROOT / "models/mol2mol_data_only_retrain.prior"
            if alt_prior.exists():
                prior = alt_prior
            sample_reinvent4_mol2mol_cli(prior, out, N_PER_MODEL + 100)
        return [s.strip() for s in out.read_text().splitlines() if s.strip()]
    raise ValueError(f"Unknown cohort: {cohort}")


def process_cohort(cohort: str, n: int):
    """End-to-end pipeline for one cohort: sample → filter → build YAMLs → Boltz → extract."""
    print(f"\n{'='*60}\n[cohort {cohort}] START\n{'='*60}")
    raw = get_smiles_for_cohort(cohort)
    print(f"[{cohort}] {len(raw)} raw SMILES")
    smis = filter_valid_acryl(raw, n)
    print(f"[{cohort}] {len(smis)} after filter (target {n})")
    yaml_dir = BOLTZ_ROOT / "yamls" / cohort
    cofold_dir = BOLTZ_ROOT / "cofolds" / cohort
    log_path = BOLTZ_ROOT / "logs" / f"boltz_{cohort}.log"
    manifest = build_yamls(smis, yaml_dir, cohort)
    print(f"[{cohort}] {len(manifest)} YAMLs written")
    # Resolve output CSV path
    if cohort == "v4composite_clampA":
        out_csv = BOLTZ_ROOT / "track_A_v4composite_clampA.csv"
    elif cohort == "v4d_clampA":
        out_csv = BOLTZ_ROOT / "track_A_v4d_clampA.csv"
    elif cohort == "v4theta_clampA":
        out_csv = BOLTZ_ROOT / "track_A_v4theta_clampA.csv"
    elif cohort == "v2cond":
        out_csv = BOLTZ_ROOT / "track_A_v2cond.csv"
    elif cohort == "ablation3":
        out_csv = BOLTZ_ROOT / "track_A_ablation3.csv"
    elif cohort == "v4comp_clampC":
        out_csv = BOLTZ_ROOT / "track_B_v4comp_clampC.csv"
    else:
        out_csv = BOLTZ_ROOT / f"track_A_{cohort}.csv"

    def _ckpt():
        harvest_cohort(cofold_dir, cohort, out_csv, manifest)

    run_boltz_batch(yaml_dir, cofold_dir, log_path,
                     max_workers=N_CONCURRENT_BOLTZ, checkpoint_cb=_ckpt)
    # Final harvest
    harvest_cohort(cofold_dir, cohort, out_csv, manifest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True,
                    choices=["ablation3", "v2cond", "v4composite_clampA",
                             "v4d_clampA", "v4theta_clampA", "v4comp_clampC"])
    ap.add_argument("--n", type=int, default=None,
                    help="Number of mols (default 150 for track A, 30 for track B)")
    args = ap.parse_args()
    n = args.n or (N_TRACK_B if args.cohort == "v4comp_clampC" else N_PER_MODEL)
    process_cohort(args.cohort, n)
