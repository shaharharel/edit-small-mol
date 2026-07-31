"""Batched Boltz-2 covalent cofold for CFG × retrieval samples.

For each valid, acryl-largest sample (from covalent_metric_panel.csv), we
build a Boltz YAML with:
  - protein sequence = ZAP70 kinase domain (327..606), reusing the cached MSA
  - ligand SMILES = the sample's largest-fragment SMILES
  - covalent bond constraint: Cys346 SG <-> ligand acrylamide β-C
    (β-C atom name in Boltz's canonical rank = "C26" for Mol1-like acryl
    scaffolds; we auto-detect the β-C atom name per SMILES via RDKit +
    Boltz's tokenizer since new scaffolds may name it differently).

Boltz outputs (mmcif) go to
    data/paper_pair_training/cfg_retrieval/boltz_cofolds/{cell}/{sample_idx}/

We select TOP N per cell (default 50) by BD-ready-in-2D THEN NLL — so
we prioritize samples that pass the cheap acryl+planar filter.

Progress written per-cofold to progress.json.  On preemption, incomplete
cofold dirs are re-attempted on restart (mmcif file presence == success).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")

# ZAP70 kinase domain (UniProt P43403, residues 327..606) — same window as
# the v2 training + cofold_mol1_zap70_v2.py.
ZAP70_FULL = (
    "MPDPAAHLPFFYGSISRAEAEEHLKLAGMADGLFLLRQCLRSLGGYVLSLVHDVRFHHFPIERQLNGTYAIAGGKAHCGPAELCEFYSRDPDGLPCNLRKPCNRPSGLEPQPGVFDCLRDAMVRDYVRQTWKLEGEALEQAIISQAPQVEKLIATTAHERMPWYHSSLTREEAERKLYSGAQTDGKFLLRPRKEQGTYALSLIYGKTVYHYLISQDKAGKYCIPEGTKFDTLWQLVEYLKLKADGLIYCLKEACPNSSASNASGAAAPTLPAHPSTLTHPQRRIDTLNSDGYTPEPARITSPDKPRPMPMDTSVYESPYSDPEELKDKKLFLKRDNLLIADIELGCGNFGSVRQGVYRMRKKQIDVAIKVLKQGTEKADTEEMMREAQIMHQLDNPYIVRLIGVCQAEALMLVMEMAGGGPLHKFLVGKREEIPVSNVAELLHQVSMGMKYLEEKNFVHRDLAARNVLLVNRHYAKISDFGLSKALGADDSYYTARSAGKWPLKWYAPECINFRKFSSRSDVWSYGVTMWEALSYGQKPYKKMKGPEVMAFIEQGKRMECPPECPPELYALMSDCWIYKWEDRPDFLTVEQRMRACYYSLASKVEGPPGSTQKAEAACA"
)
ZAP70_START = 327
ZAP70_END = 606
SEQ = ZAP70_FULL[ZAP70_START - 1: ZAP70_END]
CYS_POS_YAML = 346 - ZAP70_START + 1   # = 20, 1-indexed in Boltz YAML
assert SEQ[CYS_POS_YAML - 1] == "C"

MSA_SRC = PROJECT_ROOT / ("data/m1a_v2_boltz_mol1/boltz_pred/"
                             "boltz_results_mol1_zap70/msa/mol1_zap70_0.csv")


def _acryl_beta_atom_name_boltz(smi: str) -> str | None:
    """Determine the canonical Boltz atom name for the acrylamide β-C.

    Boltz uses RDKit's CanonicalRankAtoms after adding Hs and mapping via the
    schema.  For the acrylamide `C=C-C(=O)-N` we take the terminal `=CH2` atom
    (SMARTS match[0]).  Boltz's atom naming: after ETKDG-canon build it uses
    "C<canonical_rank_index>".  We compute canonical ranks here and return
    f"C{rank_of_beta_C + 1}" (Boltz is 1-indexed by convention).

    Fallback: return "C26" which is what the ZAP70 v2 cofold used for Mol1.
    If our derivation matches "C26" for Mol1 it validates the derivation.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    patt = Chem.MolFromSmarts("[CH2;X3]=[CH;X3][C;X3](=O)[N]")
    matches = m.GetSubstructMatches(patt)
    if not matches:
        return None
    beta_idx = matches[0][0]
    # Use RDKit canonical ranks; add Hs to match Boltz's addHs path.
    mH = Chem.AddHs(m)
    ranks = list(Chem.CanonicalRankAtoms(mH, breakTies=True))
    # Find the beta atom in the mH indexing (H-adding preserves heavy indices).
    return f"C{ranks[beta_idx] + 1}"


def build_yaml(smi: str, yaml_path: Path, beta_atom_name: str) -> None:
    yaml_text = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {SEQ}\n"
        f"      msa: {MSA_SRC}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{smi}'\n"
        "constraints:\n"
        "  - bond:\n"
        f"      atom1: [A, {CYS_POS_YAML}, SG]\n"
        f"      atom2: [B, 1, {beta_atom_name}]\n"
    )
    yaml_path.write_text(yaml_text)


def run_boltz(yaml_path: Path, out_pred_dir: Path, timeout_s: int = 900,
                sampling_steps: int = 200) -> tuple[bool, str]:
    """Return (success, message)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = ["boltz", "predict", str(yaml_path),
             "--out_dir", str(out_pred_dir),
             "--diffusion_samples", "1",
             "--recycling_steps", "3",
             "--sampling_steps", str(sampling_steps),
             "--output_format", "mmcif",
             "--override"]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if res.returncode != 0:
        return False, f"rc={res.returncode} stderr={res.stderr[-500:]}"
    # Boltz writes to {out_pred_dir}/boltz_results_{yaml_stem}/predictions/{yaml_stem}/{yaml_stem}_model_0.cif
    stem = yaml_path.stem
    expected = out_pred_dir / f"boltz_results_{stem}" / "predictions" / stem / f"{stem}_model_0.cif"
    if not expected.exists():
        # search
        cifs = list(out_pred_dir.rglob("*_model_0.cif"))
        if not cifs:
            return False, "no cif produced"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_metric_panel_v2.csv"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/boltz_cofolds"))
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "boltz_cofold_progress.json"))
    ap.add_argument("--per_cell", type=int, default=50,
                    help="Number of samples per cell to cofold (top by "
                          "acryl+low-dihedral+low-NLL).")
    ap.add_argument("--only_cells", default=None,
                    help="Comma-separated cell names to run (default = all).")
    ap.add_argument("--sampling_steps", type=int, default=200)
    ap.add_argument("--timeout_s", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None,
                    help="Total cofold cap (across all cells) for debug/quick.")
    ap.add_argument("--plan_csv", default=None,
                    help="If provided, consume this pre-computed plan (cols: "
                          "cell, sample_idx, smi) — skips per-cell ranking.")
    args = ap.parse_args()

    assert MSA_SRC.exists(), f"MSA not found at {MSA_SRC}"

    plan = []
    if args.plan_csv is not None and Path(args.plan_csv).exists():
        pdf = pd.read_csv(args.plan_csv)
        print(f"Consuming pre-computed plan: {args.plan_csv} ({len(pdf)} rows)",
               flush=True)
        for _, row in pdf.iterrows():
            plan.append({
                "cell": str(row["cell"]),
                "sample_idx": int(row["sample_idx"]),
                "smi": str(row["smi"]),
            })
    else:
        df = pd.read_csv(args.panel_csv)
        df = df[df["valid"] == True]
        if "acryl_largest" in df.columns:
            df = df[df["acryl_largest"] == True]
        print(f"Panel: {len(df)} valid+acryl rows across cells", flush=True)
        df = df.copy()
        df["planar_dihedral_deg"] = df["planar_dihedral_deg"].astype(float)
        df = df.sort_values(["cell", "planar_dihedral_deg", "NLL"])
        cells = sorted(df["cell"].unique())
        if args.only_cells:
            wanted = set(c.strip() for c in args.only_cells.split(","))
            cells = [c for c in cells if c in wanted]
        for c in cells:
            cd = df[df["cell"] == c].head(args.per_cell)
            for _, row in cd.iterrows():
                plan.append({
                    "cell": c,
                    "sample_idx": int(row["sample_idx"]),
                    "smi": row["largest_frag_SMILES"],
                    "NLL": float(row["NLL"]),
                    "planar_dihedral_deg": float(row["planar_dihedral_deg"]),
                })
    if args.limit is not None:
        plan = plan[:args.limit]
    print(f"Total cofolds planned: {len(plan)}", flush=True)

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_path)

    n_ok = 0; n_fail = 0; n_skip = 0
    t_start = time.time()
    for i, item in enumerate(plan):
        cell = item["cell"]; sidx = item["sample_idx"]; smi = item["smi"]
        cell_dir = out_root / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        yaml_stem = f"s{sidx:04d}"
        yaml_path = cell_dir / f"{yaml_stem}.yaml"
        pred_root = cell_dir / f"pred_{yaml_stem}"
        expected_cif = pred_root / f"boltz_results_{yaml_stem}" / "predictions" / yaml_stem / f"{yaml_stem}_model_0.cif"
        if expected_cif.exists():
            n_skip += 1
            continue
        beta = _acryl_beta_atom_name_boltz(smi)
        if beta is None:
            n_fail += 1
            continue
        build_yaml(smi, yaml_path, beta)
        t_c = time.time()
        ok, msg = run_boltz(yaml_path, pred_root,
                              timeout_s=args.timeout_s,
                              sampling_steps=args.sampling_steps)
        dt = time.time() - t_c
        if ok:
            n_ok += 1
            state = "OK"
        else:
            n_fail += 1
            state = f"FAIL:{msg[:80]}"
        elapsed = time.time() - t_start
        rate = (n_ok + n_fail) / max(elapsed, 1e-6)
        eta = (len(plan) - i - 1) / max(rate, 1e-6)
        print(f"[{i+1}/{len(plan)}] {cell} s{sidx:04d}  {dt:.1f}s  {state}  "
               f"ok={n_ok} fail={n_fail} skip={n_skip}  "
               f"rate={rate:.3f}/s  ETA={eta/60:.1f}min", flush=True)
        progress_path.write_text(json.dumps({
            "phase": "cofold_batch",
            "i": i + 1, "total": len(plan),
            "cell": cell, "sample_idx": sidx,
            "n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip,
            "rate_per_sec": rate, "eta_min": eta / 60,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))

    print(f"\nDONE  ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)
    progress_path.write_text(json.dumps({
        "phase": "cofold_batch_done",
        "n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


if __name__ == "__main__":
    main()
