"""Plan E: DiffSBDD baseline for ZAP70 Cys346 on local Mac (MPS).

Generates 250 covalent-conditioned mols using DiffSBDD inpaint with
warhead-at-Cys346 fixed atoms, applies the 3 gotcha fixes from
memory/diffsbdd_inpaint_gotchas.md, scores with FiLMDelta, and writes
metrics + CSV.

Output:
  data/genplan_e_diffsbdd/molecules.csv       per-mol metrics
  data/genplan_e_diffsbdd/raw_pre_fix.sdf     raw inpaint output
  data/genplan_e_diffsbdd/fixed.sdf           warhead-bond-fixed output
  data/genplan_e_summary.json                 headline summary
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSBDD_ROOT = PROJECT_ROOT / "external" / "DiffSBDD"
ANCHORDIFF_ROOT = PROJECT_ROOT / "anchordiff"
POCKET_ROOT = ANCHORDIFF_ROOT / "pockets" / "zap70_cys346"
OUT_ROOT = PROJECT_ROOT / "data" / "genplan_e_diffsbdd"
SUMMARY_PATH = PROJECT_ROOT / "data" / "genplan_e_summary.json"
PROGRESS_PATH = PROJECT_ROOT / "data" / "genplan_e_progress.log"

sys.path.insert(0, str(DIFFSBDD_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Descriptors, QED  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
RDLogger.DisableLog("rdApp.*")

import argparse as _ap  # noqa: E402
torch.serialization.add_safe_globals([_ap.Namespace])


def log_progress(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    line = f"{ts} {msg}\n"
    print(line.strip(), flush=True)
    with open(PROGRESS_PATH, "a") as f:
        f.write(line)


# ------------------------------------------------------------ gotcha #3 check
def cys_sg_warhead_distance(pdb_file: Path, warhead_sdf: Path, cys_resi: int = 346):
    """Verify warhead Cβ (atom 0) is at prereactive distance from Cys346 SG."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", str(pdb_file))[0]
    sg_coord = None
    for chain in struct:
        for residue in chain:
            if residue.get_resname() == "CYS" and residue.get_id()[1] == cys_resi:
                for atom in residue:
                    if atom.get_name() == "SG":
                        sg_coord = np.array(atom.get_coord())
    if sg_coord is None:
        return None, None
    rdmol = Chem.SDMolSupplier(str(warhead_sdf), sanitize=False)[0]
    cb = np.array(list(rdmol.GetConformer().GetAtomPosition(0)))
    d = float(np.linalg.norm(sg_coord - cb))
    return d, sg_coord


# ------------------------------------------------------------ sampling
def run_inpaint(n_total: int, batch_size: int, timesteps: int, resamplings: int,
                device: str, add_n_nodes: int):
    """Run DiffSBDD inpaint in batches; return list of RDKit mols (raw)."""
    from inpaint import inpaint_ligand
    from lightning_modules import LigandPocketDDPM

    ckpt_path = DIFFSBDD_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    pdb = POCKET_ROOT / "receptor.pdb"
    warhead_sdf = POCKET_ROOT / "warhead_at_cys.sdf"

    log_progress(f"INPAINT_LOAD device={device} ckpt={ckpt_path.name}")
    model = LigandPocketDDPM.load_from_checkpoint(
        str(ckpt_path), map_location="cpu", strict=False
    ).to(device)
    model.eval()

    all_mols: list = []
    n_batches = (n_total + batch_size - 1) // batch_size
    for b in range(n_batches):
        n_this = min(batch_size, n_total - len(all_mols))
        t0 = time.time()
        try:
            mols = inpaint_ligand(
                model, str(pdb), n_samples=n_this, ligand=str(warhead_sdf),
                fix_atoms=[str(warhead_sdf)], add_n_nodes=add_n_nodes,
                center="ligand", sanitize=False, largest_frag=False,
                relax_iter=0, timesteps=timesteps, resamplings=resamplings,
                save_traj=False,
            )
        except Exception as e:
            log_progress(f"BATCH_FAIL batch={b}/{n_batches} err={type(e).__name__}: {e}")
            traceback.print_exc()
            # MPS-specific fallback: retry once on CPU if MPS errored mid-loop
            if device == "mps":
                log_progress(f"FALLBACK_CPU batch={b}")
                model = model.to("cpu")
                device = "cpu"
                try:
                    mols = inpaint_ligand(
                        model, str(pdb), n_samples=n_this, ligand=str(warhead_sdf),
                        fix_atoms=[str(warhead_sdf)], add_n_nodes=add_n_nodes,
                        center="ligand", sanitize=False, largest_frag=False,
                        relax_iter=0, timesteps=timesteps, resamplings=resamplings,
                    )
                except Exception as e2:
                    log_progress(f"CPU_FALLBACK_ALSO_FAILED {type(e2).__name__}: {e2}")
                    continue
            else:
                continue
        dt = time.time() - t0
        all_mols.extend(mols)
        log_progress(f"BATCH {b + 1}/{n_batches} n={len(mols)} t={dt:.1f}s "
                     f"({dt / max(1, len(mols)):.2f}s/mol) total={len(all_mols)}")
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    return all_mols


# ------------------------------------------------------------ gotcha #2 fix
def apply_warhead_bond_fix(mols, debug: bool = False):
    """Use the existing anchordiff/fix_inpaint_warhead_bonds.fix_warhead_bonds.

    Raw inpaint mols are unsanitized (no implicit-valence info), which crashes
    the bond fixer. Workaround: write to an in-memory SDF and re-read, which
    forces minimal preprocessing identical to the canonical CLI path.
    """
    sys.path.insert(0, str(ANCHORDIFF_ROOT))
    from fix_inpaint_warhead_bonds import fix_warhead_bonds
    import io
    fixed = []
    for mol in mols:
        if mol is None:
            fixed.append(None)
            continue
        try:
            # Round-trip via SDF to materialize bond-perception state used by fixer
            sio = io.StringIO()
            w = Chem.SDWriter(sio)
            w.write(mol)
            w.close()
            sdf_str = sio.getvalue()
            suppl = Chem.SDMolSupplier()
            suppl.SetData(sdf_str, sanitize=False, removeHs=False)
            m2 = next(iter(suppl), None)
            if m2 is None:
                fixed.append(None)
                continue
            out = fix_warhead_bonds(m2)
            fixed.append(out)
        except Exception as e:
            if debug:
                log_progress(f"BOND_FIX_FAIL {type(e).__name__}: {e}")
            fixed.append(None)
    return fixed


# ------------------------------------------------------------ metrics
ACRYL_SMARTS_STRICT = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
ACRYL_SMARTS_RELAXED = Chem.MolFromSmarts("C=CC(=O)N")
# headline metric per memory: largest fragment match, NOT anywhere
ACRYL_SMARTS = ACRYL_SMARTS_RELAXED


def largest_fragment(mol):
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        return max(frags, key=lambda m: m.GetNumHeavyAtoms())
    except Exception:
        return None


def n_heavy(m):
    return 0 if m is None else m.GetNumHeavyAtoms()


def compute_metrics_one(mol):
    out = {
        "smiles": "",
        "n_atoms": 0,
        "connected_largest_frac": 0.0,
        "acryl_anywhere": False,
        "acryl_largest_frag": False,
        "acryl_largest_strict": False,
        "body_atom_element": "",
        "qed": np.nan,
        "mw": np.nan,
        "logp": np.nan,
        "scaffold": "",
    }
    if mol is None:
        return out
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return out
    largest = largest_fragment(mol)
    if largest is None or largest.GetNumHeavyAtoms() == 0:
        return out
    out["smiles"] = Chem.MolToSmiles(largest)
    out["n_atoms"] = largest.GetNumHeavyAtoms()
    out["connected_largest_frac"] = largest.GetNumHeavyAtoms() / max(1, n_heavy(mol))
    # gotcha #2 HEADLINE: must match on LARGEST fragment, not anywhere.
    # We report 'relaxed' (C=CC(=O)N) as headline because DiffSBDD often embeds
    # the warhead C_beta into a ring (substituted alkene), which is still a
    # functional Michael acceptor; also report strict (terminal CH2=CH) for completeness.
    out["acryl_anywhere"] = bool(mol.HasSubstructMatch(ACRYL_SMARTS_RELAXED))
    out["acryl_largest_frag"] = bool(largest.HasSubstructMatch(ACRYL_SMARTS_RELAXED))
    out["acryl_largest_strict"] = bool(largest.HasSubstructMatch(ACRYL_SMARTS_STRICT))
    # body-atom element: the C attached to Cβ via the double bond's neighbor
    # i.e., the next-non-warhead atom from atom-4 (N). Use atom indices 0-4 are
    # warhead per DiffSBDD inpaint convention.
    try:
        # find atom bonded to N (atom 4) that isn't C_carb (atom 2)
        n_atom = mol.GetAtomWithIdx(4)
        body = [nbr for nbr in n_atom.GetNeighbors() if nbr.GetIdx() not in (0, 1, 2, 3)]
        out["body_atom_element"] = body[0].GetSymbol() if body else ""
    except Exception:
        pass
    try:
        out["qed"] = float(QED.qed(largest))
    except Exception:
        pass
    try:
        out["mw"] = float(Descriptors.MolWt(largest))
        out["logp"] = float(Descriptors.MolLogP(largest))
    except Exception:
        pass
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(largest)
        out["scaffold"] = Chem.MolToSmiles(scaff) if scaff is not None else ""
    except Exception:
        pass
    return out


# ------------------------------------------------------------ FiLM scoring
def film_score(smiles_list):
    """Score with the FiLMDelta anchor checkpoint used in Plan B/C/D."""
    try:
        sys.path.insert(0, str(ANCHORDIFF_ROOT))
        from b2_film_scorer import FiLMScorer
        ckpt = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
        if not ckpt.exists():
            return [None] * len(smiles_list)
        scorer = FiLMScorer(ckpt)
        scores = scorer.score_smiles_batch(smiles_list)
        return [None if not np.isfinite(s) else float(s) for s in scores]
    except Exception as e:
        log_progress(f"FILM_SCORE_FAIL {type(e).__name__}: {e}")
        return [None] * len(smiles_list)


# ------------------------------------------------------------ orchestration
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_total", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--resamplings", type=int, default=1)
    p.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    p.add_argument("--add_n_nodes", type=int, default=15,
                   help="Atoms added beyond the 5-atom warhead. Larger -> bulkier ligand.")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    t_total = time.time()

    # gotcha #3: pocket-cys distance sanity
    log_progress("GOTCHA3_CHECK pocket-cys distance")
    d, sg = cys_sg_warhead_distance(
        POCKET_ROOT / "receptor.pdb",
        POCKET_ROOT / "warhead_at_cys.sdf",
    )
    if d is None:
        log_progress("GOTCHA3_WARN could not find Cys346 SG in receptor.pdb")
        errors.append("gotcha3_no_sg")
    else:
        log_progress(f"GOTCHA3 d(C_beta, SG) = {d:.2f} A (expected ~1.85)")
        if d > 8.0:
            log_progress(f"GOTCHA3_FAIL warhead is {d:.1f} A from Cys346 SG (>8). "
                         "Aborting; pocket misaligned.")
            errors.append(f"gotcha3_dist={d:.2f}")
            with open(SUMMARY_PATH, "w") as f:
                json.dump({"status": "FAILED", "errors": errors,
                          "d_SG_Cbeta": d}, f, indent=2)
            return

    # Sample
    log_progress(
        f"SAMPLE start n_total={args.n_total} batch={args.batch_size} "
        f"T={args.timesteps} resamplings={args.resamplings} device={args.device} "
        f"add_n_nodes={args.add_n_nodes}"
    )
    raw_mols = run_inpaint(
        n_total=args.n_total, batch_size=args.batch_size,
        timesteps=args.timesteps, resamplings=args.resamplings,
        device=args.device, add_n_nodes=args.add_n_nodes,
    )
    log_progress(f"SAMPLE done raw={len(raw_mols)} in {time.time()-t_total:.1f}s")
    # Save raw
    raw_sdf = OUT_ROOT / "raw_pre_fix.sdf"
    w = Chem.SDWriter(str(raw_sdf))
    for m in raw_mols:
        if m is not None:
            try:
                w.write(m)
            except Exception:
                pass
    w.close()

    # gotcha #2: warhead-bond fix
    log_progress("GOTCHA2 apply warhead-bond fix")
    fixed_mols = apply_warhead_bond_fix(raw_mols)
    n_fix_ok = sum(1 for m in fixed_mols if m is not None)
    log_progress(f"GOTCHA2 fixed_ok={n_fix_ok}/{len(raw_mols)}")
    fixed_sdf = OUT_ROOT / "fixed.sdf"
    w = Chem.SDWriter(str(fixed_sdf))
    for m in fixed_mols:
        if m is not None:
            try:
                w.write(m)
            except Exception:
                pass
    w.close()

    # Metrics
    log_progress("METRICS compute per-mol")
    rows = []
    smis_for_film = []
    for i, (raw, fixed) in enumerate(zip(raw_mols, fixed_mols)):
        # prefer fixed (correct warhead bonds); fall back to raw if fix failed
        m = fixed if fixed is not None else raw
        rec = compute_metrics_one(m)
        rec["idx"] = i
        rec["fix_ok"] = fixed is not None
        rows.append(rec)
        smis_for_film.append(rec["smiles"])

    # FiLM scoring
    log_progress("FILM_SCORE start")
    film_pic50 = film_score(smis_for_film)
    for rec, s in zip(rows, film_pic50):
        rec["pred_pIC50"] = s if s is not None else np.nan

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["idx", "smiles", "n_atoms", "qed", "mw", "logp",
            "connected_largest_frac",
            "acryl_largest_frag", "acryl_largest_strict", "acryl_anywhere",
            "body_atom_element", "scaffold", "fix_ok", "pred_pIC50"]
    df = df[[c for c in cols if c in df.columns]]
    csv_path = OUT_ROOT / "molecules.csv"
    df.to_csv(csv_path, index=False)
    log_progress(f"CSV written {csv_path} rows={len(df)}")

    # Headline summary
    n_sampled = len(raw_mols)
    valid = df[df["smiles"] != ""].copy()
    n_connected = int((valid["connected_largest_frac"] >= 0.80).sum()) if len(valid) else 0
    acryl_largest_pct = (float(valid["acryl_largest_frag"].mean()) * 100
                          if len(valid) else 0.0)
    acryl_largest_strict_pct = (float(valid["acryl_largest_strict"].mean()) * 100
                                  if len(valid) else 0.0)
    acryl_anywhere_pct = (float(valid["acryl_anywhere"].mean()) * 100
                           if len(valid) else 0.0)
    body_C_pct = (float((valid["body_atom_element"] == "C").mean()) * 100
                   if len(valid) else 0.0)
    qed_med = float(valid["qed"].median()) if len(valid) else float("nan")
    mw_med = float(valid["mw"].median()) if len(valid) else float("nan")
    n_scaff = int(valid["scaffold"].nunique()) if len(valid) else 0
    if len(valid):
        top10 = valid.dropna(subset=["pred_pIC50"]).sort_values(
            "pred_pIC50", ascending=False).head(10)
        top10_records = top10[["smiles", "pred_pIC50", "qed", "mw",
                               "acryl_largest_frag"]].to_dict("records")
    else:
        top10_records = []

    summary = {
        "status": "OK",
        "config": {
            "n_total": args.n_total, "batch_size": args.batch_size,
            "timesteps": args.timesteps, "resamplings": args.resamplings,
            "device": args.device, "add_n_nodes": args.add_n_nodes,
        },
        "d_SG_Cbeta": float(d) if d is not None else None,
        "n_sampled": int(n_sampled),
        "n_valid_smiles": int(len(valid)),
        "n_connected_ge_80pct": int(n_connected),
        "connected_pct": float(n_connected / max(1, n_sampled) * 100),
        "acryl_largest_pct": float(acryl_largest_pct),
        "acryl_largest_strict_pct": float(acryl_largest_strict_pct),
        "acryl_anywhere_pct": float(acryl_anywhere_pct),
        "body_C_pct": float(body_C_pct),
        "QED_median": qed_med,
        "MW_median": mw_med,
        "n_unique_murcko_scaffolds": int(n_scaff),
        "top10_pred_pIC50": top10_records,
        "errors": errors,
        "runtime_s": float(time.time() - t_total),
        "csv": str(csv_path),
        "fixed_sdf": str(fixed_sdf),
        "raw_sdf": str(raw_sdf),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log_progress(f"DONE wrote {SUMMARY_PATH} connected_pct={summary['connected_pct']:.1f}% "
                 f"acryl_largest={acryl_largest_pct:.1f}% body_C={body_C_pct:.1f}% "
                 f"QED_med={qed_med:.2f}")


if __name__ == "__main__":
    main()
