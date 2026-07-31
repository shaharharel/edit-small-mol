"""Phase 2: Fresh Boltz-2 cofold of Mol1 against ZAP70 (UniProt P43403, residues
327-606) with a Cys346 covalent restraint, then extract pose + pocket residues
through the SAME pipeline that produced the v2 training triples — so the
inference distribution matches training exactly (Fix 4).

Outputs:
  data/m1a_v2_boltz_mol1/mol1_zap70/  Boltz output (CIF, lig.sdf, etc.)
  data/m1a_v2_boltz_mol1/mol1_zap70_triple.json
      {pocket_residues: [...], warhead_pose_6d_v2_unnorm: [...],
       pocket_pseudo_seq: 'CLGCAP...', pocket_residue_embedding_path: ...}
  data/m1a_v2_boltz_mol1/mol1_zap70_esm.npz
      {residues_emb: (1, R, 320), residues_mask: (1, R)} for inference
  data/m1a_v2_boltz_mol1/mol1_zap70_pose.npz
      {pose_unnorm: (6,), pose_norm: (6,)} (using the v2 normalizer)
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))

# Re-use the enrichment / pose-fix helpers
from enrich_and_recompute_poses import (  # noqa: E402
    parse_boltz_cif_lig, map_smi_to_het_coords, get_warhead_4_atoms_from_mol,
    compute_new_pose,
)
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# ZAP70 kinase domain (UniProt P43403 residues 327-606; same window as
# build_triples.py boltz runs used; corresponds to CHEMBL2803).
ZAP70_UNIPROT = "P43403"
ZAP70_START = 327
ZAP70_END = 606
# Full-length UniProt sequence (P43403); we slice [326:606]
ZAP70_FULL = (
    "MPDPAAHLPFFYGSISRAEAEEHLKLAGMADGLFLLRQCLRSLGGYVLSLVHDVRFHHFPIERQLNGTYAIAGGKAHCGPAELCEFYSRDPDGLPCNLRKPCNRPSGLEPQPGVFDCLRDAMVRDYVRQTWKLEGEALEQAIISQAPQVEKLIATTAHERMPWYHSSLTREEAERKLYSGAQTDGKFLLRPRKEQGTYALSLIYGKTVYHYLISQDKAGKYCIPEGTKFDTLWQLVEYLKLKADGLIYCLKEACPNSSASNASGAAAPTLPAHPSTLTHPQRRIDTLNSDGYTPEPARITSPDKPRPMPMDTSVYESPYSDPEELKDKKLFLKRDNLLIADIELGCGNFGSVRQGVYRMRKKQIDVAIKVLKQGTEKADTEEMMREAQIMHQLDNPYIVRLIGVCQAEALMLVMEMAGGGPLHKFLVGKREEIPVSNVAELLHQVSMGMKYLEEKNFVHRDLAARNVLLVNRHYAKISDFGLSKALGADDSYYTARSAGKWPLKWYAPECINFRKFSSRSDVWSYGVTMWEALSYGQKPYKKMKGPEVMAFIEQGKRMECPPECPPELYALMSDCWIYKWEDRPDFLTVEQRMRACYYSLASKVEGPPGSTQKAEAACA"
)

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def write_progress(progress_path: str, phase: str, **extra):
    rec = {"phase": phase, "timestamp": time.time(),
           "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    Path(progress_path).write_text(json.dumps(rec, indent=2))


def run_boltz_cofold(out_root: Path, force: bool = False) -> Path:
    """Invoke `boltz predict` to cofold MOL1+ZAP70.

    Writes a YAML manifest, then runs `boltz predict ...`, returns the path to
    the resulting *_model_0.cif inside the output dir.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    yaml_path = out_root / "mol1_zap70.yaml"
    pred_dir = out_root / "boltz_pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    # Compose the YAML.
    # Boltz indexes atom_idx_map by slice-relative residue position (1-indexed
    # in YAML, decremented to 0 internally at parse_boltz_schema). The protein
    # sequence we pass is ZAP70 residues 327..606. So Cys346 is at slice-
    # relative position 346 - 327 + 1 = 20 in the YAML.
    seq = ZAP70_FULL[ZAP70_START - 1: ZAP70_END]  # 1-indexed inclusive
    cys346_slice_pos = 346 - ZAP70_START + 1   # = 20
    assert seq[cys346_slice_pos - 1] == "C", (
        f"slice pos {cys346_slice_pos} of ZAP70[327..606] is "
        f"{seq[cys346_slice_pos - 1]!r}, expected 'C' (Cys346)")
    n = len(seq)
    yaml_text = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
  - ligand:
      id: B
      smiles: '{MOL1_SMI}'
constraints:
  - bond:
      atom1: [A, {cys346_slice_pos}, SG]    # Cys346 -> slice-relative pos {cys346_slice_pos}
      atom2: [B, 1, C26]                     # Mol1 acrylamide beta-C (CanonicalRankAtoms-derived name; see boltz/data/parse/schema.py)
"""
    yaml_path.write_text(yaml_text)
    print(f"Wrote {yaml_path}", flush=True)

    # Look for any existing prediction
    cifs = list(pred_dir.rglob("*_model_0.cif"))
    if cifs and not force:
        print(f"Reusing existing CIF: {cifs[0]}", flush=True)
        return cifs[0]

    # Invoke boltz; use cuda
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    # boltz CLI: `boltz predict <yaml> --out_dir <dir> --use_msa_server`
    # MSA server is needed unless we provide a precomputed MSA. Use the public
    # server for a single fresh cofold (~5 min).
    cmd = [
        "boltz", "predict", str(yaml_path),
        "--out_dir", str(pred_dir),
        "--use_msa_server",
        "--diffusion_samples", "1",
        "--recycling_steps", "3",
        "--sampling_steps", "200",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=2400)
    print("BOLTZ STDOUT (tail):", res.stdout[-2000:], flush=True)
    print("BOLTZ STDERR (tail):", res.stderr[-2000:], flush=True)
    if res.returncode != 0:
        raise RuntimeError(f"boltz failed rc={res.returncode}")
    cifs = list(pred_dir.rglob("*_model_0.cif"))
    if not cifs:
        raise RuntimeError("no CIF produced by boltz")
    return cifs[0]


def extract_zap70_triple(cif_path: Path, pose_normalizer: dict) -> dict:
    """Pull pocket residues + warhead pose from the cofold CIF.

    Uses the same logic as the v2 build pipeline. We re-implement here rather
    than importing build_triples_v2.boltz_pose_to_triple because we want to
    apply the NEW pose schema (Fix 1+2+3) immediately.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    s = parser.get_structure("m", str(cif_path))
    model = next(s.get_models())

    # Collect residues from chain A; ligand HET atoms from any chain
    residues = []
    lig_atoms = []
    for chain in model:
        for res in chain:
            hetflag = res.id[0]
            if hetflag == " ":
                if chain.id != "A":
                    continue
                atoms = {a.get_name(): np.array(a.get_coord()) for a in res}
                residues.append({
                    "aa": res.get_resname(), "idx": int(res.id[1]),
                    "atoms": atoms,
                })
            else:
                if res.get_resname().startswith("LIG"):
                    for a in res:
                        name = a.get_name()
                        elem = a.element.strip() if hasattr(a, "element") else (
                            "H" if name.startswith("H") else name[0])
                        if elem == "H":
                            continue
                        lig_atoms.append({"name": name,
                                            "elem": elem.upper(),
                                            "xyz": np.array(a.get_coord())})

    # Build mol3d
    mol = Chem.MolFromSmiles(MOL1_SMI)
    mol3d = map_smi_to_het_coords(mol, lig_atoms)
    if mol3d is None:
        raise RuntimeError(f"Could not map Mol1 SMILES to {len(lig_atoms)} HET atoms")
    info = get_warhead_4_atoms_from_mol(mol3d)
    if info is None:
        raise RuntimeError("No acrylamide warhead pattern matched on Mol1")
    b_xyz = info["b"]
    a_xyz = info["a"]
    g_xyz = info["gamma"]
    d_xyz = info["delta"]

    # Find Cys346 SG -> nucleophile
    nuc_xyz = None
    for r in residues:
        if r["aa"] == "CYS" and r["idx"] == 346 and "SG" in r["atoms"]:
            nuc_xyz = r["atoms"]["SG"]
            break
    if nuc_xyz is None:
        # Fallback: nearest CYS SG
        best = None
        for r in residues:
            if r["aa"] == "CYS" and "SG" in r["atoms"]:
                d = float(np.linalg.norm(r["atoms"]["SG"] - b_xyz))
                if best is None or d < best[0]:
                    best = (d, r["atoms"]["SG"], r["idx"])
        if best is None:
            raise RuntimeError("No CYS SG in chain A")
        nuc_xyz = best[1]
        print(f"WARN: Cys346 SG missing; using nearest Cys{best[2]}@{best[0]:.2f}A",
               flush=True)

    # Pocket residues within 8A of b_xyz
    pocket = []
    for r in residues:
        if "CA" not in r["atoms"]:
            continue
        d_ca = float(np.linalg.norm(r["atoms"]["CA"] - b_xyz))
        if d_ca <= 8.0:
            one = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E",
                    "GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
                    "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
                    "TYR":"Y","VAL":"V","MSE":"M","SEC":"C"}.get(r["aa"], "X")
            pocket.append({
                "aa": one, "idx": r["idx"], "d": d_ca,
                "ca_xyz": r["atoms"]["CA"].tolist(),
            })
    pocket = sorted(pocket, key=lambda r: r["idx"])
    if len(pocket) < 4:
        raise RuntimeError(f"Pocket too small: {len(pocket)} residues")
    print(f"Pocket: {len(pocket)} residues, "
          f"sequence: {''.join(r['aa'] for r in pocket)}", flush=True)

    # BD angle = angle(nuc-b, a-b)
    v1 = nuc_xyz - b_xyz
    v2 = a_xyz - b_xyz
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        bd = 0.0
    else:
        c = float(np.dot(v1, v2) / (n1 * n2))
        c = max(-1.0, min(1.0, c))
        bd = float(np.degrees(np.arccos(c)))
    # Use compute_new_pose only to extract the planar (vinyl-amide) dihedral
    # (and 'src' for provenance). We discard its 6-dim pose vector entirely
    # because the v3 schema is a 3-dim rotation/translation-invariant pose:
    #   dim0 = d_b_nuc_A  (Å)
    #   dim1 = bd_angle_deg  (°)
    #   dim2 = planar_dihedral_deg  (°)
    _, dih, src = compute_new_pose(b_xyz, a_xyz, nuc_xyz, bd, g_xyz, d_xyz)
    d_b_nuc = float(np.linalg.norm(b_xyz - nuc_xyz))
    pose_unnorm = np.array([d_b_nuc, bd, dih], dtype=np.float32)
    mean = np.array(pose_normalizer["mean"], dtype=np.float32)
    std = np.array(pose_normalizer["std"], dtype=np.float32)
    assert pose_unnorm.shape == mean.shape == std.shape == (3,), (
        f"v3 pose normalizer must be 3-dim; got pose={pose_unnorm.shape} "
        f"mean={mean.shape} std={std.shape}")
    pose_norm = (pose_unnorm - mean) / np.maximum(std, 1e-6)

    return {
        "pocket_residues": pocket,
        "pocket_pseudo_seq": "".join(r["aa"] for r in pocket),
        "pose_v2_unnorm": pose_unnorm.tolist(),
        "pose_v2_norm": pose_norm.tolist(),
        "pose_schema": ["d_b_nuc_A", "bd_angle_deg", "planar_dihedral_deg"],
        "b_xyz": b_xyz.tolist(),
        "a_xyz": a_xyz.tolist(),
        "gamma_xyz": g_xyz.tolist(),
        "delta_xyz": d_xyz.tolist(),
        "nuc_xyz": nuc_xyz.tolist(),
        "bd_angle_deg": bd,
        "planar_dihedral_deg": dih,
        "dih_source": src,
        "d_b_nuc_A": d_b_nuc,
    }


def run_esm(pseudo_seq: str, model_id: str = "facebook/esm2_t6_8M_UR50D"):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    enc = tok([pseudo_seq], return_tensors="pt", padding=True, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (1, L+2, 320)
    seq_len = len(pseudo_seq)
    emb = out[0, 1:1 + seq_len].cpu().numpy().astype(np.float32)
    mask = np.ones((seq_len,), dtype=bool)
    return emb, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(PROJECT_ROOT /
                     "data/m1a_v2_boltz_mol1"))
    ap.add_argument("--normalizer", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/pose_normalizer_v3.json"))
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/m1a_v2_progress.json"))
    ap.add_argument("--force_cofold", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    write_progress(args.progress_path, "phase2_cofolding_mol1")
    cif = run_boltz_cofold(out_root, force=args.force_cofold)
    print(f"Got cofold CIF: {cif}", flush=True)

    write_progress(args.progress_path, "phase2_extracting_triple")
    pose_norm_dict = json.loads(Path(args.normalizer).read_text())
    triple = extract_zap70_triple(cif, pose_norm_dict)
    out_triple = out_root / "mol1_zap70_triple.json"
    out_triple.write_text(json.dumps(triple, indent=2))
    print(f"Wrote {out_triple}", flush=True)

    write_progress(args.progress_path, "phase2_running_esm")
    emb, mask = run_esm(triple["pocket_pseudo_seq"])
    print(f"ESM-2 residue embedding: shape={emb.shape}", flush=True)
    out_esm = out_root / "mol1_zap70_esm.npz"
    np.savez_compressed(out_esm,
                         residues_emb=emb[None],   # (1, R, 320)
                         residues_mask=mask[None]) # (1, R)
    print(f"Wrote {out_esm}", flush=True)

    out_pose = out_root / "mol1_zap70_pose.npz"
    np.savez_compressed(out_pose,
                         pose_unnorm=np.array(triple["pose_v2_unnorm"], dtype=np.float32),
                         pose_norm=np.array(triple["pose_v2_norm"], dtype=np.float32))
    print(f"Wrote {out_pose}", flush=True)

    write_progress(args.progress_path, "phase2_done",
                   pocket_size=len(triple["pocket_residues"]),
                   bd_angle_deg=triple["bd_angle_deg"],
                   planar_dihedral_deg=triple["planar_dihedral_deg"],
                   d_b_nuc_A=triple["d_b_nuc_A"])


if __name__ == "__main__":
    main()
