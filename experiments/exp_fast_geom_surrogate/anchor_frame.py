"""Extract a canonical Cys346 anchor frame from the best Boltz-2 cofold pose.

Picks the highest `boltz_iptm` row from `data/tier4_scored/F4_boltz_full.csv`,
resolves the corresponding cofold CIF via the f4 / f4_extra results manifest
maps, and pulls out the geometry the FastGeomScorer needs:

  - Cys346 SG (covalent anchor)
  - Cys346 CA, CB (frame vectors)
  - Met416 backbone N (hinge donor)
  - Ala417 backbone N (hinge donor)
  - The ligand warhead Cβ in this pose (so we know the "ideal" Cβ position)
  - Pocket atom cloud: all PROTEIN heavy atoms within 8 Å of any ligand atom

The output is `data/fast_geom_surrogate/anchor_frame.npz` with parallel arrays
`names`, `elements` and `coords`, plus scalar keys `sg`, `cb_ideal`,
`met416_n`, `ala417_n`, and an `anchor_row_id` / `anchor_iptm` provenance pair.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
POOL_DIR = PROJECT_ROOT / "data/boltz_f4_results"
EXTRA_DIR = PROJECT_ROOT / "data/boltz_f4_extra_results"
F4_CSV = PROJECT_ROOT / "data/tier4_scored/F4_boltz_full.csv"
OUT_NPZ = PROJECT_ROOT / "data/fast_geom_surrogate/anchor_frame.npz"

CYS_RESI = 346
MET_RESI = 416
ALA_RESI = 417
POCKET_CUT = 8.0  # Å


def build_cofold_index() -> dict[str, Path]:
    """row_id -> /predictions/<name> dir, scanning POOL_DIR and EXTRA_DIR."""
    idx: dict[str, Path] = {}
    for root in (POOL_DIR, EXTRA_DIR):
        if not root.exists():
            continue
        for vm in sorted(root.iterdir()):
            if not vm.is_dir():
                continue
            man = vm / "manifest.csv"
            if man.exists():
                try:
                    mf = pd.read_csv(man)
                    for _, r in mf.iterrows():
                        name = str(r["name"])
                        rid = str(r["row_id"])
                        pdir = vm / f"boltz_results_{name}" / "predictions" / name
                        if pdir.is_dir():
                            idx[rid] = pdir
                except Exception:
                    pass
            for sub in vm.iterdir():
                if sub.is_dir() and sub.name.startswith("boltz_results_"):
                    name = sub.name[len("boltz_results_"):]
                    pdir = sub / "predictions" / name
                    if pdir.is_dir() and name not in idx:
                        idx[name] = pdir
    return idx


def parse_cif(cif: Path) -> tuple[list[dict], list[dict]]:
    """Return (prot_atoms, lig_atoms). Chain A = protein, B = ligand."""
    st = gemmi.read_structure(str(cif))
    try:
        st.setup_entities()
    except Exception:
        pass
    prot, lig = [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    rec = {
                        "chain": chain.name,
                        "resname": res.name,
                        "resi": int(res.seqid.num),
                        "name": atom.name,
                        "element": atom.element.name,
                        "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z], float),
                    }
                    if chain.name == "A":
                        prot.append(rec)
                    elif chain.name == "B":
                        lig.append(rec)
        break  # first model only
    return prot, lig


def find_warhead_cb(lig: list[dict], sg: np.ndarray) -> dict | None:
    """Pick the carbon closest to Cys346.SG as the warhead Cβ in this pose."""
    cands = [(float(np.linalg.norm(a["pos"] - sg)), a) for a in lig if a["element"] == "C"]
    cands.sort(key=lambda x: x[0])
    if not cands:
        return None
    return cands[0][1]


def find_acrylamide_anchor_atoms(lig: list[dict], cb: dict) -> list[dict] | None:
    """Given Cβ in the ligand, find Cα, C', O, N of the (now Michael-adducted) acrylamide.

    Topology (post-Michael-addition; bond orders in cofold are ambiguous so we use distance):
        Cβ — Cα — C'(=O) — N
    Walk by nearest-carbon-then-amide-carbon-then-NO heuristic.

    Returns [Cβ, Cα, C', O, N] or None if any not findable.
    """
    cb_pos = cb["pos"]
    cb_name = cb["name"]

    # Cα: closest other carbon to Cβ in 1.3-1.8Å range
    cands = [(float(np.linalg.norm(a["pos"] - cb_pos)), a)
             for a in lig if a["element"] == "C" and a["name"] != cb_name]
    cands.sort(key=lambda x: x[0])
    ca = next((a for d, a in cands if 1.25 <= d <= 1.85), None)
    if ca is None:
        return None

    # C': closest carbon to Cα (not Cβ) in 1.3-1.8 with at least one O neighbour
    cands2 = [(float(np.linalg.norm(a["pos"] - ca["pos"])), a)
              for a in lig if a["element"] == "C" and a["name"] not in (cb_name, ca["name"])
              and 1.25 <= float(np.linalg.norm(a["pos"] - ca["pos"])) <= 1.85]
    cprime = None
    for _, c_cand in sorted(cands2, key=lambda x: x[0]):
        has_O = any(o["element"] == "O" and np.linalg.norm(o["pos"] - c_cand["pos"]) < 1.45
                    for o in lig)
        has_N = any(n["element"] == "N" and np.linalg.norm(n["pos"] - c_cand["pos"]) < 1.6
                    for n in lig)
        if has_O and has_N:
            cprime = c_cand
            break
    if cprime is None:
        return None
    oxy = None
    for o in lig:
        if o["element"] != "O":
            continue
        if np.linalg.norm(o["pos"] - cprime["pos"]) < 1.45:
            oxy = o
            break
    nit = None
    for n in lig:
        if n["element"] != "N":
            continue
        if np.linalg.norm(n["pos"] - cprime["pos"]) < 1.6:
            nit = n
            break
    if oxy is None or nit is None:
        return None
    return [cb, ca, cprime, oxy, nit]


def main():
    if not F4_CSV.exists():
        print(f"[anchor_frame] missing {F4_CSV}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(F4_CSV)
    if "boltz_iptm" not in df.columns:
        print("[anchor_frame] no boltz_iptm column", file=sys.stderr)
        sys.exit(2)
    idx = build_cofold_index()
    print(f"[anchor_frame] {len(idx)} cofold dirs indexed")

    # Rank by iptm but require sensible d_SG / BD_dev so the anchor is
    # both confident AND a real Michael-addition pose.
    rank = df.copy()
    if "d_SG" in rank.columns and "burgi_dunitz_dev_deg" in rank.columns:
        rank = rank[(rank["d_SG"] < 2.5) & (rank["burgi_dunitz_dev_deg"] < 35.0)]
        print(f"[anchor_frame] {len(rank)} rows survive d_SG<2.5 & BD_dev<35 filter")
    rank = rank.sort_values("boltz_iptm", ascending=False)

    chosen = None
    for _, r in rank.iterrows():
        rid = str(r["row_id"])
        if rid not in idx:
            continue
        pdir = idx[rid]
        cif = pdir / f"{pdir.name}_model_0.cif"
        if not cif.exists():
            continue
        chosen = (rid, r, pdir, cif)
        break
    if chosen is None:
        print("[anchor_frame] no anchor candidate found", file=sys.stderr)
        sys.exit(3)
    rid, row, pdir, cif = chosen
    print(f"[anchor_frame] picked row_id={rid} iptm={row['boltz_iptm']:.4f} "
          f"d_SG={row.get('d_SG', float('nan')):.3f} "
          f"BD_dev={row.get('burgi_dunitz_dev_deg', float('nan')):.2f}")
    print(f"[anchor_frame] cif: {cif}")

    prot, lig = parse_cif(cif)
    if not prot or not lig:
        print("[anchor_frame] empty chains", file=sys.stderr)
        sys.exit(4)

    sg = next((p["pos"] for p in prot
               if p["resi"] == CYS_RESI and p["resname"] == "CYS" and p["name"] == "SG"), None)
    cys_ca = next((p["pos"] for p in prot
                   if p["resi"] == CYS_RESI and p["resname"] == "CYS" and p["name"] == "CA"), None)
    cys_cb = next((p["pos"] for p in prot
                   if p["resi"] == CYS_RESI and p["resname"] == "CYS" and p["name"] == "CB"), None)
    met_n = next((p["pos"] for p in prot
                  if p["resi"] == MET_RESI and p["name"] == "N"), None)
    ala_n = next((p["pos"] for p in prot
                  if p["resi"] == ALA_RESI and p["name"] == "N"), None)
    if any(x is None for x in (sg, cys_ca, cys_cb, met_n, ala_n)):
        print("[anchor_frame] missing reference atoms", file=sys.stderr)
        print(f"  sg={sg is not None} cys_ca={cys_ca is not None} cys_cb={cys_cb is not None} "
              f"met_n={met_n is not None} ala_n={ala_n is not None}")
        sys.exit(5)

    cb_atom = find_warhead_cb(lig, sg)
    if cb_atom is None:
        print("[anchor_frame] no warhead Cβ in ligand", file=sys.stderr)
        sys.exit(6)
    cb_ideal = cb_atom["pos"]
    print(f"[anchor_frame] warhead Cβ atom {cb_atom['name']} at d_SG={np.linalg.norm(cb_ideal-sg):.3f}Å")

    five = find_acrylamide_anchor_atoms(lig, cb_atom)
    if five is None:
        print("[anchor_frame] could not trace Cβ–Cα–C'–O/N in ligand", file=sys.stderr)
        sys.exit(7)
    warhead_template = np.stack([a["pos"] for a in five], axis=0).astype(np.float32)
    print(f"[anchor_frame] warhead template atoms: "
          f"{[a['name'] for a in five]}  shape={warhead_template.shape}")
    # Distal ligand atom: heavy atom furthest from Cβ — used as a reference for hinge reach.
    cb_pos = cb_atom["pos"]
    heavy = [a for a in lig if a["element"] != "H"]
    distal = max(heavy, key=lambda a: float(np.linalg.norm(a["pos"] - cb_pos)))
    distal_xyz = distal["pos"].astype(np.float32)
    distal_to_met = float(np.linalg.norm(distal["pos"] - met_n))
    distal_to_ala = float(np.linalg.norm(distal["pos"] - ala_n))
    print(f"[anchor_frame] distal atom {distal['name']} ({distal['element']}): "
          f"d_to_Met416N={distal_to_met:.2f}Å, d_to_Ala417N={distal_to_ala:.2f}Å")

    # Pocket cloud: PROTEIN heavy atoms within POCKET_CUT of any ligand atom.
    lig_xyz = np.stack([a["pos"] for a in lig], axis=0)
    pocket_atoms = []
    for p in prot:
        if p["element"] == "H":
            continue
        d = float(np.min(np.linalg.norm(lig_xyz - p["pos"], axis=1)))
        if d <= POCKET_CUT:
            pocket_atoms.append(p)
    print(f"[anchor_frame] pocket cloud: {len(pocket_atoms)} protein heavy atoms "
          f"within {POCKET_CUT}Å of ligand")

    names = np.array([f"{p['resname']}{p['resi']}.{p['name']}" for p in pocket_atoms])
    elements = np.array([p["element"] for p in pocket_atoms])
    coords = np.stack([p["pos"] for p in pocket_atoms], axis=0).astype(np.float32)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    # Also save the anchor LIGAND heavy atoms — the "ideal shape cloud" for
    # shape-complementarity scoring of new candidates (post warhead alignment).
    lig_heavy = [a for a in lig if a["element"] != "H"]
    lig_names = np.array([a["name"] for a in lig_heavy])
    lig_elements = np.array([a["element"] for a in lig_heavy])
    lig_coords = np.stack([a["pos"] for a in lig_heavy], axis=0).astype(np.float32)
    print(f"[anchor_frame] anchor ligand cloud: {len(lig_heavy)} heavy atoms")

    np.savez(
        OUT_NPZ,
        names=names,
        elements=elements,
        coords=coords,
        sg=sg.astype(np.float32),
        cys_ca=cys_ca.astype(np.float32),
        cys_cb=cys_cb.astype(np.float32),
        cb_ideal=cb_ideal.astype(np.float32),
        met416_n=met_n.astype(np.float32),
        ala417_n=ala_n.astype(np.float32),
        warhead_template=warhead_template,           # (5,3): Cβ Cα C' O N
        distal_ideal=distal_xyz,                     # the anchor's far-end heavy atom
        lig_names=lig_names,
        lig_elements=lig_elements,
        lig_coords=lig_coords,                       # (Nlig, 3) anchor LIGAND heavy atoms
        anchor_row_id=np.array([rid]),
        anchor_iptm=np.array([float(row["boltz_iptm"])]),
        anchor_d_SG=np.array([float(row.get("d_SG", np.nan))]),
        anchor_BD_dev=np.array([float(row.get("burgi_dunitz_dev_deg", np.nan))]),
    )
    print(f"[anchor_frame] saved {OUT_NPZ}")


if __name__ == "__main__":
    main()
