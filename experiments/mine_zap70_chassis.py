"""Phase A — Mine chassis SMILES from the ChEMBL ZAP70 active dataset.

Identifies acrylamide-bearing actives, strips each to a "warhead + nearest
ring system" chassis, clusters by Murcko scaffold, and writes the top
6-10 families to data/zap70_chassis_families.json for the
multi-chassis ensemble runner.

Usage:
    python experiments/mine_zap70_chassis.py \
        --source data/docking_chembl_zap70/docking_results.csv \
        --output data/zap70_chassis_families.json \
        --min_cluster_size 3 --top_n 10
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ACRYL_PATTERN = Chem.MolFromSmarts("[CH2]=[CH]-C(=O)-N")
# more permissive acrylamide pattern (allows substituted vinyl)
ACRYL_PATTERN_LOOSE = Chem.MolFromSmarts("C=CC(=O)N")


def find_acrylamide_atoms(mol):
    """Return list of (Cbeta, Calpha, Ccarb, O, N) atom-index tuples."""
    matches = mol.GetSubstructMatches(ACRYL_PATTERN_LOOSE)
    return list(matches)


def extract_chassis(smi: str):
    """Strip a molecule down to its warhead + the ring system it attaches to
    (via the warhead's N).

    Returns dict {chassis_smiles_with_star, chassis_canon, attach_atom_idx}
    or None if the molecule has no acrylamide, no ring attached to N, etc.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    matches = find_acrylamide_atoms(mol)
    if not matches:
        return None
    # take first acrylamide
    cb, ca, cc, o_idx, n_idx = matches[0]
    # the N's non-warhead neighbor (the substituent attached past N)
    n_atom = mol.GetAtomWithIdx(n_idx)
    n_substituent_atoms = [a.GetIdx() for a in n_atom.GetNeighbors() if a.GetIdx() != cc]
    if not n_substituent_atoms:
        return None
    # find the nearest ring system (BFS from N substituent atoms)
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return None
    # BFS expanding ring system: start with the first ring containing any atom
    # reachable from n_substituent_atoms within 3 bonds
    visited = set([cb, ca, cc, o_idx, n_idx])
    frontier = list(n_substituent_atoms)
    ring_atoms_found = set()
    bond_radius = 0
    keep = set(visited)
    while frontier and bond_radius < 6:
        new_frontier = []
        for ai in frontier:
            if ai in visited:
                continue
            visited.add(ai)
            keep.add(ai)
            atom = mol.GetAtomWithIdx(ai)
            if atom.IsInRing():
                ring_atoms_found.add(ai)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in visited:
                    new_frontier.append(nbr.GetIdx())
        if ring_atoms_found:
            # we've hit a ring; expand to fully include this ring system
            # (every atom in any ring sharing atoms with ring_atoms_found)
            target_rings = set()
            for r in ring_info.AtomRings():
                if any(ai in ring_atoms_found for ai in r):
                    target_rings.update(r)
            # also fuse rings sharing atoms (one expansion pass)
            for _ in range(3):
                added = False
                for r in ring_info.AtomRings():
                    if any(ai in target_rings for ai in r):
                        for ai in r:
                            if ai not in target_rings:
                                target_rings.add(ai)
                                added = True
                if not added:
                    break
            keep |= target_rings
            break
        frontier = new_frontier
        bond_radius += 1
    if not ring_atoms_found:
        return None
    # determine attachment atom (where [*] will be placed): the ring atom
    # outermost from the warhead — pick the ring atom with the most heavy
    # neighbors NOT in `keep` (i.e. most decoration outside the chassis)
    candidate_attach = None
    best_outside = -1
    for ai in keep:
        atom = mol.GetAtomWithIdx(ai)
        if not atom.IsInRing():
            continue
        # only consider ring atoms in the target ring system
        outside = sum(1 for nbr in atom.GetNeighbors()
                      if nbr.GetIdx() not in keep and nbr.GetAtomicNum() > 1)
        if outside > best_outside:
            best_outside = outside
            candidate_attach = ai
    if candidate_attach is None or best_outside <= 0:
        # no external decoration — chassis is the full mol; useless as
        # a generation anchor (nothing to elaborate). Skip.
        return None
    # build chassis SMILES: keep `keep` atoms; replace the OUTSIDE neighbor
    # of candidate_attach with a single [*]
    rw = Chem.RWMol(mol)
    # remove all atoms NOT in keep, EXCEPT we need to know the [*] attachment
    # The simplest path: add a [*] atom bonded to candidate_attach, then strip
    # all out-of-keep heavy atoms.
    star_idx = rw.AddAtom(Chem.Atom(0))  # dummy
    rw.AddBond(candidate_attach, star_idx, Chem.BondType.SINGLE)
    keep.add(star_idx)
    # remove unwanted atoms in descending idx order so indices remain stable
    to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep]
    for ai in sorted(to_remove, reverse=True):
        rw.RemoveAtom(ai)
    try:
        chassis_mol = rw.GetMol()
        Chem.SanitizeMol(chassis_mol)
        chassis_smi = Chem.MolToSmiles(chassis_mol)
    except Exception as e:
        print(f"[chassis] sanitize failed: {e}")
        return None
    # require chassis to still contain a vinyl (warhead intact)
    if "C=C" not in chassis_smi:
        print(f"[chassis] missing vinyl: {chassis_smi}")
        return None
    if "*" not in chassis_smi:
        print(f"[chassis] missing dummy: {chassis_smi}")
        return None
    # require some minimum heavy atoms (e.g. >= 8 incl warhead and ring)
    n_heavy = chassis_mol.GetNumHeavyAtoms() - 1  # excluding [*]
    if n_heavy < 8:
        print(f"[chassis] n_heavy={n_heavy} < 8: {chassis_smi}")
        return None
    return {
        "chassis_smiles_with_star": chassis_smi,
        "n_heavy_scaffold": n_heavy,
    }


def murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        sca = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sca)
    except Exception:
        return ""


def load_actives(source: str) -> pd.DataFrame:
    """Load actives with smiles + pIC50 from a CSV (docking_results) or .smi."""
    p = Path(source)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        # docking_chembl_zap70/docking_results.csv: smiles,chembl_id,pIC50_exp,...
        col_smi = "smiles" if "smiles" in df.columns else df.columns[0]
        col_pic = None
        for c in ("pIC50_exp", "pIC50", "pic50", "pchembl_value"):
            if c in df.columns:
                col_pic = c
                break
        if col_pic is None:
            df["pIC50"] = float("nan")
            col_pic = "pIC50"
        out = df[[col_smi, col_pic]].rename(columns={col_smi: "smiles", col_pic: "pIC50"})
        return out
    elif p.suffix == ".smi":
        rows = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smi = parts[0]
            pic = float(parts[1]) if len(parts) >= 2 else float("nan")
            rows.append({"smiles": smi, "pIC50": pic})
        return pd.DataFrame(rows)
    else:
        raise ValueError(f"unsupported source extension: {p.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="data/docking_chembl_zap70/docking_results.csv",
        help="Path to actives source (CSV or .smi). docking_results.csv default.",
    )
    ap.add_argument(
        "--fallback_source",
        default="data/zap70_all.smi",
        help="Secondary source merged in if present (provides full chembl set).",
    )
    ap.add_argument("--output", default="data/zap70_chassis_families.json")
    ap.add_argument("--min_cluster_size", type=int, default=3)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    df = load_actives(args.source)
    if Path(args.fallback_source).exists():
        df2 = load_actives(args.fallback_source)
        df = pd.concat([df, df2], ignore_index=True)
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    print(f"[mine] loaded {len(df)} unique actives from {args.source}")

    # filter to acrylamide-bearing
    acryl_rows = []
    for i, r in df.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        if mol.HasSubstructMatch(ACRYL_PATTERN_LOOSE):
            acryl_rows.append(r)
    acryl_df = pd.DataFrame(acryl_rows).reset_index(drop=True)
    print(f"[mine] acrylamide-bearing: {len(acryl_df)} / {len(df)}")

    # extract chassis + murcko scaffold
    enriched = []
    for i, r in acryl_df.iterrows():
        ch = extract_chassis(r["smiles"])
        if ch is None:
            continue
        sca = murcko_scaffold(ch["chassis_smiles_with_star"])
        enriched.append({
            "smiles": r["smiles"],
            "pIC50": r["pIC50"],
            "chassis_smiles_with_star": ch["chassis_smiles_with_star"],
            "n_heavy_scaffold": ch["n_heavy_scaffold"],
            "murcko": sca,
        })
    en_df = pd.DataFrame(enriched)
    print(f"[mine] extracted chassis for {len(en_df)} mols")

    # cluster by murcko scaffold
    clusters = defaultdict(list)
    for i, r in en_df.iterrows():
        clusters[r["murcko"]].append(r.to_dict())
    print(f"[mine] {len(clusters)} unique Murcko scaffolds")

    # filter min_cluster_size, pick representative chassis per cluster (most common
    # canonical chassis SMILES with [*]); rank by median pIC50.
    families = []
    for sca, members in clusters.items():
        if len(members) < args.min_cluster_size:
            continue
        # within cluster, group by canonical chassis_smiles_with_star
        chassis_counter = defaultdict(int)
        chassis_examples = defaultdict(list)
        for m in members:
            chassis_counter[m["chassis_smiles_with_star"]] += 1
            chassis_examples[m["chassis_smiles_with_star"]].append(m)
        # pick the chassis SMILES with most members
        rep_chassis = max(chassis_counter.items(), key=lambda kv: kv[1])[0]
        rep_members = chassis_examples[rep_chassis]
        pics = [m["pIC50"] for m in members if not pd.isna(m["pIC50"])]
        median_pic = float(pd.Series(pics).median()) if pics else float("nan")
        pic_range = (float(min(pics)), float(max(pics))) if pics else (None, None)
        # collect example actives (up to 3 highest pIC50)
        members_sorted = sorted(members, key=lambda m: m["pIC50"] if not pd.isna(m["pIC50"]) else -1, reverse=True)
        examples = [m["smiles"] for m in members_sorted[:3]]
        families.append({
            "murcko": sca,
            "chassis_smiles_with_star": rep_chassis,
            "rep_chassis_count": chassis_counter[rep_chassis],
            "all_chassis_variants": dict(chassis_counter),
            "n_actives": len(members),
            "median_pIC50": median_pic,
            "pIC50_range": pic_range,
            "example_actives": examples,
        })

    families.sort(key=lambda f: (-(f["n_actives"]), -(f["median_pIC50"] if not pd.isna(f["median_pIC50"]) else -1)))
    families = families[: args.top_n]

    for i, f in enumerate(families):
        f["chassis_id"] = f"ZAP_C{i+1}"

    out = {
        "source": args.source,
        "fallback_source": args.fallback_source,
        "n_total_actives": int(len(df)),
        "n_acrylamide_actives": int(len(acryl_df)),
        "n_chassis_extracted": int(len(en_df)),
        "n_unique_murcko": int(len(clusters)),
        "min_cluster_size": int(args.min_cluster_size),
        "n_families": int(len(families)),
        "families": families,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    print(f"[mine] wrote {len(families)} chassis families to {args.output}")
    for f in families:
        print(f"  {f['chassis_id']:>6}  n={f['n_actives']:>3}  med_pIC50={f['median_pIC50']:.2f}  {f['chassis_smiles_with_star']}")


if __name__ == "__main__":
    main()
