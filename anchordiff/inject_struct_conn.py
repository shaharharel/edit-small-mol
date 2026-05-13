"""Inject a `_struct_conn` covalent-bond record into a Boltz output mmCIF
so 3Dmol.js (and other viewers) draw the Cys SG ↔ ligand-Cβ bond.

Boltz-2's output CIF does not write the cross-chain covalent bond into
`_struct_conn` even when the bond was specified in the input YAML. Without
that record, 3Dmol.js auto-perceives only intra-residue bonds and the
viewer shows two unconnected atoms in space.

Usage:
  conda run -n quris python -m anchordiff.inject_struct_conn \\
      --cif data/boltz_poses/mol1__zap70_cys346/mol1/mol1_model_0.cif \\
      --cys_chain A --cys_resi 346 --cys_atom SG \\
      --lig_chain B --lig_resi 1  --lig_atom C26
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path


STRUCT_CONN_BLOCK = """
#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.pdbx_dist_value
covale1 covale {cys_chain} {cys_chain} {cys_resi} {cys_resi} {cys_atom} CYS {lig_chain} {lig_chain} 1 {lig_resi} {lig_atom} LIG1 {dist:.3f}
#
"""


def get_distance(cif_path: Path, cys_chain, cys_resi, cys_atom, lig_chain, lig_atom):
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    model = st[0]
    sg = lig = None
    for chain in model:
        for res in chain:
            if chain.name == cys_chain and res.seqid.num == cys_resi and res.name == "CYS":
                for atom in res:
                    if atom.name == cys_atom: sg = atom.pos
            if chain.name == lig_chain:
                for atom in res:
                    if atom.name == lig_atom: lig = atom.pos
    if sg is None or lig is None:
        sys.exit(f"Could not find atoms: SG={sg is not None}, lig={lig is not None}")
    return sg.dist(lig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", type=Path, required=True)
    ap.add_argument("--cys_chain", default="A")
    ap.add_argument("--cys_resi", type=int, required=True)
    ap.add_argument("--cys_atom", default="SG")
    ap.add_argument("--lig_chain", default="B")
    ap.add_argument("--lig_resi", type=int, default=1)
    ap.add_argument("--lig_atom", required=True)
    args = ap.parse_args()

    txt = args.cif.read_text()
    if "_struct_conn.id" in txt:
        print(f"  {args.cif.name} already has _struct_conn; skipping")
        return
    d = get_distance(args.cif, args.cys_chain, args.cys_resi, args.cys_atom,
                     args.lig_chain, args.lig_atom)
    block = STRUCT_CONN_BLOCK.format(
        cys_chain=args.cys_chain, cys_resi=args.cys_resi, cys_atom=args.cys_atom,
        lig_chain=args.lig_chain, lig_resi=args.lig_resi, lig_atom=args.lig_atom,
        dist=d,
    )
    # Insert AFTER the last atom_site loop but BEFORE the final '#' / EOF.
    # Safe insertion point: after the last '#' line in the file.
    lines = txt.splitlines(keepends=True)
    # Find last line that is '#' or '#\n'
    insert_at = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "#":
            insert_at = i + 1
            break
    new_txt = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
    args.cif.write_text(new_txt)
    print(f"  injected _struct_conn covalent bond ({args.cys_atom}–{args.lig_atom}, "
          f"d={d:.3f} Å) into {args.cif.name}")


if __name__ == "__main__":
    main()
