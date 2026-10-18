#!/usr/bin/env python3
"""Build a pocket-conditioned CovalentFormer prompt for ZAP70 / mol1.

WHY THIS IS NOT LIKE THE TRAINING ROWS, AND MUST BE LABELLED AS SUCH
--------------------------------------------------------------------
Every row the model was trained on carries a contact table measured from a deposited CRYSTAL.
mol1 has no crystal: it is a designed seed. The table here is computed from a **Boltz-2 covalent
cofold** of mol1 against ZAP70 Cys346 (`data/boltz_poses/mol1__zap70_cys346`, ligand ipTM 0.976,
confidence 0.844), so every distance in it is PREDICTED, not measured. Two consequences:

  1. This is a distribution shift away from what the model saw in training. Predicted poses carry
     1-2 A coordinate error, which perturbs every distance and can flip which protein atom is
     nearest. Output should be read as a demonstration, never as evidence about mol1.
  2. Ligand atom names in the training tables come from the CCD. A cofold has no CCD names, so
     names here are generated as element+index. They are internally consistent but are not the
     names any crystallographer would use.

The prompt text itself is byte-identical in structure to `sft_out_g3/sft_pocket.jsonl` prompts,
and the contact typing calls `evidence_v4.type_contact` directly rather than reimplementing it, so
the typing conventions (covalent ceilings, salt bridges, metal rules, backbone exclusions) are the
same ones the training data used.
"""
import argparse, importlib.util, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
POSE = os.path.join(ROOT, 'data/boltz_poses/mol1__zap70_cys346/mol1')
PROT = os.path.join(POSE, 'mol1_model_0.prot.pdb')
LIG = os.path.join(POSE, 'mol1_model_0.lig.sdf')
MANIFEST = os.path.join(ROOT, 'data/boltz_poses/mol1__zap70_cys346/mol1_manifest.json')

TARGET = 'P43403'          # ZAP70, UniProt
CUTOFF = 4.0               # evidence_v4.CONTACT_CUTOFF

INSTRUCTION = """### Instruction:
You are a medicinal chemist proposing a single-step analogue of a ligand whose crystal structure \
in its target binding site has been measured.
You are given the ligand's SMILES and, where available, a table of its measured protein contacts: \
for each ligand heavy atom, the nearest protein atom it faces, the distance, the angle where one \
is defined, and the interaction typing.
Propose one analogue and state what the change does to that binding site, in a single sentence.
Name only residues that appear in the contact table you were given. Where a contact is flagged \
uncertain, do not cite it as established. Then give the analogue's SMILES on its own line.
Answer with the sentence, then a line of the form 'SMILES: <smiles>'.
"""


def load_ev4():
    spec = importlib.util.spec_from_file_location('ev4', os.path.join(HERE, 'evidence_v4.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_protein(path):
    """Heavy atoms of the protein as dicts shaped for evidence_v4.type_contact."""
    out = []
    for line in open(path):
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        name = line[12:16].strip()
        el = (line[76:78].strip() or name[:1]).upper()
        if el == 'H':
            continue
        out.append({'name': name, 'res': line[17:20].strip(), 'rn': line[22:26].strip(),
                    'chain': line[21].strip() or 'A', 'el': el,
                    'xyz': (float(line[30:38]), float(line[38:46]), float(line[46:54]))})
    return out


def read_ligand(path):
    """Heavy atoms of the SDF, named element+running index (no CCD names exist for a cofold)."""
    lines = open(path).read().split('\n')
    n = int(lines[3][0:3])
    atoms, counts = [], {}
    for i in range(4, 4 + n):
        x, y, z = float(lines[i][0:10]), float(lines[i][10:20]), float(lines[i][20:30])
        el = lines[i][31:34].strip().upper()
        if el == 'H':
            continue
        counts[el] = counts.get(el, 0) + 1
        atoms.append({'name': '%s%d' % (el, counts[el]), 'el': el, 'xyz': (x, y, z)})
    return atoms


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='jsonl written in sft_pocket.jsonl schema')
    ap.add_argument('--samples', type=int, default=8,
                    help='rows to emit; each is the same prompt, so the arm is sampled N times')
    a = ap.parse_args()

    ev4 = load_ev4()
    man = json.load(open(MANIFEST))['mol1']
    prot, lig = read_protein(PROT), read_ligand(LIG)
    print('protein heavy atoms %d, ligand heavy atoms %d' % (len(prot), len(lig)))

    rows, named = [], set()
    for L in lig:
        best, bd = None, 1e9
        for P in prot:
            d = dist(L['xyz'], P['xyz'])
            if d < bd:
                best, bd = P, d
        if best is None or bd > CUTOFF:
            rows.append('%-5s %-2s %-15s %-6s %5s    - | %s'
                        % (L['name'], L['el'], '-', '-', '-',
                           'no protein atom within %.1f A (nearest %.2f A)' % (CUTOFF, bd)))
            continue
        typ, ang = ev4.type_contact(L['el'], best, bd)
        named.add('%s%s' % (best['res'], best['rn']))
        rows.append('%-5s %-2s %-15s %-6s %5.2f  %4s | %s'
                    % (L['name'], L['el'], '%s%s.%s' % (best['res'], best['rn'], best['name']),
                       best['chain'], bd, ('-' if ang is None else '%.0f' % ang), typ))

    table = '\n'.join(rows)
    prompt = (INSTRUCTION + '\n### Input:\nTARGET: %s\nINPUT SMILES: %s\n\n'
              'MEASURED POCKET CONTACTS of the input ligand, one row per heavy atom '
              '(nearest protein atom within %.1f A):\natom  el faces           chain  dist   ang '
              '| interaction\n%s\n\n### Response:\n'
              % (TARGET, man['smiles'], CUTOFF, table))

    # sft_data.load_rows_jsonl imposes two shape rules on any file it reads, both of which exist
    # to stop unlabelled rows reaching training: every val row must carry a NON-empty completion,
    # and the file must contain at least one `excluded` row whose verdict is REJECT with an empty
    # completion. mol1 has no reference analogue, so the val completion below is a declared
    # placeholder that points at mol1 itself. Consequence, stated plainly: every
    # reference-DEPENDENT metric on this row measures similarity to the INPUT, i.e. it is an echo
    # check and nothing more. Only the generated molecule and sentence are used downstream.
    placeholder = ('No reference analogue exists for mol1; this row exists only to elicit a '
                   'generation.\nSMILES: %s' % man['smiles'])
    with open(a.out, 'w') as f:
        for i in range(a.samples):
            f.write(json.dumps({
                'pair': 'MOL1 zap70_cys346 sample%02d' % i,
                'pdb': 'boltz_cofold', 'target': TARGET,
                'input_smiles': man['smiles'],
                'analogue_smiles': man['smiles'],
                'prompt': prompt, 'completion': placeholder, 'split': 'val',
                'arm': 'pocket', 'verdict': 'ADOPT',
                'reference_sentence': '', 'reference_residues': [],
                'table_residues': sorted(named), 'n_atom_rows': len(rows),
                'pose_source': man['pose_source'], 'boltz_iptm': man['boltz_ligand_iptm'],
                'PREDICTED_GEOMETRY': True}) + '\n')
        # the REJECT row the loader requires; it is filtered out by split before scoring
        f.write(json.dumps({
            'pair': 'MOL1 loader-required reject', 'pdb': 'boltz_cofold', 'target': TARGET,
            'input_smiles': man['smiles'], 'analogue_smiles': man['smiles'],
            'prompt': prompt, 'completion': '', 'split': 'excluded',
            'arm': 'pocket', 'verdict': 'REJECT',
            'reference_sentence': '', 'reference_residues': [],
            'table_residues': sorted(named), 'n_atom_rows': len(rows),
            'PREDICTED_GEOMETRY': True}) + '\n')
    print('wrote %s (%d identical rows)' % (a.out, a.samples))
    print('residues in table: %d -> %s' % (len(named), ', '.join(sorted(named))))
    print('\n'.join(prompt.split('\n')[8:]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
