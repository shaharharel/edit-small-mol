#!/usr/bin/env python3
"""Verify ownership of every cited contact by rebuilding the ligand from its CCD definition.

This is the check that should have gated the report's showcase rows and did not. The author's hand
check verified a cited (residue, distance) was a real table row and then matched ELEMENTS -- an "O"
row satisfied any oxygen claim -- and passed all ten candidates. An independent chemist review
failed eight of them, with misattributions up to 14 A, and its method is what this script
automates:

  1. Rebuild the ligand from its own CCD atom/bond block (evidence_v4.ccd_mol).
  2. Require that rebuild to match input_smiles EXACTLY, atom for atom. This is the step the
     project previously lacked: reward_verifiers notes CCD-name mapping matches ambiguously for 79%
     of ligands, which is true of an unvalidated match, and is why `directional` carries weight
     0.00. Validating the rebuild against the SMILES collapses the ambiguity -- the reviewer got an
     exact match on all ten ligands it tried.
  3. From that mapping, every table row has a known molecule position.
  4. MCS the input against the generated analogue to get the atoms the edit actually touches.
  5. For each cited contact: is the owning atom one the edit touched? And, from the deposited
     coordinates, how far is the cited residue from the nearest edited atom?

Step 5's distance is what catches the failures element-matching cannot see. In the reviewed ten,
6UVF cited residues 14.3 and 13.4 A from its edit and 8U6G cited one 10.1 A away: the rows were
real, the elements matched, and the claims were nonsense.

Also checks covalent claims against the deposition's own struct_conn, because 8U6G asserts a
"1.48 A LYS102 covalent anchor" that its entry never records -- evidence_v4.cov_check inferred it
from distance and the table printed it as fact.
"""
import argparse
import collections
import gzip
import importlib.util
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REACH = 5.0          # A from the nearest edited atom; beyond this a "redirect toward" is fiction


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RETENTION = re.compile(r'\b(while|whilst|and the|,\s*the)\s+[^,]{0,40}?'
                       r'(retain|retains|retaining|held|holds|keeps?|keeping|conserv|unchanged|'
                       r'intact)', re.I)


def split_clause(sent):
    """-> (edit clause, retention clause).

    Both the ownership and the REACHABILITY test must be scoped to the edit clause. A retained
    contact belongs to an atom the edit leaves alone and is therefore legitimately far from the
    edit -- applying the 5 A reach test to it rejects correct sentences. Unscoped, this filter
    rejected the two candidates an independent chemist passed (7R2L, 28LM) while passing two it
    failed.
    """
    m = RETENTION.search(sent)
    return (sent[:m.start()], sent[m.start():]) if m else (sent, '')


def covale_residues(pdb, mirror):
    """Residues the deposition itself records as covalently bonded to a ligand, from struct_conn."""
    path = os.path.join(mirror, 'cif', '%s.cif.gz' % pdb)
    if not os.path.exists(path):
        return None
    txt = gzip.open(path, 'rt', errors='ignore').read()
    if '_struct_conn.' not in txt:
        return set()
    out = set()
    for ln in txt.split('\n'):
        if 'covale' in ln and not ln.startswith(('_', '#')):
            out |= {m.group(0) for m in re.finditer(r'\b[A-Z]{3}\b', ln)}
        elif ln.startswith('_struct_conn.conn_type_id') and 'covale' in ln:
            out.add('COVALE_DECLARED')
    return out


def name_to_index(code, input_smiles, ev4):
    """{CCD atom name: molecule atom index}, or None if the rebuild does not match the SMILES."""
    from rdkit import Chem
    cm, names = ev4.ccd_mol(code)
    if cm is None:
        return None
    a = Chem.MolFromSmiles(input_smiles)
    if a is None:
        return None
    match = cm.GetSubstructMatch(a)
    # the rebuild must cover the whole molecule: a partial match is the ambiguity that made this
    # approach unusable before
    if not match or len(match) != a.GetNumAtoms():
        return None
    return {names[match[i]]: i for i in range(a.GetNumAtoms())}


def edited_atoms(inp, gen):
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    a, b = Chem.MolFromSmiles(inp), Chem.MolFromSmiles(gen)
    if a is None or b is None:
        return None
    m = rdFMCS.FindMCS([a, b], timeout=5, ringMatchesRingOnly=True)
    core = Chem.MolFromSmarts(m.smartsString) if m.smartsString else None
    if core is None:
        return None
    kept = set(a.GetSubstructMatch(core))
    return {i for i in range(a.GetNumAtoms()) if i not in kept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', default=os.path.join(HERE, 'g3g4_eval_results/chemdfm/generations.jsonl'))
    ap.add_argument('--data-dir', default=os.path.join(HERE, 'sft_out_g3g4'))
    ap.add_argument('--out', default=os.path.join(HERE, 'heldout_ownership_verified.json'))
    ap.add_argument('--reach', type=float, default=REACH)
    a = ap.parse_args()
    import numpy as np
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    ev4 = load('ev4', 'evidence_v4.py')
    rw = load('rw', 'reward_verifiers.py')
    ext = os.path.join(ev4.D, 'ccd_atoms_ext.json')
    if os.path.exists(ext):
        ev4.CCD.update({k: v for k, v in json.load(open(ext)).items() if k not in ev4.CCD})

    ev = {}
    for line in open(os.path.join(ev4.D, 'v7_evidence.jsonl')):
        r = json.loads(line)
        ev[r['pair']] = r['MODEL_INPUT']
    meta = {r['pair']: r for r in (json.loads(l) for l in open(
        os.path.join(a.data_dir, 'sft_pocket.jsonl'))) if r['split'] == 'val'}

    def sp(t):
        t = re.sub(r'<\|[^|>]*\|>', '', t or '').replace('</s>', '')
        return (' '.join(t.split('SMILES:')[0].split()),
                t.split('SMILES:')[-1].strip().split()[0] if 'SMILES:' in t else '')

    stage = collections.Counter()
    clean, dirty = [], []
    for r in (json.loads(l) for l in open(a.gens)):
        pair = r['pair']
        mi, md = ev.get(pair), meta.get(pair)
        if mi is None or md is None:
            continue
        sent, smi = sp(r.get('completion'))
        if not smi or Chem.MolFromSmiles(smi) is None:
            continue
        stage['parses'] += 1
        s = rw.score(mi, sent, smi, mi['input_smiles'])
        if s['total'] < 1.0:
            continue
        stage['verifier_1000'] += 1
        code = pair.split()[1].split('->')[0]
        n2i = name_to_index(code, mi['input_smiles'], ev4)
        if n2i is None:
            dirty.append({'pair': pair, 'fail': 'CCD rebuild does not match input_smiles'})
            continue
        stage['ccd_rebuild_exact'] += 1
        ed = edited_atoms(mi['input_smiles'], smi)
        if not ed:
            dirty.append({'pair': pair, 'fail': 'no detectable edit'})
            continue
        P, L, _ = ev4.read_cx(md['pdb'], code)
        if not P:
            dirty.append({'pair': pair, 'fail': 'could not read the complex'})
            continue
        pos = {at['name']: np.array(at['xyz']) for at in L}
        byres = collections.defaultdict(list)
        for at in P:
            byres['%s%s' % (at['res'], at['rn'])].append(np.array(at['xyz']))
        # table row -> owning atom name
        own = {}
        for at in mi['atoms']:
            own.setdefault((str(at['faces']).split('.')[0], round(at['distance_A'], 2)),
                           at['atom'])
        edit_xyz = [pos[nm] for nm, i in n2i.items() if i in ed and nm in pos]
        problems = []
        edit_part, _ret = split_clause(sent)
        edit_claims = {(x['res'], x['d']) for x in rw.parse_claims(edit_part)
                       if x.get('d') is not None}
        for c in [x for x in rw.parse_claims(sent) if x.get('d') is not None]:
            key = (c['res'], c['d'])
            nm = own.get(key)
            if nm is None:
                problems.append('%s %.2f is not a row' % key); continue
            idx = n2i.get(nm)
            if idx is None:
                problems.append('row atom %s not in the CCD mapping' % nm); continue
            in_edit = key in edit_claims
            if idx not in ed and in_edit:
                problems.append('%s %.2f is owned by %s, which the edit does not touch'
                                % (key[0], key[1], nm))
            # reachability: how far is the cited residue from the nearest EDITED atom?
            tgt = byres.get(c['res'])
            if tgt and edit_xyz and in_edit:
                dmin = min(float(np.linalg.norm(p - q)) for p in edit_xyz for q in tgt)
                if dmin > a.reach:
                    problems.append('%s is %.1f A from the nearest edited atom' % (c['res'], dmin))
        # covalent claims against struct_conn
        if re.search(r'covalent', sent, re.I):
            cov = covale_residues(md['pdb'], ev4.D)
            if cov is not None and not cov:
                problems.append('claims a covalent bond; the deposition records no covale')
        rec = {'pair': pair, 'target': md.get('target'), 'pdb': md.get('pdb'),
               'input_smiles': mi['input_smiles'], 'proposed_smiles': smi, 'sentence': sent,
               'reward': round(s['total'], 3), 'n_edited_atoms': len(ed)}
        if problems:
            rec['fail'] = '; '.join(problems[:4]); dirty.append(rec)
        else:
            stage['ownership_and_reach_ok'] += 1; clean.append(rec)

    for k, v in stage.items():
        print('  %-24s %d' % (k, v))
    json.dump({'stages': dict(stage), 'clean': clean, 'rejected': dirty},
              open(a.out, 'w'), indent=1)
    print('\nCLEAN: %d  -> %s' % (len(clean), a.out))
    for c in clean[:20]:
        print('  %-22s %-9s %s' % (c['pair'], c['target'], c['sentence'][:100]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
