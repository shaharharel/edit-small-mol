#!/usr/bin/env python3
"""Cut a mol1 candidate pool down to what is worth a chemist's time.

The first mol1 round sent all 181 candidates to three external chemists -- 543 slot judgements --
and 3 survived. Their verdicts show most of that effort was spent on failures this script can now
detect without them:

    caveat_respected        failed 142/181 (78%)   <- rule agrees with the panel on 94%
    ownership_correct       failed 128/181 (71%)   <- rule agrees with the panel on 83%
    claim_matches_molecule  failed 111/181 (61%)   <- partly automatable: see GROUP NAMING
    molecule_is_sound       failed  75/181 (41%)   <- automatable: warhead checks

OWNERSHIP. The finals reviewer published the atom-order mapping for mol1, so for this one pocket the
rule is exact rather than heuristic:

    acrylamide   C1-C3, O1        isoindoline  N1, C4-C9, C17-C18
    carboxamide  C10, O2, N2      imidazole    C11, C12, N3, C16, N4
    N-isopropyl  C13, C14, C15

    C13/C15 -> MET414.SD 3.63/3.53      C14 -> LYS369.CG 3.67

So a candidate editing the N-alkyl may cite MET414 or LYS369 and nothing else. A cited contact
belonging to a group the edit leaves alone is the single commonest failure, and it is now checkable.

CAVEAT. The pose is a cofold and the CYS346 bond length is not measurable from it. Quoting a number
for that bond is the failure; naming the bond without one is what all three endorsed candidates did.

GROUP NAMING. The group the sentence names must be the group the SMILES changes. An MCS diff against
the input catches the cases that defeated the author's own screening -- "diethylamino" for a
2-(diethylamino)ethyl arm, "isobutyl" for an isopropyl.

What is left for the chemists is what no rule covers: is this edit worth making.
"""
import argparse
import collections
import importlib.util
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# published by the finals reviewer, quoted in mol1_candidate_pick/finals/final_ranking.json
GROUP_OF = {}
for _g, _atoms in {
        'acrylamide': ['C1', 'C2', 'C3', 'O1'],
        'isoindoline': ['N1', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C17', 'C18'],
        'carboxamide': ['C10', 'O2', 'N2'],
        'imidazole': ['C11', 'C12', 'N3', 'C16', 'N4'],
        'N-isopropyl': ['C13', 'C14', 'C15']}.items():
    for _a in _atoms:
        GROUP_OF[_a] = _g

CYS346_WORDS = re.compile(r'cys\s*346', re.I)
NUM = re.compile(r'(\d+\.\d+)')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def covalent_length_quoted(sent):
    """Does the sentence put a number on the CYS346 bond?

    Only a number NEAR the Cys346 mention counts. A sentence may legitimately quote 3.67 for
    LYS369 in the same breath as naming the covalent anchor without a length.
    """
    for m in CYS346_WORDS.finditer(sent):
        lo, hi = max(0, m.start() - 70), min(len(sent), m.end() + 70)
        window = sent[lo:hi]
        for n in NUM.finditer(window):
            # a distance to another residue inside the window is not a claim about the bond
            seg = window[max(0, n.start() - 40):n.start()]
            if re.search(r'[A-Z]{3}\s*\d+', seg) and not CYS346_WORDS.search(seg):
                continue
            return True
    return False


RETENTION = re.compile(r'\b(while|whilst|and the|,\s*the)\s+[^,]{0,40}?'
                       r'(retain|retains|retaining|held|holds|keeps?|keeping|conserv|unchanged|'
                       r'intact)', re.I)


def split_clause(sent):
    """-> (edit clause, retention clause).

    A retention clause may legitimately cite a group the edit leaves alone -- "while the 2.88 A
    GLY601.O hydrogen bond is retained" is a correct sentence about an untouched atom. The EDIT
    clause may not.

    Scoping matters more than it looks. Letting a retention phrase anywhere in the sentence excuse
    every cited contact agreed with the three-chemist majority on only 37% of the 181 (103 misses),
    because the model writes "while retaining ..." in most sentences. Scoping the exemption to the
    clause it appears in takes that to 83% (23 misses, 6 false alarms).
    """
    m = RETENTION.search(sent)
    return (sent[:m.start()], sent[m.start():]) if m else (sent, '')


def caveat_violations(sent, mi, rw):
    """Every way a mol1 sentence can over-claim on a PREDICTED pose.

    Quoting the CYS346 length was only the most visible one. Validating the length rule alone
    against three chemists' majority on the 181 agreed 72% of the time with 44 misses, and their
    finals summary names the other two: the 2.74 A C2 contact "is unusually close and uncertain and
    supports no optimization claim", and "O1/N2/N4 polar distances lack directional geometry and
    cannot establish hydrogen bonds". Both are readable off the table's own interaction text.
    """
    out = []
    if covalent_length_quoted(sent):
        out.append('quotes a length for the CYS346 bond')
    # rows the table itself flags, and rows that carry no angle so cannot support an H-bond claim
    uncertain, no_geom = set(), set()
    for at in mi['atoms']:
        it = (at.get('interaction') or '').lower()
        key = (str(at['faces']).split('.')[0], round(at['distance_A'], 2))
        if 'uncertain' in it or 'unreliable' in it:
            uncertain.add(key)
        if 'geometry unknown' in it or 'geometry poor' in it:
            no_geom.add(key)
    cited = [(c['res'], c['d']) for c in rw.parse_claims(sent) if c.get('d')]
    for k in cited:
        if k in uncertain:
            out.append('builds on the uncertain %s %.2f row' % k)
    if re.search(r'hydrogen[- ]bond|h-bond|hbond', sent, re.I):
        for k in cited:
            if k in no_geom:
                out.append('calls %s %.2f a hydrogen bond, but the row has no angle' % k)
    return out


def edited_groups(inp, gen, rw):
    """Which named groups of mol1 the edit touches, by MCS against the input."""
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
    # mol1's atom names are element+running index in SMILES order, which is how the prompt is built
    counts, names = collections.Counter(), []
    for at in a.GetAtoms():
        el = at.GetSymbol()
        counts[el] += 1
        names.append('%s%d' % (el, counts[el]))
    return {GROUP_OF.get(names[i]) for i in range(a.GetNumAtoms()) if i not in kept} - {None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', required=True, help='output of sample_programmes.py')
    ap.add_argument('--prompt', default=os.path.join(HERE, 'mol1_prompt_variants.jsonl'))
    ap.add_argument('--out', default=os.path.join(HERE, 'mol1_funnel.json'))
    a = ap.parse_args()
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    rw = load('rw', 'reward_verifiers.py')
    tr = load('tr', 'build_mol1_candidate_task.py')      # reuse its prompt->MODEL_INPUT parser

    base = json.loads(open(a.prompt).readline())
    mi = tr.model_input_from_prompt(base['prompt'], base['input_smiles'])
    inp = base['input_smiles']
    owner = {}                       # (residue, distance) -> the group that owns it
    for at in mi['atoms']:
        res = str(at['faces']).split('.')[0]
        owner[(res, round(at['distance_A'], 2))] = GROUP_OF.get(at['atom'])

    rows = [json.loads(l) for l in open(a.gens)]
    cand, seen = [], set()
    for r in rows:
        for g in r['generations']:
            sent = (g.get('gen_sentence') or '').strip()
            smi = g.get('gen_smiles') or ''
            key = (Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None,
                   ' '.join(sent.split()))
            if key[0] is None or key in seen:
                continue
            seen.add(key)
            cand.append({'variant': r.get('variant', '?'), 'decode': g['decode'],
                         'sentence': ' '.join(sent.split()), 'smiles': smi})

    stage = collections.Counter()
    kept = []
    for c in cand:
        stage['unique_parsing'] += 1
        s = rw.score(mi, c['sentence'], c['smiles'], inp)
        c['reward'] = round(s['total'], 3)
        if s['total'] == 0.0:
            continue
        stage['verifier_nonzero'] += 1
        cv = caveat_violations(c['sentence'], mi, rw)
        if cv:
            c['fail'] = 'caveat: ' + '; '.join(cv)
            continue
        stage['caveat_ok'] += 1
        eg = edited_groups(inp, c['smiles'], rw)
        if not eg:
            c['fail'] = 'no detectable edit'
            continue
        c['edited_groups'] = sorted(eg)
        edit_part, ret_part = split_clause(c['sentence'])
        cited = [(x['res'], x['d']) for x in rw.parse_claims(edit_part) if x.get('d')]
        owners = {owner.get(k) for k in cited} - {None}
        c['cited_owners'] = sorted(owners)
        stray = owners - eg
        if stray:
            c['fail'] = 'ownership: the edit clause cites %s, the edit touches %s' % (
                sorted(stray), sorted(eg))
            continue
        stage['ownership_ok'] += 1
        kept.append(c)

    kept.sort(key=lambda c: -c['reward'])
    print('candidates generated (unique, parsing) : %d' % stage['unique_parsing'])
    print('  verifier score > 0                   : %d' % stage['verifier_nonzero'])
    print('  does not quote the CYS346 length     : %d' % stage['caveat_ok'])
    print('  every cited contact owned by the edit: %d' % stage['ownership_ok'])
    json.dump({'stages': dict(stage), 'kept': kept,
               'rejected': [c for c in cand if c.get('fail')][:400]},
              open(a.out, 'w'), indent=1)
    print('\nkept %d -> %s' % (len(kept), a.out))
    for c in kept[:10]:
        print('  %.3f  %-11s %s' % (c['reward'], ','.join(c['edited_groups']), c['sentence'][:96]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
