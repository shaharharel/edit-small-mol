#!/usr/bin/env python3
"""Measure how often a chair sentence under-describes the edit it is labelling.

Three chemists auditing 18 g3 training rows found 6 whose sentence described a single-site change
while the pair actually differed in more ways -- a silent stereocentre inversion, an unmentioned
halogen, a regiochemical move. 6 of 18 is enough to show the failure exists and far too small to
size it. This measures the whole corpus.

This is NOT a code bug. The pipeline faithfully records what the labeller wrote; the issue is that
the labeller was shown a matched pair and described the change they noticed, while the pair also
differed elsewhere. It matters because the model is trained to emit that sentence given that pair,
so wherever the sentence omits a difference the model is being taught to omit it too.

What counts as a multi-site difference here is deliberately conservative -- only things RDKit can
establish without judgement:

  * heavy-atom count change of more than one
  * a change in the number of rings (SSSR)
  * a change in the Bemis-Murcko scaffold
  * a change in the multiset of CIP stereocentre labels
  * a change in halogen composition

And a sentence is credited with disclosing it when it says so, via a small vocabulary of phrases
the labelling guide produced ("two-site", "also", "as well", "in addition", "plus", "while also",
"aromatises", "relocat", "migrat", "and the ... also"). The measurement is therefore a LOWER bound
on disclosure and an UPPER bound on silent multi-site edits: a sentence that describes the second
change in words not on this list is counted as silent, so the true rate is no worse than reported.
"""
import argparse, json, os, re, sys, collections
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))

DISCLOSE = re.compile(
    r'two[- ]site|three[- ]site|multi[- ]site|\balso\b|as well|in addition|\bplus\b|'
    r'aromatis|aromatiz|relocat|migrat|\bboth\b|simultaneous|at the same time|'
    r'second (change|edit|site)|while .* also', re.I)
HALOGENS = ('F', 'Cl', 'Br', 'I')


def scaffold(m):
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def stereo(m):
    try:
        return tuple(sorted(l for _, l in Chem.FindMolChiralCenters(
            m, includeUnassigned=True, useLegacyImplementation=False)))
    except Exception:
        return None


def halogens(m):
    c = collections.Counter(a.GetSymbol() for a in m.GetAtoms())
    return tuple(c[h] for h in HALOGENS)


def differences(a, b):
    """The list of conservative, RDKit-establishable differences between input and analogue."""
    out = []
    dh = abs(a.GetNumHeavyAtoms() - b.GetNumHeavyAtoms())
    if dh > 1:
        out.append('heavy_atoms_%+d' % (b.GetNumHeavyAtoms() - a.GetNumHeavyAtoms()))
    ra, rb = rdMolDescriptors.CalcNumRings(a), rdMolDescriptors.CalcNumRings(b)
    if ra != rb:
        out.append('rings_%d_to_%d' % (ra, rb))
    sa, sb = scaffold(a), scaffold(b)
    if sa is not None and sb is not None and sa != sb:
        out.append('scaffold_changed')
    ta, tb = stereo(a), stereo(b)
    if ta is not None and tb is not None and ta != tb:
        out.append('stereo_%s_to_%s' % (''.join(ta) or '-', ''.join(tb) or '-'))
    ha, hb = halogens(a), halogens(b)
    if ha != hb:
        out.append('halogens_%s_to_%s' % (ha, hb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'sft_out_g3/sft_pocket.jsonl'))
    ap.add_argument('--split', default='train')
    ap.add_argument('--json', default=None)
    ap.add_argument('--show', type=int, default=8)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if r['split'] == a.split]
    n = unparsed = 0
    multi = disclosed = silent = 0
    kinds = collections.Counter()
    examples = []
    for r in rows:
        A = Chem.MolFromSmiles(r.get('input_smiles') or '')
        B = Chem.MolFromSmiles(r.get('analogue_smiles') or '')
        if A is None or B is None:
            unparsed += 1
            continue
        n += 1
        d = differences(A, B)
        if not d:
            continue
        multi += 1
        for x in d:
            kinds[x.split('_')[0]] += 1
        s = (r.get('reference_sentence') or '')
        if DISCLOSE.search(s):
            disclosed += 1
        else:
            silent += 1
            if len(examples) < a.show:
                examples.append((r['pair'], d, s[:150]))

    print('%s  split=%s' % (a.data, a.split))
    print('rows parsed              %5d   (%d unparseable, excluded)' % (n, unparsed))
    print('multi-site by RDKit      %5d   %5.1f%% of parsed' % (multi, 100.0*multi/max(n,1)))
    print('  sentence discloses it  %5d   %5.1f%% of multi-site' % (disclosed, 100.0*disclosed/max(multi,1)))
    print('  SILENT                 %5d   %5.1f%% of multi-site, %5.1f%% of ALL rows'
          % (silent, 100.0*silent/max(multi,1), 100.0*silent/max(n,1)))
    print('\ndifference kinds among multi-site rows (a row may have several):')
    for k, v in kinds.most_common():
        print('  %-12s %5d' % (k, v))
    if examples:
        print('\nexamples of silent multi-site rows:')
        for pair, d, s in examples:
            print('  %-24s %s\n      %s' % (pair, ','.join(d), s))
    if a.json:
        json.dump({'data': a.data, 'split': a.split, 'parsed': n, 'unparseable': unparsed,
                   'multi_site': multi, 'disclosed': disclosed, 'silent': silent,
                   'kinds': dict(kinds)}, open(a.json, 'w'), indent=2)
        print('\nwrote %s' % a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
