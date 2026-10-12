#!/usr/bin/env python3
"""Record how each adjudicated label stands against REVIEWER.md's mechanical rules. A COLUMN, NEVER A GATE.

    python check_label_rules.py                        # every label_out/batch_*.json
    python check_label_rules.py --write                 # add the flags to the label files
    python check_label_rules.py --only 0021 0022        # just these batches
    python check_label_rules.py --violations r2         # print the sentences that break one rule

This exists because the project's standing instruction is that nothing is dropped: a REJECT verdict
is kept as a column rather than deleting the pair, and rule compliance is the same kind of fact. So
this script never blocks a labelling run, never halts a batch, and always exits 0 unless it cannot
read its inputs. It annotates. The training filter decides later what to select on, which is the
only place that decision belongs -- a gate here would silently shrink the corpus at the moment of
collection, when the information is most expensive to recover.

WHAT IS CHECKED, and what each rule is really asking

  r1   every residue named in the sentence appears on the INPUT side of the evidence
       (input atom table `faces`, its `polar partner:` line, or a conserved row's input column).
       A residue only in `analogue_contacts` fails: at inference there is no analogue crystal, so
       naming it teaches the model to invent a partner. 24% of the first 2,000 labels break this,
       and 100% of those mentions trace to the analogue's table with 0% invented -- the labellers
       were reading real evidence under an instruction that was wrong.

  r2   no number in the sentence is a measured ANALOGUE distance. Implemented as "no number matches
       an analogue-side measured value unless it also matches an input-side one", which is the
       rule's intent rather than the stricter "every number must be input-side": a difference the
       labeller computed from two input values ("0.2 A tighter") is legitimate and the strict form
       would fail it. 92% of the first 2,000 usable labels break this one; it is the binding
       constraint on converting the old corpus, not a marginal check.

  r2b  at least one named residue is contacted by a `site_of_change` atom -- the anchor sits at the
       edit's own location rather than on a bystander residue. This is what keeps the reasoning
       tied to the input complex instead of drifting to whatever is easiest to cite.

  r3   no analogue-provenance phrasing ("in the analogue's crystal", "measured at", "the crystal
       shows"). A sentence may describe an expected consequence; it may not report an observation
       the model will not have.

  r5   none of REVIEWER.md rule 4's banned claims: burial, solvent-facing, room, channel, potency.
       The first 2,000 labels pass this at 100% -- it is here to keep it that way, not to find
       problems.

  r6   at most 35 words.

Rounding: distances are compared at two decimals, which is how both the packet and the sentences
print them. A sentence citing 3.4 where the table says 3.39 therefore counts as a miss; that is
deliberate, because a labeller quoting a number should quote the number.

Not checked: rule 4's expectation vocabulary (r4 in the brief). Whether a hedge is present is easy
to test and whether it is honest is not, and a mechanical hedge check would reward "should" as
punctuation. The chair judges that one.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

AA = frozenset('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL '
               'MSE SEC PYL HID HIE HIP'.split())
_RES = re.compile(r'\b([A-Z]{3})\s?(\d+)')
_NUM = re.compile(r'\d+\.\d{1,2}')
_PROV = re.compile(r"analogue'?s? crystal|in the analogue'?s? (crystal|structure)|measured at"
                   r"|observed at|crystal shows|crystallis?ed at", re.I)
_BANNED = re.compile(r'\bburi(ed|al)\b|solvent[- ]facing|solvent[- ]exposed|\bchannel\b|\broom\b'
                     r'|\bcavity\b|potenc|\bIC50\b|\bKi\b|affinit', re.I)

HERE = os.path.dirname(os.path.abspath(__file__))
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
RULES = ('r1', 'r2', 'r2b', 'r3', 'r5', 'r6')


def _res(s):
    m = _RES.match(str(s or ''))
    return '%s%s' % (m.group(1), m.group(2)) if m and m.group(1) in AA else None


def _res_all(s):
    return {'%s%s' % (m.group(1), m.group(2)) for m in _RES.finditer(s or '')
            if m.group(1) in AA}


def sides(rec):
    """The input side and analogue side of one pair, as the rules need to see them."""
    mi = rec['MODEL_INPUT']['atoms']
    lo = rec['LABEL_ONLY']
    inp_res, inp_num = set(), set()
    for a in mi:
        for k in ('faces', 'polar_partner'):
            r = _res(a.get(k))
            if r:
                inp_res.add(r)
        for k in ('distance_A', 'polar_distance_A'):
            if a.get(k) is not None:
                inp_num.add(round(a[k], 2))
    for c in (lo.get('conserved_interactions') or []):
        r = _res(c.get('input_faces'))
        if r:
            inp_res.add(r)
        if c.get('input_distance_A') is not None:
            inp_num.add(round(c['input_distance_A'], 2))
    ana_num = set()
    for c in (lo.get('analogue_contacts') or []):
        if c.get('distance_A') is not None:
            ana_num.add(round(c['distance_A'], 2))
    for c in (lo.get('conserved_interactions') or []):
        if c.get('analogue_distance_A') is not None:
            ana_num.add(round(c['analogue_distance_A'], 2))
    soc = {x['atom'] for x in (lo.get('site_of_change') or [])}
    soc_res = set()
    for a in mi:
        if a['atom'] in soc:
            for k in ('faces', 'polar_partner'):
                r = _res(a.get(k))
                if r:
                    soc_res.add(r)
    return inp_res, inp_num, ana_num, soc_res


def check(sentence, rec):
    inp_res, inp_num, ana_num, soc_res = sides(rec)
    named = _res_all(sentence)
    nums = {round(float(x), 2) for x in _NUM.findall(sentence)}
    bad_nums = sorted(x for x in nums if x in ana_num and x not in inp_num)
    return {
        'r1': bool(named) and named <= inp_res,
        'r2': not bad_nums,
        'r2b': bool(named & soc_res),
        'r3': not _PROV.search(sentence),
        'r5': not _BANNED.search(sentence),
        'r6': len(sentence.split()) <= 35,
        'ungrounded_residues': sorted(named - inp_res),
        'analogue_numbers_cited': bad_nums,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help='add a `rules` block to each label record in place')
    ap.add_argument('--only', nargs='*', default=None, help='batch numbers, e.g. 0021 0022')
    ap.add_argument('--violations', default=None, choices=RULES,
                    help='print the sentences breaking this rule')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(HERE, 'label_out', 'batch_*.json')))
    if a.only:
        files = [f for f in files if any(('batch_%s.json' % b) in f for b in a.only)]
    if not files:
        print('no label files found', file=sys.stderr)
        return 0

    labels = {}
    for f in files:
        for pid, v in json.load(open(f)).items():
            labels[pid] = (f, v)
    print('label files %d   labels %s' % (len(files), format(len(labels), ',')))

    ev = {}
    for line in open(EV):
        r = json.loads(line)
        if r['pair'] in labels:
            ev[r['pair']] = r
    print('evidence matched %s' % format(len(ev), ','))

    tally = collections.Counter()
    per_file = collections.defaultdict(dict)
    shown = 0
    n = 0
    for pid, (f, v) in labels.items():
        rec = ev.get(pid)
        if rec is None:
            tally['no_evidence_row'] += 1
            continue
        sent = (v.get('final') or '').strip()
        if not sent:
            tally['empty_sentence'] += 1
            continue
        res = check(sent, rec)
        per_file[f][pid] = res
        # Only usable labels are tallied: a REJECT's sentence is deliberately the most defensible
        # thin thing that can be said, so holding it to the shape rules would measure nothing.
        if v.get('verdict') in ('ADOPT', 'REWRITE'):
            n += 1
            for k in RULES:
                tally[k] += res[k]
            tally['ALL'] += all(res[k] for k in RULES)
            if a.violations and not res[a.violations] and shown < 12:
                shown += 1
                extra = (res['analogue_numbers_cited'] if a.violations == 'r2'
                         else res['ungrounded_residues'])
                print('\n  %-24s %s' % (pid, ('%s' % extra) if extra else ''))
                print('      %s' % sent[:200])

    if n:
        print('\nusable (ADOPT+REWRITE) labels checked: %s\n' % format(n, ','))
        for k in RULES:
            print('  %-5s %6s  (%5.1f%%)' % (k, format(tally[k], ','), 100.0 * tally[k] / n))
        print('  %-5s %6s  (%5.1f%%)' % ('ALL', format(tally['ALL'], ','),
                                         100.0 * tally['ALL'] / n))
    for k in ('no_evidence_row', 'empty_sentence'):
        if tally[k]:
            print('  %-5s %6d' % (k, tally[k]))

    if a.write:
        for f, flags in per_file.items():
            d = json.load(open(f))
            for pid, res in flags.items():
                d[pid]['rules'] = res
            json.dump(d, open(f, 'w'), indent=1)
        print('\nwrote a `rules` block into %d label files' % len(per_file))

    # Always 0. This annotates; it does not gate. See the module docstring.
    return 0


if __name__ == '__main__':
    sys.exit(main())
