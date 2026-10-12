#!/usr/bin/env python3
"""Build the review packet a medicinal-chemist cohort judges the written labels against.

Ten chemists each read the SAME packet, independently. That is the point: a single chemist's
verdict is one opinion and we have already been burned twice by treating it as ground truth
(round 1 called 4 of 20 WRONG; round 2 called 2 WRONG, and on 7IIY the measurement did not
support the chemist). Ten independent reads turn "is this label right" into a distribution we
can act on -- unanimous WRONG is a defect, 5-5 is an ambiguous pair, and a lone dissent is noise.

The packet deliberately carries the SMILES and the full evidence the labeller saw, so a chemist
can contradict the label from the chemistry rather than only from our own fields. It does NOT
carry the gate's verdict: the gate is mechanical and agrees with the brief by construction, so
showing it would anchor the cohort on the brief instead of on the molecules.
"""
import glob, json, os, random, sys

SP = os.path.dirname(os.path.abspath(__file__))
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

ev = {}
for line in open(EV):
    r = json.loads(line)
    ev[r['pair']] = r

labels = {}
for p in sorted(glob.glob(os.path.join(SP, 'label_out', 'batch_*.json'))):
    try:
        labels.update(json.load(open(p)))
    except Exception as e:
        print('UNPARSEABLE %s: %s' % (os.path.basename(p), e))

# A REJECT has no sentence to judge, so the cohort cannot score it. Keep the rejects out of the
# packet but report them -- whether the panel was RIGHT to reject is a separate question, and the
# user has one open design call on it (solvent-only edits: reject, or label as solvent-directed).
written = {k: v for k, v in labels.items() if (v.get('final') or '').strip()}
rejected = sorted(set(labels) - set(written))

print('labels collected   %d' % len(labels))
print('  with a sentence  %d' % len(written))
print('  REJECTed         %d  %s' % (len(rejected), ', '.join(rejected[:6])))

missing = [k for k in written if k not in ev]
if missing:
    print('\nNOT IN EVIDENCE (packet would be unjudgeable): %s' % missing[:5])
    written = {k: v for k, v in written.items() if k in ev}

random.seed(11)
pick = sorted(random.sample(sorted(written), min(N, len(written))))

out = []
for i, pid in enumerate(pick, 1):
    r, lab = ev[pid], written[pid]
    mi, lo = r['MODEL_INPUT'], r['LABEL_ONLY']
    b = []
    b.append('=' * 96)
    b.append('PAIR %d of %d        %s' % (i, len(pick), pid))
    b.append('=' * 96)
    b.append('')
    # analogue_smiles lives in r['ANSWER'], and target at the top level -- reading them from
    # LABEL_ONLY/MODEL_INPUT printed "?" for all 42 pairs, so the first cohort judged these labels
    # without ever seeing the analogue molecule or the protein. The whole point of the review is
    # chemistry, and chemistry is not auditable from the contact tables alone.
    b.append('input molecule     %s' % mi.get('input_smiles', '?'))
    b.append('analogue molecule  %s' % ((r.get('ANSWER') or {}).get('analogue_smiles') or '?'))
    b.append('protein (UniProt)  %s' % (r.get('target') or '?'))
    b.append('input crystal      %s        analogue crystal  %s'
             % (pid.split()[0], lo.get('analogue_pdb') or '?'))
    b.append('tanimoto           %s        heavy atoms added %s'
             % (r.get('tanimoto'), lo.get('heavy_atoms_added')))
    if lo.get('unlabellable'):
        b.append('FLAGGED UNLABELLABLE: %s' % '; '.join(lo.get('unlabellable_reason') or []))
    b.append('')
    b.append('--- THE SENTENCE UNDER REVIEW -------------------------------------------------')
    b.append('  "%s"' % lab['final'].strip())
    b.append('')
    b.append('--- site of change (which atoms the edit touched) -----------------------------')
    for s in lo.get('site_of_change', []):
        b.append('  %-6s %s' % (s.get('atom', '?'), s.get('role', '?')))
    b.append('')
    b.append('--- what the INPUT molecule touches in its own crystal ------------------------')
    for a in mi['atoms']:
        b.append('  %-5s %-2s  %-14s %-6s %s' % (
            a.get('atom', ''), a.get('element', ''), a.get('faces', ''),
            a.get('dist', ''), a.get('interaction', '')))
        if a.get('polar_partner'):
            b.append('        %-14s %-6s %s' % (a['polar_partner'], a.get('polar_dist', ''),
                                                a.get('polar_interaction', '')))
    b.append('')
    b.append('--- contacts RETAINED in both crystals ----------------------------------------')
    for c in (lo.get('conserved_interactions') or []):
        b.append('  %-5s %-12s input: %-28s analogue: %s' % (
            c.get('input_atom', ''), c.get('faces', ''),
            c.get('input_interaction', ''), c.get('analogue_interaction', '')))
    b.append('')
    b.append('--- what the ANALOGUE touches in ITS crystal (label-only; not shown to a model) ')
    for c in (lo.get('analogue_contacts') or []):
        b.append('  %-5s %-2s  %-14s %-6s %s' % (
            c.get('atom', ''), c.get('element', ''), c.get('faces', ''),
            c.get('dist', ''), c.get('interaction', '')))
    b.append('')
    out.append('\n'.join(b))

hdr = """MEDICINAL-CHEMIST COHORT REVIEW -- %d analogue pairs

You are one of ten chemists reading this packet independently. Do not try to guess what the other
nine will say, and do not try to be lenient or harsh to balance them out; score what you see.

For EACH pair, judge the ONE sentence under review on a single question:

    does this sentence correctly describe what the edit does to the binding site?

  GOOD   a chemist reading it would learn something true and useful about the edit
  WEAK   nothing in it is false, but it misses the point of the edit, states something trivial,
         or omits the change that actually matters
  WRONG  it asserts something the structures do not support, or contradicts the chemistry

Judge from the CHEMISTRY FIRST. The evidence tables below each pair are our pipeline's output and
may themselves be wrong -- that is one of the things we are trying to find out. If the tables and
the molecules disagree, say so explicitly and tell us which you trust; that is the most valuable
thing you can report. If a sentence is right but for a reason the tables do not show, say that too.

Known conventions, so you do not mark these as errors:
  * There is NO hit-to-lead claim. The two molecules are analogues from the same paper and the
    direction is an arbitrary coin flip, so the analogue is as often the smaller molecule.
  * No affinity or potency data is available, so a sentence must not claim improvement.
  * The sentence is capped at 35 words, so it cannot describe everything; judge whether it picked
    the RIGHT thing to say, not whether it said everything.
""" % len(pick)

path = os.path.join(SP, 'cohort_packet.txt')
with open(path, 'w') as f:
    f.write(hdr + '\n\n' + '\n\n'.join(out) + '\n')

print('\nwrote %s  (%d pairs, %.0f KB)' % (path, len(pick), os.path.getsize(path) / 1024.0))
with open(os.path.join(SP, 'cohort_pairs.json'), 'w') as f:
    json.dump({'pairs': pick, 'rejected': rejected,
               'sentences': {k: written[k]['final'] for k in pick}}, f, indent=1)
print('wrote cohort_pairs.json')
