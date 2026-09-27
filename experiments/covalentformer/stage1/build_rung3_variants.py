#!/usr/bin/env python
"""RUNG 3, THREE OUTPUT VARIANTS -- the ablation the user asked for.

THE QUESTION. Four independent experiments say the pocket conditioning is inert when it enters
through the PROMPT. None of them tested whether making the model EMIT pocket-grounded tokens
changes what it generates. These three arms differ ONLY in what the answer must contain; the hit,
the pocket block and the split are identical across all three, so any difference is attributable to
the output format and nothing else.

  R3_SMILES    Analogue: <SMILES>
               No pocket-related output at all. The CONTROL. If the other two do not beat it, the
               pocket-in-output idea is dead by the same standard that killed pocket-in-prompt.

  R3_COV       Residue: <NUC>
               Analogue: <SMILES>
               Adds the covalent site -- the one pocket-derived label measured to carry real signal
               (+7 to +13 pts over the constant-CYS floor on both informative folds).

  R3_COVPOCK   Residue: <NUC>
               Pocket: most room <quadrant> (wall <d> A)      <- COPYABLE, see below
               Change: <grow|trim> <n> heavy atoms            <- NOT copyable
               Analogue: <SMILES>

THE TRAP IN R3_COVPOCK, NAMED BEFORE IT IS BUILT. The prompt already renders every quadrant's wall
distance, so "most room" is a deterministic function OF THE PROMPT. A model can score 100% on that
line by copying, learning nothing -- which is exactly how the v3 aux head came to be reading its own
label. It is kept anyway, because the hypothesis is about whether EMITTING the grounding changes the
SMILES, not about whether the line is hard. Two protections:
  * the copy floor is COMPUTED HERE and printed, so the line can never be quoted as a capability;
  * the `Change:` line is NOT prompt-derivable -- it is determined by the answer -- so it forces the
    model to commit to a plan before emitting the molecule. That is the part that can actually move
    generation.

PRE-REGISTERED READING, stated before any model is trained:
  H1  R3_COVPOCK > R3_COV > R3_SMILES on analogue quality -> emitting pocket grounding helps.
  H2  all three equal                                     -> the pocket is inert in the OUTPUT too,
                                                             and the pocket direction closes.
  H3  R3_SMILES wins                                      -> the extra tokens are a tax; the site
                                                             head should be a separate head, not
                                                             part of the generation target.

SPLIT: inherits the shared cross-rung Murcko scaffold split, so a rung-3 test series is unseen by
rung 1 and rung 2 as well. Rung-3-only scaffolds (never seen by the other rungs) keep rung 3's
ORIGINAL split assignment -- dropping them would gut a 373-row test set. Both routes are counted.
"""
from __future__ import annotations
import os, sys, json, re, collections

REPO = '/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer'
os.chdir(REPO)
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

OUT = 'data/ladder_v3'
SRC = {s: 'data/ladder/rung3_%s.jsonl' % s for s in ('train', 'valid', 'test', 'mpro')}
WALL = re.compile(r'quadrant ([-+]x[-+]y):.*?nearest wall ([\d.]+) A')


def heavy(s):
    m = Chem.MolFromSmiles(s) if s else None
    return m.GetNumHeavyAtoms() if m else None


def main():
    assign = json.load(open(os.path.join(OUT, 'scaffold_split.json')))
    scaf = json.load(open(os.path.join(OUT, 'mol_scaffold.json')))
    print('shared split: %d scaffolds | molecule->scaffold map %d' % (len(assign), len(scaf)))

    def scaffold_of(s):
        if s in scaf:
            return scaf[s]
        m = Chem.MolFromSmiles(s)
        v = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ''
        v = v or ('ACYCLIC::' + s)
        scaf[s] = v
        return v

    rows, src_split = [], {}
    for sp, p in SRC.items():
        for line in open(p):
            d = json.loads(line)
            rows.append(d)
            src_split[id(d)] = sp

    route = collections.Counter()
    out_rows = collections.defaultdict(list)
    for d in rows:
        hit, ana = d.get('hit'), d.get('analogue')
        if not (hit and ana):
            route['no_molecule'] += 1
            continue
        sh, sa = scaffold_of(hit), scaffold_of(ana)
        ah, aa = assign.get(sh), assign.get(sa)
        orig = src_split[id(d)]
        if ah and aa:
            if ah != aa:
                route['dropped_straddle'] += 1
                continue
            sp = ah if orig != 'mpro' else 'mpro'   # Mpro stays its OWN fold, always
            route['by_shared_split'] += 1
        else:
            sp = orig
            route['rung3_only_scaffold_kept_original'] += 1
        out_rows[sp].append(d)
    print('routing:', dict(route))

    # copy floor for the "most room" line -- the fraction a pure argmax-of-prompt copier gets right
    stats = collections.Counter()
    made = collections.Counter()
    for variant in ('R3_SMILES', 'R3_COV', 'R3_COVPOCK'):
        for sp, ds in out_rows.items():
            with open(os.path.join(OUT, 'rung3_%s_%s.jsonl' % (variant, sp)), 'w') as fo:
                for d in ds:
                    nuc, ana, hit = d['nucleophile'], d['analogue'], d['hit']
                    ha, hh = heavy(ana), heavy(hit)
                    if variant == 'R3_SMILES':
                        task = 'Task: %s. Give the analogue.' % _op(d)
                        out = 'Analogue: %s' % ana
                    elif variant == 'R3_COV':
                        task = ('Task: %s. State which protein residue the warhead reacts with, '
                                'then give the analogue.' % _op(d))
                        out = 'Residue: %s\nAnalogue: %s' % (nuc, ana)
                    else:
                        q = WALL.findall(d['instruction'])
                        if q:
                            best = max(q, key=lambda t: float(t[1]))
                            stats['copyable_rows'] += 1
                        else:
                            best = ('?', '0.0')
                        delta = (ha - hh) if (ha is not None and hh is not None) else 0
                        verb = 'grow' if delta > 0 else ('trim' if delta < 0 else 'hold')
                        task = ('Task: %s. State which protein residue the warhead reacts with, '
                                'then which quadrant has most room and how the analogue changes, '
                                'then give the analogue.' % _op(d))
                        out = ('Residue: %s\nPocket: most room %s (wall %s A)\n'
                               'Change: %s %d heavy atoms\nAnalogue: %s'
                               % (nuc, best[0], best[1], verb, abs(delta), ana))
                        stats['verb_' + verb] += 1
                    instr = re.sub(r'Task: .*$', task, d['instruction'], flags=re.S)
                    fo.write(json.dumps({**d, 'instruction': instr, 'output': out,
                                         'variant': variant, 'split': sp}) + '\n')
                    made[(variant, sp)] += 1

    print()
    print('%-12s %8s %7s %6s %6s' % ('variant', 'train', 'valid', 'test', 'mpro'))
    for v in ('R3_SMILES', 'R3_COV', 'R3_COVPOCK'):
        print('%-12s %8d %7d %6d %6d' % (v, made[(v, 'train')], made[(v, 'valid')],
                                         made[(v, 'test')], made[(v, 'mpro')]))
    print()
    print('R3_COVPOCK FLOORS -- quote these beside any score on those lines:')
    print('  "Pocket: most room" is a deterministic argmax of the PROMPT -> copy floor 1.0000')
    tot = sum(stats['verb_' + v] for v in ('grow', 'trim', 'hold'))
    if tot:
        mx = max(stats['verb_' + v] for v in ('grow', 'trim', 'hold'))
        print('  "Change:" verb distribution %s -> majority floor %.4f'
              % ({v: stats['verb_' + v] for v in ('grow', 'trim', 'hold')}, mx / tot))
    print('ALL_DONE_RUNG3_VARIANTS')
    return 0


def _op(d):
    return {'GROW': 'grow the substituent', 'TRIM': 'trim the substituent',
            'REPLACE': 'replace the substituent'}.get(d.get('op', ''), 'modify the substituent')


if __name__ == '__main__':
    sys.exit(main())
