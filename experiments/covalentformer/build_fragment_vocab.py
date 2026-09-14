"""EXPANDED FRAGMENT VOCABULARY for Phase A pre-training.

WHY. Phase A's ceiling is not the number of (fragment, gap) rows -- gaps are free -- it is the number
of distinct REACTIVE ENDS the model ever sees. tierA_v3 had 3,477. Millions of rows over 3,477
distinct outputs teaches the gap->shape map and no new chemistry; the model can only recombine what
it was shown.

SOURCES, and why each is defensible:
  CovInDB 2.0            ~10.4k real covalent inhibitors -- ground truth chemistry, 74 warheads
  prior RL cohorts       ~200k generated kinase/covalent molecules from validated runs in this repo
                         (exp6_v5, murcko_rl_*, thiq_rl_*). These are NOT real molecules, but they
                         are drug-like kinase chemotypes produced by priors trained on ChEMBL, and
                         they are used here ONLY as a source of LINKER TOPOLOGY -- never as targets
                         to imitate. Their properties are not inherited: every fragment is re-cut
                         and re-profiled from scratch.

TWO CUT MODES, because the roles differ:
  warhead_directed   the electrophile lands in the FRAGMENT (what v3 did) -> teaches "write a
                     reactive end that spans a gap"
  warhead_retained   the electrophile stays in the RETAINED half -> teaches "rewrite the scaffold
                     that positions a FIXED electrophile". This is the role the hit-to-lead use case
                     actually needs and the one that makes acrylamide retention 100% by construction
                     rather than the 52.2% we measured.

Every fragment carries exactly one attachment point and, for the warhead_directed set, exactly one
electrophile from the 5-class vocabulary. Deduplicated by canonical SMILES.
"""
import os, sys, csv, json, glob, argparse, collections
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import electrophile_index

MIN_FRAG_HEAVY, MAX_FRAG_HEAVY = 4, 26
MIN_KEEP_HEAVY = 8


def smiles_from(path, limit=0):
    """-> (smiles list, n_read_before_cap, error_or_None).

    The first version wrapped the whole read in `except Exception: pass` and returned whatever it
    had accumulated. A malformed or truncated CSV then looked identical to a genuinely small one and
    the source quietly vanished from the report -- the same swallow-and-emit-a-default pattern that
    made vocab.decode report 0% validity as a model failure. It also applied --per-source silently,
    so a source capped at 60,000 was indistinguishable from one that happens to have 60,000 rows;
    four RL cohorts were being truncated with no record of it, which shifts the real-vs-generated
    composition of the vocabulary without saying so.
    """
    out, err, n_total = [], None, 0
    try:
        with open(path) as fh:
            r = csv.DictReader(fh)
            cols = [c for c in (r.fieldnames or []) if c and
                    c.lower() in ('smiles', 'target_mol', 'smi', 'canonical_smiles', 'mol')]
            col = cols[0] if cols else (r.fieldnames or [None])[0]
            for i, row in enumerate(r):
                v = (row.get(col) or '').strip()
                if v:
                    n_total += 1
                    if not limit or len(out) < limit:
                        out.append(v)
    except Exception as e:
        err = '%s: %s' % (type(e).__name__, str(e)[:80])
    return out, n_total, err


def cuts(mol, mode):
    """Yield (retained_smiles_with_star, fragment_smiles_with_star) for every single BRICS cut."""
    bonds = list(BRICS.FindBRICSBonds(mol))
    for (a1, a2), _lab in bonds:
        b = mol.GetBondBetweenAtoms(a1, a2)
        if b is None:
            continue
        try:
            pieces = Chem.GetMolFrags(
                Chem.FragmentOnBonds(mol, [b.GetIdx()], addDummies=True),
                asMols=True, sanitizeFrags=True)
        except Exception:
            continue
        if len(pieces) != 2:
            continue
        wh = [electrophile_index(p)[0] is not None for p in pieces]
        if sum(wh) != 1:
            continue                      # electrophile must sit cleanly on exactly one side
        ei = wh.index(True)
        if mode == 'warhead_directed':
            frag, keep = pieces[ei], pieces[1 - ei]
        else:                              # warhead_retained: generate the NON-reactive half
            frag, keep = pieces[1 - ei], pieces[ei]
        nf = frag.GetNumHeavyAtoms() - 1
        nk = keep.GetNumHeavyAtoms() - 1
        if not (MIN_FRAG_HEAVY <= nf <= MAX_FRAG_HEAVY) or nk < MIN_KEEP_HEAVY:
            continue
        try:
            yield Chem.MolToSmiles(keep), Chem.MolToSmiles(frag)
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['warhead_directed', 'warhead_retained'],
                    default='warhead_directed')
    ap.add_argument('--out', default='experiments/covalentformer/data/vocab')
    ap.add_argument('--per-source', type=int, default=60000)
    a = ap.parse_args()

    srcs = ['data/covbinder/raw_covindb2/CovInDB_All.csv', 'models/train.csv']
    srcs += sorted(glob.glob('data/_backups/*/tier1_scored_cohorts/*_scored.csv'))
    smis, per, capped, errs = [], {}, [], []
    for s in srcs:
        got, n_total, err = smiles_from(s, a.per_source)
        if err:
            errs.append((os.path.basename(s), err))
        if got:
            per[os.path.basename(s)] = len(got)
            if a.per_source and n_total > a.per_source:
                capped.append((os.path.basename(s), n_total, a.per_source))
            smis += got
    smis = list(dict.fromkeys(smis))
    print('sources: %s' % per)
    if capped:
        print('  CAPPED (--per-source %d) -- composition is shifted by this, not an accident:'
              % a.per_source)
        for n, tot, cap in capped:
            print('     %-42s %7d -> %6d  (dropped %d)' % (n, tot, cap, tot - cap))
    if errs:
        print('  READ ERRORS (source silently shrank):')
        for n, e in errs:
            print('     %-42s %s' % (n, e))
    print('unique input molecules: %d' % len(smis))

    frags, keeps, stats = collections.Counter(), set(), collections.Counter()
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            stats['unparseable'] += 1
            continue
        if electrophile_index(m)[0] is None:
            stats['no_single_warhead'] += 1
            continue
        n = 0
        for keep, frag in cuts(m, a.mode):
            frags[frag] += 1
            keeps.add(keep)
            n += 1
        stats['molecules_with_cuts' if n else 'no_valid_cut'] += 1
        if (i + 1) % 20000 == 0:
            print('  %d/%d scanned, %d distinct fragments' % (i + 1, len(smis), len(frags)),
                  flush=True)

    print('\nmode=%s' % a.mode)
    print('  distinct FRAGMENTS : %d   (tierA_v3 had 3,477)' % len(frags))
    print('  distinct RETAINED  : %d' % len(keeps))
    print('  %s' % dict(stats))
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, 'fragments_%s.smi' % a.mode)
    with open(fp, 'w') as fh:
        for f, c in frags.most_common():
            fh.write('%s\t%d\n' % (f, c))
    kp = os.path.join(a.out, 'retained_%s.smi' % a.mode)
    with open(kp, 'w') as fh:
        for k in sorted(keeps):
            fh.write('%s\n' % k)
    print('  wrote %s (%d) and %s (%d)' % (fp, len(frags), kp, len(keeps)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
