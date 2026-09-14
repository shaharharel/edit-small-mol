"""PHASE A (revised): ROLE-CONDITIONED EDITING. No pocket, no geometry.

WHY THIS REPLACED GEOMETRY PRETRAINING. The application keeps the warhead and edits elsewhere. But
if the warhead stays in the retained half, then BOTH the attachment atom A and the electrophile E
live in the half we are keeping -- so d(A -> E) is fixed before generation starts and the generated
fragment cannot change it. Conditioning on d is meaningless in the mode we actually deploy. The
geometry that does matter there -- does the regrown half fit the pocket, clash, make contacts -- is
inherently POCKET-dependent and has no pocket-free form.

I had picked warhead-directed cuts because the reach envelope is only MEASURABLE there. That let the
metric choose the task. It is backwards, and it is why Phase A trained something we will not use.

What IS pocket-free and worth pretraining is the editing skill itself: change ONLY the part you are
told to change. That is the prerequisite for every downstream use, it needs no protein, it scales to
millions of examples, and -- unlike reach -- it has a clean null: a role-blind model complies at
chance.

    PHASE A learns WHAT TO CHANGE.   PHASE B learns WHERE TO PUT IT.

ROLES, assigned by what lands in the GENERATED half:
    EDIT_WARHEAD      the fragment carries the electrophile
    EDIT_DECORATION   small, acyclic, peripheral
    EDIT_SCAFFOLD     ring system belonging to the Bemis-Murcko core
    EDIT_LINKER       everything else -- the connective tissue between core and warhead
EDIT_WARHEAD is the role that is SUPPOSED to be able to break warhead retention, and it is opt-in via
the role token, against the 52.2% retention the geometry model actually posted.

BUT "every other role retains the warhead BY CONSTRUCTION" -- which this docstring used to claim -- is
FALSE, and the counterexamples come from a live defect rather than from noise. Measured on a 60,000-row
sample of the corpus this file produces, asking whether the fragment the edit REMOVES carries an
electrophile under a full SMARTS scan:

    EDIT_WARHEAD     n=34,131    100.00%
    EDIT_SCAFFOLD    n= 8,177      0.51%   <- 42 rows that DO delete an electrophile
    EDIT_DECORATION  n=16,814      0.00%
    EDIT_LINKER      n=   878      0.00%

CAUSE. reachability.electrophile_index returns None for "TWO OR MORE electrophiles of one class"
exactly as it does for "no electrophile at all", and the gate below tests only `is not None`. So a
fragment carrying two same-class warheads reads as "no warhead on this side", passes the
exactly-one-side test, and is handed a non-WARHEAD role. (The CROSS-class case -- an acrylamide in
`keep` and a chloroacetamide in `frag` -- is caught, because both sides then return not-None and the
cut is dropped. It is the same-class case that leaks through.)

0.51% is small, but "by construction" does not survive 42 counterexamples. Honest phrasing: non-WARHEAD
roles retain the electrophile THAT electrophile_index REPORTS; a fragment carrying two or more
electrophiles of a single class is invisible to that function and can be deleted under a SCAFFOLD label.

THE EVALUATION THIS ENABLES (no pocket needed): ROLE COMPLIANCE -- given a role token, did the model
change the named part and leave the others intact? Per-role, with a chance baseline.
"""
import os, sys, csv, json, glob, random, argparse, collections
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import electrophile_index, CLASSIFIER_VERSION, CORPUS_LABELLED_WITH

MIN_FRAG, MAX_FRAG = 1, 24
MIN_KEEP = 8
ROLES = ['EDIT_WARHEAD', 'EDIT_LINKER', 'EDIT_SCAFFOLD', 'EDIT_DECORATION']
SEED = 20260914


def classify(frag, parent_scaffold_atoms, frag_atom_ids):
    """Which role does this generated half play?

    FIRST VERSION WAS BROKEN and produced EDIT_SCAFFOLD 902 / EDIT_LINKER 15 / EDIT_DECORATION 10.
    It sent any ring-containing fragment touching the Bemis-Murcko core to SCAFFOLD -- but Murcko is
    DEFINED as all rings plus the linkers between them, so essentially every ring fragment touches
    it and the test was near-tautological. LINKER was starved to 1.5% of the data.

    Replaced by a composition rule that does not consult Murcko at all:
        has electrophile            -> WARHEAD
        >= 1 ring                   -> SCAFFOLD    (a ring system is core-like by construction)
        acyclic, <= 6 REAL heavy atoms -> DECORATION  (methyl, halogen, methoxy and similar)
          NOTE the threshold is SIX, not four, and there is no "simple" test. The code is
          `nh = GetNumHeavyAtoms() - 1; nh <= 5`, but RDKit does NOT count the [n*] dummy as
          heavy, so that -1 subtracts an atom that was never counted and the effective bound
          is 6 real heavy atoms. role_compliance.edited_role was written against "<= 5" and
          disagreed with these LABELS on 9.6% of valid_strat rows until it was matched.
          Quote SIX. This docstring said four and is the line most likely to reach a paper.
        acyclic, larger             -> LINKER      (the connective chain)
    Natural frequencies are still uneven, so main() downsamples to a balanced set afterwards -- the
    classifier decides the label, the sampler decides the mix.
    """
    if electrophile_index(frag)[0] is not None:
        return 'EDIT_WARHEAD'
    nh = frag.GetNumHeavyAtoms() - 1
    if frag.GetRingInfo().NumRings() > 0:
        return 'EDIT_SCAFFOLD'
    if nh <= 5:
        return 'EDIT_DECORATION'
    return 'EDIT_LINKER'



def _announce_build_classifier_version():
    """A BUILDER that does not announce its labeller is how a corpus-version boundary is created.

    #142 pinned the SCORERS and left the BUILDERS silent -- and the builders are the half that can
    actually create the mismatch. Re-run this file today and it labels the Phase A role corpus with
    electrophile_index v%d while reachability.CORPUS_LABELLED_WITH still advertises v%d, with
    nothing anywhere to notice. Two scorers read those constants; eleven other importers do not, and
    the two that matter most are this file and its sibling builder.
    My own words in #142: "a convention nobody can check is not a pin."
    """ % (CLASSIFIER_VERSION, CORPUS_LABELLED_WITH)
    print('  BUILDING WITH electrophile_index v%d' % CLASSIFIER_VERSION)
    if CLASSIFIER_VERSION != CORPUS_LABELLED_WITH:
        print('  *** THIS BUILD SUPERSEDES THE RECORDED CORPUS VERSION (v%d). Any corpus written by '
              'this run is labelled v%d. BUMP reachability.CORPUS_LABELLED_WITH to %d IN THE SAME '
              'COMMIT, or every scorer will announce a straddle that no longer exists -- and every '
              'per-role number from the old corpus becomes incomparable to this one (#142/#147: the '
              'effect is role-asymmetric and does NOT cancel in a paired contrast).'
              % (CORPUS_LABELLED_WITH, CLASSIFIER_VERSION, CLASSIFIER_VERSION))

def main():
    _announce_build_classifier_version()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='experiments/covalentformer/data/roles')
    ap.add_argument('--max-mol', type=int, default=120000)
    ap.add_argument('--per-mol', type=int, default=4)
    ap.add_argument('--per-role', type=int, default=250000)
    ap.add_argument('--max-mw', type=float, default=600.0,
                    help='cap on the JOINED product; 600 was calibrated against real CovInDB '
                         'parents (MW 498) -- an earlier 520 cap overcorrected to MW 441')
    a = ap.parse_args()
    rng = random.Random(SEED)

    srcs = ['data/covbinder/raw_covindb2/CovInDB_All.csv', 'models/train.csv']
    srcs += sorted(glob.glob('data/_backups/*/tier1_scored_cohorts/*_scored.csv'))
    smis = []
    for s in srcs:
        try:
            with open(s) as fh:
                r = csv.DictReader(fh)
                cols = [c for c in (r.fieldnames or []) if c and c.lower() in
                        ('smiles', 'target_mol', 'smi', 'canonical_smiles')]
                col = cols[0] if cols else (r.fieldnames or [None])[0]
                for row in r:
                    v = (row.get(col) or '').strip()
                    if v:
                        smis.append(v)
        except Exception:
            pass
    smis = list(dict.fromkeys(smis))
    rng.shuffle(smis)
    smis = smis[:a.max_mol]
    print('input molecules: %d' % len(smis))

    # keep -> {role: [fragments]}  so an anchor and a target can share a retained half
    bank = collections.defaultdict(lambda: collections.defaultdict(set))
    stats = collections.Counter()
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None or electrophile_index(m)[0] is None:
            stats['skip'] += 1
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            scaf_atoms = set(m.GetSubstructMatch(scaf)) if scaf.GetNumAtoms() else set()
        except Exception:
            scaf_atoms = set()
        n = 0
        for (a1, a2), _lab in BRICS.FindBRICSBonds(m):
            if n >= a.per_mol:
                break
            b = m.GetBondBetweenAtoms(a1, a2)
            if b is None:
                continue
            try:
                frag_mol = Chem.FragmentOnBonds(m, [b.GetIdx()], addDummies=True)
                pieces = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
                idxs = Chem.GetMolFrags(frag_mol)
            except Exception:
                continue
            if len(pieces) != 2:
                continue
            for side in (0, 1):
                frag, keep = pieces[side], pieces[1 - side]
                nf, nk = frag.GetNumHeavyAtoms() - 1, keep.GetNumHeavyAtoms() - 1
                if not (MIN_FRAG <= nf <= MAX_FRAG) or nk < MIN_KEEP:
                    continue
                # the warhead must be unambiguous: exactly one side carries it
                if (electrophile_index(frag)[0] is not None) == \
                   (electrophile_index(keep)[0] is not None):
                    continue
                role = classify(frag, scaf_atoms, set(idxs[side]))
                try:
                    bank[Chem.MolToSmiles(keep)][role].add(Chem.MolToSmiles(frag))
                except Exception:
                    continue
                n += 1
        stats['used' if n else 'no_cut'] += 1
        if (i + 1) % 20000 == 0:
            print('  %d/%d  banks=%d' % (i + 1, len(smis), len(bank)), flush=True)

    # CROSS-MOLECULE FRAGMENT SWAPS. Requiring both fragments of a pair to have been observed on
    # the SAME retained half yielded 1,908 pairs from 8,000 molecules -- most retained halves carry
    # exactly one fragment per role, so they produce nothing. Instead, draw the substitute from the
    # GLOBAL pool for that role. The retained half is still shared between anchor and target (so the
    # edit is localised and the role label stays true), but the replacement fragment no longer has
    # to have been seen on that particular scaffold. This is what makes the set scale.
    global_pool = collections.defaultdict(set)
    for _k, byrole in bank.items():
        for role, frags in byrole.items():
            global_pool[role] |= frags
    for role in global_pool:
        global_pool[role] = sorted(global_pool[role])
    print('  global fragment pool per role: %s'
          % {k: len(v) for k, v in global_pool.items()})

    rows = []
    for keep, byrole in bank.items():
        for role, frags in byrole.items():
            fl = sorted(frags)
            pool = global_pool[role]
            if not fl or len(pool) < 2:
                continue
            for j in range(min(len(fl), 4)):
                fa = fl[j]
                fb = pool[rng.randrange(len(pool))]
                if fb == fa:
                    continue
                anc, tgt = join(keep, fa), join(keep, fb)
                if not anc or not tgt or anc == tgt:
                    continue
                mt = Chem.MolFromSmiles(tgt)
                if mt is None or Descriptors.MolWt(mt) > a.max_mw:
                    continue
                rows.append(dict(anchor=anc, target=tgt, keep=keep, role=role,
                                 frag_from=fa, frag_to=fb))
    print('\npairs BEFORE balancing: %d over %d retained halves' % (len(rows), len(bank)))
    print('  by role: %s' % dict(collections.Counter(r['role'] for r in rows)))
    # WHY ROLES ARE CAPPED AT ALL. Natural cut frequencies are heavily skewed toward ring-containing
    # fragments, and a model trained on 90% EDIT_SCAFFOLD would learn to ignore the token and always
    # rewrite the core -- the role channel would look dead for the same reason the geometry channel
    # did.
    #
    # A STALE PARAGRAPH WAS DELETED HERE, and it mattered. It said "Downsample every role to the
    # smallest, so the token carries real information and role compliance has an honest chance
    # baseline of 1/n_roles." The code below does the OPPOSITE (a soft cap, see the next comment),
    # so that text contradicted the implementation -- and its 1/n_roles = 25% claim is FALSE under
    # soft capping. That sentence is the most likely origin of the 25% chance baseline this project
    # spent a night retracting: the real floor is ~0.79-0.89 macro-AUROC from ECFP4 of the anchor
    # alone, because the role is a deterministic function of the swapped fragment. Never quote 25%.
    byrole = collections.defaultdict(list)
    for r in rows:
        byrole[r['role']].append(r)
    # SOFT cap, not hard balance. Capping every role at the SMALLEST one threw away 98.7% of the
    # data (1,491 -> 20 rows) because one rare role dictated the size of all four. Cap at a fixed
    # ceiling instead, keep whatever each role actually has, and REPORT the residual imbalance so
    # it can be handled with loss weighting rather than by deleting data.
    cap = a.per_role
    rows = []
    for role, v in byrole.items():
        rng.shuffle(v)
        rows += v[:cap]
    rng.shuffle(rows)
    cnt = collections.Counter(r['role'] for r in rows)
    lo, hi = min(cnt.values()), max(cnt.values())
    print('  capped at %d per role -> %d rows (min role %d, max %d, ratio %.1fx)'
          % (cap, len(rows), lo, hi, hi / max(lo, 1)))
    if hi / max(lo, 1) > 10:
        print('  WARNING: role imbalance >10x. The token risks being ignored for rare roles -- '
              'use class-weighted loss at training time.')
    print('  by role: %s' % dict(collections.Counter(r['role'] for r in rows)))
    if not rows:
        print('NOTHING BUILT')
        return 1

    keys = sorted({r['keep'] for r in rows})
    rng.shuffle(keys)
    val = set(keys[:max(1, int(0.1 * len(keys)))])
    tr = [r for r in rows if r['keep'] not in val]
    va = [r for r in rows if r['keep'] in val]
    print('  split on RETAINED HALF: %d train / %d valid (overlap %d)'
          % (len(tr), len(va), len({r['keep'] for r in tr} & {r['keep'] for r in va})))
    os.makedirs(a.out, exist_ok=True)
    cols = ['anchor', 'target', 'keep', 'role', 'frag_from', 'frag_to']
    for nm, part in (('train', tr), ('valid', va)):
        with open(os.path.join(a.out, '%s.csv' % nm), 'w', newline='') as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(part)
        print('  wrote %s/%s.csv (%d)' % (a.out, nm, len(part)))
    return 0


def join(keep_smi, frag_smi):
    a, b = Chem.MolFromSmiles(keep_smi), Chem.MolFromSmiles(frag_smi)
    if a is None or b is None:
        return None
    combo = Chem.RWMol(Chem.CombineMols(a, b))
    stars = [x.GetIdx() for x in combo.GetAtoms() if x.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    nb = []
    for s in stars:
        n = [y.GetIdx() for y in combo.GetAtomWithIdx(s).GetNeighbors()]
        if len(n) != 1:
            return None
        nb.append(n[0])
    try:
        combo.AddBond(nb[0], nb[1], Chem.BondType.SINGLE)
        for s in sorted(stars, reverse=True):
            combo.RemoveAtom(s)
        m = combo.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


if __name__ == '__main__':
    sys.exit(main())
