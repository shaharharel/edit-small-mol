"""CROSS-ROLE v2: the steering test, with the artifact that killed v1 removed by construction.

WHY v1 WAS WITHDRAWN. build_roles forces exactly one side of the BRICS cut to carry the
electrophile, so on an EDIT_WARHEAD-anchor row the retained half has NONE. v1 joined that retained
half to a fragment of a different role, which deleted the warhead entirely. Measured:

    target contains an electrophile     train_strat 99.95% | valid_clean 99.97% | cross_role 15.38%
    inside cross_role, anchor=WARHEAD (n=3384):                                            0.56%

84% of v1's rows had a WARHEAD anchor, so v1 asked the model to produce a molecule class that is
0.05% of its training data. The gap collapsed, and "the token points rather than steers" was the
natural reading -- but a floor-level gap on impossible requests measures the dataset, not the model.

THE STRUCTURAL FACT THAT DICTATES THIS DESIGN. For a WARHEAD-anchor row, retaining the electrophile
while requesting a non-warhead role is IMPOSSIBLE: the electrophile lives only in the fragment being
replaced, and the replacement is by definition not a warhead. So warhead-anchor rows cannot be made
electrophile-retaining, and they are excluded rather than patched. v2 draws only from NON-WARHEAD
anchors, where the retained half carries the electrophile and any cross-role swap preserves it.

That is a real narrowing and it must be stated with any result: v2 tests "can the model be told to
change a non-warhead fragment's class while the warhead stays put". It does NOT test "can the model
be told to install a warhead where there is none" -- nothing in this corpus can, because
build_roles never produced such a row.

QUOTA, NOT LUCK. v1 sampled the requested role uniformly and let a 600 Da cap decide what survived,
which deleted 83% of WARHEAD requests and kept exactly those with an unusually small retained half.
v2 sets a per-requested-role quota and reports achieved vs requested, so selection is visible.

FRAGMENT SUPPLY IS THE REAL CEILING, NOT ROW COUNT. v1's 1,458 LINKER rows were built from 26
distinct fragments -- 56 rows each. For a question that is a property of the FRAGMENT, effective n
is the fragment count. v2 reports DISTINCT fragments per role, and any statistic off this file must
cluster its errors on frag_to.
"""
import os, sys, csv, json, random, argparse, collections, hashlib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_roles import classify, join
from reachability import electrophile_index

ROLES = ['EDIT_WARHEAD', 'EDIT_LINKER', 'EDIT_SCAFFOLD', 'EDIT_DECORATION']


def has_elec(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return m is not None and electrophile_index(m)[0] is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='experiments/covalentformer/data/roles/valid_clean.csv')
    ap.add_argument('--out', default='experiments/covalentformer/data/roles/cross_role_v2.csv')
    ap.add_argument('--per-role', type=int, default=400)
    ap.add_argument('--max-mw', type=float, default=600.0)
    ap.add_argument('--seed', type=int, default=20260914)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = list(csv.DictReader(open(a.src)))
    print('source: %s (%d rows)' % (os.path.basename(a.src), len(rows)))
    print('  source role mix: %s' % dict(collections.Counter(r['role'] for r in rows)))

    pool = collections.defaultdict(set)
    for r in rows:
        for k in ('frag_from', 'frag_to'):
            f = r.get(k)
            if f:
                m = Chem.MolFromSmiles(f)
                if m is not None:
                    pool[classify(m, set(), set())].add(f)
    pool = {k: sorted(v) for k, v in pool.items()}
    print('  DISTINCT fragments per role: %s' % {k: len(v) for k, v in pool.items()})
    print('  (row counts cannot exceed this in information terms -- cluster errors on frag_to)')

    # ONLY non-warhead anchors: their retained half carries the electrophile, so a cross-role swap
    # preserves it. Warhead anchors are excluded by construction, not filtered post hoc.
    elig = [r for r in rows if r.get('role') in ROLES and r['role'] != 'EDIT_WARHEAD'
            and r.get('keep') and has_elec(r['keep'])]
    dropped_wh = sum(1 for r in rows if r.get('role') == 'EDIT_WARHEAD')
    no_elec_keep = sum(1 for r in rows if r.get('role') not in (None, 'EDIT_WARHEAD')
                       and r.get('keep') and not has_elec(r['keep']))
    print('  eligible anchors (non-warhead, keep carries the electrophile): %d' % len(elig))
    print('    excluded: %d warhead-anchor rows (electrophile retention is IMPOSSIBLE there)'
          % dropped_wh)
    print('    excluded: %d non-warhead rows whose keep lacks an electrophile' % no_elec_keep)
    if not elig:
        print('NOTHING ELIGIBLE')
        return 1

    have = collections.Counter()
    out, stats = [], collections.Counter()
    order = elig[:]
    rng.shuffle(order)
    # round-robin over requested roles so the quota drives the mix, not the MW cap
    for r in order:
        x = r['role']
        cands = [y for y in ROLES if y != x and pool.get(y) and have[y] < a.per_role]
        if not cands:
            continue
        y = min(cands, key=lambda z: (have[z], rng.random()))
        frags = pool[y][:]
        rng.shuffle(frags)
        placed = False
        for fy in frags[:60]:                      # bounded retry, reported below
            tgt = join(r['keep'], fy)
            if not tgt or tgt == r['anchor']:
                continue
            m = Chem.MolFromSmiles(tgt)
            if m is None:
                stats['parse_fail'] += 1
                continue
            if Descriptors.MolWt(m) > a.max_mw:
                stats['mw_reject'] += 1
                continue
            if not has_elec(tgt):
                # THIS COUNTER FIRED AT 2,235 AND I MISREAD IT. The comment here used to say
                # "must be ~0 by construction; counted to prove it", and when it came back 2,235 I
                # explained it as chemistry -- a BRICS cut next to the Michael acceptor. That was
                # wrong. The real cause: a requested-WARHEAD target carries TWO electrophiles (one
                # in `keep`, one in frag_to), and reachability.electrophile_index returns
                # (None,'multiple_warheads') when a pattern matches more than once, so has_elec is
                # False. Essentially all 2,235 are requested-WARHEAD attempts: 216 accepted of
                # ~2,451 = 8.8%.
                # CORRECTION, second pass: I first wrote that the survivors are HETERO-class pairs.
                # They are not. All 216 classify as plain 'michael' (checked directly). The real
                # cause is that electrophile_index is FIRST-PATTERN-WINS and 'michael_sub'
                # ([CX3]=[CX3][CX3](=O)[NX3]) is a strict SUPERSET of 'michael'
                # ([CH2]=[CH]C(=O)[NX3]). A bis-acrylamide where exactly one acrylamide is TERMINAL
                # matches 'michael' once, so the loop returns before ever evaluating 'michael_sub',
                # which would have matched twice. Demonstrated: C=CC(=O)NCCCNC(=O)/C=C/C has
                # michael:1, michael_sub:2, and electrophile_index returns (0,'michael') -- a
                # genuine BIS-Michael molecule reported as a SINGLE warhead.
                # So the 8.8% acceptance is a SMARTS-ORDERING artifact, not a chemical selection.
                # The cell is still unusable; the reason is different from what I first wrote.
                stats['elec_lost'] += 1
                continue
            out.append(dict(anchor=r['anchor'], target=tgt, keep=r['keep'], role=y,
                            anchor_role=x, frag_from=r.get('frag_from', ''), frag_to=fy))
            have[y] += 1
            placed = True
            break
        if not placed:
            stats['no_fragment_fit'] += 1

    print('\n  built %d rows | %s' % (len(out), dict(stats)))
    if not out:
        print('NOTHING BUILT')
        return 1
    print('  %-16s %9s %9s   %s' % ('requested', 'quota', 'achieved', 'distinct frag_to'))
    for y in ROLES:
        got = [r for r in out if r['role'] == y]
        if not got and not pool.get(y):
            continue
        print('  %-16s %9d %9d   %d' % (y, a.per_role, len(got),
                                        len({r['frag_to'] for r in got})))

    # THE THREE CHECKS THAT USED TO SIT HERE WERE ALL TAUTOLOGIES, under a comment claiming they
    # were verification. None could fail, so none was evidence of anything:
    #   "requested != anchor role"  -- y is drawn from [y for y in ROLES if y != x]. Cannot differ.
    #   "target retains electrophile" -- rows failing has_elec hit `continue` and never reach `out`.
    #     This is the WORST of the three: the 2,235 elec_lost rows ARE the signal, and they are
    #     excluded BEFORE the check runs, so it only ever reports its own survivors. That is the
    #     same counter I already misread once.
    #   "classify(frag_to) == requested" -- fy is drawn from pool[y], and pool was KEYED by classify.
    #     Recomputing classify on the same molecule is the identity function.
    # Replaced with the check that CAN fail, and which would have caught the frag_from bug that
    # shipped in v1 and again in v2 before it was found by hand.
    def _canon(s):
        m = Chem.MolFromSmiles(s) if s else None
        return Chem.MolToSmiles(m) if m is not None else None

    # THE ASYMMETRY BETWEEN THESE TWO IS THE WHOLE POINT, and I got it wrong once already.
    # anc_ok crosses a FILE BOUNDARY: `anchor` is read from the source CSV while keep/frag_from are
    # recombined here, so the two sides have independent provenance and disagreeing is possible.
    # That is why it caught the frag_from bug.
    # The old `tgt_ok` did NOT cross any boundary. Rows are appended above as
    #     target=tgt, frag_to=fy   where   tgt = join(r['keep'], fy)
    # so it recomputed join(keep, frag_to) and compared it against the stored value of that exact
    # expression -- _canon(X) == _canon(X), true on every input that can reach the line, and true
    # even when BOTH sides are None because the SMILES does not parse. It was the same
    # identity-function shape as the "classify(frag_to) == requested" tautology it replaced, printed
    # under a banner promising it could fail. Removed, and replaced by the two properties of the
    # constructed target that genuinely can fail.
    # THIRD TIME I HAVE SHIPPED A TAUTOLOGY UNDER THIS BANNER, so the reasoning is written out.
    # A check belongs here ONLY if some row that REACHES `out` could fail it. Rows are appended at
    # the bottom of a loop that has already `continue`d on: tgt falsy or tgt == r['anchor'] (:107),
    # MolFromSmiles(tgt) is None (:109-111), MolWt > max_mw (:113), not has_elec (:115).
    # So anything that merely restates one of those guards CANNOT fail.
    #   REMOVED, it was a tautology: "target is a PARSEABLE molecule". Line 109 already ran
    #   MolFromSmiles and skipped the row on None, so _canon(target) is not None is guaranteed.
    #   It replaced an earlier tautology and was itself printed as falsifiable. Gone.
    # KEPT, and it is NOT a tautology despite looking like one -- the distinction is exact:
    #   the :107 guard rejects on a RAW STRING comparison (tgt == r['anchor']); this check compares
    #   CANONICAL forms. A join that emits a different SMILES string which canonicalises to the
    #   anchor passes :107 and fails here. Demonstrated: anchor 'C1=CC=CC=C1CC(=O)N' vs tgt
    #   'c1ccccc1CC(N)=O' are unequal as strings and identical as molecules
    #   (both canonicalise to NC(=O)Cc1ccccc1). That is a real no-op edit that only the canonical
    #   form catches, and it is exactly the class of bug the raw-string guard lets through.
    #   REMOVED, a tautology: "CANONICAL target != anchor". I kept this last round and DEFENDED it,
    #   arguing the :107 guard compares raw strings while this compares canonical form, so a
    #   raw-different / canonically-identical target could fail it. That escape hatch is real in
    #   principle and UNREACHABLE in this build, established by execution rather than argument:
    #   over ~29,600 candidate joins the raw guard fired ZERO times and "canon-equal but raw-differs"
    #   fired ZERO times; join(keep, frag_from) is BYTE-identical to the stored anchor on 4000/4000
    #   rows, so there is no format gap for canonicalisation to catch; and the only route to
    #   target == anchor is canon(frag_to) == canon(frag_from), impossible because `pool` is KEYED BY
    #   classify(), so a fragment drawn from pool[y] is never the same molecule as a frag_from that
    #   classified as x != y. Unreachable is the same as cannot-fail for this banner's purposes.
    # The real verification now happens AFTER the write, in the READ-BACK block at the end of main().
    anc_ok = sum(1 for r in out
                 if r['frag_from'] and _canon(join(r['keep'], r['frag_from'])) == _canon(r['anchor']))
    print()
    print('  FALSIFIABLE CHECK (this can fail, and did before the frag_from fix):')
    print('    anchor == join(keep, frag_from)     : %d / %d   %s'
          % (anc_ok, len(out), 'OK' if anc_ok == len(out) else '*** MISMATCH ***'))
    print('      (crosses the build_roles -> build_crossrole_v2 file boundary: `anchor` is read from')
    print('       the source CSV while keep/frag_from are recombined here, so the two sides have')
    print('       independent provenance. That is why it caught the frag_from bug.)')
    if anc_ok != len(out):
        print('    the (anchor, keep, frag_from, frag_to) tuple is INCOHERENT -- do not use this file')
    # reported as data, not as a check: these are properties of the construction, not tests of it
    print('  by construction (NOT verification): requested != anchor on all rows; every target'
          ' carries an electrophile because non-retaining candidates were rejected at build time'
          ' (%d of them); classify(frag_to)==requested because the pool is keyed by classify.'
          % stats['elec_lost'])
    print()
    print('  *** DO NOT AVERAGE OVER ALL FOUR CELLS. The requested-WARHEAD cell is not comparable:')
    print('      - its targets are BIS-electrophiles that SURVIVED a first-pattern-wins SMARTS')
    print('        check: michael_sub is a superset of michael, so a bis-acrylamide with exactly')
    print('        one TERMINAL acrylamide matches michael once and never reaches michael_sub.')
    print('        Acceptance ~8.8%%, selected by SMARTS ordering, not by chemistry.')
    print('      - fragment novelty is wildly unbalanced across cells (frag_to seen in train:')
    print('        WARHEAD 100%%, DECORATION 100%%, LINKER 52%%, SCAFFOLD 29%%), so a per-role ordering')
    print('        here is partly a have-I-seen-this-fragment ordering.')
    print('      Report the three non-warhead cells, and that one separately or not at all. ***')
    print('  NOTE: v2 cannot test "install a warhead where there is none" -- see docstring.')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    # ORDER-SENSITIVE DIGEST. I hunted for a corruption class the four read-back checks miss and
    # found one: ROW ORDER. Every check is row-wise or aggregate, so shuffling all 865 rows passes
    # all four cleanly (measured: shuffle, and a two-row transposition, both PASS everything).
    # That matters because downstream evaluation subsamples by INDEX under a fixed seed --
    # role_compliance does rng.choice(len(rows), n) -- so a silently reordered file draws a
    # DIFFERENT subsample with the SAME seed and changes the numbers with nothing firing. This
    # digest is over the rows IN ORDER, so any permutation breaks it.
    _digest = hashlib.sha256(
        '\n'.join('\t'.join(r[k] for k in out[0].keys()) for r in out).encode()).hexdigest()
    json.dump({'src': a.src, 'seed': a.seed, 'per_role': a.per_role, 'n': len(out),
               'achieved': {y: sum(1 for r in out if r['role'] == y) for y in ROLES},
               'distinct_frag_to': {y: len({r['frag_to'] for r in out if r['role'] == y})
                                    for y in ROLES},
               'row_order_sha256': _digest,
               'stats': dict(stats)},
              open(a.out.replace('.csv', '_provenance.json'), 'w'), indent=1)
    print('  wrote %s' % a.out)

    # READ-BACK VERIFICATION -- the only check in this file besides anc_ok that CAN fail.
    # Everything computed before the write is in-memory and therefore restates a guard the loop
    # already applied; I shipped THREE such tautologies tonight and defended one of them after being
    # told. This one crosses the DISK boundary: it re-reads what was actually written and recomputes
    # from those bytes. It can fail on a quoting bug, a column-order bug, a truncated write, a
    # dropped row, or an encoding problem -- none of which any in-memory assertion can see.
    # NOTE ON WHAT IS CHECKED HERE. My first version of this block called classify(frag_to) and
    # CRASHED: classify takes (frag, parent_scaffold_atoms, frag_atom_ids), not a bare mol, and I
    # wrote the call without reading its signature. It cannot be recomputed from CSV columns alone
    # because the atom-id arguments do not survive the write. Replaced with checks that CAN be
    # recomputed from the written bytes.
    # THE KEY POINT, and it is why join-identity belongs HERE and not before the write: in memory
    # `target == join(keep, frag_to)` is a tautology, because that is how target was constructed.
    # Re-read from disk it is NOT, because the write can corrupt it -- a quoting bug, a shifted
    # column, a truncated field or an encoding problem all break the identity without touching the
    # in-memory objects. Same arithmetic, different provenance, and the provenance is what makes a
    # check falsifiable.
    rb = list(csv.DictReader(open(a.out)))
    n_rows = len(rb) == len(out)
    n_cls = sum(1 for r in rb
                if r['frag_to'] and _canon(join(r['keep'], r['frag_to'])) == _canon(r['target']))
    n_join = sum(1 for r in rb
                 if r['frag_from'] and _canon(join(r['keep'], r['frag_from'])) == _canon(r['anchor']))
    prov = json.load(open(a.out.replace('.csv', '_provenance.json')))
    ach_ok = all(sum(1 for r in rb if r['role'] == y) == prov['achieved'][y] for y in ROLES)
    rb_digest = hashlib.sha256(
        '\n'.join('\t'.join(r[k] for k in out[0].keys()) for r in rb).encode()).hexdigest()
    order_ok = rb_digest == prov['row_order_sha256']
    # FALSIFICATION TEST RUN, because passing on good data is exactly what a tautology also does.
    # /tmp/xr_falsify.py applies these four checks to CORRUPTED copies of the shipped CSV. Result:
    #   unmodified control ........... all four pass
    #   drop one row ................. rowcount FIRES, provenance FIRES
    #   swap keep/frag_to in a row ... anchor_join FIRES
    #   copy another row's target .... target_join FIRES
    #   mangle an anchor ............. anchor_join FIRES
    #   flip a role .................. provenance FIRES
    #   mangle a frag_from ........... anchor_join FIRES
    # Six corruptions, six caught, control clean. These are real checks.
    # ONE MEASURED BLIND SPOT, worth knowing rather than assuming: `join` is SYMMETRIC in its two
    # halves (join(keep, frag_to) and join(frag_to, keep) canonicalise identically -- verified), so
    # target_join CANNOT see a keep/frag_to transposition. anchor_join catches that case because
    # frag_from is untouched by the swap. The two checks cover each other; neither alone suffices.
    print()
    print('  READ-BACK CHECKS (re-read from disk; these cross the write boundary and CAN fail):')
    print('    row count survived the write        : %d / %d   %s'
          % (len(rb), len(out), 'OK' if n_rows else '*** ROWS LOST ***'))
    print('    target == join(keep, frag_to) FROM DISK: %d / %d   %s'
          % (n_cls, len(rb), 'OK' if n_cls == len(rb) else '*** MISLABELLED ON DISK ***'))
    print('    anchor == join(keep, frag_from)     : %d / %d   %s'
          % (n_join, len(rb), 'OK' if n_join == len(rb) else '*** MISMATCH ***'))
    print('    per-role counts match provenance    : %s'
          % ('OK' if ach_ok else '*** PROVENANCE DISAGREES WITH THE CSV ***'))
    print('    ROW ORDER digest matches provenance : %s'
          % ('OK' if order_ok else '*** ROWS REORDERED OR ALTERED ***'))
    if not (n_rows and n_cls == len(rb) and n_join == len(rb) and ach_ok and order_ok):
        print('    THE WRITTEN FILE DOES NOT MATCH WHAT WAS BUILT -- do not use it')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
