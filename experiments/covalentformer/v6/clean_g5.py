#!/usr/bin/env python3
"""FLAG the pair-level defects that ten independent labellers found, and repair what can be repaired.

Nothing is dropped. An earlier version removed 634 pairs on three rules; keeping them with an
explicit `unlabellable_reason` is strictly better -- the flag is information the model and the
analysis can both use, and a corpus that contains only edits with pocket consequences would teach
that every substituent must buy a contact. The chair's own verdict (ADOPT / REWRITE / REJECT) rides
alongside as a column for the same reason.

Every rule here exists because a labeller hit it and could not write a defensible sentence. The
counters are named and must reconcile to the input count -- the same discipline gen_ev5 uses --
because an unnamed drop is how v6 lost 10,381 pairs to a rule nobody could later identify.

g5 variant: identical rules, paths repointed at the g5 corpus (clean_v7.py stays
reproducible for g1-g4). Run AFTER gen_ev5_g5.py. Reads g5_evidence.NEW.jsonl, writes
g5_evidence.jsonl.
"""
import collections, json, os, re, sys

from rdkit import Chem

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SRC = os.path.join(D, 'g5_evidence.NEW.jsonl')
DST = os.path.join(D, 'g5_evidence.jsonl')
RES = re.compile(r'^([A-Z]{1,3})(-?\d+)\.')

# A partial NEW file must never reach the live corpus. A stray waiter ran this against a 51-pair
# fragment of an interrupted rebuild and wrote `kept 0 -> v7_evidence.jsonl`, emptying the corpus --
# the same class of accident as the earlier truncation, and for the same reason: a script that
# rewrites the live artifact without checking its input is whole.
MIN_IN = 1200  # g5 is a 1,249-pair build; the 6000 floor belonged to the 8,359-pair one
_n_in = sum(1 for _ in open(SRC))
if _n_in < MIN_IN:
    sys.exit('REFUSING TO RUN: %s holds only %d pairs (expected >=%d). The rebuild that produced '
             'it is incomplete or still running; the live corpus has NOT been touched.'
             % (os.path.basename(SRC), _n_in, MIN_IN))

flag = collections.Counter()
fix = collections.Counter()
kept = []
n = 0

for line in open(SRC):
    r = json.loads(line)
    n += 1
    mi, lo = r['MODEL_INPUT'], r['LABEL_ONLY']
    cons = lo.get('conserved_interactions') or []
    acs = lo.get('analogue_contacts') or []

    # ---- DROP 1. The two crystals share no frame of reference. `numbering_consistent` is already
    # False and nothing is conserved, so every residue the analogue names is unrelatable to the
    # input's numbering -- 7YD2's two residue sets are entirely disjoint. No gain, loss or retained
    # claim is possible, which is precisely what two labellers independently concluded.
    reasons = []
    if lo.get('numbering_consistent') is False and not cons:
        reasons.append('the two crystals share no residue numbering and nothing is conserved, so no '
                       'gain, loss or retained contact can be attributed to the edit')
        flag['no shared numbering AND nothing conserved'] += 1

    # ---- DROP 1b. Zero conserved interactions at ALL, whatever the numbering says. If not one
    # atom-pair survives across the two structures then no effect is attributable to the edit --
    # 7Z6A's input sits unengaged in solvent while the analogue's base and sugar contact the
    # protein, which is a different binding mode, not an edit effect.
    if not cons and acs:
        reasons.append('no interaction survives between the two crystals, so nothing is '
                       'attributable to the edit rather than to a different binding mode')
        flag['no interaction survives between the two crystals'] += 1

    # ---- DROP 2. A residue NUMBER that denotes different residue TYPES on the two sides: 7MMJ has
    # ALA1168 in the input and ASP1168 in the analogue. Either a numbering offset or two different
    # constructs; either way neither side's row for that residue can be trusted, and a labeller
    # cannot tell which.
    inp_t, ana_t = {}, {}
    for a in mi['atoms']:
        m = RES.match(a.get('faces') or '')
        if m: inp_t.setdefault(m.group(2), set()).add(m.group(1))
    for c in acs:
        m = RES.match(c.get('faces') or '')
        if m: ana_t.setdefault(m.group(2), set()).add(m.group(1))
    if any(not (inp_t[k] & ana_t[k]) for k in set(inp_t) & set(ana_t)):
        reasons.append('a residue number denotes DIFFERENT residues in the two crystals, so '
                       'neither side\'s row for it can be trusted')
        flag['a residue number means different residues'] += 1

    # ---- DROP 3. The analogue's atom names are a different namespace and the mapping silently
    # failed: nothing conserved, and every analogue row is NEW with no input atom. 6TBG reports a
    # one-heavy-atom edit at tanimoto 0.64 like this, which also invalidates its MOVED site rows
    # since MOVED asserts a correspondence that does not exist.
    if acs and not cons and all(c.get('status') == 'NEW' for c in acs) \
            and lo.get('heavy_atoms_added', 0) in (-1, 0, 1):
        reasons.append('the atom mapping failed: a near-identical analogue shares no mapped atom, '
                       'so every role and every conserved claim in this pair is unreliable')
        flag['atom mapping failed'] += 1

    # ---- REPAIR 1. `status: conserved` with no input atom asserts conservation against an atom
    # that has no input coordinates. The status is the lie, not the row: the contact is real, it
    # simply cannot be compared. Say so instead of dropping the pair.
    for c in acs:
        if c.get('status') == 'conserved' and str(c.get('input_atom') or '-').strip() in ('-', ''):
            c['status'] = 'present, NOT COMPARABLE (the matching input atom is not modelled)'
            c['input_atom'] = None
            fix['conserved status asserted against an unmodelled input atom'] += 1

    # ---- REPAIR 2. A ligand heavy atom cannot be 0.28 A from an ordered water. Those waters are
    # modelled at the ligand's own position (alternate conformer or an apo water list), so the
    # count and the distance are both meaningless -- and "displaces N waters" is a rationale a
    # labeller would otherwise reach for.
    for a in mi['atoms']:
        w = a.get('nearest_water_A')
        if w is not None and w < 2.4:        # shortest credible ligand-water H-bond is ~2.4 A
            a['nearest_water_A'] = None
            a['water_note'] = ('water data unusable for this atom: the nearest modelled water is '
                               '%.2f A away, closer than any real ligand-water hydrogen bond -- an alternate '
                               'conformer or an apo water list. Do not reason about displaced '
                               'waters here.' % w)
            fix['ligand atom closer to a water than physically possible'] += 1

    # ---- FLAG. Two independent transformations sold as one edit: 7UEP moves a group at one end of
    # the molecule AND deletes a methyl at the other, at tanimoto 0.51. A one-sentence label has to
    # pick one and will read as incomplete. Flagged rather than dropped -- it is a real pair, and
    # the chair can decide whether to describe the dominant change or reject.
    # The two-transformation heuristic moved into evidence_v4, where it measures TOPOLOGICAL
    # distance between the edited atoms. The version that lived here judged it by whether the site
    # atoms' pocket linings were disjoint, which was wrong in both directions: a single
    # phosphate->acetate swap "spanned 3 regions" because its three oxygens line different
    # residues, while genuine two-site edits at tanimoto 0.50 went unflagged. Contiguity is a
    # property of the molecule, not of the pocket.

    if reasons:
        lo['unlabellable_reason'] = reasons
        lo['unlabellable'] = True
    else:
        lo['unlabellable'] = False

    kept.append(r)

tmp = DST + '.tmp'
with open(tmp, 'w') as out:
    for r in kept:
        out.write(json.dumps(r) + '\n')
assert sum(1 for _ in open(tmp)) == len(kept), 'row count changed on write'
os.replace(tmp, DST)

print('read %s pairs from %s' % (format(n, ','), os.path.basename(SRC)))
print('\nFLAGGED UNLABELLABLE (kept, with a reason, NOT dropped)')
for k, v in flag.most_common():
    print('  %-66s %5d' % (k, v))
nflag = sum(1 for r in kept if r['LABEL_ONLY'].get('unlabellable'))
print('  %-66s %5d  (%.1f%%)' % ('distinct pairs flagged', nflag, 100.0 * nflag / max(len(kept), 1)))
print('\nREPAIRED IN PLACE (pair kept)')
for k, v in fix.most_common():
    print('  %-66s %5d' % (k, v))
print('\nkept %s -> %s' % (format(len(kept), ','), os.path.basename(DST)))
assert len(kept) == n, 'every pair must be kept: %d in, %d out' % (n, len(kept))
print('every pair kept: %s in, %s out' % (format(n, ','), format(len(kept), ',')))
