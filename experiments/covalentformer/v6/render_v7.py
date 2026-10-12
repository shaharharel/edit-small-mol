#!/usr/bin/env python3
"""Render a REPRESENTATIVE audit sample of v7 evidence, exactly as a labeller reads it.

Representative, not stratified. The v6 audit sampled one pair per distinct target and forced in a
covalent, a metal and an uncertain-flag case. That over-weighted odd chemistry so badly that both
reviewers reported the corpus was ~30% non-medicinal-chemistry when corpus-wide it is 4.7%, and
their scores could not be compared with the previous round's. A plain random draw answers the
question we are actually asking: what will a labeller see.

A second file carries the forced covalent/metal/positional cases, clearly labelled as a separate
targeted set, so those classes still get eyes on them without distorting the headline sample.
"""
import collections, json, os, random, sys

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SP = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


import re


# Row-level boilerplate -> tags. The brief's caveat legend holds the reasoning; a row needs only to
# say WHICH caveat fired and the pair-specific numbers.
_TAGS = [
    (re.compile(r'polar contact, donor/acceptor INCOMPATIBLE \(([^)]*)\): not a hydrogen bond'),
     r'polar contact [INCOMPAT: \1]'),
    (re.compile(r' \[RESIDUE SUSPECT: (\d+) other atoms of this ligand are modelled implausibly '
                r'close to ([A-Z0-9]+),[^\]]*\]'), r' [SUSPECT \2 x\1]'),
    (re.compile(r' \[nearest INDEPENDENT contact; ([A-Z0-9]+) is excluded as the residue its '
                r'covalent link already describes\]'), r' [INDEP, \1 excl]'),
    (re.compile(r' -- unusually close \(([0-9.]+) A vs ([0-9.]+) A vdW sum\); treat the crystal '
                r'model as uncertain unless a neighbouring atom coordinates the metal'),
     r' [CLOSE \1/\2 uncertain]'),
    (re.compile(r' -- unusually close \(([0-9.]+) A vs ([0-9.]+) A vdW sum\); treat the crystal '
                r'model as uncertain'), r' [CLOSE \1/\2 uncertain]'),
    (re.compile(r' -- unusually short \(([0-9.]+) A vs ([0-9.]+) A minimum for [A-Z]+\); treat '
                r'the crystal model as uncertain'), r' [SHORT \1/\2 uncertain]'),
    (re.compile(r'unusually short polar contact \(([0-9.]+) A\) -- treat the crystal model as '
                r'uncertain'), r'short polar [SHORT \1 uncertain]'),
    (re.compile(r' \(modelled at ([0-9.]+) A vs ([0-9.]+) A ideal -- disclosed as uncertain; NOT '
                r'citable as a covalent bond\)'), r' [\1 vs \2 ideal, NOT citable]'),
    (re.compile(r'near ([A-Z]+) metal(?: in [A-Z0-9]+)?'), r'near \1'),
]


def _tag(t, lean=True):
    """Collapse a typed interaction's explanatory suffixes to tags."""
    if not t or not lean:
        return t or ''
    for pat, rep in _TAGS:
        t = pat.sub(rep, t)
    return t


_ROW_CAVEAT = {
    'moved_caveat': lambda c: 'MOVED atom - not a before/after of the same group',
    'typing_changed': lambda c: 'TYPING CHANGED - not retained',
    'chain_caveat': lambda c: 'CHAINS %s/%s' % (c.get('input_chain'), c.get('analogue_chain')),
    'numbering_caveat': lambda c: 'HETATM-LABEL MATCH ONLY',
}


# `unlabellable` is computed by clean_v7.py on 634 pairs (9.0%) and was rendered NOWHERE, in lean
# mode or full -- so every one of those pairs read as an ordinary pair, and CHAIR_FULL.md's second
# REJECT ground ("`unlabellable` is set") keyed on a field no reader could see. The long reason
# strings collapse to tags like every other caveat; an UNRECOGNISED reason falls through to its full
# text rather than being dropped, so a new reason can never go silent the way this flag did.
_UNLAB = [
    ('no interaction survives', 'no interaction survives between the two crystals'),
    ('the two crystals share no residue numbering', 'no shared numbering AND nothing conserved'),
    ('a residue number denotes DIFFERENT residues', 'a residue number means different residues'),
    ('the atom mapping failed', 'atom mapping failed: no mapped atom at all'),
]


def _unlab(reasons, lean=True):
    out = []
    for x in reasons or []:
        if not lean:
            out.append(x)
            continue
        for pre, short in _UNLAB:
            if x.startswith(pre):
                out.append(short)
                break
        else:
            out.append(x)
    return out


# Strength ordering of the interaction classes, used ONLY to detect a typing flip whose distance
# moved the other way. The typing is ANGLE-driven; a sentence about it is distance-flavoured, and on
# 647 of 1,527 `typing_changed` rows (42%) the two disagree -- an H-bond "degrading" to a polar
# contact while the contact got 0.12 A SHORTER. That is verbatim the 9GPO sentence all five chemists
# marked WRONG, the only unanimous WRONG in the cohort.
_STRENGTH = [('COVALENT', 6), ('METAL', 5), ('SALT', 4), ('H-bond,', 3),
             ('unusually short polar', 2), ('polar contact', 2), ('probable', 2),
             ('contact with', 1), ('polar/apolar', 1), ('hydrophobic', 1), ('near ', 1)]


def _strength(t):
    for pre, v in _STRENGTH:
        if (t or '').startswith(pre):
            return v
    return 0


_STATUS = {
    'conserved': 'conserved',
    'NEW': 'NEW',
    'MOVED (present in both molecules but its bonds differ, so there is no valid before/after '
    'comparison)': 'MOVED',
    'present, NOT COMPARABLE (the matching input atom is not modelled)': 'NOTCOMPARABLE',
}


def _direction_tag(c):
    """Flag a conserved row whose retyping and whose distance point opposite ways."""
    di, da = c.get('input_distance_A'), c.get('analogue_distance_A')
    si, sa = _strength(c.get('input_interaction')), _strength(c.get('analogue_interaction'))
    if di is None or da is None or si == sa or abs(da - di) < 0.01:
        return ''
    if sa < si and da < di:
        return '  [ANGLE-ONLY: typing weakens but the contact SHORTENS %.2f->%.2f A]' % (di, da)
    if sa > si and da > di:
        return '  [ANGLE-ONLY: typing strengthens but the contact LENGTHENS %.2f->%.2f A]' % (di, da)
    return ''


_AA = frozenset('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR '
                'VAL MSE SEC PYL HID HIE HIP'.split())
_RESNUM = re.compile(r'([A-Z0-9]+?)(\d+)\.')
# A C-S bond is 1.81 A and a C-N 1.47; nothing non-bonded sits inside 1.90 A. So a sub-1.90 contact
# typed as packing is a typing failure, not a close approach, and 42 pairs (0.6%) have one while
# COVALENCY MISMATCH tells the labeller the bond was lost. 9GC6 is the mechanism in plain sight:
# input CYS797.SG 1.79 A COVALENT BOND, analogue CYS800.SG 1.66 A "packing" -- a +3 numbering
# offset between the depositions, so the same cysteine failed its own identity test. Two of the
# four such pairs kept in the first 1,000 assert a bond loss that is probably this bug.
_BOND_LEN_MAX = 1.90


def _covalency_unreliable(r, lo):
    if not lo.get('covalency_mismatch'):
        return ''
    hits = [c for c in (lo.get('analogue_contacts') or [])
            if (c.get('distance_A') or 9) < _BOND_LEN_MAX
            and 'COVALENT' not in (c.get('interaction') or '').upper()]
    if not hits:
        return ''
    h = min(hits, key=lambda c: c['distance_A'])
    return ('*** COVALENCY MISMATCH IS UNRELIABLE HERE -- the analogue models a:%s at %.2f A from '
            '%s, typed as a non-covalent contact. Nothing sits that close without a bond, so the '
            'mismatch above is probably a failed bond perception or a residue-numbering offset '
            'between the two depositions, NOT a lost bond. Do not write that the bond was lost.\n'
            % (h['atom'], h['distance_A'], h['faces']))


def _hetero_renumbered(r, lo):
    def het(rows, key):
        out = collections.defaultdict(set)
        for c in rows:
            m = _RESNUM.match(c.get(key) or '')
            if m and m.group(1) not in _AA:
                out[m.group(1)].add(m.group(2))
        return out

    inp = het(r['MODEL_INPUT']['atoms'], 'faces')
    ana = het(lo.get('analogue_contacts') or [], 'faces')
    joined = {m.group(1) for c in (lo.get('conserved_interactions') or [])
              for m in [_RESNUM.match(c.get('input_faces') or '')] if m}
    bad = sorted(k for k in set(inp) & set(ana)
                 if inp[k] != ana[k] and k not in joined)
    if not bad:
        return ''
    return ('*** HETERO PARTNER RENUMBERED -- %s. The two crystals number the same cofactor or ion '
            'differently, so it is matched by LABEL and the match failed: no conserved row was '
            'produced for it even though both sides contact it. Its absence from '
            'conserved_interactions is a join failure, NOT a lost interaction -- do not write that '
            'the coordination or the cofactor contact was lost, and do not write that it held '
            'unless you can cite both distances from the two tables yourself.\n'
            % '; '.join('%s is %s in the input and %s in the analogue'
                        % (k, '/'.join(sorted(inp[k])), '/'.join(sorted(ana[k])))
                        for k in bad))


def render(r, lean=True):
    mi, lo = r['MODEL_INPUT'], r['LABEL_ONLY']
    L = ['=' * 100,
         'PAIR %s   target %s   tanimoto %.3f   analogue crystal %s'
         % (r['pair'], r['target'], r['tanimoto'], lo.get('analogue_pdb')),
         '=' * 100, '',
         '--- MODEL_INPUT: everything the model gets at inference (input molecule + its own pocket) ---',
         'input SMILES: %s' % mi['input_smiles']]
    if mi.get('suspect_residues'):
        L.append(('SUSPECT RESIDUES: %s -- see brief' % ', '.join(mi['suspect_residues'])) if lean
                 else ('SUSPECT RESIDUES in this crystal: %s -- each is modelled implausibly close '
                       'to two or more ligand atoms, so its position is unreliable. Every row '
                       'naming one carries [RESIDUE SUSPECT]; treat those claims as you would an '
                       'uncertain row.' % ', '.join(mi['suspect_residues'])))
    if mi.get('input_atoms_listed'):
        L.append('ATOM TABLE COVERAGE: %s%s' % (mi['input_atoms_listed'],
                 '  ** %s' % mi['input_atom_warning'] if mi.get('input_atom_warning') else ''))
    if mi.get('atoms_dropped_as_clashes'):
        L.append('OMITTED AS CLASHES (present in the crystal but overlapping a protein atom, so '
                 'their rows are unreliable and absent below): %s -- %s'
                 % (', '.join(mi['atoms_dropped_as_clashes']), mi.get('clash_note') or ''))
    if mi.get('atoms_not_modelled'):
        L.append('NOT MODELLED in this crystal (absent from the coordinates, so they appear in no '
                 'row below): %s' % ', '.join(mi['atoms_not_modelled']))
    # NOTHING in the interaction column may be truncated. It was cut at [:60] (and polar at [:30]),
    # which hid the word `uncertain` in 1,116 of 1,116 rows that carry it -- 100%, across 10.3% of
    # the corpus. Rule 9 keys on that word, so the brief demanded compliance with a caveat the
    # packet made invisible in every single instance. The interaction text therefore goes LAST and
    # unpadded, and the polar partner moves to its own continuation line rather than being squeezed.
    if lean:
        L += ['',
              '%-5s %-2s %-15s %-5s %5s %5s | %s'
              % ('atom', 'el', 'faces', 'chain', 'dist', 'ang',
                 'interaction (never truncated)')]
        _nc = []
        for a in mi['atoms']:
            _it = a.get('interaction') or ''
            if _it.startswith('no contact within') and '[' not in _it:
                _nc.append(a['atom'])
                continue
            L.append('%-5s %-2s %-15s %-5s %5.2f %5s | %s'
                     % (a['atom'], a['element'], a['faces'], a['chain'] or '-', a['distance_A'],
                        ('%.0f' % a['angle_deg']) if a['angle_deg'] else '-',
                        _tag(a['interaction'], lean)))
            # The polar partner is the nearest protein N/O/S when that is NOT already `faces`. The
            # lean cut dropped it, which silently deleted 1,748 typed H-bonds across 54% of pairs --
            # a polar atom whose nearest neighbour is a carbon still has an H-bond partner, and
            # rules 1 and 2 both license a claim from this field. It costs 1 KB per 14 pairs.
            if a.get('polar_partner') and a['polar_partner'] != a['faces']:
                L.append('%-5s %-2s %-15s %-5s %5s %5s | polar partner: %s'
                         % ('', '', a['polar_partner'], '',
                            ('%.2f' % a['polar_distance_A']) if a.get('polar_distance_A') else '-',
                            ('%.0f' % a['polar_angle_deg']) if a.get('polar_angle_deg') else '-',
                            _tag(a.get('polar_interaction'), lean) or '-'))
        if _nc:
            L.append('NOTHING WITHIN 4.0 A (listed, so the table is complete): %s' % ', '.join(_nc))
    else:
        L += ['',
              '%-5s %-2s %-15s %-5s %5s %5s | %3s %5s %-26s %5s %-14s | %s'
              % ('atom', 'el', 'faces', 'chain', 'dist', 'ang',
                 'H2O', 'd_H2O', 'lining', 'open', 'reading', 'interaction (never truncated)')]
        for a in mi['atoms']:
            L.append('%-5s %-2s %-15s %-5s %5.2f %5s | %3d %5s %-26s %5.2f %-14s | %s'
                     % (a['atom'], a['element'], a['faces'], a['chain'] or '-', a['distance_A'],
                        ('%.0f' % a['angle_deg']) if a['angle_deg'] else '-',
                        a['waters_within_3.5A'],
                        ('%.2f' % a['nearest_water_A']) if a['nearest_water_A'] else '-',
                        a['lining'], a['open_fraction'], a['reading'],
                        a['interaction']))
            # This block sat OUTSIDE the loop, so full mode printed the polar partner of the LAST
            # atom only, once. Full mode is not what anyone labels from, but the bug would have
            # transferred the moment we rendered a full packet again.
            if a.get('polar_partner'):
                L.append('      polar partner %-15s %5s A  %-5s | %s'
                         % (a['polar_partner'],
                            ('%.2f' % a['polar_distance_A']) if a['polar_distance_A'] else '-',
                            ('%.0f' % a['polar_angle_deg']) if a['polar_angle_deg'] else '-',
                            a['polar_interaction'] or '-'))
    ci_pre = lo.get('conserved_interactions') or []
    lc_pre = lo.get('analogue_contacts') or []
    L += ['', '--- LABEL_ONLY: for writing the label; the model never sees any of this ---',
          (('*** PAIR FLAGGED UNLABELLABLE -- %s\n'
            % '; '.join(_unlab(lo.get('unlabellable_reason'), lean)))
           if lo.get('unlabellable') else ''),
          # Derived, not stored: nothing maps, so there is no input<->analogue join and no claim can
          # be attributed to the edit rather than to a different pose. 150 pairs (2.1%), mostly
          # nucleoside/cofactor swaps the MCS mapper refuses to align.
          (('*** NO ATOM-LEVEL JOIN: conserved_interactions is empty and EVERY analogue atom is '
            'NEW, so the two molecules share no mapped atom. Nothing here is attributable to the '
            'edit.\n')
           if (lc_pre and not ci_pre
               and all((c.get('status') or '').startswith('NEW') for c in lc_pre)) else ''),
          # Derived, not stored. Both of these disclose the SAME underlying defect: identity across
          # the two crystals is established by LABEL (residue number, HETATM number), never by
          # position. When the two depositions number the same thing differently, the join silently
          # fails and the packet then asserts something false.
          #
          # They disclose; they do not repair. Re-joining by position needs a superposition, and a
          # wrong join is worse than a disclosed gap -- it would move a measured distance onto the
          # wrong residue, which is the one error a labeller cannot detect.
          _covalency_unreliable(r, lo),
          _hetero_renumbered(r, lo),
          'EDIT: %s' % lo.get('edit_size_note', '(not annotated)'),
          'analogue SMILES: %s' % r['ANSWER']['analogue_smiles'],
          # The packet states the FACT; REVIEWER.md rule 11 owns the policy. It used to print the
          # policy and drop the fact: `re.sub(r'^.*?: ', ...)` cut at the first ': ', which sits
          # after "One sentence cannot cover both", so everything measured -- how many groups, how
          # many bonds apart -- was discarded and all that survived was the stored instruction
          # "describe the change with the larger pocket consequence and say nothing about the
          # other, or reject". That directly contradicts rule 11, which requires disclosing a
          # multi-site edit, and two chairs reported their labellers splitting on it. A packet
          # should never carry editorial policy: it goes stale in a place nobody looks.
          (('*** TWO TRANSFORMATIONS -- %s. See rule 11.\n'
            % (lo['multi_edit_warning'].split('. ')[1].split(': ')[0].strip()
               if len(lo['multi_edit_warning'].split('. ')) > 1
               else 'the edited atoms form separate groups far apart in the molecule'))
           if lo.get('multi_edit_warning') else ''),
          (('*** STEREO -- %s\n' % lo['stereo_note'].split('. ')[0])
           if lo.get('stereo_note') else ''),
          (('*** COVALENCY MISMATCH -- %s\n' % lo['covalency_mismatch'].split('. ')[0])
           if lo.get('covalency_mismatch') else ''),
          (('*** SYMMETRY -- %s\n' % lo['symmetry_note'].split('. ')[0])
           if lo.get('symmetry_note') else ''),
          (('*** ROOM PROBE CONTRADICTS THE EDIT\n') if lo.get('room_caveat') else ''),
          (('*** NO ATTACHMENT POINT FOUND despite NEW atoms\n')
           if lo.get('no_attachment_point_warning') else ''),
          (('*** MULTI-CHAIN POCKET -- %s\n' % lo['multi_chain_note'].split('. ')[0])
           if lo.get('multi_chain_note') else ''),
          (('*** CHAIN TERMINUS CONTACT -- %s\n' % lo['terminus_note'].split('. ')[0])
           if lo.get('terminus_note') else ''),
          (('*** NO SHARED RESIDUE NUMBERING (overlap %.2f)\n'
            % (lo.get('residue_numbering_overlap') or 0.0))
           if lo.get('numbering_note') else ''),
          (('*** CHAINS RELABELLED CONSISTENTLY -- %s\n'
            % (lo.get('chain_relabel_note') or '').split(':')[-1].split('.')[0].strip())
           if lo.get('chain_relabel_note') else ''),
          'numbering_consistent: %s   room_contradicts_edit: %s'
          % (lo.get('numbering_consistent'), lo.get('room_contradicts_edit')), '',
          ('site_of_change -- which rows above the edit touches:' if lean
           else 'site_of_change -- which rows above the edit touches, and the room there:')]
    for s in lo['site_of_change']:
        if lean:
            L.append('   %-6s %s' % (s['atom'], s['role']))
        else:
            L.append('   %-6s %-66s room %-5s %-34s lining %s'
                     % (s['atom'], s['role'], s.get('pocket_room_A'),
                        s.get('room_reading') or '-',
                        ','.join(s.get('pocket_room_lining') or []) or '-'))
    ci = lo.get('conserved_interactions') or []
    if lean:
        L += ['', 'conserved_interactions (%d) -- see brief' % len(ci)]
    else:
        L += ['', 'conserved_interactions -- contacts the edit did NOT disturb: the same input atom faces',
              '  the same protein atom in BOTH crystals. Measured on both sides, not inferred. You may',
              '  describe these as retained/held. %d here.' % len(ci)]
    if ci:
        # Was `ci[:14]` plus "... and N more", which truncated 26.7% of pairs. Rules 1/2/3 are
        # gated on these rows, so hiding them made LEGAL sentences look illegal -- a labeller could
        # see a retained H-bond in the atom table and be unable to cite it. Print all of them.
        # Both typings are shown: "conserved" means the atom pair is in contact in both crystals,
        # NOT that both measurements are sound, and three labellers independently hit pairs where
        # the input side reads clean while the analogue side is flagged uncertain.
        L.append('   %-6s %-8s %-15s %14s   %s'
                 % ('input', 'analog', 'both face', 'input -> analog', 'typing on each side'))
        for c in ci:
            L.append('   %-6s %-8s %-15s %5.2f -> %5.2f   input: %s'
                     % (c['input_atom'], 'a:' + (c['analogue_atom'] or '?'), c['faces'],
                        c['input_distance_A'], c['analogue_distance_A'],
                        _tag(c['input_interaction'], lean)))
            an = c.get('analogue_interaction')
            if an and an != c['input_interaction']:
                L.append('   %-6s %-8s %-15s %14s   analogue: %s%s'
                         % ('', '', '', '', _tag(an, lean),
                            _direction_tag(c) if lean else ''))
            if c.get('input_faces') and c.get('input_faces') != c.get('faces'):
                L.append('        (the INPUT table calls that protein atom %s)' % c['input_faces'])
            for _k in ('moved_caveat', 'typing_changed', 'chain_caveat', 'numbering_caveat'):
                if c.get(_k):
                    L.append('        !! %s' % (_ROW_CAVEAT[_k](c) if lean else c[_k]))
    lc = lo.get('analogue_contacts') or []
    # Which rows this table will actually PRINT, decided before the header announces a count. The
    # header used to count `status != conserved`, which stopped being the printed set the moment
    # conserved rows whose contact MOVED were restored below: it said "2 of 22" over twelve rows.
    # Two labellers read the mismatch as a truncated table. One predicate, evaluated once.
    _consk = {(x.get('input_atom'), x.get('faces')) for x in ci}

    def _is_dup(c):
        # Drop a row ONLY when it is genuinely duplicated in conserved_interactions -- same input
        # atom AND same protein atom. Dropping every `status: conserved` row removed 60% more than
        # that: an atom can be MAPPED to an input atom while its CONTACT changed, so it never
        # enters the conserved table. Those rows were deleted with a pointer to a table that did
        # not contain them, across 97% of pairs, and a labeller could see the contact existed but
        # could not cite it. That is the one failure mode this whole trimming exercise had to avoid.
        return (lean and c.get('status') == 'conserved'
                and (c.get('input_atom'), c.get('faces')) in _consk)

    # Partitioned by row, not by atom name, so the split cannot depend on names being unique.
    _dup = [c['atom'] for c in lc if _is_dup(c)]
    _show = [c for c in lc if not _is_dup(c)]
    if lean:
        L += ['', 'analogue_contacts -- %d of %d analogue atoms: every atom whose contact CHANGED '
                  '(NEW, MOVED, NOTCOMPARABLE) and every mapped atom whose contact moved to a '
                  'different protein atom (status conserved, and the row says which).'
                  % (len(_show), len(lc)),
              '  Only atoms whose contact is unchanged AND already in conserved_interactions are '
              'omitted; they are listed at the end of the table. An ABSENCE HERE IS NOT A LOSS. '
              'Nearest protein atom only. See the brief\'s legend.',
              '  Names are prefixed a: because they are the ANALOGUE\'s namespace. a:C19 and the '
              'input table\'s C19 are DIFFERENT ATOMS unless input_atom says so.']
    else:
        L += ['', 'analogue_contacts -- the ANALOGUE in its own crystal, same 4.0 A cutoff as above.',
              '  `input_atom` is the INPUT molecule\'s name for atoms the edit conserved, so gains and losses can be',
              '  read off directly. `NEW` atoms exist only in the analogue.']
        if lo.get('analogue_polar_caveat'):
            L += ['  ** ' + lo['analogue_polar_caveat']]
    if lo.get('analogue_contacts_complete') is False:
        L += ['  !! INCOMPLETE TABLE (%s) -- absence is NOT evidence; see brief'
              % (lo.get('analogue_atoms_listed') or 'fewer atoms listed than the molecule has')
              ] if lean else ['  ' + (lo.get('analogue_contact_warning') or 'INCOMPLETE table.')]
    elif lo.get('analogue_contacts_complete') is True:
        # This used to read "complete, so an absence IS evidence here", which overstates what the
        # table can support and is the one line in the packet that invited a false loss claim.
        # Complete means every ATOM is listed. Each atom contributes its NEAREST protein atom only,
        # so a residue can be absent from the whole column while still contacting the ligand -- it
        # just is not the nearest neighbour of anything. Absence of an ATOM is evidence; absence of
        # a RESIDUE is not. A labeller caught this on 6PP0; the other two did not.
        L += ['  %s -- complete, so an ATOM missing here is evidence. A RESIDUE missing here is '
              'NOT: each atom contributes only its nearest protein atom, so a real contact that '
              'is nobody\'s nearest neighbour appears nowhere in this column.'
              % lo.get('analogue_atoms_listed', 'all analogue heavy atoms listed')]
    L += ['',
          '   %-6s %-2s %-10s %-14s %-15s %-5s %5s %5s %s'
          % ('atom', 'el', 'input_atom', 'status', 'faces', 'chain', 'dist', 'ang',
             'interaction')]
    for c in _show:
        # `a:` marks the ANALOGUE's namespace. A NEW analogue atom can reuse an input atom's name --
        # 10,155 rows, and in 2,630 pairs (37.2%) BOTH sides carry a real contact, so reading them
        # as one atom produces a confident false sentence ("C20's contact moves from Leu983 to
        # Gly935" when the analogue's C20 is a chemically different atom that merely shares a name).
        #
        # The status was cut at [:9], which printed "MOVED (pr" 51 times in one batch of 100. The
        # four values now render whole, as short labels with legend entries in the brief.
        L.append('   %-6s %-2s %-10s %-14s %-15s %-5s %5.2f %5s %s'
                 % ('a:' + c['atom'], c['el'], c['input_atom'] or '-',
                    (_STATUS.get(c['status'], c['status']) if lean else c['status']),
                    c['faces'] or '-', c.get('chain') or '-', c['distance_A'],
                    ('%.0f' % c['angle_deg']) if c.get('angle_deg') else '-',
                    _tag(c['interaction'], lean)))
    if _dup:
        L.append('   (%d unchanged analogue atoms are in conserved_interactions, not repeated here: %s)'
                 % (len(_dup), ', '.join('a:' + x for x in _dup)))
    # unfired caveat conditionals append '' -- drop them rather than printing blank lines
    out = []
    for _x in L:
        if _x == '' and out and out[-1] == '':
            continue
        out.append(_x)
    return '\n'.join(out)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(D + '/v7_evidence.jsonl')]
    random.seed(17)


    def has(r, s):
        return any(s in a['interaction'].upper() for a in r['MODEL_INPUT']['atoms'])


    def has_role(r, s):
        return any(x['role'].startswith(s) for x in r['LABEL_ONLY']['site_of_change'])


    main = random.sample(rows, N)
    with open(SP + '/audit_v7_representative.txt', 'w') as f:
        f.write('REPRESENTATIVE AUDIT SAMPLE - %d pairs drawn uniformly at random from %s v7 evidence '
                'rows.\nNo stratification: this is what a labeller will actually see.\n\n'
                % (N, format(len(rows), ',')))
        for r in main:
            f.write(render(r) + '\n\n')

    targeted = []
    for tag, pred in (('covalent', lambda r: has(r, 'COVALENT BOND')),
                      ('metal', lambda r: has(r, 'METAL COORDINATION')),
                      ('positional isomer (MOVED)', lambda r: has_role(r, 'MOVED')),
                      ('probable covalent, disclosed', lambda r: has(r, 'PROBABLE COVALENT')),
                      ('unmodelled atoms disclosed',
                       lambda r: bool(r['MODEL_INPUT'].get('atoms_not_modelled')))):
        pool = [r for r in rows if pred(r)]
        if pool:
            targeted.append((tag, random.choice(pool)))
    with open(SP + '/audit_v7_targeted.txt', 'w') as f:
        f.write('TARGETED CASES - one pair per class we specifically want eyes on. NOT representative; '
                'do not infer corpus composition from this file.\n\n')
        for tag, r in targeted:
            f.write('[CLASS: %s]\n%s\n\n' % (tag, render(r)))

    print('representative sample: %d pairs -> audit_v7_representative.txt (%d KB)'
          % (N, os.path.getsize(SP + '/audit_v7_representative.txt') / 1024))
    print('targeted classes:      %d pairs -> audit_v7_targeted.txt (%d KB)'
          % (len(targeted), os.path.getsize(SP + '/audit_v7_targeted.txt') / 1024))
    print('\ncorpus composition, measured on all %s rows (NOT on the sample):' % format(len(rows), ','))
    tot = len(rows)
    for tag, pred in (('has a covalent bond', lambda r: has(r, 'COVALENT BOND')),
                      ('has metal coordination', lambda r: has(r, 'METAL COORDINATION')),
                      ('has a MOVED (positional) site atom', lambda r: has_role(r, 'MOVED')),
                      ('discloses unmodelled ligand atoms',
                       lambda r: bool(r['MODEL_INPUT'].get('atoms_not_modelled')))):
        c = sum(1 for r in rows if pred(r))
        print('   %-38s %5d  (%4.1f%%)' % (tag, c, 100.0 * c / tot))
    print('\nrepresentative sample drawn: %s'
          % ', '.join('%s' % r['pair'] for r in main))
