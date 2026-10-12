#!/usr/bin/env python3
"""Automatic gate: does a written sentence claim only what this pair's evidence supports?

Rejection is mechanical, so a violation is caught when the sentence is written rather than
discovered after training the way v5's 24% phantom-residue rate was.

SCHEMA NOTE. This was rewritten for the current evidence schema. The previous version read
`CHANGED_ATOMS`, `ATTACHMENT_ATOMS`, `full_pocket_residues` and `contact_type`, none of which
exist any more -- the evidence is now MODEL_INPUT.atoms (with an `interaction` field) plus
LABEL_ONLY.site_of_change. Against the current corpus the old gate resolved an EMPTY set of
allowed residues, so it rejected every sentence that named any residue at all, which would have
made the whole run look like a labelling failure.

Usage:  gate.py <evidence.jsonl> <sentences.json>
        sentences.json maps a pair id ("8ARY NQU->NG9") to the one-sentence label.
"""
import json, re, sys

AA = ('ALA|ARG|ASN|ASP|CYS|GLN|GLU|GLY|HIS|ILE|LEU|LYS|MET|PHE|PRO|SER|THR|TRP|TYR|VAL')
RES = re.compile(r'\b(%s)\s*[- ]?(\d{1,5})\b' % AA, re.I)
HB = re.compile(r'\bhydrogen[- ]bonds?\b|\bh-bonds?\b', re.I)
COV = re.compile(r'\bcovalent\w*\b', re.I)
METAL = re.compile(r'\b(metal|zinc|magnesium|manganese|iron|copper|nickel|cobalt|calcium)\b', re.I)
BURIED = re.compile(r'\bburied|fully enclosed|deeply enclosed|occluded\b', re.I)
SOLV = re.compile(r'\bsolvent[- ]?(facing|exposed)|bulk solvent\b', re.I)
# "the edit removes N1", "deletes C22", "loses the methoxy" -- a claim that a NAMED LIGAND ATOM
# was removed or changed. 7IIY asserted "the edit removes N1, whose 2.56 A hydrogen bond to
# ASP22.OD1 anchors the adenine site" when site_of_change held only BR1, C1 and C14: N1 was not
# touched at all, the rings were merely renumbered, and the label claimed the loss of an anchor
# that is retained. Every rule the gate checked passed, because none of them looked at whether the
# atom named as edited is actually an edited atom.
REMOVED = re.compile(r'\b(remov\w+|delet\w+|los[et]\w*|lose|dropp?\w*|excis\w+|strip\w*)\b'
                     r'[^.;]{0,60}?\b([A-Z]{1,3}\d{1,3})\b', re.I)
LIGATOM = re.compile(r'\b([A-Z]{1,3}\d{1,3})\b')
BETTER = re.compile(r'\bimprov\w*|better|more potent|increas\w+ (potency|affinity)|'
                    r'optimis\w*|optimiz\w*|gain\w* potency|boost\w*|enhanc\w*\b', re.I)
# Rule 9's second half keys on interaction verbs. `contact` is included as a verb form only;
# "a packing contact" is a legitimate noun phrase and must not trip it.
REACHES = re.compile(r'\b(reach\w*|pair\w*|engag\w*|interact\w*|contacts|contacting|'
                     r'hydrogen[- ]bond\w*|h-bond\w*|bond\w*\s+(?:to|with))\b', re.I)

# A claim that the ANALOGUE no longer makes a contact. Distinct from REMOVED, which is about an
# atom the edit deleted; this is about a contact going away.
LOSS = re.compile(r'\\b(no longer|loses|lost|loss of|drops?|dropped|abolish\\w*|breaks?|broken|forfeit\\w*|sacrific\\w*|gives up|relinquish\\w*)\\b', re.I)

ATTRIB = re.compile(r"\banalogue'?s?\b|\bin the analogue\b|\banalogue crystal\b", re.I)
LANDS = re.compile(r'\b(new|added|incoming|appended)\s+\w*\s*(group|substituent|ring|chain|atoms?)\b'
                   r'.{0,40}\b(contacts?|touch\w*|reach\w*|engag\w*|interact\w*|binds?)\b', re.I)


def check(sent, e):
    """Return a list of reasons the sentence is not supported. Empty list means it passes."""
    bad = []
    mi, lo = e['MODEL_INPUT'], e['LABEL_ONLY']
    atoms = mi['atoms']
    site_names = {s['atom'] for s in lo.get('site_of_change', [])}
    site_rows = [a for a in atoms if a['atom'] in site_names]

    # Rule 1 -- only residues present in THIS pair's pocket table may be named. The hit-side table
    # is the right authority: the sentence is anchored at the hit's atoms.
    # Every field that names a residue counts, not just `faces`. polar_partner carries the
    # hydrogen-bond partner that nearest-atom-only reporting used to hide, pocket_room_lining names
    # the residues lining the growth vector, and conserved_interactions names retained contacts --
    # gating on `faces` alone rejected sentences built from any of the three.
    allowed = set()
    def _add(tok):
        m = re.match(r'([A-Z]{1,3})(-?\d+)', tok or '')
        if m:
            allowed.add((m.group(1) + m.group(2)).upper())
    for a in atoms:
        _add(a['faces'])
        _add(a.get('polar_partner'))
    for srow in lo.get('site_of_change', []):
        for lin in (srow.get('pocket_room_lining') or []):
            _add(lin)
    for c in (lo.get('conserved_interactions') or []):
        _add(c['faces'])
    # Rule 5 explicitly permits naming what the ANALOGUE gained, attributed to its own crystal, so
    # rule 1 must allow those residues too. It did not, and the labellers silently dropped
    # legitimate gain descriptions in 4 of the first 20 pairs rather than trip the gate
    # (ARG330/LYS601 in 6KA6, ASN102 in 5SB8, GLU108 in 7I9H, GLY226 in 7L0K). Same contradiction
    # as rules 2/3 had, in the one field left unreconciled.
    input_side = set(allowed)          # everything nameable from the INPUT molecule's own evidence
    for c in (lo.get('analogue_contacts') or []):
        _add(c.get('faces'))
    analogue_only = allowed - input_side
    for m in RES.finditer(sent):
        tok = (m.group(1) + m.group(2)).upper()
        if tok not in allowed:
            bad.append('names %s, which is not in this pair\'s pocket table' % tok)

    # Rules 2-3 -- an interaction may be claimed only where a SITE atom actually has it. Claiming
    # an H-bond that exists elsewhere in the molecule is not a statement about the edit.
    # A SITE atom's typing, or a CONSERVED interaction. A conserved contact is by definition not at
    # the site of change, so gating on the site alone rejected every retained-contact sentence --
    # including the brief's own worked example, where 8ARO's covalently-bonded C16 is not a site
    # atom. `polar_interaction` counts too: it holds the H-bond partner that the nearest-atom-only
    # table used to omit entirely.
    cons = lo.get('conserved_interactions') or []

    # A conserved row carries BOTH sides' typing, and "conserved" only means the same atom pair is
    # in contact in both crystals -- not that both measurements are trustworthy. Gating on
    # `input_interaction` alone let the input side launder an analogue geometry the evidence itself
    # disowns: 6PUE reads `input: COVALENT BOND to LYS43.NZ | analogue: ... 1.22 A ... uncertain`,
    # so "the LYS43 covalent bond is held" was positively licensed by a 1.22 A artifact. If either
    # side of a conserved row is uncertain, that row cannot support a claim of retention.
    def _ok(c):
        return 'uncertain' not in ((c.get('input_interaction') or '') +
                                   (c.get('analogue_interaction') or '')).lower()

    # An ANALOGUE row supports a claim too, when the sentence attributes it to the analogue's
    # crystal -- the same allowance rule 1 makes. Without this, a gained interaction was named but
    # could not be typed: 6W0W's new hydroxyl makes a 3.09 A good-geometry H-bond to GLY257 and the
    # label could not call it a hydrogen bond.
    acs = [c for c in (lo.get('analogue_contacts') or [])
           if 'uncertain' not in (c.get('interaction') or '').lower()]

    def supported(pred):
        if any(pred(a['interaction']) or pred(a.get('polar_interaction') or '') for a in site_rows):
            return True
        if any(pred(c.get('input_interaction') or '') for c in cons if _ok(c)):
            return True
        # Any INPUT atom, not only a site atom, when the sentence is about that interaction
        # CHANGING. 46% of all salt bridges sat on an atom that is neither a site-of-change row nor
        # in conserved_interactions, which made the most important consequence of some edits
        # unsayable -- 8C1D's carboxylate/LYS166 ion pair going 3.30 -> 4.08 A is the whole story of
        # that pair, and rule 2 forbade the words while rule 5 invited the observation.
        if any(pred(a.get('interaction') or '') or pred(a.get('polar_interaction') or '')
               for a in atoms):
            return True
        return ATTRIB.search(sent) and any(pred(c.get('interaction') or '') for c in acs)

    if HB.search(sent) and not supported(lambda t: t.startswith('H-bond')):
        bad.append('claims a hydrogen bond, but neither a site-of-change atom nor a conserved '
                   'interaction has an H-bond typing')
    # `probable covalent bond` is a disclosed, stretched adduct -- rule 3 does not let a label call
    # it covalent, so match the confirmed string only.
    if COV.search(sent) and not supported(lambda t: t.startswith('COVALENT BOND')):
        bad.append('claims covalent chemistry, but neither a site-of-change atom nor a conserved '
                   'interaction has a COVALENT BOND (a `probable covalent bond` does not count)')
    if METAL.search(sent) and not supported(lambda t: 'METAL COORDINATION' in t):
        bad.append('claims a metal interaction, but neither a site-of-change atom nor a conserved '
                   'interaction coordinates a metal')

    readings = {a['reading'] for a in site_rows}
    if BURIED.search(sent) and 'buried' not in readings:
        bad.append('claims buried/enclosed, but no site atom reads "buried" (reads: %s)'
                   % (', '.join(sorted(readings)) or 'none'))
    if SOLV.search(sent) and 'solvent-facing' not in readings:
        bad.append('claims solvent-facing, but no site atom reads "solvent-facing" (reads: %s)'
                   % (', '.join(sorted(readings)) or 'none'))

    # Rule 5, properly enforced. Naming a residue only the ANALOGUE touches is legal, but only
    # when the claim is attributed to the analogue's own crystal rather than asserted as a property
    # of the edit -- at inference there is no analogue structure to check it against. The LANDS
    # regex cannot do this: it keys on the literal word "new|added|incoming|appended", so
    # "the analogue's fluorines contact GLU108" sailed through. Key on the residue instead.
    named_analogue_only = {m for m in (
        (mm.group(1) + mm.group(2)).upper() for mm in RES.finditer(sent)) if m in analogue_only}
    if named_analogue_only and not ATTRIB.search(sent):
        bad.append('names %s, which only the analogue contacts, without attributing it to the '
                   'analogue\'s crystal (rule 5)' % ', '.join(sorted(named_analogue_only)))

    # Rule 9 -- a contact whose typing carries `uncertain` models the ligand and the residue closer
    # together than any real bond; it is a refinement artifact. The brief only forbade quoting its
    # DISTANCE, so a sentence could keep the claim and drop the number and still pass -- which is
    # what 6OXP did. Forbid the residue instead: if every row that reaches a residue is uncertain,
    # that residue is not a measurement and may not be named at all. A residue that also appears in
    # a trustworthy row stays nameable, since the pair's other contacts are fine.
    untrusted, trusted = set(), set()
    def _sort(tok, txt):
        m = re.match(r'([A-Z]{1,3})(-?\d+)', tok or '')
        if m:
            (untrusted if 'uncertain' in (txt or '').lower() else trusted).add(
                (m.group(1) + m.group(2)).upper())
    for a in atoms:
        _sort(a['faces'], a.get('interaction'))
        _sort(a.get('polar_partner'), a.get('polar_interaction'))
    for c in (lo.get('analogue_contacts') or []):
        _sort(c.get('faces'), c.get('interaction'))
    for c in cons:
        _sort(c['faces'], '%s %s' % (c.get('input_interaction') or '',
                                     c.get('analogue_interaction') or ''))
    uncertain_only = untrusted - trusted
    named_uncertain = {m for m in (
        (mm.group(1) + mm.group(2)).upper() for mm in RES.finditer(sent)) if m in uncertain_only}
    if named_uncertain:
        bad.append('names %s, which this pair reaches only through a contact typed `uncertain` -- a '
                   'refinement artifact, not a measurement (rule 9)'
                   % ', '.join(sorted(named_uncertain)))

    # Rule 9's other half. A row typed `packing` is proximity, not chemistry, so wording it as
    # reaching/pairing/engaging a residue upgrades a van der Waals contact into an interaction the
    # evidence never measured -- 8Q2W did exactly this on a 3.72 A packing row, in the same protein
    # where 8Q2T correctly refused the residue. If every row reaching a residue is packing, the
    # sentence may place the group NEAR it but may not describe an interaction with it.
    packing, engaged = set(), set()
    def _sort2(tok, txt):
        m = re.match(r'([A-Z]{1,3})(-?\d+)', tok or '')
        if m:
            (packing if 'packing' in (txt or '').lower() else engaged).add(
                (m.group(1) + m.group(2)).upper())
    for a in atoms:
        _sort2(a['faces'], a.get('interaction'))
        _sort2(a.get('polar_partner'), a.get('polar_interaction'))
    for c in (lo.get('analogue_contacts') or []):
        _sort2(c.get('faces'), c.get('interaction'))
    for c in cons:
        _sort2(c['faces'], '%s %s' % (c.get('input_interaction') or '',
                                      c.get('analogue_interaction') or ''))
    packing_only = packing - engaged
    if packing_only and REACHES.search(sent):
        hit = {m for m in ((mm.group(1) + mm.group(2)).upper()
                           for mm in RES.finditer(sent)) if m in packing_only}
        if hit:
            bad.append('words %s as an interaction (reaches/pairs/engages/contacts), but every row '
                       'reaching it is typed `packing` -- proximity, not chemistry (rule 9)'
                       % ', '.join(sorted(hit)))

    # 12.5% of pairs list fewer analogue heavy atoms than the molecule has, so for those the
    # brief's promise that "an absence is real" is false and a loss claim is unfalsifiable: the
    # contact may exist and simply not be modelled. Gains stay legal -- a listed row is a listed row.
    if lo.get('analogue_contacts_complete') is False and LOSS.search(sent):
        bad.append('claims the analogue loses/drops a contact, but this pair\'s analogue table is '
                   'INCOMPLETE (%s), so an absence is not evidence'
                   % lo.get('analogue_atoms_listed', 'fewer atoms listed than the molecule has'))

    # Rule 4 -- the analogue has no structure here, so nothing may be claimed about what its NEW
    # atoms touch. Where the change is ANCHORED is fair; where it lands is not.
    if LANDS.search(sent):
        bad.append('claims what the NEW atoms contact; the analogue has no structure in this '
                   'evidence, so only the anchor point may be described')

    # An atom named as removed/deleted/lost must actually be an edited atom. Checked against
    # site_of_change, which is the only authority on what the edit touched.
    changed_atoms = {s2['atom'].upper() for s2 in lo.get('site_of_change', [])
                     if s2['role'].startswith(('CHANGED', 'MOVED'))}
    all_lig = {a['atom'].upper() for a in atoms}
    for m in REMOVED.finditer(sent):
        tok = m.group(2).upper()
        if tok in all_lig and tok not in changed_atoms:
            bad.append('says %s was removed/lost, but site_of_change does not list it as CHANGED '
                       'or MOVED (it lists: %s)'
                       % (tok, ', '.join(sorted(changed_atoms)) or 'nothing'))

    # Rule 5 -- direction of potency is not in the evidence.
    if BETTER.search(sent):
        bad.append('claims an improvement; the evidence carries no affinity direction')

    # Rule 6
    if len(sent.split()) > 35:
        bad.append('over 35 words (%d)' % len(sent.split()))

    # Rule 7 -- if the room probe contradicts the edit for this pair, a "no room" claim is wrong.
    if lo.get('room_contradicts_edit') and re.search(r'\bno room|cannot grow|too tight|no space\b',
                                                     sent, re.I):
        bad.append('claims there is no room, but this pair is flagged room_contradicts_edit: the '
                   'analogue demonstrably adds atoms at this site')
    return bad


if __name__ == '__main__':
    ev = {}
    for l in open(sys.argv[1]):
        r = json.loads(l)
        ev[r['pair']] = r
    sents = json.load(open(sys.argv[2]))
    npass = 0
    for k, s in sents.items():
        e = ev.get(k)
        if e is None:
            print('\n%s  SKIP (pair not in evidence file)' % k)
            continue
        bad = check(s, e)
        print('\n%s  %s' % (k, 'PASS' if not bad else 'REJECT'))
        print('   "%s"' % s)
        for b in bad:
            print('     ! %s' % b)
        npass += not bad
    print('\n%d/%d passed the gate' % (npass, len(sents)))
