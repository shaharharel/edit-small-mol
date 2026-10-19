#!/usr/bin/env python3
"""Programmatic reward for GRPO over the verifiable parts of a pocket-grounded edit rationale.

DESIGN NOTE 1 -- THE EXPLOIT THIS MUST NOT HAVE
-----------------------------------------------
`attrib 1b` (every cited (residue, distance) co-occurs in one table row) has a MODEL-CONTROLLED
denominator: an arm that cites nothing scores 1.0. This is not hypothetical. ether0-24B base scores
100% attribution on the benchmark precisely because it never cites a distance, and SmileyLlama base
scores 100% because it emits a SMILES where the rationale belongs. A naive reward on that ratio has
silence as its global optimum, and GRPO will find it within a few dozen steps.

So attribution is coverage-floored: `correct / max(cited, MIN_CLAIMS)`. Full credit requires at
least MIN_CLAIMS asserted claims; silence scores zero, not one. The reference band is 2.42 residues
per sentence, so over-citing is penalised too -- otherwise machine-gunning every table row inflates
the numerator.

DESIGN NOTE 2 -- WHAT THE VERIFIER MAY SEE
------------------------------------------
A verifier is allowed to use information the model cannot. But everything here is computed from
(a) the prompt's own contact table and (b) the model's own output, with ONE exception noted at
`directional_attribution`, which derives the changed/retained atom sets by MCS between the input and
the model's OWN proposal rather than from the analogue crystal. That choice is deliberate: it makes
the reward self-contained, so it works on any pocket+ligand prompt, not only on MMP pairs that have
a measured analogue. The training pool is therefore not the labelled corpus.

DESIGN NOTE 3 -- WHAT IS DELIBERATELY *NOT* IN THE REWARD
---------------------------------------------------------
`banned_vocabulary` and `stereo_disclosure` are computed and returned but carry weight 0.0 by
default. They are the reward-hacking tripwire: train on attribution/grounding/self-consistency, and
watch these. If they degrade while the trained components climb, the model is gaming the checker
rather than learning to be grounded. Promoting them into the reward destroys that instrument.
"""
import argparse, json, os, re, sys, collections

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS, rdMolDescriptors, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
      'PHE','PRO','SER','THR','TRP','TYR','VAL'}
RES_TOK = re.compile(r'\b([A-Z]{3})(\d+)(?:\.([A-Z0-9]+))?')
NUM = re.compile(r'(\d+\.\d+)')
WINDOW = 80            # chars after a residue token in which its distance may appear

# Two different thresholds, because "says nothing" and "says one true thing" are not the same
# failure. The GATE kills only genuine silence; the FLOOR grades terseness without erasing it. A
# single hard gate at 2 zeroed 40% of ChemDFM g3's rows -- including correct single-claim sentences
# -- which over-punishes brevity and risks inverting the arm ranking this reward is validated
# against.
MIN_CLAIMS_GATE = 1   # below this the reward is 0: nothing was asserted, so nothing can be right
MIN_CLAIMS = 2        # divisor floor; full credit needs >= 2 claims. See design note 1.
MAX_CLAIMS = 5         # beyond the reference band, extra claims stop earning credit

GIVES_UP = re.compile(r'gives? up|loses?|losing|removes?|removing|deletes?|deleting|'
                      r'trades? away|forfeits?|sacrific|abolish|destroy', re.I)
RETAINS = re.compile(r'retain|retains|retaining|holds?|held|keeps?|keeping|preserv|'
                     r'maintain|conserv|unchanged|intact', re.I)
HBOND_CLAIM = re.compile(r'hydrogen bond|h-bond|hbond', re.I)
SALT_CLAIM = re.compile(r'salt bridge', re.I)
COVALENT_CLAIM = re.compile(r'covalent', re.I)

# Motifs the chemist panels actually rejected as water-labile or strained. SMARTS, so cheap.
LIABILITIES = [
    ('N,O-acetal',        '[NX3][CX4;H1,H2]([OX2])[#6,#1]'),
    ('aminal',            '[NX3][CX4;H1,H2][NX3]'),
    ('acyl halide',       '[CX3](=O)[F,Cl,Br,I]'),
    ('anhydride',         '[CX3](=O)[OX2][CX3]=O'),
    ('three-member-fused','[R2;r3]'),           # housane-like strained fusion
]
# the acrylamide Michael acceptor, and the alpha-methylated form that is much less thiol-reactive
ACRYLAMIDE = Chem.MolFromSmarts('[CH2]=[CH]C(=O)[NX3]')
METHACRYLAMIDE = Chem.MolFromSmarts('[CH2]=[C]([CH3])C(=O)[NX3]')

WEIGHTS = {            # see design note 3 for the two zeros
    'attribution': 1.00,
    'interaction_type': 0.50,
    # DISABLED, and parked rather than fixed. Three attempts; current state, measured:
    #   * name->index mapping now succeeds on 408/408 molecules via the CCD graph, BUT 324 of those
    #     (79%) match AMBIGUOUSLY. Making bonds generic -- which is what let the match succeed at
    #     all -- also erases the bond-order information that would pin a unique correspondence, so
    #     "take the first match" is arbitrary wherever the generic-bond graph has more automorphisms
    #     than the real molecule.
    #   * even with mapping fixed the component only FIRES on 7.5% (g3+g4) to 25.5% (g3) of rows,
    #     because direction parsing reads 37 of 251 sentences as entirely neutral and the owner
    #     lookup misses on many of the rest.
    # It is the most targeted verifier available in principle -- "gives up" claims should cite atoms
    # the edit changed, "retains" claims atoms it kept -- and it is not usable at any weight until
    # both the symmetry and the direction parsing are solved. Left computed and instrumented so the
    # coverage stays visible; reinstating it means fixing those two things, not tuning this number.
    'directional': 0.00,
    'grounded': 0.50,
    'edit_self_consistency': 0.75,
    'uncertainty_respect': 0.50,
    'chemistry_clean': 0.50,
    'banned_vocabulary': 0.00,
    'stereo_disclosure': 0.00,
}


# ----------------------------------------------------------------- table


def index_table(model_input):
    """-> (rows, by_residue, nothing_within, suspect, typing) from one MODEL_INPUT block."""
    rows, typing = [], {}
    for a in (model_input.get('atoms') or []):
        for dk, fk, ik in (('distance_A', 'faces', 'interaction'),
                           ('polar_distance_A', 'polar_partner', 'polar_interaction')):
            if a.get(dk) is None or not a.get(fk):
                continue
            res = str(a[fk]).split('.')[0]
            d = round(a[dk], 2)
            rows.append({'atom': a['atom'], 'res': res, 'full': str(a[fk]),
                         'd': d, 'type': (a.get(ik) or '')})
            typing.setdefault((res, d), []).append(a.get(ik) or '')
    by_res = collections.defaultdict(list)
    for r in rows:
        by_res[r['res']].append(r)
    nothing = set(model_input.get('input_atoms_listed') or []) if False else set()
    # atoms explicitly declared contact-free or unmodelled: a claim about them is invention
    for key in ('atoms_not_modelled', 'atoms_dropped_as_clashes'):
        v = model_input.get(key)
        if isinstance(v, (list, tuple)):
            nothing |= {str(x) for x in v}
    suspect = {str(x).split('.')[0] for x in (model_input.get('suspect_residues') or [])}
    return rows, by_res, nothing, suspect, typing


def parse_claims(sent):
    """Residue-anchored claims: (residue, distance|None, direction, span).

    A claim is a residue token plus the first distance within WINDOW characters after it. Direction
    comes from the nearest preceding give-up/retain verb in the same clause. This is the same
    heuristic `table_extra.attrib` uses, kept identical on purpose so the reward and the reported
    metric cannot drift apart.
    """
    out = []
    s = sent or ''
    for m in RES_TOK.finditer(s.upper()):
        if m.group(1) not in AA:
            continue
        res = '%s%s' % (m.group(1), m.group(2))
        tail = s[m.end():m.end() + WINDOW]
        nm = NUM.search(tail)
        d = round(float(nm.group(1)), 2) if nm else None
        head = s[max(0, m.start() - 120):m.start()]
        direction = ('gives_up' if (GIVES_UP.search(head) and not RETAINS.search(head[-40:]))
                     else 'retains' if RETAINS.search(head) else 'neutral')
        span = s[max(0, m.start() - 60):m.end() + WINDOW]
        out.append({'res': res, 'd': d, 'dir': direction, 'span': span})
    return out


# ----------------------------------------------------------------- molecule


def mol(smi):
    return Chem.MolFromSmiles(smi) if smi else None


_MCS_CACHE = {}
_MCS_STATS = collections.Counter()


def changed_conserved(inp, gen):
    """Input atom indices the model's OWN edit touched, and those it kept. See design note 2.

    Memoised on the canonical SMILES pair: MCS dominates the cost of this reward and GRPO rescores
    the same prompt many times per step, so the same pair recurs constantly.
    """
    if inp is None or gen is None:
        return None, None
    try:
        ck = (Chem.MolToSmiles(inp), Chem.MolToSmiles(gen))
    except Exception:
        ck = None
    if ck is not None and ck in _MCS_CACHE:
        _MCS_STATS['hit'] += 1
        return _MCS_CACHE[ck]
    _MCS_STATS['miss'] += 1
    try:
        res = rdFMCS.FindMCS([inp, gen], timeout=5, ringMatchesRingOnly=True,
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareAny)
        patt = Chem.MolFromSmarts(res.smartsString) if res and res.smartsString else None
        keep = set(inp.GetSubstructMatch(patt)) if patt else set()
    except Exception:
        if ck is not None:
            _MCS_CACHE[ck] = (None, None)
        return None, None
    allidx = set(range(inp.GetNumAtoms()))
    out = (allidx - keep, keep)
    if ck is not None:
        _MCS_CACHE[ck] = out
    return out


CCD_PATH = ('/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/ccd_atoms.json')
_CCD = None
_MAP_STATS = collections.Counter()


def _ccd():
    global _CCD
    if _CCD is None:
        try:
            _CCD = json.load(open(CCD_PATH))
        except Exception:
            _CCD = {}
    return _CCD


def atom_name_to_index(model_input, inp, ligand_code=None):
    """Map contact-table atom NAMES onto input-molecule atom indices.

    THE POSITIONAL VERSION OF THIS FUNCTION WAS WRONG AND SILENTLY SO. It assumed the table listed
    one row per input heavy atom in molecule order. Two things are false about that:

      * the table omits atoms with no contact within 4.0 A (they are named in a separate
        "NOTHING WITHIN 4.0 A" line), so it is often SHORTER than the molecule -- 42 of 251 rows in
        one arm; and
      * the rows are sorted ALPHABETICALLY by atom name ('C', 'C1', 'C10', 'C11', ...), not in
        molecule order.

    The second defect is the dangerous one: whenever the counts happened to coincide the positional
    zip returned a confident, wrong mapping rather than failing, so `directional` was not merely
    vacuous on the rows it skipped -- it was wrong on many of the rows it scored.

    The correct mapping needs the chemical-component definition, which names every atom. Build a
    graph from the CCD entry, match it onto the input molecule, and read the correspondence off the
    match. Returns None when that cannot be done, and records why in _MAP_STATS so the caller can
    see how often the component is actually measurable.
    """
    if inp is None:
        _MAP_STATS['no_input'] += 1
        return None
    rows = model_input.get('atoms') or []
    names = [a['atom'] for a in rows]
    if not names:
        _MAP_STATS['no_rows'] += 1
        return None
    entry = _ccd().get((ligand_code or '').upper()) if ligand_code else None
    if not entry:
        _MAP_STATS['no_ccd_entry'] += 1
        return None
    heavy = [(n, e) for n, e in entry['atoms'] if e.upper() != 'H']
    idx = {n: i for i, (n, _) in enumerate(heavy)}
    em = Chem.RWMol()
    for _, e in heavy:
        em.AddAtom(Chem.Atom(e.capitalize() if len(e) > 1 else e.upper()))
    for b in entry.get('bonds') or []:
        a1, a2 = b[0], b[1]
        if a1 in idx and a2 in idx:
            em.AddBond(idx[a1], idx[a2], Chem.BondType.SINGLE)
    ccd_mol = em.GetMol()
    try:
        Chem.SanitizeMol(ccd_mol, Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                         | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
    except Exception:
        pass
    # Match on the heavy-atom skeleton ONLY. The CCD entry carries bond orders we deliberately did
    # not transcribe (every bond above is SINGLE), so a SMARTS derived from this graph can never
    # match an aromatic ring -- which is why the first attempt failed on 408 of 408 molecules.
    # AdjustQueryProperties with makeBondsGeneric turns every bond into "any bond", so the query is
    # pure connectivity plus element, which is exactly the correspondence we need.
    try:
        qp = Chem.AdjustQueryParameters()
        qp.makeBondsGeneric = True
        qp.makeAtomsGeneric = False
        qp.adjustDegree = False
        qp.aromatizeIfPossible = False
        q = Chem.AdjustQueryProperties(ccd_mol, qp)
        matches = inp.GetSubstructMatches(q, uniquify=False, maxMatches=64,
                                          useChirality=False) if q is not None else ()
        # a symmetric ligand can match several ways; any one of them is a valid naming of
        # equivalent atoms, so take the first, but record when it was ambiguous
        if len(matches) > 1:
            _MAP_STATS['ambiguous_symmetry'] += 1
        match = matches[0] if matches else ()
    except Exception:
        match = ()
    if not match or len(match) != len(heavy):
        _MAP_STATS['ccd_match_failed'] += 1
        return None
    order = list(Chem.CanonicalRankAtoms(ccd_mol)) if False else list(range(len(heavy)))
    name_by_pos = {i: n for n, i in idx.items()}
    out = {name_by_pos[i]: match[i] for i in order if i < len(match)}
    _MAP_STATS['ok'] += 1
    return out


# ----------------------------------------------------------------- components


def score(model_input, sent, gen_smi, inp_smi, ligand_code=None):
    """-> dict of component scores in [0,1] (plus booleans), and the weighted total."""
    rows, by_res, nothing, suspect, typing = index_table(model_input)
    claims = parse_claims(sent)
    inp, gen = mol(inp_smi), mol(gen_smi)
    c = {}

    # --- hard gates -------------------------------------------------
    gates = {
        'parses': gen is not None,
        'not_echo': (gen is not None and inp is not None
                     and Chem.MolToSmiles(gen) != Chem.MolToSmiles(inp)),
        'has_sentence': bool((sent or '').strip()),
        'within_word_budget': len((sent or '').split()) <= 35,
        'single_step': (gen is not None and inp is not None
                        and abs(gen.GetNumHeavyAtoms() - inp.GetNumHeavyAtoms()) <= 6),
    }
    c['gates'] = gates
    if not all(gates.values()):
        c['components'] = {k: 0.0 for k in WEIGHTS}
        c['total'] = 0.0
        c['n_claims'] = len(claims)
        return c

    numeric = [x for x in claims if x['d'] is not None]
    tbl_pairs = {(r['res'], r['d']) for r in rows}

    # Validation caught a back door into the silence exploit that the coverage floor alone did not
    # close. Several components are vacuously 1.0 when there is nothing to check -- an arm citing no
    # distances has no wrong interaction type, no wrong direction and no abused caveat -- so
    # SmileyLlama base, which emits a SMILES where the rationale belongs, collected 0.324 total
    # while scoring 0.000 on attribution and grounding. Paying anything for an empty rationale is
    # exactly the gradient this reward exists to avoid, so the coverage floor is a HARD GATE too.
    if len(numeric) < MIN_CLAIMS_GATE:
        c['components'] = {k: 0.0 for k in WEIGHTS}
        c['total'] = 0.0
        c['n_claims'] = len(numeric)
        c['gates']['min_claims'] = False
        return c
    c['gates']['min_claims'] = True

    # --- attribution, coverage-floored (design note 1) --------------
    ok = sum(1 for x in numeric if (x['res'], x['d']) in tbl_pairs)
    denom = max(len(numeric), MIN_CLAIMS)
    c_attrib = ok / denom
    if len(numeric) > MAX_CLAIMS:                 # no credit for machine-gunning the table
        c_attrib *= MAX_CLAIMS / len(numeric)
    c['attribution'] = c_attrib

    # --- interaction-type fidelity ---------------------------------
    # the panels kept finding packing contacts described as hydrogen bonds, and rows the table
    # itself types "donor/acceptor INCOMPATIBLE ... not a hydrogen bond" asserted as H-bonds
    tot = good = 0
    for x in numeric:
        types = typing.get((x['res'], x['d']))
        if not types:
            continue
        t = ' '.join(types).lower()
        if HBOND_CLAIM.search(x['span']):
            tot += 1
            good += ('h-bond' in t or 'hydrogen bond' in t) and 'not a hydrogen bond' not in t
        elif SALT_CLAIM.search(x['span']):
            tot += 1
            good += 'salt bridge' in t
        elif COVALENT_CLAIM.search(x['span']):
            tot += 1
            good += 'covalent' in t
    c['interaction_type'] = (good / tot) if tot else 1.0
    c.setdefault('measured', {})['interaction_type'] = tot > 0

    # --- directional attribution (design note 2) -------------------
    chg, kept = changed_conserved(inp, gen)
    name2idx = atom_name_to_index(model_input, inp, ligand_code)
    tot = good = 0
    if chg is not None and name2idx:
        owner = {}
        for r in rows:
            i = name2idx.get(r['atom'])
            if i is not None:
                owner[(r['res'], r['d'])] = i
        for x in numeric:
            if x['dir'] == 'neutral':
                continue
            i = owner.get((x['res'], x['d']))
            if i is None:
                continue
            tot += 1
            good += (i in chg) if x['dir'] == 'gives_up' else (i in kept)
    c['directional'] = (good / tot) if tot else 1.0
    c.setdefault('measured', {})['directional'] = tot > 0
    c['directional_n'] = tot

    # --- grounded: named residues exist, and none is contact-free ---
    named = {x['res'] for x in claims}
    present = {r['res'] for r in rows}
    c['grounded'] = (len(named & present) / len(named)) if named else 0.0
    invented_atom_claim = any(x['res'] not in present for x in claims)

    # --- edit self-consistency --------------------------------------
    # does the prose name the element/direction of the change the model actually made?
    c['edit_self_consistency'] = 1.0
    if chg is not None and inp is not None and gen is not None:
        dh = gen.GetNumHeavyAtoms() - inp.GetNumHeavyAtoms()
        says_add = re.search(r'add|grow|extend|append|introduc|install', sent or '', re.I)
        says_cut = re.search(r'remov|delet|trunc|shorten|trim|cut|replac|swap|exchang',
                             sent or '', re.I)
        if dh > 0 and not (says_add or says_cut):
            c['edit_self_consistency'] = 0.0
        elif dh < 0 and not says_cut:
            c['edit_self_consistency'] = 0.0

    # --- uncertainty respect ----------------------------------------
    # the mol1 A/B showed both trained arms quoting a length the row flagged unreliable
    bad = 0
    for x in numeric:
        if x['res'] in suspect:
            bad += 1
            continue
        types = typing.get((x['res'], x['d'])) or []
        t = ' '.join(types).lower()
        if ('uncertain' in t or 'unreliable' in t or 'suspect' in t) and not re.search(
                r'uncertain|unreliable|not (established|citable)|may not', x['span'], re.I):
            bad += 1
    c['uncertainty_respect'] = 1.0 - (bad / len(numeric)) if numeric else 1.0
    c.setdefault('measured', {})['uncertainty_respect'] = bool(numeric)

    # --- chemistry cleanliness --------------------------------------
    liab = []
    for name, sma in LIABILITIES:
        p = Chem.MolFromSmarts(sma)
        if p is not None and gen.HasSubstructMatch(p) and not (
                inp is not None and inp.HasSubstructMatch(p)):
            liab.append(name)                      # only NEW liabilities count against the edit
    warhead_lost = (inp is not None and inp.HasSubstructMatch(ACRYLAMIDE)
                    and not gen.HasSubstructMatch(ACRYLAMIDE))
    warhead_deactivated = gen.HasSubstructMatch(METHACRYLAMIDE) and not (
        inp is not None and inp.HasSubstructMatch(METHACRYLAMIDE))
    mw = Descriptors.MolWt(gen)
    props_sane = 120.0 <= mw <= 900.0
    c['chemistry_clean'] = float(not liab and not warhead_lost
                                 and not warhead_deactivated and props_sane
                                 and not invented_atom_claim)
    c['liabilities'] = liab
    c['warhead_lost'] = warhead_lost
    c['warhead_deactivated'] = warhead_deactivated

    # --- tripwires, weight 0 (design note 3) ------------------------
    banned = re.search(r'\bburied in solvent|\bmore room\b|\bchannel\b|\bpotenc|\bIC50|\baffinity '
                       r'will\b', sent or '', re.I)
    c['banned_vocabulary'] = float(not banned)
    inv = _stereo_changed(inp, gen)
    says_stereo = re.search(r'stereo|chiral|epimer|invert|configuration|enantiomer', sent or '',
                            re.I)
    c['stereo_disclosure'] = float(not inv or bool(says_stereo))

    c['components'] = {k: float(c.get(k, 0.0)) for k in WEIGHTS}
    c['total'] = sum(WEIGHTS[k] * c['components'][k] for k in WEIGHTS) / sum(WEIGHTS.values())
    c['n_claims'] = len(numeric)
    c.setdefault('measured', {})
    return c


def _stereo_changed(inp, gen):
    """True only for an inversion at CONSTANT centre count -- a count change can follow from the
    edit itself, a label flip at constant count cannot. Same rule as audit_g3_edit_claims.py."""
    def centres(m):
        try:
            return tuple(sorted(l for _, l in Chem.FindMolChiralCenters(
                m, includeUnassigned=True, useLegacyImplementation=False)))
        except Exception:
            return None
    a, b = centres(inp), centres(gen)
    if a is None or b is None:
        return False
    return len(a) == len(b) and a != b


# ----------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser(description='Score generations with the GRPO reward components.')
    ap.add_argument('generations', nargs='+')
    ap.add_argument('--evidence',
                    default='/Users/shaharharel/Documents/github/edit-small-mol/'
                            'data/pdb_mirror/v7_evidence.jsonl')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    ev = {}
    for line in open(a.evidence):
        r = json.loads(line)
        ev[r['pair']] = r['MODEL_INPUT']

    out = {}
    for path in a.generations:
        agg = collections.defaultdict(list)
        gated = n = 0
        for line in open(path):
            r = json.loads(line)
            mi = ev.get(r['pair'])
            if mi is None:
                continue
            g = None
            for k in ('gen_smiles_canonical', 'gen_smiles'):
                v = r.get(k)
                if isinstance(v, str) and v.strip() and v.strip() != 'None':
                    g = v.strip(); break
            code = None
            try:
                code = r['pair'].split()[1].split('->')[0]
            except Exception:
                pass
            s = score(mi, (r.get('gen_sentence') or '').strip(), g,
                      mi.get('input_smiles'), ligand_code=code)
            n += 1
            gated += (s['total'] == 0.0)
            for k, v in s['components'].items():
                agg[k].append(v)
            for k, v in (s.get('measured') or {}).items():
                agg['MEASURED_' + k].append(float(v))
            if s.get('directional_n') is not None:
                agg['directional_n'].append(s['directional_n'])
            agg['total'].append(s['total'])
            agg['n_claims'].append(s['n_claims'])
        name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        print('\n%s   n=%d  gated-to-zero=%d (%.0f%%)' % (name, n, gated, 100.0*gated/max(n,1)))
        for k in list(WEIGHTS) + ['n_claims', 'directional_n', 'total',
                                  'MEASURED_interaction_type', 'MEASURED_directional',
                                  'MEASURED_uncertainty_respect']:
            v = agg[k]
            if v:
                print('  %-24s %6.3f' % (k, sum(v)/len(v)))
        out[name] = {k: (sum(v)/len(v) if v else None) for k, v in agg.items()}
    print('\nMCS cache: %d hits, %d misses' % (_MCS_STATS['hit'], _MCS_STATS['miss']))
    print('name->index mapping: %s' % dict(_MAP_STATS))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=2)
        print('\nwrote %s' % a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
