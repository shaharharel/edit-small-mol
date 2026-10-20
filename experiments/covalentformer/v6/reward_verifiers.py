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

# --------------------------------------------------------------- reward profile
#
# 'v1' is the reward every number in STATUS.md, covalentformer_report.html and the chemist panels
# was computed with. It does not change. 'v2' exists because v1 is a poor OPTIMISATION target even
# though it is a serviceable measuring stick, and GRPO exposed that:
#
#   GRPO v3 ran 150 steps and moved total reward by -0.002. Decomposing the 150-step component
#   trace: PCA on the six weighted components puts 72.9% of all variance on PC1, with near-uniform
#   loadings, and the effective dimensionality is 1.81 of 6. The cause is the gate block below --
#   ANY failed gate sets ALL components to 0.0 simultaneously, so the components move together and
#   the reward is mostly a pass/fail indicator. Since SFT already passes ~88% of the time there is
#   very little variance left for group-relative advantages to work with.
#
#   The one axis that is genuinely free is claim count, and it is gameable: n_claims rose over
#   training (r=+0.32 with step) while interaction_type fell (r=-0.20), because MIN_CLAIMS=2 halves
#   attribution on any single-claim sentence and so pays for padding. Net reward change: zero.
#
# v2 makes four changes, each targeting one of those findings:
#
#   1. Gates split into BLOCKING and GRADED. A blocking gate failure means the output cannot be
#      scored at all (it does not parse, there is no sentence, it asserts nothing, or the molecule
#      is the input unchanged so there is no edit to describe) and still returns 0. A graded gate
#      failure is a real defect in an otherwise scoreable output, and now multiplies the total
#      instead of erasing every component. This is what breaks the PC1 collapse.
#   2. MIN_CLAIMS floor drops 2 -> 1, removing the padding incentive. It also stops halving 39% of
#      chemist-ADOPTED gold (2,680 of 6,865 single-claim sentences), which was never intended.
#   3. Reweighted onto the components that carry independent information. attribution,
#      interaction_type and edit_self_consistency are the least mutually correlated (0.45-0.60) and
#      are the three the chemist rubric actually asks about. grounded, uncertainty_respect and
#      chemistry_clean correlate 0.67-0.91 with each other, so they share one low weight between
#      them rather than three middling ones.
#   4. single_step loosens from 6 to 10 heavy atoms. At 6 it zeroed 453 of 6,865 chemist-ADOPTED
#      gold pairs (6.6%) -- real chemists make bigger single-step edits than that -- and it is a
#      graded gate in v2, so exceeding even 10 costs rather than erases.
#
# Everything v2 changes is a weighting or a gate policy; no verifier logic differs, so the two
# profiles score the same parse of the same sentence and remain directly comparable.
PROFILE = os.environ.get('REWARD_PROFILE', 'v1')
assert PROFILE in ('v1', 'v2'), 'REWARD_PROFILE must be v1 or v2, got %r' % PROFILE

# A failed gate here means the output genuinely cannot be scored, so the total is 0 in both
# profiles. 'not_echo' belongs here: if the molecule is the input, no edit happened and every
# sentence describing one is false, which is not a matter of degree.
BLOCKING_GATES = ('parses', 'has_sentence', 'not_echo')
# v2 only: a defect in an otherwise scoreable output. The value multiplies the total.
GRADED_GATES = {'single_step': 0.40, 'within_word_budget': 0.70}
SINGLE_STEP_MAX = 6 if PROFILE == 'v1' else 10

if PROFILE == 'v2':
    MIN_CLAIMS = 1

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
    # ABANDONED after three attempts, with a measured negative result rather than a hunch.
    # QA against ground truth: three chemists scored ownership on 36 model generations
    # (panel_gen_exemplars). The fixed component is measurable on 4 of those 36 slots (11%) and
    # agrees with the chemist majority on 2 of 4 -- one false negative, one false positive. Coin
    # flip on a ninth of cases.
    # The two fixes themselves worked: transcribing real CCD bond orders made ordered matches
    # possible (16 of 21 successful maps) and cut ambiguity from 79% to 67%, and bidirectional
    # clause parsing recovered the 37 all-neutral sentences. Coverage still collapses because the
    # check composes three lossy steps -- name->atom mapping, which group the prose means, and
    # direction classification -- and the product is small.
    # It also cannot work at all on designed molecules: mol1 has no CCD entry, so ownership
    # verification is structurally unavailable for the case study we care most about.
    # A reliable version almost certainly needs an LLM judge reasoning from the molecule to the
    # sentence, which is what the chemists do, not a regex reading the sentence and trying to
    # recover the molecule. That is a different and far more expensive design.
    # Earlier notes, kept because they record what was tried:
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

# v2 concentrates weight on the three components that carry independent information, measured on
# v3's 150-step trace: attribution/interaction_type correlate 0.46 and edit_self_consistency/
# chemistry_clean 0.45, while grounded/uncertainty_respect correlate 0.91 and grounded/
# chemistry_clean 0.88. Three near-copies of one signal at 0.50 each outvoted attribution at 1.00;
# now they share 0.50 between them. The zero-weighted instruments stay zero-weighted -- promoting
# them would destroy their value as an un-gamed tripwire, which is the whole point of the note in
# the v1 block above.
WEIGHTS_V2 = {
    'attribution': 1.00,
    'interaction_type': 0.75,
    'edit_self_consistency': 0.75,
    'grounded': 0.25,
    'chemistry_clean': 0.25,
    'uncertainty_respect': 0.00,   # 0.91 correlated with grounded; keeping both double-counts
    'directional': 0.00,
    'banned_vocabulary': 0.00,
    'stereo_disclosure': 0.00,
}

if PROFILE == 'v2':
    WEIGHTS = WEIGHTS_V2


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


def _lastpos(rx, text):
    """Proximity score of the LAST match in text before the token: closer => larger."""
    last = None
    for mm in rx.finditer(text):
        last = mm
    return -(len(text) - last.end()) if last else -10 ** 9


def _firstpos(rx, text, _unused=0):
    """Proximity score of the FIRST match after the token: closer => larger."""
    mm = rx.search(text)
    return -mm.start() if mm else -10 ** 9


def parse_claims(sent):
    """Residue-anchored claims: (residue, distance|None, direction, span).

    Each distance literal is assigned to the NEAREST residue token within WINDOW characters, on
    either side. Direction comes from the nearest give-up/retain verb in the same clause.

    The first version searched only FORWARD from each residue, which matches "LYS114 at 3.14 A" but
    not the equally standard "while the 2.82 A GLY601.O hydrogen bond holds". Audited against 6,865
    chemist-adopted gold sentences, it attached just 63.9% of the distances actually written: 13.5%
    of gold sentences registered no numeric claim at all and would have scored a hard zero, and
    64.9% fell under the MIN_CLAIMS divisor floor while being entirely correct. As a reward that is
    worse than merely noisy -- the cheapest way to raise it is to put distances after residue names,
    so it pays for word order rather than for correct chemistry, and it caps the score of the house
    style the SFT model was trained to produce.

    Nearest-token assignment also fixes a quieter error: with a forward-only window, one residue
    swallowed the distance belonging to the NEXT residue in the sentence, which produced a confident
    (residue, distance) pair that the chemist never claimed.
    """
    out = []
    s = sent or ''
    up = s.upper()
    residues = []
    for m in RES_TOK.finditer(up):
        if m.group(1) not in AA:
            continue
        residues.append(('%s%s' % (m.group(1), m.group(2)), m.start(), m.end()))
    if not residues:
        return out

    # "makes no contact within 4.0 A" states the search threshold, not a contact. Attaching it to
    # the nearest residue invents a claim the sentence does not make -- and the gold sentences that
    # use this phrasing would then be penalised for a distance that is correctly absent from the
    # table.
    lits = [(mm.start(), mm.end(), round(float(mm.group(1)), 2)) for mm in NUM.finditer(s)
            if 'WITHIN' not in up[max(0, mm.start() - 14):mm.start()]]

    taken = {}
    prev = None                     # (index of literal's residue, literal end) for conjunctions
    for pos, end, d in lits:
        # "packs LYS575.CG at 3.86 and 3.88 A" -- the second literal belongs to the SAME residue as
        # the first. Measured on gold, nearest-token alone handed it to whatever residue came next
        # in the sentence, inventing a contact the chemist never claimed.
        if prev is not None and not re.search(r'[A-Z]{3}\d', up[prev[1]:pos]) \
                and re.fullmatch(r'\s*(?:A|Å)?\s*(?:and|,|or|/)\s*', s[prev[1]:pos] or ' '):
            taken.setdefault(prev[0], []).append(d)
            prev = (prev[0], end)
            continue
        best, bestgap = None, None
        for i, (res, rs, re_) in enumerate(residues):
            # Measure from the LITERAL'S NEAR EDGE on both sides. Comparing a forward gap taken
            # from the literal's start against a backward gap taken from its end biased every
            # choice by the width of the number itself, which is how "the 2.44 A SER289 and 2.54 A
            # GLU308" credited 2.54 to SER289.
            gap = (rs - end) if end <= rs else (pos - re_)
            if gap <= WINDOW and (bestgap is None or gap < bestgap):
                best, bestgap = i, gap
        if best is not None:
            taken.setdefault(best, []).append(d)
            prev = (best, end)
        else:
            prev = None

    for i, (res, rs, re_) in enumerate(residues):
        ds = taken.get(i) or [None]
        for d in ds:
            lo = max((s.rfind(x, 0, rs) for x in (';', ',', ' while ', ' and ')), default=-1)
            hi = min((x for x in (s.find(x, re_) for x in (';', ',', ' while ', ' and '))
                      if x != -1), default=len(s))
            before, after = s[lo + 1:rs], s[re_:hi]
            gu = max(_lastpos(GIVES_UP, before), _firstpos(GIVES_UP, after, re_ - hi))
            rt = max(_lastpos(RETAINS, before), _firstpos(RETAINS, after, re_ - hi))
            direction = 'neutral' if gu == rt == -10 ** 9 else ('gives_up' if gu > rt else 'retains')
            out.append({'res': res, 'd': d, 'dir': direction,
                        'span': s[max(0, rs - 60):re_ + WINDOW]})
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


# Resolve data relative to the repo, not to one laptop. The first version hardcoded an absolute
# /Users path, which on any other machine silently returns no_ccd_entry for every molecule -- the
# component appears to run while scoring nothing. COVF_DATA overrides for odd layouts.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data():
    """First existing of: COVF_DATA, a flat data/ beside this file, or the repo tree.

    The training box keeps a FLAT copy of this directory (~/covf_v6/) rather than the repo's
    experiments/covalentformer/v6/ path, so a single relative guess resolves to a directory that
    does not exist and every lookup returns no_ccd_entry -- the component would appear to run while
    scoring nothing, which is the failure this function exists to prevent.
    """
    env = os.environ.get('COVF_DATA')
    cands = ([env] if env else []) + [
        os.path.join(_HERE, 'data', 'pdb_mirror'),
        os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'data', 'pdb_mirror')),
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[-1]


DATA = _find_data()
CCD_PATH = os.path.join(DATA, 'ccd_atoms.json')
_CCD = None
_MAP_STATS = collections.Counter()
_AMBIG = {}


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
    # Transcribe the REAL bond orders. The first version made every bond SINGLE and then had to
    # match with generic bonds to succeed at all, which erased the very information that pins a
    # unique correspondence: 324 of 408 molecules matched ambiguously. The CCD gives Kekule orders
    # (SING/DOUB/TRIP, no aromatic flag), which is enough.
    BO = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE,
          'TRIP': Chem.BondType.TRIPLE}
    em = Chem.RWMol()
    for _, e in heavy:
        em.AddAtom(Chem.Atom(e.capitalize() if len(e) > 1 else e.upper()))
    for b in entry.get('bonds') or []:
        a1, a2 = b[0], b[1]
        if a1 in idx and a2 in idx:
            em.AddBond(idx[a1], idx[a2], BO.get(b[2] if len(b) > 2 else 'SING',
                                                Chem.BondType.SINGLE))
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
    # With real bond orders the CCD graph is Kekule while the input SMILES is usually aromatic, so
    # compare after kekulising both. Fall back to generic bonds only if the ordered match fails, and
    # record which path was taken so the QA can tell a clean match from a degenerate one.
    matches = ()
    try:
        probe = Chem.Mol(inp)
        Chem.Kekulize(probe, clearAromaticFlags=True)
        qp = Chem.AdjustQueryParameters()
        qp.makeBondsGeneric = False
        qp.makeAtomsGeneric = False
        qp.adjustDegree = False
        qp.aromatizeIfPossible = False
        q = Chem.AdjustQueryProperties(ccd_mol, qp)
        if q is not None:
            matches = probe.GetSubstructMatches(q, uniquify=False, maxMatches=64,
                                                useChirality=False)
        if matches:
            _MAP_STATS['ordered_match'] += 1
    except Exception:
        matches = ()
    if not matches:
        try:
            qp = Chem.AdjustQueryParameters()
            qp.makeBondsGeneric = True
            qp.makeAtomsGeneric = False
            qp.adjustDegree = False
            qp.aromatizeIfPossible = False
            q = Chem.AdjustQueryProperties(ccd_mol, qp)
            matches = inp.GetSubstructMatches(q, uniquify=False, maxMatches=64,
                                              useChirality=False) if q is not None else ()
            if matches:
                _MAP_STATS['generic_fallback'] += 1
        except Exception:
            matches = ()
    if len(matches) > 1:
        _MAP_STATS['ambiguous'] += 1
    match = matches[0] if matches else ()
    _AMBIG[(ligand_code or '').upper()] = len(matches)
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
                        and abs(gen.GetNumHeavyAtoms() - inp.GetNumHeavyAtoms())
                        <= SINGLE_STEP_MAX),
    }
    c['gates'] = gates
    # v1: any failed gate zeroes everything. v2: only a blocking failure does, and the graded
    # failures are collected into a multiplier applied to the total at the end -- so a long-winded
    # or two-step proposal still earns for the parts of its reasoning that are right, and the nine
    # components stop moving as one. See the reward-profile note at the top of this file.
    blocking = [k for k in BLOCKING_GATES if not gates.get(k, True)]
    if PROFILE == 'v1':
        blocking = [k for k, v in gates.items() if not v]
    if blocking:
        c['components'] = {k: 0.0 for k in WEIGHTS}
        c['total'] = 0.0
        c['n_claims'] = len(claims)
        c['blocked_by'] = blocking
        return c
    gate_mult = 1.0
    for k, pen in GRADED_GATES.items():
        if not gates.get(k, True):
            gate_mult *= pen
    c['gate_multiplier'] = gate_mult

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
    # v2's graded gates land here rather than on the components, so the component breakdown stays
    # readable as "how good was the reasoning" independently of "did it trip a gate". gate_mult is
    # 1.0 under v1 and under a clean v2 generation.
    c['total'] *= c.get('gate_multiplier', 1.0)
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
                    default=os.path.join(DATA, 'v7_evidence.jsonl'))
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
