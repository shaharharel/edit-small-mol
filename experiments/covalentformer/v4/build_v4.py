#!/usr/bin/env python
"""Build the three v4 corpora, with the QA gate as a precondition for writing anything.

TWO DEFECTS IN THE PREVIOUS ATTEMPT, both fixed here.

(1) The QA gate tested tautologies. complexes.jsonl coordinates are ALREADY in a canonical reaction
    frame written by extract_complexes.py (electrophile at the origin in 400/400 sampled complexes,
    nucleophile on the z-axis in 98.8%). Recomputing a frame and then asserting "electrophile at
    origin" tested the upstream extractor, not this code -- it would have passed with the frame code
    deleted. The fix is to stop recomputing: use the upstream frame as given. The gate now tests what
    is actually at risk -- determinism, leakage, and accounting.

(2) The +x rule reintroduced a retired defect. It picked the nearest oxygen ANYWHERE in the ligand with
    no distance cut, falling back to the furthest atom when no oxygen existed. extract_complexes.py:165
    records that this exact rule was replaced because it found a carbonyl for only 54.3% of complexes
    and the other 45.7% "had their octants randomly rotated about z -- and azimuth is exactly where the
    surviving pocket signal lives". Not recomputing the frame removes this entirely.

THREE CORPORA:
  rung2_mmp   hit -> transformation -> analogue. Teaches the editing vocabulary. No structure needed.
  rung3_v4    pocket geometry + hit -> residue, transformation, analogue. The pocket comes from the
              HIT'S OWN crystal (cdl.jsonl carries pdb_a and pdb_b; the builder uses pdb_a), so the
              pose belongs to the molecule being edited.
  contacts    self-supervised: an observed contact is masked by truncating the ligand group that makes
              it, and the model must restore a group. Includes NEGATIVE sites -- solvent-facing groups
              that make no polar contact, where the correct answer is to restore nothing. Without those
              the task rewards bolting a donor onto everything.
"""
from __future__ import annotations
import argparse, collections, json, math, os, random, sys

REPO = '/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer'
HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 0.1

def q(v):
    return round(round(v / GRID) * GRID, 1) + 0.0

def fmt_atoms(lig, poc, max_poc=200, mode='atom'):
    """Coordinates AS GIVEN -- they are already in the canonical reaction frame.

    mode='atom'    one line per pocket ATOM    (~2,000 tokens at max_poc=60)
    mode='residue' one line per pocket RESIDUE (~300 tokens)

    WHY RESIDUE MODE EXISTS. Measured on the built v4 mix, the pocket block is 87% of all training
    compute: contacts 57.3% + rung3 29.8%, against 2.0% for rung1. One epoch costs 374M tokens and
    77 h on one A100-40GB -- and a roofline check puts that at 60% of a realistic ceiling, so it is
    the representation that is expensive, not the implementation. Residue granularity takes the
    corpus to ~97M tokens, a 3.9x cut, at full data exposure.

    This is NOT the retired 56-d summary vector: every number here is still a real coordinate read
    out of the PDB in the canonical frame. What changes is resolution, not provenance. Per residue we
    keep identity, sequence number, the closest-approach distance to the ligand, and two positions --
    the nearest atom (where the contact actually is) and the sidechain centroid (which way the
    residue points). For contacts that is exactly the granularity of the question being asked, since
    the label is a property of the residue. For rung3 it is the actionable signal for deciding where
    to grow or trim, which is what the rung-3 target is."""
    def d2(a):
        return min(sum((a['xyz'][i] - b['xyz'][i]) ** 2 for i in range(3)) for b in lig)

    if mode == 'invariant':
        # ---- v4b: NO FRAME AT ALL. This is the only mode with no coordinate system to choose. ----
        # Every number printed is an interatomic DISTANCE, which is invariant to rotation and
        # translation by construction -- there is nothing to canonicalise, so the anchor question
        # that dogged the covalent frame simply does not arise, and it extends to non-covalent
        # complexes where no bond exists to anchor on. This is the standard move in the
        # pocket-conditioned literature (coordinates entering only through relative geometry) rather
        # than an invention of ours, and it is what survives the crystal -> Boltz/AlphaFold transfer:
        # predicted poses degrade in absolute orientation long before they degrade in contact
        # distances.
        #
        # The ligand block carries TOPOLOGY ONLY (index + element); its conformation enters
        # implicitly through the pocket distances. That also makes this the cheapest mode -- no xyz
        # triples anywhere -- which matters because the pocket block was 87% of v4's compute.
        L = ['LIGAND idx el']
        for i, at in enumerate(lig):
            L.append('  L%-3d %-2s%s' % (i, at.get('el', '?'), '  ELEC' if at.get('is_elec') else ''))
        by = collections.defaultdict(list)
        for p in poc:
            by[(p.get('res', '?'), str(p.get('resnum', '')))].append(p)
        rows = []
        for key, ats in by.items():
            # the K nearest (ligand atom, distance) pairs for this residue: partial trilateration,
            # enough to place the residue relative to the ligand without naming a frame
            best = []
            for p in ats:
                for j, b in enumerate(lig):
                    dd = math.sqrt(sum((p['xyz'][k] - b['xyz'][k]) ** 2 for k in range(3)))
                    best.append((dd, j))
            best.sort()
            seen, trip = set(), []
            for dd, j in best:
                if j in seen:
                    continue
                seen.add(j); trip.append((dd, j))
                if len(trip) == 3:
                    break
            rows.append((trip[0][0], key, trip, any(x.get('is_nuc_res') for x in ats)))
        rows.sort(key=lambda t: (round(t[0], 3), t[1]))
        kept = rows[:max_poc]
        L.append('POCKET res num  (distance to the 3 nearest ligand atoms, Angstrom)   '
                 '(%d of %d nearest residues)' % (len(kept), len(rows)))
        for _, key, trip, isnuc in kept:
            cells = ' '.join('%4.1f L%-3d' % (dd, j) for dd, j in trip)
            L.append('  %-4s %-5s %s%s' % (key[0], key[1], cells, '  NUC' if isnuc else ''))
        return '\n'.join(L)

    L = ['LIGAND idx el x y z']
    for i, a in enumerate(lig):
        x, y, z = (q(c) for c in a['xyz'])
        L.append('  L%-3d %-2s %6.1f %6.1f %6.1f%s'
                 % (i, a.get('el', '?'), x, y, z, '  ELEC' if a.get('is_elec') else ''))

    if mode == 'residue':
        by = collections.defaultdict(list)
        for a in poc:
            by[(a.get('res', '?'), str(a.get('resnum', '')))].append(a)
        rows = []
        for key, ats in by.items():
            near = min(ats, key=d2)
            dmin = math.sqrt(d2(near))
            n = len(ats)
            cx = tuple(sum(a['xyz'][i] for a in ats) / n for i in range(3))
            rows.append((dmin, key, near, cx, any(a.get('is_nuc_res') for a in ats)))
        rows.sort(key=lambda t: (round(t[0], 3), t[1]))
        kept = rows[:max_poc]
        # FOUR NUMBERS PER RESIDUE, NOT SEVEN. The first cut of this block carried dmin + nearest-atom
        # xyz + sidechain-centroid xyz, and MEASURED NO CHEAPER THAN ATOM MODE: 2,549 tokens/row at 40
        # residues against 2,469 at 60 atoms. Widening each line by 4 numbers cancelled the 1.5x drop
        # in line count exactly. Token cost is numbers-on-the-page, so the only way to pay less is to
        # print fewer of them -- reorganising the same information is free of benefit. The centroid is
        # the one droppable field: dmin says how close the residue reaches and near_xyz says from which
        # direction, which is what both tasks turn on.
        L.append('POCKET res num dmin near_x near_y near_z   (%d of %d nearest residues)'
                 % (len(kept), len(rows)))
        for dmin, key, near, cx, isnuc in kept:
            nx, ny, nz = (q(c) for c in near['xyz'])
            L.append('  %-4s %-5s %4.1f %6.1f %6.1f %6.1f%s'
                     % (key[0], key[1], dmin, nx, ny, nz, '  NUC' if isnuc else ''))
        return '\n'.join(L)

    poc = sorted(poc, key=lambda a: (round(d2(a), 4), a.get('res', ''), str(a.get('resnum')),
                                     a.get('name', '')))
    kept = poc[:max_poc]
    L.append('POCKET res num atom el x y z   (%d of %d nearest shown)' % (len(kept), len(poc)))
    for a in kept:
        x, y, z = (q(c) for c in a['xyz'])
        L.append('  %-4s %-5s %-4s %-2s %6.1f %6.1f %6.1f%s'
                 % (a.get('res', '?'), a.get('resnum', ''), a.get('name', '?'), a.get('el', '?'),
                    x, y, z, '  NUC' if a.get('is_nuc_res') else ''))
    return '\n'.join(L)

def _worker_init():
    global Chem, rdFMCS
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    lfc = rdMolStandardize.LargestFragmentChooser()

def _label_one(t):
    h, an, _wh = t
    try:
        return _mcs(h, an)
    except Exception:
        return None

def _mcs(h, an):
    mh, ma = Chem.MolFromSmiles(h), Chem.MolFromSmiles(an)
    if mh is None or ma is None:
        return None
    res = rdFMCS.FindMCS([mh, ma], ringMatchesRingOnly=True, completeRingsOnly=True,
                         timeout=5, matchValences=False)
    if res.canceled or res.numAtoms < 4:
        return 'scaffold_hop'
    patt = Chem.MolFromSmarts(res.smartsString)
    m1, m2 = mh.GetSubstructMatch(patt), ma.GetSubstructMatch(patt)
    if not m1 or not m2:
        return 'scaffold_hop'
    lost = sorted(set(range(mh.GetNumAtoms())) - set(m1))
    gain = sorted(set(range(ma.GetNumAtoms())) - set(m2))
    ls = Chem.MolFragmentToSmiles(mh, atomsToUse=lost) if lost else 'H'
    gs = Chem.MolFragmentToSmiles(ma, atomsToUse=gain) if gain else 'H'
    verb = ('substitute' if lost and gain else 'grow' if gain else 'trim' if lost else 'rearrange')
    anc = 'core'
    src, changed, core = (mh, lost, set(m1)) if lost else (ma, gain, set(m2))
    for idx in changed:
        for nb in src.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in core:
                anc = ('aromatic-' if nb.GetIsAromatic() else '') + nb.GetSymbol()
                break
        else:
            continue
        break
    return '%s %s -> %s at %s' % (verb, ls, gs, anc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='v4')
    ap.add_argument('--max-pocket', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--pocket-mode', choices=('atom','residue','invariant'), default='atom')
    ap.add_argument('--contacts-n', type=int, default=0, help='cap contacts rows (0 = no cap)')
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    lfc = rdMolStandardize.LargestFragmentChooser()

    # ---------------- complexes, keyed by PDB ----------------
    # UNION of the original 1,693 and the 5,256 newly parsed out of the 76,840-structure mirror.
    # rung 3 is unaffected by the expansion (it joins on cdl.jsonl's pdb_a, which is always in the
    # original set -- pocket_pairs_v3 references only 1,749 PDB ids in total). CONTACTS is a
    # single-complex task and therefore scales with the union directly, which is the whole reason
    # the new complexes are worth parsing.
    # PRECEDENCE MATTERS -- setdefault keeps the FIRST source a PDB appears in.
    #   complexes_v5.jsonl  re-extracted under the centroid +x rule (85.7% Murcko core, 14.3%
    #                       ligand centroid, 0% discrete fallback); verified SE(3)-invariant end to
    #                       end on 60 rotated structures, median deviation 0.003 A. rung 3 depends
    #                       on azimuth (pocket DIRECTION is the signal) so it must come from here.
    #   complexes_76k.jsonl old frame, whose azimuth was decided by the atom-name tie-break in
    #                       19.8% of rows. Retained ONLY because it feeds contacts, whose label is
    #                       an interaction TYPE -- a chemical property invariant to rotation about
    #                       z, so the differing azimuth convention cannot corrupt it. Every rung-3
    #                       pdb_a is in the v5 set, so no rung-3 row draws on the old frame.
    cx = {}
    n_src = {}
    for src in ('v5', None, 'data/complexes.jsonl'):
        path = (os.path.join(HERE, 'complexes_v5.jsonl') if src == 'v5' else
                os.path.join(REPO, src) if src else os.path.join(HERE, 'complexes_76k.jsonl'))
        if not os.path.exists(path):
            print('MISSING complex source %s -- continuing without it' % path, flush=True)
            continue
        c0 = len(cx)
        for l in open(path):
            r = json.loads(l)
            if r.get('status') == 'ok':
                cx.setdefault(r['pdb'], r)
        n_src[os.path.basename(path)] = len(cx) - c0
    print('complexes ok: %d  (%s)' % (len(cx), n_src), flush=True)

    # ---------------- rung 3 : pocket + hit -> residue, transform, analogue ----------------
    acct = collections.Counter()
    rung3 = []
    for l in open(os.path.join(REPO, 'data/cdl.jsonl')):
        r = json.loads(l)
        acct['cdl_rows'] += 1
        ca = cx.get(r.get('pdb_a'))
        if ca is None:
            acct['skip_no_hit_complex'] += 1; continue
        h, an = r.get('hit'), r.get('analogue')
        if not h or not an:
            acct['skip_bad_pair'] += 1; continue
        mh, ma = Chem.MolFromSmiles(h), Chem.MolFromSmiles(an)
        if mh is None or ma is None:
            acct['skip_unparseable'] += 1; continue
        if Chem.MolToSmiles(lfc.choose(mh)) == Chem.MolToSmiles(lfc.choose(ma)):
            acct['skip_identical_after_canonicalisation'] += 1; continue
        nuc = (ca.get('nuc') or {})
        block = fmt_atoms(ca.get('ligand_atoms') or [], ca.get('pocket') or [], a.max_pocket, a.pocket_mode)
        bond = ca.get('bond') or {}
        # THE HEADER MUST NOT CLAIM A FRAME THAT DOES NOT EXIST. In invariant mode nothing is at an
        # origin and there is no +z; saying otherwise would be a false statement in every prompt,
        # and the covalent bond is demoted to a named feature rather than the coordinate system.
        hdr = ('Covalent hit-to-lead. The pocket below is from this hit\'s own co-crystal (%s), '
               'described by interatomic distances only -- no coordinate frame.\n\n'
               if a.pocket_mode == 'invariant' else
               'Covalent hit-to-lead. The pocket below is from this hit\'s own co-crystal (%s), in '
               'the reaction frame: the electrophilic carbon is the origin and the attack axis is +z.\n\n')
        # PARENTHESES AROUND THE CONCATENATION ARE LOAD-BEARING: `%` binds tighter than `+`, so
        # without them the format applied only to the literal following hdr, which has six
        # placeholders against seven arguments -> "not all arguments converted during string
        # formatting". It crashed the build immediately rather than producing a wrong prompt.
        prompt = (hdr +
                  'Hit: %s\n\nCOVALENT partner=%s%s.%s angle=%s\n\n%s\n\n'
                  'Task: propose an analogue that keeps the warhead and the binding scaffold.'
                  ) % (r['pdb_a'], h, nuc.get('res', '?'), nuc.get('resnum', ''),
                       nuc.get('atom', '?'),
                       ('%.0f' % bond['angle_elec_nuc_cb']) if bond.get('angle_elec_nuc_cb') else 'na',
                       block)
        op = (r.get('cdl') or {}).get('OP')
        # THE TARGET. The previous build wrote no 'output' at all on 100% of rows while the trainer
        # does str(r['output']) -- it would have raised KeyError on the first rung-3 batch. Target is
        # the docstring's contract: transformation, then analogue.
        #
        # NO 'Residue:' LINE, deliberately, and this differs from R3_COVPOCK. This prompt already
        # states 'COVALENT partner=SER203.OG', so a Residue: line in the response is a verbatim copy
        # of the input -- H(residue | prompt) = 0. derive_cdl.py's own gate says a field computable
        # from the input alone is decoration and belongs on the source side, and scoring it would
        # inflate exact-match with free tokens. R3_COVPOCK could carry it because its prompt withheld
        # the partner; this one does not.
        out = 'Change: %s\nAnalogue: %s' % (op or 'NONE', an)
        rung3.append({'instruction': prompt, 'output': out, 'hit': h, 'analogue': an,
                      'nucleophile': nuc.get('res'), 'pdb_a': r['pdb_a'], 'pdb_b': r.get('pdb_b'),
                      'protein': r.get('protein'), 'op': op,
                      'tc': r.get('tc')})
        acct['rung3_built'] += 1
    print('rung3: %s' % dict(acct), flush=True)

    # ---------------- contacts : INTERACTION TYPE from the real fingerprint ----------------
    # THE PREVIOUS VERSION WAS VOID AND MEASURED SO. It asked "which protein interaction does ligand
    # atom L{i} make?" and answered with residue identity + distance, while the prompt printed BOTH
    # the ligand and the pocket coordinates. A 20-line regex + Euclidean distance script scored
    # 96.1% exact-match on it (96.8% residue, median distance error 0.03 A) and 100% on the
    # negatives. H(answer | prompt) = 0: it taught coordinate arithmetic, not chemistry, and it was
    # the largest auxiliary corpus in v4.
    #
    # It also ignored r['ifp'] -- the actual interaction fingerprint, which carries the one field
    # that is NOT derivable: the interaction TYPE. Measured on 125,981 entries with a 70/30 split, a
    # residue-class lookup table scores 59.9% against a 59.1% always-hydrophobic floor, i.e. ~40
    # points of headroom. Geometry alone does not settle whether a contact is hydrophobic, an
    # h-bond, aromatic or ionic; that is a chemical judgement.
    #
    # So: name the residue in the PROMPT (it is visible in the pocket block anyway -- withholding it
    # would be theatre) and ask for the type. Negatives are pocket residues with no fingerprint
    # entry, subsampled so the task cannot be won by answering NONE.
    con = collections.Counter(); contacts = []
    for pdb, r in cx.items():
        lig = r.get('ligand_atoms') or []
        poc = r.get('pocket') or []
        if not lig or not poc:
            con['skip_empty'] += 1; continue
        by_res = collections.defaultdict(set)
        for s in (r.get('ifp') or []):
            p = s.split('.')
            if len(p) >= 4:
                by_res[(p[0], p[2])].add(p[-1])
        if not by_res:
            con['skip_no_ifp'] += 1; continue
        block = fmt_atoms(lig, poc, a.max_pocket, a.pocket_mode)
        present = sorted({(p.get('res'), str(p.get('resnum'))) for p in poc})
        neg = [k for k in present if k not in by_res]
        random.shuffle(neg)
        # cap negatives at 1/3 of the positives for this complex, floor of 1, so NONE stays a
        # minority answer and the majority-class floor is the positive type distribution
        keep_neg = set(neg[:max(1, len(by_res) // 3)])
        for key in present:
            types = by_res.get(key)
            if types:
                lab = '+'.join(sorted(types)); con['positive'] += 1
            elif key in keep_neg:
                lab = 'NONE'; con['negative'] += 1
            else:
                con['skip_unsampled_negative'] += 1; continue
            contacts.append({'pdb': pdb, 'res': key[0], 'resnum': key[1],
                             'instruction': 'Which interaction type(s) does residue %s%s make with the '
                                            'ligand? Answer with one or more of hydrophobic, hbond, '
                                            'aromatic, ionic, or NONE.\n\n%s' % (key[0], key[1], block),
                             'output': lab,
                             'label': 'positive' if types else 'negative'})
    if a.contacts_n and len(contacts) > a.contacts_n:
        # contacts is 57.3% of training compute for a one-word auxiliary label. Capping it is the
        # single highest-leverage cut; sample WITHOUT regard to label so the class balance and the
        # measured baselines below are unchanged by the cut.
        random.shuffle(contacts)
        contacts = contacts[:a.contacts_n]
        con['capped_to'] = len(contacts)
    print('contacts: %s' % dict(con), flush=True)

    # ---------------- rung 2 : MMP transformations ----------------
    def mcs_transform(h, an):
        mh, ma = Chem.MolFromSmiles(h), Chem.MolFromSmiles(an)
        if mh is None or ma is None:
            return None
        res = rdFMCS.FindMCS([mh, ma], ringMatchesRingOnly=True, completeRingsOnly=True,
                             timeout=5, matchValences=False)
        if res.canceled or res.numAtoms < 4:
            return 'scaffold_hop'
        patt = Chem.MolFromSmarts(res.smartsString)
        m1, m2 = mh.GetSubstructMatch(patt), ma.GetSubstructMatch(patt)
        if not m1 or not m2:
            return 'scaffold_hop'
        lost = sorted(set(range(mh.GetNumAtoms())) - set(m1))
        gain = sorted(set(range(ma.GetNumAtoms())) - set(m2))
        ls = Chem.MolFragmentToSmiles(mh, atomsToUse=lost) if lost else 'H'
        gs = Chem.MolFragmentToSmiles(ma, atomsToUse=gain) if gain else 'H'
        verb = ('substitute' if lost and gain else 'grow' if gain else 'trim' if lost else 'rearrange')
        anc = 'core'
        src, changed, core = (mh, lost, set(m1)) if lost else (ma, gain, set(m2))
        for idx in changed:
            for nb in src.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetIdx() in core:
                    anc = ('aromatic-' if nb.GetIsAromatic() else '') + nb.GetSymbol()
                    break
            else:
                continue
            break
        return '%s %s -> %s at %s' % (verb, ls, gs, anc)

    r2acct = collections.Counter(); rung2 = []
    src2 = 'rung2_shared_train.jsonl'
    todo = []
    for l in open(src2):
        r = json.loads(l)
        r2acct['rows'] += 1
        h, an = r.get('input'), r.get('output')
        if not isinstance(h, str) or not isinstance(an, str) or not h or not an:
            r2acct['skip_bad_pair'] += 1; continue
        mh, ma = Chem.MolFromSmiles(h), Chem.MolFromSmiles(an)
        if mh is None or ma is None:
            r2acct['skip_unparseable'] += 1; continue
        if Chem.MolToSmiles(lfc.choose(mh)) == Chem.MolToSmiles(lfc.choose(ma)):
            r2acct['skip_identical_after_canonicalisation'] += 1; continue
        todo.append((h, an, r.get('warhead', 'covalent')))
    print('rung2: %d pairs to label, parallel' % len(todo), flush=True)
    import multiprocessing as mp
    cache = os.path.join(a.out, '_mmp_cache.json')
    if os.path.exists(cache):
        res = json.load(open(cache))
        if len(res) == len(todo):
            print('  reusing cached MMP labels (%d)' % len(res), flush=True)
            todo_res = res
        else:
            res = None
    else:
        res = None
    ctx = mp.get_context('fork')
    if res is None:
        res = []
        with ctx.Pool(8, initializer=_worker_init) as pool:
            for i, t in enumerate(pool.imap(_label_one, todo, chunksize=200), 1):
                res.append(t)
                if i % 10000 == 0:
                    print('  rung2 labelled %d/%d' % (i, len(todo)), flush=True)
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(cache, 'w'))
        print('  cached MMP labels -> %s' % cache, flush=True)
    for (h, an, wh), t in zip(todo, res):
        if t is None:
            r2acct['skip_unparseable'] += 1; continue
        if t == 'scaffold_hop':
            r2acct['scaffold_hop'] += 1; continue      # excluded, not written as a fake transform
        rung2.append({'instruction': 'Here is a covalent ligand with a %s warhead. Propose a close '
                                     'analogue that keeps the warhead and the binding scaffold.\n\nHit: %s'
                                     % (wh, h),
                      'output': 'TRANSFORM: %s\nANALOGUE: %s' % (t, an),
                      'hit': h, 'analogue': an, 'transform': t, 'warhead': wh})
        r2acct['built'] += 1
    print('rung2: %s' % dict(r2acct), flush=True)

    # ---------------- QA GATE: nothing is written unless this passes ----------------
    fails = collections.Counter()
    TOKRE = __import__('re').compile(r'[A-Za-z0-9@\\[\\]\\(\\)=#\\-\\+\\\\/%\\.]{6,}')
    def leak_check(prompt, answer):
        """Is the answer RETRIEVABLE from the prompt? Compare canonical molecules token by token.
        A raw substring test is wrong: for a `trim` edit the analogue is a fragment of the hit, so its
        canonical SMILES legitimately appears inside the hit's SMILES string without the answer being
        recoverable. That false positive blocked a 94k-row build over one benign row."""
        m = Chem.MolFromSmiles(answer or '')
        if m is None:
            return
        c = Chem.MolToSmiles(m)
        for tok in TOKRE.findall(prompt):
            mt = Chem.MolFromSmiles(tok)
            if mt is not None and Chem.MolToSmiles(mt) == c:
                fails['ANSWER_IN_PROMPT'] += 1
                return
    for r in random.sample(rung3, min(300, len(rung3))):
        leak_check(r['instruction'], r['analogue'])
        if r['hit'] not in r['instruction']:
            fails['hit_missing_from_prompt'] += 1
    for r in random.sample(rung2, min(300, len(rung2))):
        leak_check(r['instruction'], r['analogue'])
    # CONTACTS GATE. The old check was `if output.split()[0] in instruction -> FAIL`, which is the
    # wrong test for a classification task: the label vocabulary has to be stated in the question,
    # so that check now fires on every row by construction. And it never caught the defect that
    # mattered -- the previous contacts corpus passed it while being 96.1% solvable by regex and
    # Euclidean distance.
    #
    # The real question is whether a model-free baseline already wins. Fit the strongest cheap
    # baseline (majority type per residue class) on 70% of the built rows and score the other 30%.
    # If a lookup table clears 75% there is no chemistry left to learn and the corpus is decoration.
    if contacts:
        CHARGED = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        AROM = {'PHE', 'TYR', 'TRP', 'HIS'}
        PLR = {'SER', 'THR', 'ASN', 'GLN', 'CYS', 'TYR'}
        def rcls(x):
            return ('charged' if x in CHARGED else 'aromatic' if x in AROM
                    else 'polar' if x in PLR else 'apolar')
        # The feature set the baseline is allowed must match what the PROMPT actually reveals.
        # residue mode prints `dmin` per residue, so a distance-aware lookup is available to any
        # model and must therefore be the thing we fail against -- residue class alone would be too
        # weak a baseline and would let a newly-trivial task pass. 0.5 A buckets.
        import re as _re
        def feat(c):
            k = rcls(c['res'])
            # BOTH residue AND invariant modes print a distance for every residue, so both must be
            # scored against a distance-aware lookup. Gating this on 'residue' alone made the
            # invariant build report 46.0% when the same table reaches ~73% once it can read the
            # number that is sitting on the page -- a baseline too weak to fail a trivial task.
            if a.pocket_mode in ('residue', 'invariant'):
                m = _re.search(r'^\s+%s\s+%s\s+([\d.]+)' % (c['res'], c['resnum']),
                               c['instruction'], _re.M)
                if m:
                    k = (k, round(float(m.group(1)) * 2) / 2)
            return k
        idx = list(range(len(contacts)))
        random.shuffle(idx)
        cut = int(0.7 * len(idx))
        tab = collections.defaultdict(collections.Counter)
        for i in idx[:cut]:
            tab[feat(contacts[i])][contacts[i]['output']] += 1
        maj = collections.Counter(contacts[i]['output'] for i in idx[:cut]).most_common(1)[0][0]
        te = idx[cut:]
        acc = sum(1 for i in te
                  if (tab[feat(contacts[i])].most_common(1)[0][0]
                      if tab.get(feat(contacts[i])) else maj) == contacts[i]['output'])
        acc = acc / max(len(te), 1)
        flr = sum(1 for i in te if contacts[i]['output'] == maj) / max(len(te), 1)
        print('  contacts baselines: majority(%s) %.1f%%  residue-class lookup %.1f%%'
              % (maj, 100 * flr, 100 * acc))
        if acc > 0.75:
            fails['contacts_solved_by_lookup_table_%.0f%%' % (100 * acc)] += 1
    # determinism: rebuilding one block twice must be identical
    k = next(iter(cx))
    if fmt_atoms(cx[k]['ligand_atoms'], cx[k]['pocket'], a.max_pocket) != \
       fmt_atoms(cx[k]['ligand_atoms'], cx[k]['pocket'], a.max_pocket):
        fails['nondeterministic'] += 1
    print('\n=== QA GATE ===')
    print('  rung2 %d | rung3 %d | contacts %d (%d positive / %d negative)'
          % (len(rung2), len(rung3), len(contacts), con['positive'], con['negative']))
    for k2, v in fails.most_common():
        print('  FAIL %-28s %d' % (k2, v))
    if fails:
        print('QA FAILED -- nothing written'); sys.exit(1)
    print('  QA PASSED')
    for name, rows in (('rung2_mmp', rung2), ('rung3_v4', rung3), ('contacts_v4', contacts)):
        p = os.path.join(a.out, name + '.jsonl')
        with open(p, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        print('  wrote %-16s %7d rows' % (p, len(rows)))
    json.dump({'rung3': dict(acct), 'contacts': dict(con), 'rung2': dict(r2acct)},
              open(os.path.join(a.out, 'accounting.json'), 'w'), indent=2)
    print('BUILD_V4_DONE')

if __name__ == '__main__':
    main()
