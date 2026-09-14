"""REACHABILITY — can this molecule present its electrophile to the target cysteine?

THE ONE METRIC THAT IS NOT A PROXY. Deterministic, CPU-only, no model, no decoys, nothing to leak.

WHY THIS AND NOT A COFOLD METRIC. A campaign of ~1000 Boltz cofolds established that no confidence
or geometry statistic from a cofold separates covalent fit on a contrast a free descriptor cannot
already solve. Concretely: deleting the warhead entirely moves complex_iplddt by AUROC 0.442 --
wrong-signed; and on the wrong-cysteine contrast, free volume around the site separated the groups
at 0.947 against the cofold metric's 0.938, with zero GPU. The error decomposition explains why:
Boltz's receptor is crystal-quality (protein CA RMSD 0.40 A, and dropping the CRYSTAL ligand into the
PREDICTED protein gives 1.98 A to the cysteine) but its own ligand lands a median 8.43 A away. The
receptor is excellent; the placement is not. So: take the receptor from the cofold, and solve
placement analytically.

THE QUESTION IT ASKS, per molecule:
    holding the scaffold in its binding pose, does ANY low-energy conformer of the linker+warhead
    place the electrophilic carbon within striking distance of SG, at the right attack angle,
    without clashing?
Pass/fail. A cohort statistic is then just the pass rate.

GEOMETRIC CRITERIA, and where each number comes from:
  distance   3.0-4.5 A from SG to the electrophilic carbon. A formed C-S bond is 1.81 A; a
             PRE-reactive Michaelis complex sits further out. 4.5 A is the generous edge.
  angle      angle(SG, C_el, C_distal) in 110 +- 35 deg. The 110 is EMPIRICAL: measured over 24
             deposited covalent adducts (michael n=7 median 109.3, haloacetamide n=17 median 111.7,
             overall 111.6). ONE coordinate serves both warhead families because post-reaction the
             electrophilic carbon is sp3 tetrahedral regardless of mechanism. This corrects an
             earlier claim of ours that the coordinate had to be class-conditional -- it does not.
  clash      no ligand heavy atom within 2.8 A of a protein heavy atom (excluding the target SG,
             which is supposed to be close).

WHAT IT DOES NOT CLAIM. Reachability is necessary, not sufficient. It says the electrophile CAN be
presented. It says nothing about k_inact, K_I, potency or selectivity. Pair it with a reactivity
term and say so explicitly.

KILL CRITERION, fixed before use: on real deposited covalent complexes the engine must recover the
true warhead placement. If it cannot, the bug is ours and no learned model will do better on the
same geometry.
"""
import os, sys, json, math, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
RDLogger.DisableLog('rdApp.*')

# NO LOWER BOUND ON DISTANCE (fixed after the kill test). The first version demanded
# 3.0 <= d <= 4.5, which REJECTS a molecule for reaching too well: a formed C-S bond is 1.81 A, so
# on the deposited adducts used for validation the correct answer sits BELOW the floor. 46% of real
# complexes failed on distance alone. Reachability is "can the electrophile get there", so only an
# upper bound is meaningful.
D_MIN, D_MAX = 0.0, 4.5
ANGLE_TARGET, ANGLE_TOL = 110.0, 35.0
# CLASH BY vdW OVERLAP, NOT A FLAT CUT (fixed after the kill test). The first version flagged any
# protein-ligand contact under 2.8 A -- which is an N...O HYDROGEN BOND distance (2.7-3.0 A), so
# every real hydrogen bond counted as a steric clash. 86% of deposited complexes failed on it,
# median 4 "clashes" each. Same convention as PoseCheck: overlap = d < r_i + r_j - tol.
# CLASH BUDGET CALIBRATED AGAINST REAL STRUCTURES, not assumed (fixed after the kill test).
# Demanding ZERO vdW overlaps admits only 46.8% of the 141 DEPOSITED covalent ligands measured in
# their OWN crystals (median 1 overlap, 75th pct 3, 90th pct 6). A zero-overlap rule therefore
# rejects the majority of reality, and the original 70% recovery bar was unreachable by
# construction. MAX_CLASH = 2 admits 71.6% of real crystal poses; it was chosen by measuring that
# distribution, not by tuning until the recovery rate looked acceptable.
MAX_CLASH = 2
CLASH_TOL = 0.5
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'F': 1.47, 'CL': 1.75,
       'BR': 1.85, 'I': 1.98, 'P': 1.80}
CLASH_CUT = 2.8          # retained only for the module docstring; not used in the test
N_CONF = 300
PRUNE_RMS = 0.5

# CLASSIFIER VERSION. #142 decided to PIN rather than straddle, and this is what makes the pin
# visible instead of a convention nobody can check. Every corpus on disk (train_strat, train,
# valid_strat, valid_clean) was LABELLED by v1 -- first-SMARTS-wins, multi-warhead detected only
# WITHIN a class. v2 subtracts overlapping michael_sub matches and counts distinct electrophilic
# atoms across classes. Measured difference on the corpora: 1.35% of train_strat rows flip
# EDIT_WARHEAD -> non-WARHEAD, 100% role-asymmetric (LINKER/SCAFFOLD/DECORATION flip 0.00%), and
# realised on generations it is -0.80 pts on the WARHEAD requested arm, -2.26 pts at the
# ground-truth ceiling.
# WHY A CONSTANT AND NOT A COMMENT: the flips do NOT cancel in a paired contrast (#147) -- they
# land 1 in one arm and 3 in the other, so the steering GAP moves and a post-hoc constant
# subtraction cannot recover it. A scorer therefore has to know which version labelled its corpus,
# and the only way that survives a /tmp clear or a new reader is a stamped value.
CLASSIFIER_VERSION = 2
# The version every corpus currently on disk was built with. If you rebuild a corpus, bump this
# in the SAME commit as the rebuild, never separately.
CORPUS_LABELLED_WITH = 1


def announce_classifier_version(where=''):
    """THE scorer-side straddle announcement. One definition, imported by role_compliance and
    warhead_retention.

    THIS DOCSTRING PREVIOUSLY SAID "retyped by nobody" WHILE warhead_retention STILL RETYPED IT.
    Only role_compliance had been converted, so the count went 4 copies -> 2, not -> 1, and the
    sentence asserting completion was written in the same edit that left it incomplete. That is
    #157's failure -- a comment claiming a refactor that was only partly performed -- recurring
    three hours later inside the fix for #157. It is now true: both scorers delegate here.
    The two BUILDER announcements (build_roles, build_phaseB) are deliberately separate and say
    something different -- a builder DEFINES the corpus label and must tell you to bump
    CORPUS_LABELLED_WITH; a scorer CONSUMES a mismatched one and must tell you the numbers are not
    comparable. Different direction, different remedy, correctly not merged.

    It was retyped in three files (build_roles, build_phaseB, role_compliance) with a fourth inline
    copy in warhead_retention -- the identical four-copies shape that de_leak_holdout had, one
    constant over, and found the same way: by an auditor counting call sites rather than by anyone
    reading the code. Worse, those four are the only files that announce anything: NINETEEN files
    call electrophile_index. The boundary I decided "must be named in every affected number"
    (#142) was being named in 4 of 19.

    Two of the silent fifteen are load-bearing for the shape headline:
      - sweep_test.bond_count() calls electrophile_index, and bond_count IS the length variable in
        both the binned Q2 and the partial rho. A v2 relabel moves which atom is the electrophile,
        hence the attachment->electrophile path, hence the strata AND the partial's control.
      - nulls/build_length_null_pool.py rebuilds (bond count, reach) from envelopes*.jsonl, which
        were LABELLED v1, using v2 -- so the model statistic and the null it is judged against are
        BOTH recomputed across the boundary, and neither said so.
    """
    tag = (' [%s]' % where) if where else ''
    if CLASSIFIER_VERSION != CORPUS_LABELLED_WITH:
        print('  CLASSIFIER STRADDLE%s: this code runs electrophile_index v%d, but every corpus on '
              'disk was LABELLED with v%d. Measured effect: EDIT_WARHEAD compliance falls ~0.80 pts '
              'on generations and ~2.26 pts at the ground-truth ceiling, 100%% role-asymmetric, and '
              'it does NOT cancel in the requested-minus-shuffled contrast (#142/#147). Numbers '
              'from this run are NOT comparable to any filed before the classifier changed.'
              % (tag, CLASSIFIER_VERSION, CORPUS_LABELLED_WITH))
    else:
        print('  classifier v%d matches the corpus label version -- no straddle.%s'
              % (CLASSIFIER_VERSION, tag))

WARHEADS = [
    ('michael',       '[CH2]=[CH]C(=O)[NX3]',            0),
    ('michael_sub',   '[CX3]=[CX3][CX3](=O)[NX3]',       0),
    ('haloacetamide', '[F,Cl,Br,I][CH2]C(=O)[NX3]',      1),
    ('vinylsulfone',  '[CX3]=[CX3][SX4](=O)(=O)',        0),
    ('propiolamide',  'C#CC(=O)[NX3]',                   0),
]
PATS = [(n, Chem.MolFromSmarts(s), i) for n, s, i in WARHEADS]


def electrophile_index(mol):
    """-> (atom index of the electrophilic carbon, class) or (None, None).

    THE CLASS WAS DECIDED BY SMARTS ORDER, NOT BY CHEMISTRY, in two separate ways:

    (1) michael_sub '[CX3]=[CX3][CX3](=O)[NX3]' is a STRICT SUPERSET of michael
        '[CH2]=[CH]C(=O)[NX3]' -- a plain acrylamide matches both. Because the loop returned on the
        FIRST class with exactly one match and michael is listed first, a plain acrylamide could
        NEVER be labelled michael_sub. "michael_sub" therefore did not mean "substituted Michael
        acceptor", it meant "matched the general pattern and happened not to be listed earlier".
    (2) the multiple-warhead guard only ever fired WITHIN one class. A molecule carrying an
        acrylamide AND a haloacetamide matched michael once, returned immediately, and was reported
        as a single-warhead michael -- the second electrophile never looked at.

    WHERE THIS PUT A WRONG NUMBER IN A CONCLUSION: build_phaseB.py states as validated fact that of
    52 sub-vdW rows "All 52 were vinylsulfone (33) or michael_sub (19) and none were michael or
    haloacetamide, so it is warhead-class-systematic, not noise." Under the old rule "none were
    michael" was PARTLY GUARANTEED BY ORDERING rather than observed, because anything michael-like
    was labelled michael and anything labelled michael_sub was by construction not michael. The
    sub-vdW conclusion survives; the class attribution behind it does not.

    FIXED HERE: michael_sub now means what its name says (matches the general pattern but is NOT the
    terminal CH2=CH- case), and multiple warheads are detected ACROSS classes, not only within one.
    This CHANGES the warhead_class column for molecules that previously fell through the ordering,
    so any existing artifact carrying that column predates the fix and must not be compared to a
    freshly generated one without re-deriving it.
    """
    hits = []
    for name, p, idx in PATS:
        if p is None:
            continue
        mt = mol.GetSubstructMatches(p)
        if mt:
            hits.append([name, list(mt), idx])
    if not hits:
        return None, None
    # SUBTRACT OVERLAPPING MATCHES, NOT THE WHOLE CLASS. My first version dropped the entire
    # michael_sub class whenever michael matched anywhere, which is wrong as soon as michael_sub's
    # match COUNT exceeds michael's. An acrylamide plus a crotonamide gives {michael: 1,
    # michael_sub: 2} -- two genuine Michael acceptors -- and the class-level drop reported it as a
    # single 'michael'. So the bis-electrophile case the rewrite existed to catch was still missed
    # for michael + substituted-michael, which is the COMMONEST bis-Michael arrangement; it only
    # worked for michael + haloacetamide / vinylsulfone / propiolamide. The docstring above claimed
    # the fix was general. It was not, and six hand-picked tests did not reach the case.
    # Correct rule: a michael_sub match that covers the same electrophilic atom as a michael match
    # IS that electrophile seen through a looser pattern; any OTHER michael_sub match is a second
    # electrophile and must count.
    _by = {n: (m, i) for n, m, i in hits}
    if 'michael' in _by and 'michael_sub' in _by:
        m_mt, m_idx = _by['michael']
        s_mt, s_idx = _by['michael_sub']
        seen = {t[m_idx] for t in m_mt}
        kept = [t for t in s_mt if t[s_idx] not in seen]
        hits = [h for h in hits if h[0] != 'michael_sub']
        if kept:
            hits.append(['michael_sub', kept, s_idx])
    # Count DISTINCT electrophilic atoms across all surviving classes.
    atoms = set()
    for name, mt, idx in hits:
        for t in mt:
            atoms.add(t[idx])
    if len(atoms) > 1:
        return None, 'multiple_warheads'
    name, mt, idx = hits[0]
    return mt[0][idx], name


def distal_index(mol, el):
    """The atom the attack angle is measured to -- the distal partner of the electrophile.

    For a Michael acceptor that is C-alpha (the other alkene carbon); for a haloacetamide the
    carbonyl carbon. In both cases it is the bonded heavy neighbour that is NOT a leaving group,
    which is what makes a single angular criterion valid across families.
    """
    a = mol.GetAtomWithIdx(el)
    cands = [n.GetIdx() for n in a.GetNeighbors()
             if n.GetSymbol() == 'C' and n.GetIdx() != el]
    if not cands:
        cands = [n.GetIdx() for n in a.GetNeighbors() if n.GetSymbol() != 'H']
    return cands[0] if cands else None


def reachable(mol, scaffold_match, scaffold_ref_xyz, sg_xyz, protein_xyz,
              protein_elem=None, n_conf=N_CONF, seed=20260912):
    """Does ANY low-energy conformer present the electrophile to SG?

    scaffold_match   tuple of atom indices in `mol` corresponding to the frozen scaffold
    scaffold_ref_xyz (len(scaffold_match), 3) target coordinates for those atoms
    Returns dict with hit/distance/angle/strain of the BEST conformer found.
    """
    el, cls = electrophile_index(mol)
    if el is None:
        return dict(hit=False, reason='no_single_warhead', warhead_class=cls)
    dist = distal_index(mol, el)
    if dist is None:
        return dict(hit=False, reason='no_distal_atom', warhead_class=cls)

    mh = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.pruneRmsThresh = PRUNE_RMS
    ps.useSmallRingTorsions = True
    cids = AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=ps)
    if not len(cids):
        return dict(hit=False, reason='embed_failed', warhead_class=cls)
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=300)
    except Exception:
        pass

    # relaxed reference energy, for the strain term
    e_rel = None
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mh)
        if mp is not None:
            es = [AllChem.MMFFGetMoleculeForceField(mh, mp, confId=c).CalcEnergy() for c in cids]
            e_rel = float(min(es))
    except Exception:
        pass

    P = np.asarray(protein_xyz) if protein_xyz is not None and len(protein_xyz) else None
    best = None
    for cid in cids:
        conf = mh.GetConformer(cid)
        X = np.array([list(conf.GetAtomPosition(i)) for i in range(mh.GetNumAtoms())])
        # superpose the conformer's scaffold onto the reference pose
        A = X[list(scaffold_match)]
        B = np.asarray(scaffold_ref_xyz)
        if len(A) < 3:
            return dict(hit=False, reason='scaffold_too_small', warhead_class=cls)
        ac, bc = A.mean(0), B.mean(0)
        U, _S, Vt = np.linalg.svd((A - ac).T @ (B - bc))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        Xa = (X - ac) @ R.T + bc
        scaf_rmsd = float(np.sqrt(((Xa[list(scaffold_match)] - B) ** 2).sum(1).mean()))

        e_xyz, d_xyz = Xa[el], Xa[dist]
        dd = float(np.linalg.norm(e_xyz - sg_xyz))
        v1, v2 = sg_xyz - e_xyz, d_xyz - e_xyz
        ang = float(np.degrees(np.arccos(
            np.clip(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))

        clash = 0
        if P is not None:
            heavy = [i for i in range(mh.GetNumAtoms())
                     if mh.GetAtomWithIdx(i).GetSymbol() != 'H']
            L = Xa[heavy]
            lr = np.array([VDW.get(mh.GetAtomWithIdx(i).GetSymbol().upper(), 1.7) for i in heavy])
            pr = (np.array([VDW.get(e, 1.7) for e in protein_elem])
                  if protein_elem is not None else np.full(len(P), 1.7))
            dm = np.linalg.norm(L[:, None, :] - P[None, :, :], axis=2)
            thr = lr[:, None] + pr[None, :] - CLASH_TOL
            # EXCLUDE THE WHOLE TARGET CYSTEINE, not just its sulfur (fixed after the kill test).
            # Masking SG alone still counted the bonded geometry as a clash: CB sits 1.8 A from SG,
            # so an electrophile at bonding distance is ~2.8 A from CB, which trips the C-C vdW
            # threshold of 2.9 A. The criterion was therefore penalising the exact arrangement it
            # exists to detect -- 70% of deposited complexes failed on it. Any protein atom within
            # 3 A of SG belongs to the target residue's reactive end and is excluded.
            near_sg = np.linalg.norm(P - sg_xyz, axis=1) < 3.0
            ok_mask = ~np.broadcast_to(near_sg[None, :], dm.shape)
            clash = int(((dm < thr) & ok_mask).sum())

        ok = (D_MIN <= dd <= D_MAX and abs(ang - ANGLE_TARGET) <= ANGLE_TOL
              and clash <= MAX_CLASH)
        score = (0 if ok else 1, abs(ang - ANGLE_TARGET) + 10 * max(0, dd - D_MAX)
                 + 10 * max(0, D_MIN - dd) + clash)
        rec = dict(hit=ok, d=dd, angle=ang, clash=clash, scaf_rmsd=scaf_rmsd,
                   cid=int(cid), warhead_class=cls)
        if best is None or score < best[0]:
            best = (score, rec)
        if ok:
            break

    out = best[1]
    if e_rel is not None:
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mh)
            e_best = AllChem.MMFFGetMoleculeForceField(mh, mp, confId=out['cid']).CalcEnergy()
            out['strain'] = float(e_best - e_rel)
            out['strain_per_heavy'] = out['strain'] / max(mol.GetNumHeavyAtoms(), 1)
        except Exception:
            pass
    out['n_conf'] = len(cids)
    return out


def main():
    print(__doc__.split('KILL CRITERION')[0])
    print('criteria: d in [%.1f, %.1f] A | angle %.0f +- %.0f deg | no contact < %.1f A | %d confs'
          % (D_MIN, D_MAX, ANGLE_TARGET, ANGLE_TOL, CLASH_CUT, N_CONF))
    return 0


if __name__ == '__main__':
    sys.exit(main())
