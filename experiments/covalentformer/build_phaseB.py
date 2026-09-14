"""PHASE B: the conditioning vector computed from a REAL POCKET, not invented.

WHY THIS FILE EXISTS. Everything in Phase A is pocket-free: d and cos(theta) are SAMPLED from the
fragments' own conformer envelopes and v_free is pinned at 1.0. That taught the model a ligand-side
prior, but it is NOT pocket-conditioned generation. The model has never seen a cysteine.

WHAT THIS PARAGRAPH USED TO SAY, AND WHY IT WAS WRONG. It read "which the sweep control now shows
is real (rho +0.510 with the true requirement, -0.020 when a constant d is fed instead)". The
-0.020 HAS NEVER BEEN MEASURED. sweep_test.py:99 says so in its own words -- "THE CONTROL I FAILED
TO RUN" -- and there is no artifact for it anywhere: no file contains a "control" key, no
"*_recs.json" exists, and --control appears only in sweep_test's own argparse. Two files in one
directory asserted opposite things and the artifact record sided against this one. Stating an
unmeasured number as a settled result, in the docstring of the script that builds the next stage's
dataset, is the same failure as hardcoding a conclusion into a print before seeing the output.

WHAT IS ACTUALLY ESTABLISHED, recomputed from data/replicates/sweep_rep_{101,202,303}.json:
  rho(requested d, achieved reach) = +0.5097 (sd 0.0023 over 3 seeds)   <- this number is real
  rho(requested d, BOND COUNT)     = +0.4992  -> 97.9% of the above is pure LENGTH; the increment
                                                of reach over bond count is +0.0105, i.e. 2.1%.
  binned within-stratum rho        = +0.1389 (sd 0.0175), which sits INSIDE a LENGTH-MATCHED
                                     shape-blind null of +0.0971..+0.1495 -- i.e. INDISTINGUISHABLE
                                     from a policy that never looks at shape at all.
  achievable range: floor 0.0000 (ignore d -- true BY CONSTRUCTION, not measured), ceiling +0.9860.
      The ceiling is the tie cap from d taking only 6 values, and it is DERIVED, two ways that agree:
      closed form for g equal tied levels of size m, rho_max = sqrt(m^2(g^2-1)/12 / ((N^2-1)/12))
      -> sqrt(35/36) = 0.986013 for g=6; and simulation against scipy spearmanr on the sweep's actual
      counts (n=1680 = 6x280 exactly, and n=1671 = [279,279,279,278,278,278]) gives +0.9860 for both.
      THIS LINE SAID +0.9555 UNTIL NOW, AND I REWROTE THIS DOCSTRING TWICE WITHOUT TOUCHING IT.
      0.9555 was unsourced -- a repo-wide grep found it in this line and nowhere else, computed by no
      code and recorded in no artifact -- and it corresponds to no plausible level structure here
      (5 levels 0.9798, 6 levels 0.9860, 7 levels 0.9897, 6 levels with one dominant 0.8868).
      It is the denominator of the "fraction of achievable" framing, so it propagates: the model is at
      +0.5097/0.9860 = 51.7% of achievable, not 53.3%. Small, but it was stated as measured.

THIS PARAGRAPH HAS NOW BEEN WRONG TWICE, IN OPPOSITE DIRECTIONS, AND THE LESSON IS THE WIDTH.
First it said BELOW (from the hardcoded 0.2917). I corrected it to ABOVE, against a null of
+0.0992..+0.1198. That range was also too confident: it was derived by sweeping POOL and n, both
DATASET knobs, while holding the POLICY fixed at one implementation -- so a dataset-sensitivity
interval was presented as total uncertainty. An independent implementation of the same definition
returns +0.0971..+0.1495 and disagrees with the first by 0.05 on the SAME pool file, which is more
than twice the model's margin. The correct verdict is INDISTINGUISHABLE, and the correct fix was
WIDENING rather than a third reversal. The derivation now lives in nulls/ rather than /tmp, because
"derived today, unsourced after the next reboot" is how 0.2917 happened in the first place.

I WROTE "BELOW ITS OWN NULL" HERE AN HOUR AGO AND IT WAS WRONG IN SIGN. That claim came from
LOOKUP_NULL_BINNED = 0.2917, a constant hardcoded at sweep_test.py:210 that NO script derives and no
artifact records. It has now been derived from the fragment population on disk, and the problem is
that the binned metric is MONOTONE in the null policy's own length response, so a null is meaningless
unless it is LENGTH-MATCHED to the model being judged. Calibrated to the model's measured length
response of rho(d,bonds)=+0.4992, the shape-blind null is +0.0992 / +0.1174 / +0.1198 depending on
which fragment pool it is drawn from -- not 0.2917. Reaching 0.2917 requires a lookup policy with
rho(d,bonds)=+0.8394, i.e. 1.68x as length-obedient as the model it is being used to judge, on a
statistic monotone in exactly that. So the "47.6% of its own null" line was comparing the model
against a strictly stronger policy and reading the difference as failure.
Correcting a false claim with an unverified replacement is the same error one level down, and this
paragraph is the instance. The replacement numbers above are recomputed from disk and seeded (200
replicates per point); the 0.2917 was not.

CAP ON WHAT ANY SHAPE STATISTIC CAN RETURN HERE, also measured rather than asserted: over the
fragment population rho(bond count, median reach) = +0.9665, and only 5.8% of reach variance is
WITHIN bond count (4.9% on the 20k pool). sweep_test's own docstring motivates the test with
"four-bond fragments span 3.2 A purely by topology (ortho-phenyl 4.99 A ... para-phenyl 8.21 A)" --
those are hand-picked extremes. At bonds=4 the real spread is sd 0.21 A, p10-p90 span 0.52 A. So a
small partial rho is the CEILING asserting itself, not necessarily a model failure.
And it is not even a Phase A measurement: the replicates are dated 18 h BEFORE ckpt_A_role_strat/ep0
existed, and load_model builds a tierA_v3 GeomEncoderDecoder, not the role model.

NOTE ON THE CONTROL, if anyone runs it. It would genuinely sever the channel -- geom_mlp is the
only path for d and the anchor carries no distance -- but it CANNOT FAIL: with d held constant all
six sweep levels feed one identical conditional distribution, so rho -> 0 by construction. A
non-zero control would indicate a coding bug. "+0.510 vs -0.020" would therefore test only that the
scorer does not leak the label; it is NOT evidence that the model responds usefully to d.

Phase B supplies the missing half. For each deposited covalent complex we measure the requirement
the way it would actually be measured at deployment:

    SG        cysteine sulfur, from the structure
    A         the ATTACHMENT atom on the retained half, from the ligand's crystal POSE
    N         A's neighbour inside the retained half -> exit vector (A -> N)
    d         |SG - A|                                            real Angstroms
    cos theta angle between (N - A) and (SG - A)                  real direction
    v_free    fraction of a cone from A toward SG that is NOT occupied by protein heavy atoms.
              THIS IS THE CHANNEL PHASE A COULD NOT HAVE AT ALL -- occlusion only exists with a
              protein, and it is pinned at 1.0 in every Phase A row.
    phi       the warhead's vinyl-amide dihedral IN THE CRYSTAL. v2-cond conditions on this and
              reaches 2.62 deg planar deviation; we dropped it and measured 24.15 deg, the worst
              geometry number in the panel. It costs nothing to add.

WARHEAD-RETAINED CUTS. Unlike Phase A, the electrophile stays in the RETAINED half and the model
regrows the rest. That makes the requirement well posed -- "reposition THIS electrophile" rather
than "invent a fragment that spans a gap" -- and makes warhead retention 100% by construction
instead of the 52.2% measured on the Phase A model.

SMALL DATA, ON PURPOSE. ~141 complexes is tiny, and Phase B is not asked to learn chemistry. It has
to learn only the mapping from pocket geometry to the requirement vector; the ligand-side prior comes
from Phase A. Whether that transfer works is exactly what the B-scratch vs B-from-A arms test.
"""
import os, sys, json, csv, argparse, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdMolTransforms, AllChem
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import electrophile_index, CLASSIFIER_VERSION, CORPUS_LABELLED_WITH

SCR = '/private/tmp/claude-501/-Users-shaharharel-Documents-github-quris-ml-play-CB1/160c04b4-9b18-4865-b718-cd0ea00a4622/scratchpad'
CIF = os.path.join(SCR, 'ladder/cif')
ACR = Chem.MolFromSmarts('[CH2]=[CH]C(=O)[NX3]')
VDW_PROBE = 1.7


def load_struct(pid, cys, ch, lig):
    import gemmi
    st = gemmi.read_structure(os.path.join(CIF, '%s.cif.gz' % pid))
    st.setup_entities()
    sg, ligatoms, prot = None, [], []
    lig_copies = collections.defaultdict(list)
    for c in st[0]:
        for r in c:
            is_target_cys = (r.name == 'CYS' and str(r.seqid.num) == str(cys) and c.name == ch)
            if is_target_cys:
                a = next((a for a in r if a.name == 'SG'), None)
                if a is not None:
                    sg = np.array([a.pos.x, a.pos.y, a.pos.z])
                # F2 PROPER: skip the WHOLE target residue. The previous radius>3.5 A sphere left
                # the cysteine's own backbone N/C/O in place (SG-N is 3.0-4.3 A, SG-C 3.3-4.5 A,
                # chi1-dependent) so v_free was biased DOWN in a conformation-dependent way, while
                # simultaneously deleting genuinely occluding atoms of NEIGHBOURING residues,
                # biasing it UP. reachability.py's lesson was 'exclude the whole target cysteine'
                # and the sphere did not do that.
                continue
            if r.name == lig:
                # F4 PROPER: group by (chain, seqid) so each ligand COPY stays separate.
                key = (c.name, int(r.seqid.num))
                for a in r:
                    if a.element.name != 'H':
                        # CARRY THE ELEMENT. Deriving it from the PDB atom NAME (CAX, CBJ, NAE...)
                        # by taking the first two alpha chars maps 124 carbons to CALCIUM, 18 to
                        # cadmium, 16 nitrogens to sodium -- all valid symbols, so RDKit accepts
                        # them silently. FindMCS uses CompareElements, so a carbon labelled calcium
                        # can never match and the MCS runs on a mutilated graph.
                        lig_copies[key].append((a.name, np.array([a.pos.x, a.pos.y, a.pos.z]),
                                                a.element.name))
            elif r.name != 'HOH':
                for a in r:
                    if a.element.name != 'H':
                        prot.append([a.pos.x, a.pos.y, a.pos.z])
    # F4 PROPER: pick ONE copy, keep ALL of its atoms. The previous version filtered ATOMS within
    # 12 A of SG with no (chain, seqid) grouping, so two copies at a crystal contact both survived
    # and `cry` became a CHIMERIC two-component molecule -- exactly the failure F4 was meant to
    # remove, since GetSubstructMatch returns the FIRST match. It also truncated distal atoms of
    # the genuine ligand (the d cap admits attachments to 18 A), mutilating the MCS graph, and its
    # `len(near) >= 8` fallback silently reverted to ALL copies precisely when the mapping was
    # least trustworthy.
    if sg is not None and lig_copies:
        best = min(lig_copies,
                   key=lambda k: min(np.linalg.norm(t[1] - sg) for t in lig_copies[k]))
        ligatoms = lig_copies[best]
    else:
        ligatoms = [t for v in lig_copies.values() for t in v]
    return sg, ligatoms, (np.array(prot) if prot else None)


def free_volume(A, SG, prot, n_slices=8, radius=3.2):
    # RADIUS CALIBRATED, not guessed. At 2.5 A (after the cysteine-exclusion fix) the channel came
    # out median 1.00 with 82% of rows at exactly 1.0 -- a near-constant, which is the SAME defect
    # as Phase A pinning v_free=1.0, just at the other end. A radius sweep over real complexes
    # showed the median response halving between 2.5 and 3.0 and collapsing to 0 by 4.0, so 3.2
    # sits where the channel has actual spread. The value is chosen for INFORMATION CONTENT, which
    # is legitimate for a conditioning input (it is not a result being tuned).

    """Fraction of a cylinder from A to SG that no protein heavy atom intrudes into.

    A blunt but honest occlusion measure: walk from the attachment point toward the sulfur and ask,
    at each step, whether anything is in the way. 1.0 = clear path, 0.0 = fully walled.
    """
    if prot is None or len(prot) == 0:
        return 1.0
    v = SG - A
    L = np.linalg.norm(v)
    if L < 1e-6:
        return 1.0
    u = v / L
    # EXCLUDE THE TARGET RESIDUE. Every probe near SG is otherwise occluded by the cysteine itself,
    # so v_free could never exceed 0.875 and 'fully clear' was hardcoded to 0%. reachability.py
    # already learned and documented this exact lesson ("EXCLUDE THE WHOLE TARGET CYSTEINE, not
    # just its sulfur ... 70% of deposited complexes failed on it"); free_volume never applied it.
    # The target residue is now excluded at load time (by residue identity), so no radius filter
    # is applied here -- the previous >3.5 A sphere also deleted neighbouring residues that
    # genuinely occlude the path.
    if len(prot) == 0:
        return 1.0
    free = 0
    for k in range(1, n_slices + 1):
        p = A + u * (L * k / (n_slices + 1.0))
        if np.linalg.norm(prot - p, axis=1).min() > radius:
            free += 1
    return free / float(n_slices)



def mcs_anchor(mol_pre, keep_frag, lig_xyz, ligatoms, sg):
    """Map the RETAINED half onto crystal coordinates and return (attachment xyz, exit vector).

    Returns (None, None, reason) on failure. The attachment atom is the retained-half atom that
    carried the dummy; the exit vector points from it INTO the retained half, which is the direction
    the regrown fragment must leave from.
    """
    from rdkit.Chem import rdFMCS
    blk = []
    for i, t in enumerate(ligatoms):
        nm, xyz = t[0], t[1]
        el = (t[2] if len(t) > 2 and t[2] else 'C').capitalize()
        blk.append('HETATM%5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s'
                   % (i + 1, nm[:4], 'LIG', 1, xyz[0], xyz[1], xyz[2], el))
    cry = Chem.MolFromPDBBlock('\n'.join(blk) + '\nEND\n', removeHs=True, sanitize=False)
    if cry is None:
        return None, None, 'crystal_unparseable'
    try:
        Chem.SanitizeMol(cry, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE)
    except Exception:
        pass
    core = Chem.DeleteSubstructs(keep_frag, Chem.MolFromSmarts('[#0]'))
    try:
        Chem.SanitizeMol(core)
    except Exception:
        pass
    if core.GetNumHeavyAtoms() < 5:
        return None, None, 'core_too_small'
    try:
        r = rdFMCS.FindMCS([core, cry], timeout=10,
                           atomCompare=rdFMCS.AtomCompare.CompareElements,
                           bondCompare=rdFMCS.BondCompare.CompareAny,
                           ringMatchesRingOnly=True, completeRingsOnly=False)
    except Exception:
        return None, None, 'failed'
    if r.numAtoms < 5:
        return None, None, 'too_small'
    patt = Chem.MolFromSmarts(r.smartsString)
    if patt is None:
        return None, None, 'bad_smarts'
    cm = cry.GetSubstructMatch(patt)
    km = core.GetSubstructMatch(patt)
    if not cm or not km or len(cm) != len(km):
        return None, None, 'match_failed'
    conf = cry.GetConformer()
    core2cry = {k: c for k, c in zip(km, cm)}
    # the dummy's neighbour in keep_frag IS the attachment atom, in keep_frag indexing
    star = [a.GetIdx() for a in keep_frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(star) != 1:
        return None, None, 'n_dummy_%d' % len(star)
    nbrs = [x.GetIdx() for x in keep_frag.GetAtomWithIdx(star[0]).GetNeighbors()]
    if not nbrs:
        return None, None, 'no_attach_neighbour'
    # keep_frag index -> core index (core is keep_frag minus the dummy, so indices shift by one
    # for atoms after the dummy)
    def to_core(i):
        return i - 1 if i > star[0] else i
    a_core = to_core(nbrs[0])
    if a_core not in core2cry:
        return None, None, 'attach_not_in_mcs'
    A = np.array(list(conf.GetAtomPosition(core2cry[a_core])))
    # exit vector: attachment -> mean of its retained-half neighbours, in crystal coords
    inner = [to_core(x.GetIdx()) for x in keep_frag.GetAtomWithIdx(nbrs[0]).GetNeighbors()
             if x.GetAtomicNum() != 0]
    pts = [np.array(list(conf.GetAtomPosition(core2cry[i]))) for i in inner if i in core2cry]
    if not pts:
        return None, None, 'no_inner_neighbour'
    # *** THE F1 SIGN "FIX" IS RETRACTED. THIS CONVENTION IS 180 DEGREES FROM THE SPEC. ***
    #
    # The authoritative definition is reach_envelope.py:6-10, which every Phase A d/theta and the
    # deployment requirement are built on:
    #     A = attachment point; N = THE FIRST FRAGMENT ATOM, so A->N is the exit vector,
    #     "the direction the fragment grows"; theta = angle((N-A),(E-A)).
    # Phase A's exit vector therefore points INTO THE GENERATED HALF. The line below points into
    # the RETAINED half -- the opposite direction.
    #
    # HOW THE ERROR WAS MADE, because the mechanism matters more than the line: I justified the
    # flip by observing Phase B's cos_theta was the exact negation of Phase A's (-0.559 vs +0.556)
    # and flipped until the medians AGREED (now +0.545 vs +0.556). But TWO inversions are in play --
    # the exit-vector convention, AND the fact that the electrophile-bearing half is GENERATED in
    # Phase A and RETAINED in Phase B. Two inversions cancel, so agreeing medians are NOT evidence
    # the channels mean the same thing. The flip moved the convention AWAY from the spec while
    # making the summary statistic look right.
    # Worse: the comment this replaces NAMED the second inversion explicitly and then used median
    # agreement as the acceptance test anyway. And it validated against THIS FILE's own docstring
    # rather than reach_envelope's -- letting the artifact define the convention it must satisfy.
    #
    # NOT CHANGING THE SIGN NOW, deliberately: phaseB.csv and all 13 Phase B checkpoints were built
    # from this convention, and flipping it would silently invalidate every one of them while a
    # comparison is mid-flight. The arm comparisons are UNAFFECTED -- geom_mlp is never transferred
    # by the warm start and both arms consume the same conditioning, so the error is common-mode.
    # What IS invalid is any comparison of Phase B geometry against Phase A or against
    # reach_envelope's requirement, and any "steer Phase B by pose" claim. Re-derive before using.
    exitv = np.mean(pts, axis=0) - A
    # A ring attachment atom with two neighbours across the ring gives a mean that nearly
    # coincides with A: short and direction-noisy. 1e-6 was far too permissive.
    if np.linalg.norm(exitv) < 0.5:
        return None, None, 'degenerate_exit_vector'
    return A, exitv, None



def _announce_build_classifier_version():
    """A BUILDER that does not announce its labeller is how a corpus-version boundary is created.

    #142 pinned the SCORERS and left the BUILDERS silent -- and the builders are the half that can
    actually create the mismatch. Re-run this file today and it labels the Phase B row set with
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
    ap.add_argument('--out', default='experiments/covalentformer/data/phaseB')
    a = ap.parse_args()
    # SOURCE CHANGED from the 141-entry P rung to the full cysteine set in CovInDB's complex table.
    # The 141 was a curated subset from an earlier cofold campaign that I inherited without
    # questioning; the database actually holds 1,538 cysteine complexes over 1,479 PDB entries --
    # ten times what the plan had been built around, and enough to make the episodic (MAML) arm
    # viable at 107 targets with >=3 structures.
    import pandas as pd
    cc = pd.read_csv('data/covbinder/raw_covindb2/Covalent_Complex_Records.csv', low_memory=False)
    cc = cc[cc.Resi_name == 'CYS']
    # THE SPLIT COLUMN'S ONLY SOURCE. `target` is what train_phaseB splits on, and it came from a
    # file in VOLATILE /tmp with a silent `{}` default. If /tmp were cleared and phaseB.csv rebuilt,
    # every row would get target='UNK', all 1,332 rows would collapse into ONE target, and
    # train_phaseB's split would degenerate: targets=['UNK'], n_val=max(2,0)=2, val_t={'UNK'},
    # tr_t=set() -> tr_rows EMPTY -> the training loop runs zero batches and reports
    # `tot/max(n,1)` = 0.0000 train loss. The `if not va_rows` guard does NOT catch it, because
    # va_rows is full. A silent fake result from a bare os.path.exists default.
    # Now: prefer the in-repo copy, fall back to /tmp, and FAIL LOUDLY if neither exists.
    _tm_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'phaseB', 'pdb_target.json')
    _tm_path = _tm_repo if os.path.exists(_tm_repo) else '/tmp/pdb_target.json'
    if not os.path.exists(_tm_path):
        raise SystemExit(
            'FATAL: no pdb_target.json at %s or /tmp. That file is the ONLY source of the `target`\n'
            'column, which is what train_phaseB splits on. Without it every row becomes UNK, the\n'
            'split collapses to a single target, and training silently reports 0.0000 loss.'
            % _tm_repo)
    tmap = json.load(open(_tm_path))
    print('  target map: %d pids from %s' % (len(tmap), _tm_path))
    P = {}
    for _i, r in cc.iterrows():
        pid = str(r['PDB']).upper()
        if not os.path.exists(os.path.join(CIF, '%s.cif.gz' % pid)):
            continue
        P[pid] = dict(pid=pid, pre=str(r['SMILES']), cys=str(r['Resi_posi']),
                      cys_ch=str(r['Resi_chain']), lig=str(r['Ligand_name']),
                      target=tmap.get(pid, 'UNK'))
    print('cysteine complexes with a local CIF: %d  over %d distinct targets'
          % (len(P), len({v['target'] for v in P.values()})))

    rows, stats, rejects = [], collections.Counter(), []
    for pid, e in sorted(P.items()):
        pre, cys, ch, lig = e.get('pre'), e.get('cys'), e.get('cys_ch', 'A'), e.get('lig')
        if not all([pre, cys, lig]):
            stats['no_fields'] += 1
            continue
        m = Chem.MolFromSmiles(pre)
        if m is None:
            stats['unparseable'] += 1
            continue
        el, cls = electrophile_index(m)
        if el is None:
            stats['warhead_%s' % cls] += 1
            continue
        try:
            sg, ligatoms, prot = load_struct(pid, cys, ch, lig)
        except Exception:
            stats['cif_unreadable'] += 1
            continue
        if sg is None or len(ligatoms) < 8:
            stats['no_partners'] += 1
            continue

        # embed the pre-reactive SMILES once; the crystal ligand is the post-reaction adduct so its
        # coordinates cannot be used directly, but the POSE-derived quantities we need (which atom is
        # near SG, how far, how occluded) are read from the crystal ligand atoms themselves.
        lig_xyz = np.array([t[1] for t in ligatoms])
        # the attachment point for a WARHEAD-RETAINED cut: split at each BRICS bond, keep the side
        # carrying the electrophile, and take the cut atom as A
        n_emitted = 0
        for (a1, a2), _lab in BRICS.FindBRICSBonds(m):
            b = m.GetBondBetweenAtoms(a1, a2)
            if b is None:
                continue
            try:
                pieces = Chem.GetMolFrags(
                    Chem.FragmentOnBonds(m, [b.GetIdx()], addDummies=True),
                    asMols=True, sanitizeFrags=True)
            except Exception:
                continue
            if len(pieces) != 2:
                continue
            wh = [electrophile_index(p)[0] is not None for p in pieces]
            if sum(wh) != 1:
                continue
            keep, regrow = pieces[wh.index(True)], pieces[1 - wh.index(True)]
            if keep.GetNumHeavyAtoms() - 1 < 8 or regrow.GetNumHeavyAtoms() - 1 < 4:
                continue
            # ATOM CORRESPONDENCE VIA MCS -- replaces a placeholder that was picking the ligand atom
            # FARTHEST FROM SG as the attachment point and the ligand centroid as the exit vector.
            # Neither is a chemical position: "farthest from the sulfur" is wherever the molecule
            # happens to be longest, so d and cos(theta) were being measured from an arbitrary atom.
            # The crystal ligand is the POST-reaction adduct while `pre` is pre-reactive, so a direct
            # template match fails -- but the scaffold is common to both, which is exactly what MCS
            # recovers. This is the same fix that moved an earlier superposition from 8.47 A to
            # 0.40 A when the identical class of bug appeared in decompose_error.py.
            A, exitv, mcs_err = mcs_anchor(m, keep, lig_xyz, ligatoms, sg)
            if A is None:
                stats['mcs_%s' % mcs_err] += 1
                continue
            d = float(np.linalg.norm(sg - A))
            # SANITY CAP. The first run produced d up to 52.4 A, which no linker spans -- that is
            # MCS matching onto a symmetry mate or a second copy of the ligand in the asymmetric
            # unit, not a real attachment point. Real covalent linkers run 2-15 bonds, reaching at
            # most ~15 A extended, so anything past 18 A is a mis-mapped atom and must be dropped
            # rather than fed to the model as a requirement it can never satisfy.
            # FLOOR RAISED 1.5 -> 3.0 A. 52 rows (4.2%) came out below 3.0 and six at 1.73-1.84 A,
            # which is an S-C BOND LENGTH -- the "attachment atom" was sitting where the sulfur's
            # bonded partner is, i.e. mcs_anchor had mapped the ELECTROPHILIC CARBON itself as the
            # attachment point. All 52 were vinylsulfone (33) or michael_sub (19) and none were
            # michael or haloacetamide, so it is warhead-class-systematic, not noise. An attachment
            # point cannot be bonded to SG by definition: that atom is the electrophile.
            if not (3.0 <= d <= 18.0):
                stats['d_out_of_range'] += 1
                # PERSIST THE REJECTS. The claim that all 52 sub-vdW rows were vinylsulfone or
                # michael_sub lives only in a source comment; the pre-floor CSV is gone and there is
                # no stats dump, so by this project's own rule it is NOT validated. Recording them
                # makes it recomputable.
                rejects.append(dict(pid=pid, d=round(d, 3), warhead_class=cls,
                                    target=e.get('target', 'UNK')))
                continue
            v1, v2 = exitv, sg - A
            nv1 = np.linalg.norm(v1)
            if nv1 < 1e-6:
                stats['degenerate_exit'] += 1
                continue
            ct = float(v1.dot(v2) / (nv1 * np.linalg.norm(v2) + 1e-9))
            vf = free_volume(A, sg, prot)
            rows.append(dict(pid=pid, parent=pre, keep=Chem.MolToSmiles(keep),
                             regrow=Chem.MolToSmiles(regrow), warhead_class=cls,
                             d=round(d, 3), cos_theta=round(ct, 4), v_free=round(vf, 3),
                             cys=cys, chain=ch, target=e.get('target', 'UNK')))
            n_emitted += 1
            if n_emitted >= 6:
                break
        stats['entries_used' if n_emitted else 'no_valid_cut'] += 1

    print('rows %d from %d entries | %s' % (len(rows), stats['entries_used'], dict(stats)))
    if not rows:
        print('NOTHING BUILT')
        return 1
    d = np.array([r['d'] for r in rows])
    vf = np.array([r['v_free'] for r in rows])
    ct = np.array([r['cos_theta'] for r in rows])
    print('\nREAL pocket geometry (vs Phase A where v_free was pinned at 1.0):')
    print('  d        median %.2f A   range %.2f-%.2f' % (np.median(d), d.min(), d.max()))
    print('  cos th   median %+.2f     range %+.2f..%+.2f' % (np.median(ct), ct.min(), ct.max()))
    print('  v_free   median %.2f      range %.2f-%.2f   <- REAL occlusion, %d%% fully clear'
          % (np.median(vf), vf.min(), vf.max(), 100 * float((vf == 1.0).mean())))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'phaseB.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('  wrote %s/phaseB.csv' % a.out)
    if rejects:
        with open(os.path.join(a.out, 'rejects_d_out_of_range.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(rejects[0].keys()))
            w.writeheader()
            w.writerows(rejects)
        byc = collections.Counter(r['warhead_class'] for r in rejects)
        print('  wrote rejects_d_out_of_range.csv (%d rows) | by warhead class: %s'
              % (len(rejects), dict(byc)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
