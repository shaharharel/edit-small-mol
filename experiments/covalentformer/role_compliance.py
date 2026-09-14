"""PHASE A ENDPOINT: does the model edit the part it was TOLD to edit?

THE TEST. Give the model an anchor and a role token, sample, then work out which part of the
molecule actually changed and whether it matches the requested role. Paired against the SAME anchor
with a DIFFERENT role, so the only thing that varies is the instruction.

THE CHANCE BASELINE IS NOT 25%. Two measured facts make the naive baseline wrong:
  - `role` is a deterministic function of the swapped fragment. NOTE the ~36% figure I first wrote
    here was WRONG and understated ambiguity by ~2.8x: enumerating each anchor's own BRICS cuts
    under build_roles' gates shows 99.9% of anchors admit MORE THAN ONE role (3 roles for 94.5%),
    and the true role is always among them. So the anchor does NOT determine the answer -- the
    token is doing real disambiguation work, and ECFP4's 0.885 is reading the anchor's role PRIOR
    rather than a forced answer.
  - ECFP4 of the anchor predicts the role at macro-AUROC 0.884 (0.904 on a leakage-corrected set).
So a model that ignores the token entirely and just reproduces the anchor's most likely edit already
scores far above 25%. The honest baseline is the SHUFFLED-ROLE arm measured here, not 1/4.

THE CONTROL IS PAIRED AND DILUTED, and the dilution is corrected. Permuting a 4-way categorical
leaves ~25% of rows with their original role by chance, which drags the shuffled arm toward the true
arm and understates the effect. Rows where the permutation happened to be a no-op are therefore
EXCLUDED from the paired comparison rather than silently averaged in.

EVALUATE ON valid_clean.csv, NOT valid.csv. 83.5% of valid anchors also appear in train -- one
molecule is cut at several BRICS bonds, so splitting on the retained half does not separate
molecules. valid_clean holds out the anchor, the target and the retained half.
"""
import os, sys, csv, json, math, argparse, collections
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from train_phaseA import PhaseA, subsequent_mask, ROLES, R2I
from build_roles import classify
from reachability import electrophile_index, CLASSIFIER_VERSION, CORPUS_LABELLED_WITH


def assert_eval_split(ck, valid_path, data_dir=None, max_leak=0.05):
    """MEASURE the anchor overlap between the checkpoint's training file and the eval file.

    THE PREVIOUS VERSION WAS A CHECK THAT COULD NOT FAIL WHERE IT MATTERED. It tested one axis --
    whether 'strat' appeared in both filenames -- so train.csv vs valid.csv, which is 82.4%
    ANCHOR-LEAKED, passed and printed "eval split verified". That is the exact configuration this
    module's own header forbids in capitals. A guard that pattern-matches filenames certifies
    whatever the filenames happen to look like; this one reads the rows.

    Measured overlaps, for reference (fraction of eval rows whose ANCHOR appears in train):
        valid.csv   vs train.csv        82.4%      valid_clean.csv vs train.csv         0.0%
        valid.csv   vs train_strat.csv  98.8%      valid_clean.csv vs train_strat.csv  99.8%
        valid_strat vs train.csv        99.7%      valid_strat.csv vs train_strat.csv   0.0%
    Every eval set is clean for exactly ONE training corpus. There is no globally safe default.

    Fails OPEN, loudly, when the checkpoint predates provenance stamping (ckpt_A_role/* carry no
    train_file) or when the training file is not on disk -- those genuinely cannot be checked, and
    saying so is different from saying "verified".
    """
    import csv as _csv
    tf = ck.get('train_file')
    vb = os.path.basename(valid_path)
    if tf is None:
        print('  UNVERIFIABLE: checkpoint records no train_file (written before provenance '
              'stamping). The eval split CANNOT be checked. Scoring %s anyway.' % vb)
        return
    tp = os.path.join(data_dir or os.path.dirname(os.path.abspath(valid_path)), tf)
    if not os.path.exists(tp):
        print('  UNVERIFIABLE: training file %s not on disk; cannot measure overlap.' % tp)
        return
    tr_anchor = {r['anchor'] for r in _csv.DictReader(open(tp))}
    va = list(_csv.DictReader(open(valid_path)))
    leak = sum(1 for r in va if r['anchor'] in tr_anchor) / max(len(va), 1)
    if leak > max_leak:
        raise SystemExit(
            '  FATAL: %.1f%% of %s rows have an anchor that appears in %s (threshold %.0f%%).\n'
            '  These are not mutually held out -- the score would report memorisation as\n'
            '  performance. Pass the eval file matching this checkpoint.'
            % (100 * leak, vb, tf, 100 * max_leak))
    print('  eval split MEASURED: %.1f%% anchor overlap between %s and %s (threshold %.0f%%)'
          % (100 * leak, vb, tf, 100 * max_leak))


def _announce_classifier_version():
    """DELEGATES. This was the 2nd/3rd/4th retyped copy of the same
    announcement -- the identical four-copies shape as de_leak_holdout,
    one constant over. One definition now lives in reachability."""
    from reachability import announce_classifier_version
    announce_classifier_version('role_compliance.py')


def load(ckpt, device):
    _announce_classifier_version()
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    npm = dict(ck['network_parameter'])
    mode = ck.get('mode', 'role')
    # READ THE ROLE ORDER BACK, or stamping it was pointless. A checkpoint whose `roles` list
    # disagrees with this process's ROLES has role_emb indices that mean something different from
    # what the scorer assumes -- shapes match, load_state_dict is silent, and every per-role number
    # comes out permuted with no artifact from which to detect it. Older checkpoints predate the
    # stamp and carry no key; those are UNVERIFIABLE rather than wrong, and say so out loud instead
    # of passing quietly, because a guard that cannot distinguish "checked" from "no data" is the
    # defect this project has now hit four separate times.
    _r = ck.get('roles')
    if _r is None:
        print('  ROLE ORDER UNVERIFIABLE: %s predates the roles stamp. Per-role numbers from it '
              'assume the CURRENT order %s and cannot be checked.' % (os.path.basename(ckpt), ROLES))
    elif list(_r) != list(ROLES):
        raise SystemExit('FATAL role-order mismatch: checkpoint was trained with %s but this '
                         'process uses %s. Every per-role number would be silently permuted.'
                         % (list(_r), list(ROLES)))
    else:
        # ANNOUNCE THE MATCH. The comment above names the defect -- a guard that cannot distinguish
        # "checked" from "no data" -- and then this branch said nothing, so a PASS looked exactly
        # like a guard that never executed. That matters from now on: run 3 is the first checkpoint
        # carrying the stamp, so this is the first time the branch can be reached at all.
        print('  role order VERIFIED against checkpoint stamp: %s' % (list(_r),))
    m = PhaseA(mode=mode, **npm)
    m.load_state_dict(ck['model_state'])
    m.to(device).eval()
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    return m, vocab, SMILESTokenizer(), ck.get('epoch', -1), mode


@torch.no_grad()
def sample(model, vocab, tok, anchors, cond, device, max_len=128, temp=1.0):
    B = len(anchors)
    enc = [np.asarray(vocab.encode(tok.tokenize(s))).astype(np.int64) for s in anchors]
    K = max(len(e) for e in enc)
    src = torch.zeros(B, K, dtype=torch.long, device=device)
    sm = torch.zeros(B, 1, K, dtype=torch.bool, device=device)
    for i, e in enumerate(enc):
        src[i, :len(e)] = torch.from_numpy(e).to(device)
        sm[i, 0, :len(e)] = True
    mem, m2 = model.encode_cond(src, sm, cond.to(device))
    toks = vocab.tokens()
    bos = vocab['^'] if '^' in toks else 1
    eos = vocab['$'] if '$' in toks else 2
    ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        lg = model.generator(model.decode(mem, m2, ys, subsequent_mask(ys.size(1), device)))[:, -1]
        nxt = torch.multinomial(torch.softmax(lg / temp, -1), 1).squeeze(-1) if temp > 0 \
            else lg.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
        done = done | (nxt == eos)
        if bool(done.all()):
            break
    out, n_decode_fail = [], [0]
    for i in range(B):
        ids = ys[i, 1:].tolist()
        if eos in ids:
            ids = ids[:ids.index(eos)]
        try:
            out.append(tok.untokenize(vocab.decode(np.array(ids))))
        except Exception:
            # COUNTED, not swallowed. A bare except emitting '' here is the exact pattern that made
            # vocab.decode's token-list return look like 0% validity, i.e. a model failure, for a
            # whole night. warhead_retention.py counts this; this file did not. A tokenizer or
            # vocab regression must not be able to masquerade as low compliance.
            out.append('')
            n_decode_fail[0] += 1
    if n_decode_fail[0]:
        print('  DECODE FAILURES: %d/%d -- harness bug, NOT model failure. Do not read these as '
              'non-compliance.' % (n_decode_fail[0], B))
    return out


# Why edited_role returned None. Only 'invalid_smiles' is a MODEL failure; the rest are the SCORER
# failing on a perfectly valid molecule. See the ceiling note: on ground-truth pairs this function
# fails to label 59/400 rows (14.8%), dominated by SanitizeMol.
_WHY = collections.Counter()


def edited_role(anchor, gen):
    """Which role does the change from anchor -> gen correspond to?

    The changed part is what the generation has that the MCS with the anchor does not. Returns None
    when the molecule is unparseable, unchanged, or the diff is not a single clean piece.
    """
    # WHY THE CAUSE IS RECORDED. The caller used to report every None from this function as
    # "unparseable N", and that label is WRONG for six of the seven routes below: MCS exception,
    # MCS-too-small, no substructure match, empty keep, SanitizeMol failure on the excised piece,
    # and a multi-fragment diff all return None from a PERFECTLY VALID generated molecule.
    # I read that counter as "the model emits 21% invalid SMILES" and reported it as a model
    # property in two findings. It is mostly the SCORER failing. This file's own ceiling note
    # measures it: on GROUND-TRUTH pairs edited_role fails to label 59/400 rows (14.8%), dominated
    # by SanitizeMol (31/59) -- so a PERFECT model scores ~15% "unparseable" here.
    # _WHY counts the routes so the caller can separate model failure from metric failure.
    a, g = Chem.MolFromSmiles(anchor), Chem.MolFromSmiles(gen) if gen else None
    if a is None or g is None:
        _WHY['invalid_smiles'] += 1
        return None
    if Chem.MolToSmiles(a) == Chem.MolToSmiles(g):
        return 'UNCHANGED'
    try:
        r = rdFMCS.FindMCS([a, g], timeout=5, atomCompare=rdFMCS.AtomCompare.CompareElements,
                           bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True)
    except Exception:
        _WHY['mcs_exception'] += 1
        return None
    if r.numAtoms < 4:
        _WHY['mcs_too_small'] += 1
        return None
    patt = Chem.MolFromSmarts(r.smartsString)
    if patt is None:
        _WHY['no_smarts_match'] += 1
        return None
    keep = set(g.GetSubstructMatch(patt))
    if not keep:
        _WHY['empty_keep'] += 1
        return None
    changed = [i for i in range(g.GetNumAtoms()) if i not in keep]
    if not changed:
        return 'UNCHANGED'
    em = Chem.RWMol(g)
    for i in sorted(keep, reverse=True):
        em.RemoveAtom(i)
    try:
        piece = em.GetMol()
        Chem.SanitizeMol(piece)
    except Exception:
        # THE DOMINANT LOSS. The ceiling note below measures this at 31 of 59 failures on
        # GROUND-TRUTH pairs -- i.e. it fires on molecules that are known-good, and it is the single
        # biggest reason a row is unscorable. Nothing to do with the model's SMILES being invalid.
        _WHY['sanitize_failed'] += 1
        return None
    frs = Chem.GetMolFrags(piece, asMols=True, sanitizeFrags=False)
    if len(frs) != 1:
        _WHY['multifragment_diff'] += 1
        return None
    f = frs[0]
    if electrophile_index(f)[0] is not None:
        return 'EDIT_WARHEAD'
    # OFF-BY-ONE, FIXED. build_roles.classify computes `nh = GetNumHeavyAtoms() - 1` on a fragment
    # that carries a dummy attachment atom -- but RDKit does NOT count a dummy as heavy
    # ([11*]CCCCC and CCCCC both report 5). So the -1 subtracts an atom that was never counted, and
    # the training labels put the DECORATION/LINKER boundary at <=6 real heavy atoms while this
    # harness had it at <=5. Every acyclic non-warhead fragment with EXACTLY 6 heavy atoms was
    # DECORATION in training and LINKER to the scorer: 304 disagreements on valid_strat (9.6%),
    # 29.7% of all DECORATION rows. Any "steers to DECORATION worse than LINKER" claim was an
    # artifact of this line. Matched to build_roles here (the cheap side -- changing build_roles
    # would mean relabelling the whole corpus).
    nh = f.GetNumHeavyAtoms()
    if f.GetRingInfo().NumRings() > 0:
        return 'EDIT_SCAFFOLD'
    return 'EDIT_DECORATION' if nh <= 6 else 'EDIT_LINKER'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--valid', default='experiments/covalentformer/data/roles/valid_clean.csv')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model, vocab, tok, ep, mode = load(a.ckpt, dev)
    assert_eval_split(torch.load(a.ckpt, map_location='cpu', weights_only=False), a.valid)

    rows = list(csv.DictReader(open(a.valid)))
    if a.n and len(rows) > a.n:
        rows = [rows[i] for i in rng.choice(len(rows), a.n, replace=False)]
    anchors = [r['anchor'] for r in rows]
    true_r = [R2I[r['role']] for r in rows]
    perm = rng.permutation(len(rows))
    shuf_r = [true_r[i] for i in perm]
    print('ROLE COMPLIANCE (ckpt epoch %d, mode=%s, seed %d, n=%d, %s)'
          % (ep, mode, a.seed, len(rows), os.path.basename(a.valid)))

    res = {}
    for tag, cond_ids in (('requested', true_r), ('shuffled', shuf_r)):
        gens = []
        for i in range(0, len(rows), a.bs):
            c = torch.tensor(cond_ids[i:i + a.bs], dtype=torch.long)
            gens += sample(model, vocab, tok, anchors[i:i + a.bs], c, dev, temp=a.temp)
        got = [edited_role(an, g) for an, g in zip(anchors, gens)]
        # KEEP THE GENERATIONS. Until now this discarded `gens` the moment `got` was derived, so the
        # only trace of ~2,500 sampled molecules was their ROLE LABEL. That made three things
        # impossible after the fact: recovering WHY a row was unscorable, re-scoring rows with a
        # fixed edited_role, and evaluating the generated cohort on anything at all -- property
        # distributions, novelty, uniqueness, synthesizability. The compliance number was
        # reproducible from disk (that fix landed earlier) while the MOLECULES it was computed from
        # were not, which is a strange place to have stopped.
        res[tag] = dict(got=got, asked=cond_ids, gens=gens)
        n_ok = sum(1 for x in got if x not in (None, 'UNCHANGED'))
        hit = sum(1 for x, r in zip(got, cond_ids) if x == ROLES[r])
        # DENOMINATOR FIXED. This printed `scorable n_ok/len(got)` while dividing by len(got),
        # i.e. it labelled the row with one denominator and computed with another, scoring every
        # unrecoverable row as a model failure. Measured on GROUND-TRUTH pairs, a PERFECT model
        # scores 78.2-78.4% under len(got) and 92.5% under n_ok -- the losses are metric artifacts
        # (SanitizeMol failing on the excised piece, and genuinely disconnected diffs), not model
        # failures, and they are role-dependent (66% ceiling for DECORATION vs 82% for SCAFFOLD).
        # Both denominators are printed now so a raw number can never be read against the wrong one.
        print('  %-10s scorable %4d/%d  compliance %5.1f%% of scorable  (%5.1f%% of all rows)  '
              'unchanged %d  UNSCORABLE %d'
              % (tag, n_ok, len(got), 100 * hit / max(n_ok, 1), 100 * hit / max(len(got), 1),
                 sum(1 for x in got if x == 'UNCHANGED'), sum(1 for x in got if x is None)))
        # RENAMED from "unparseable", which was wrong and which I reported as a model property.
        # Only the invalid_smiles route is the model emitting bad SMILES; the rest are this metric
        # failing on a valid molecule. Ground-truth ceiling: 14.8% unscorable with a PERFECT model.
        print('        unscorable causes: %s   (only invalid_smiles is a MODEL failure)'
              % (dict(_WHY) or '{}'))
        _WHY.clear()

    # PAIRED, with the no-op permutations removed
    ta, sa = res['requested'], res['shuffled']
    keep = [i for i in range(len(rows)) if ta['asked'][i] != sa['asked'][i]]
    b = sum(1 for i in keep if ta['got'][i] == ROLES[ta['asked'][i]]
            and sa['got'][i] != ROLES[sa['asked'][i]])
    c = sum(1 for i in keep if sa['got'][i] == ROLES[sa['asked'][i]]
            and ta['got'][i] != ROLES[ta['asked'][i]])
    chi = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    p = math.erfc(math.sqrt(chi / 2.0)) if (b + c) else 1.0
    print()
    print('  permutation was a no-op on %d/%d rows -- EXCLUDED from the paired test'
          % (len(rows) - len(keep), len(rows)))
    print('  McNemar on %d discordant-eligible rows: requested-only %d, shuffled-only %d, p=%.4f'
          % (len(keep), b, c, p))
    print()
    # The "~64% admit only ONE role" line that stood here was WRONG and printed directly under the
    # headline every run. It was the un-updated complement of the retracted ~36% figure, and it
    # argued the OPPOSITE of what was measured: 99.9% of anchors admit MORE than one role (3 roles
    # for 94.5%), i.e. the anchor almost never determines the answer and the token is doing real
    # disambiguation. Off by ~640x, and the single line most likely to be copied into prose.
    print('  NOTE the baseline: ECFP4 of the anchor alone predicts role at 0.884 AUROC, while 99.9%%')
    print('  of anchors admit MORE than one role -- so the fingerprint is reading a PRIOR, not a')
    print('  forced answer. The shuffled arm above -- not 25%% -- is the honest floor.')
    # CEILING, measured on GROUND-TRUTH pairs: edited_role recovers the true role on only 78.2% of
    # rows (341/400 labelled, 313/400 correct). Dominant loss is SanitizeMol failing after the MCS
    # atoms are deleted (31/59), NOT the MCS-size guard, which never fires. 47 of 59 losses are
    # EDIT_WARHEAD rows, so the per-role ORDERING is distorted, not just the level. Report against
    # n_ok and quote 78.2%% as the ceiling: a raw 60%% is ~77%% of achievable, a different claim.
    if a.out:
        # PERSIST THE PER-ROW PAIRS, not just the McNemar counts. The previous dump held only
        # {epoch, seed, n, b, c, p, n_excluded_noop}, so the two HEADLINE numbers this script exists
        # to produce -- the per-arm compliance rates -- could not be recomputed from any artifact on
        # disk, and neither could the confusion matrix, the per-role breakdown, or the
        # ignore-the-token floor. By this project's own standard a metric that cannot be recomputed
        # from disk is NOT VALIDATED, which made every compliance figure a 35-minute rerun away from
        # being checkable. Storing asked/emitted per row makes all of them derivable offline.
        json.dump({'epoch': ep, 'seed': a.seed, 'n': len(rows), 'b': b, 'c': c, 'p': p,
                   'n_excluded_noop': len(rows) - len(keep),
                   'ckpt': a.ckpt, 'valid': a.valid, 'temp': a.temp, 'mode': mode,
                   'roles': list(ROLES),
                   'rows': [{'anchor': anchors[i],
                             'asked_requested': ROLES[ta['asked'][i]],
                             'asked_shuffled': ROLES[sa['asked'][i]],
                             'emitted_requested': ta['got'][i],
                             'emitted_shuffled': sa['got'][i],
                             # THE GENERATED MOLECULES THEMSELVES. Without these the file records
                             # what the model was ASKED and what CLASS it produced, but not WHAT it
                             # produced -- so the cohort cannot be evaluated on any chemical axis,
                             # and an unscorable row cannot be diagnosed. '' means the decoder
                             # emitted nothing recoverable (counted separately as a DECODE FAILURE).
                             'gen_requested': ta['gens'][i],
                             'gen_shuffled': sa['gens'][i]}
                            for i in range(len(rows))]},
                  open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
