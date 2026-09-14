"""THE PRE-REGISTERED KILL CRITERION: is the geometry channel alive or decorative?

Registered BEFORE any model existed, and not negotiable after the fact:

  H1 SHUFFLE       permute g across the validation set. The oracle reach rate of generations MUST
                   fall. If it does not, the decoder ignores the channel, the model is LibInvent
                   with extra inputs, and scaling to 10^6 examples is wasted money.

  H2 SWEEP         same anchor, same target bond count, sweep d from 4 to 9 A. Does the SHAPE move
                   -- ortho -> meta -> para? Length tracking is a lookup table and proves nothing;
                   shape tracking AT FIXED LENGTH is the actual result, because that is the part no
                   descriptor can supply.

  H3 BIDIRECTIONAL condition on r=0 and check the model complies by emitting NON-reaching fragments.
                   A model that only ever moves one way is biased, not conditioned. Real conditioning
                   steers in both directions.

WHY REACH RATE AND NOT LIKELIHOOD. Validation loss under shuffled g is the cheap in-training proxy
and is already logged every epoch, but it can move for reasons unrelated to geometry. The claim is
about generated molecules, so the endpoint is computed on generated molecules, scored by the
envelope oracle -- the exact part whose positives are witnessed by a real conformer.

CIRCULARITY, STATED PLAINLY. The model was trained on labels from this oracle, so "our generations
score well on the oracle" is NOT evidence the molecules are good. It is only evidence the
conditioning channel is WIRED UP. That is all H1-H3 claim. Any claim about molecule quality has to
come from a scorer we did not train against -- held-out deposited complexes, or a target-disjoint
DTA model.

THE DESCRIPTOR NULL RUNS FIRST, as everywhere else in this campaign: if a free descriptor separates
the shuffled generations from the real ones, the difference is not geometry.
"""
import os, sys, csv, json, math, argparse, collections
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from train_geom import GeomEncoderDecoder, subsequent_mask
from reach_envelope import envelope
from reachability import electrophile_index

D_TOL, A_TOL = 0.5, 20.0


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    npm = dict(ck['network_parameter'])
    # cond_repeat MUST come from the checkpoint. It is not an nn.Parameter, so constructing with
    # the default k=1 and then calling load_state_dict SUCCEEDS SILENTLY -- the weights load fine
    # and nothing errors, but encode_cond() then builds a memory with 2 conditioning rows for a
    # model whose weights were trained against 16. Every k=8 evaluation would have been measuring
    # a configuration that never existed, with no warning anywhere.
    # EVERY architecture switch stored in the checkpoint must be replayed here. cond_repeat was
    # fixed earlier; geom_rbf was not, and the RBF sweep died on a 65-vs-3 shape mismatch. This
    # one at least FAILED LOUDLY -- unlike cond_repeat, which loaded silently and would have
    # evaluated a k=8 model as k=1. Any future switch added to the constructor must be added here.
    k = int(ck.get('cond_repeat', 1))
    rbf = int(ck.get('geom_rbf', 0))
    m = GeomEncoderDecoder(cond_repeat=k, geom_rbf=rbf, **npm)
    m.load_state_dict(ck['model_state'])
    m.to(device).eval()
    if k != 1 or rbf:
        print('  loaded with cond_repeat=%d geom_rbf=%d (read from checkpoint)' % (k, rbf))
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    return m, vocab, SMILESTokenizer(), ck.get('epoch', -1)


@torch.no_grad()
def sample(model, vocab, tok, anchors, geoms, labels, device, max_len=128, temp=1.0):
    """Greedy-with-temperature autoregressive sampling. -> list of SMILES strings."""
    B = len(anchors)
    enc = [vocab.encode(tok.tokenize(s)) for s in anchors]
    Ks = max(len(e) for e in enc)
    src = torch.zeros(B, Ks, dtype=torch.long, device=device)
    src_mask = torch.zeros(B, 1, Ks, dtype=torch.bool, device=device)
    for i, e in enumerate(enc):
        src[i, :len(e)] = torch.tensor(e, dtype=torch.long, device=device)
        src_mask[i, 0, :len(e)] = True
    g = torch.tensor(geoms, dtype=torch.float, device=device)
    r = torch.tensor(labels, dtype=torch.long, device=device)
    mem, m2 = model.encode_cond(src, src_mask, g, r)

    bos = vocab['^'] if '^' in vocab.tokens() else 1
    eos = vocab['$'] if '$' in vocab.tokens() else 2
    ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        out = model.decode(mem, m2, ys, subsequent_mask(ys.size(1), device))
        logits = model.generator(out)[:, -1, :]
        if temp <= 0:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(torch.softmax(logits / temp, dim=-1), 1).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == eos)
        if bool(done.all()):
            break
    out, fails = [], collections.Counter()
    for i in range(B):
        ids = ys[i, 1:].tolist()
        if eos in ids:
            ids = ids[:ids.index(eos)]
        # vocab.decode() returns a LIST OF TOKENS (with '^'/'$' sentinels), NOT a string. Calling
        # .replace() on it raises AttributeError; a bare `except: out.append('')` around this turned
        # every generation into the empty string and the whole test reported 0.0% validity as though
        # the MODEL had failed. untokenize() is the correct inverse and strips the sentinels itself.
        try:
            out.append(tok.untokenize(vocab.decode(np.array(ids))))
        except Exception as exc:
            fails[type(exc).__name__] += 1
            out.append('')
    if fails:
        # Surfaced, not swallowed: a decode failure is a bug in this file, not a model result.
        print('    WARNING: %d/%d generations failed to DECODE (%s) -- this is a harness bug, '
              'not a model failure' % (sum(fails.values()), B, dict(fails)), flush=True)
    return out


def recut(full_smi, scaffold_smi):
    """Re-cut a GENERATED full molecule at the scaffold boundary to recover a [*]-tagged fragment.

    NECESSARY, NOT COSMETIC. envelope() measures d and theta from the ATTACHMENT POINT -- the [*]
    marking where the scaffold was -- to the electrophilic carbon. Generated molecules carry no
    dummy atom, so handing them to envelope() directly returns 'n_attachment_0' for every single
    one and the whole test silently reports n/a rather than failing.

    The scaffold (with its own [*]) is known per row. Strip its dummy to get a substructure query,
    locate it in the generation, and cut the single bond that joins the matched core to the rest.
    The fragment side is returned with a dummy at the cut, which is exactly the input envelope()
    expects and is the same convention the training fragments were built with.

    -> (fragment SMILES with [*], None) or (None, reason)
    """
    m = Chem.MolFromSmiles(full_smi)
    if m is None:
        return None, 'gen_unparseable'
    sc = Chem.MolFromSmiles(scaffold_smi)
    if sc is None:
        return None, 'scaffold_unparseable'
    core = Chem.DeleteSubstructs(sc, Chem.MolFromSmarts('[#0]'))
    try:
        Chem.SanitizeMol(core)
    except Exception:
        pass

    # EXACT MATCH FIRST, MCS AS FALLBACK. Demanding the exact scaffold as a substructure discarded
    # 28 of 40 generations as 'core_absent' -- but that is not a defect in the generations, it is
    # the point of this architecture: unlike LibInvent the model MAY rewrite the core, and often
    # does. Scoring only the generations that happened to leave the core untouched would silently
    # restrict the measurement to the most conservative edits and bias the reach rate.
    # MCS recovers the retained part whatever it is, so partial-core generations stay in the sample.
    inside = set(m.GetSubstructMatch(core))
    if not inside:
        from rdkit.Chem import rdFMCS
        try:
            r = rdFMCS.FindMCS([core, m], timeout=5,
                               atomCompare=rdFMCS.AtomCompare.CompareElements,
                               bondCompare=rdFMCS.BondCompare.CompareAny,
                               ringMatchesRingOnly=True, completeRingsOnly=False)
        except Exception:
            return None, 'mcs_failed'
        if r.numAtoms < 6:
            return None, 'mcs_too_small'
        patt = Chem.MolFromSmarts(r.smartsString)
        inside = set(m.GetSubstructMatch(patt)) if patt is not None else set()
        if not inside:
            return None, 'core_absent'

    el, _cls = electrophile_index(m)
    if el is None:
        return None, 'no_single_warhead'
    if el in inside:
        return None, 'warhead_inside_core'   # nothing was generated at the reactive end

    cut = [b.GetIdx() for b in m.GetBonds()
           if (b.GetBeginAtomIdx() in inside) != (b.GetEndAtomIdx() in inside)]
    if not cut:
        return None, 'no_cut_bond'
    try:
        frag = Chem.FragmentOnBonds(m, cut, addDummies=True)
        pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    except Exception:
        return None, 'fragment_failed'
    # With several attachment points the molecule breaks into >2 pieces. The one we want is
    # unambiguous regardless: the piece carrying the electrophile, with exactly one dummy.
    cands = [p for p in pieces
             if electrophile_index(p)[0] is not None
             and sum(1 for a in p.GetAtoms() if a.GetAtomicNum() == 0) == 1]
    if len(cands) != 1:
        return None, 'ambiguous_side_%d' % len(cands)
    return Chem.MolToSmiles(cands[0]), None


def reaches(full_smi, scaffold_smi, d, theta):
    """Does the generated molecule present its electrophile at the REQUIRED (d, theta)?

    A PASS is witnessed by a real embedded, MMFF-minimised conformer -- not inferred from a protein
    model -- so the error is one-sided: under-sampling can only turn a true positive into a
    negative, never manufacture a false positive.
    """
    frag, err = recut(full_smi, scaffold_smi)
    if frag is None:
        return None, err
    e = envelope(frag, n_conf=60)
    if 'err' in e:
        return None, e['err']
    d_arr, th_arr = np.array(e['d']), np.array(e['theta'])
    hit = bool(np.any((np.abs(d_arr - d) <= D_TOL) & (np.abs(th_arr - theta) <= A_TOL)))
    return hit, None


def descriptor_null(sm_a, sm_b):
    """Can a free descriptor separate two generated populations? Runs BEFORE any conclusion."""
    from sklearn.metrics import roc_auc_score
    def feats(ss):
        out = []
        for s in ss:
            m = Chem.MolFromSmiles(s) if s else None
            if m is None:
                continue
            out.append([m.GetNumHeavyAtoms(), Descriptors.MolWt(m), Descriptors.MolLogP(m),
                        Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m),
                        Descriptors.FractionCSP3(m), Descriptors.RingCount(m)])
        return np.array(out)
    A, B = feats(sm_a), feats(sm_b)
    if len(A) < 20 or len(B) < 20:
        return {}
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    names = ['heavy', 'MW', 'logP', 'TPSA', 'rotB', 'fsp3', 'rings']
    return {n: float(max(roc_auc_score(y, X[:, i]), roc_auc_score(y, -X[:, i])))
            for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--valid',
                    # DEFAULT CHANGED from data/tierA/valid.csv, which is the VOID v1 set:
                    # 99.49% of its anchors have exactly one target, so I(target; g, r | anchor)=0
                    # and any conditioning result scored against it is guaranteed null BY THE DATA.
                    # A run launched without an explicit --valid was silently scoring on a dead
                    # dataset and would have looked like a model failure.
                    default='experiments/covalentformer/data/tierA_v3/valid.csv')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--out', default='experiments/covalentformer/data/shuffle_test.json')
    # MPS is ONE serialising queue on this machine; running an eval on it alongside a training
    # job throttled training from 143 to under 50 steps/min. Evals go on CPU by default here.
    ap.add_argument('--device', default='auto')
    # TORCH MUST BE SEEDED. Every sampling-based number produced tonight (H1, H3, the baselines,
    # best-of-N) seeded numpy but NOT torch, so the candidate pools differed run to run and none of
    # them were replicated. The cost was concrete: best-of-N measured +16.5 points (p=0.0011) on one
    # draw and +4.1 on the next, from the same script and the same numpy seed. Reported CIs assumed
    # the only uncertainty was WITHIN a run; the between-run variance was invisible and larger.
    ap.add_argument('--seed', type=int, default=20260913)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)
    model, vocab, tok, ep = load_model(a.ckpt, dev)
    print('model %s (epoch %d) on %s' % (os.path.basename(a.ckpt), ep, dev))

    rows = [r for r in csv.DictReader(open(a.valid)) if r['label'] == '1']
    rng = np.random.default_rng(20260913)
    idx = rng.choice(len(rows), min(a.n, len(rows)), replace=False)
    rows = [rows[i] for i in idx]
    print('validation anchors (label=1 only): %d' % len(rows))

    anchors = [r['anchor'] for r in rows]
    scaffolds = [r['scaffold'] for r in rows]
    geoms = [[float(r['d']), float(r['cos_theta']), float(r['v_free'])] for r in rows]
    thetas = [float(r['theta']) for r in rows]
    labels = [1] * len(rows)
    perm = rng.permutation(len(rows))
    geoms_sh = [geoms[i] for i in perm]
    # theta MUST be permuted with d. Scoring a shuffled-arm generation against the row's OWN theta
    # while the model was shown a different requirement would mark it wrong for satisfying exactly
    # what it was asked for, manufacturing a drop that has nothing to do with the geometry channel.
    thetas_sh = [thetas[i] for i in perm]

    res = {}
    for arm, G, TH in (('true_g', geoms, thetas), ('shuffled_g', geoms_sh, thetas_sh)):
        gens = []
        for i in range(0, len(rows), a.bs):
            gens += sample(model, vocab, tok, anchors[i:i + a.bs], G[i:i + a.bs],
                           labels[i:i + a.bs], dev, temp=a.temp)
            print('  %s %d/%d' % (arm, min(i + a.bs, len(rows)), len(rows)), flush=True)
        valid = [s for s in gens if s and Chem.MolFromSmiles(s) is not None]
        # Reach is scored against the requirement THE ANCHOR WAS GIVEN -- for the shuffled arm that
        # is the permuted requirement the model was actually shown, not the row's own geometry.
        hits, errs = [], collections.Counter()
        outcome = []          # one entry per anchor, in order: True / False / None(unscorable)
        # THETA-STRATIFIED, because the training negatives are theta-biased: sample_gaps() drew
        # theta uniformly over [0,180] while real envelopes concentrate at LOW theta, so positives
        # sit at a median 58.6 deg and negatives at 96.8 deg (cos_theta predicts the label at
        # AUROC 0.657). High-theta requirements -- the electrophile folding BACK toward the
        # scaffold, which is the hard case -- are therefore under-represented among positives.
        # An aggregate reach rate would hide that: a model competent only on easy forward-pointing
        # geometry would post a respectable average. Report the bands separately.
        band_hits = collections.defaultdict(list)

        def band(t):
            return 'theta<60' if t < 60 else ('theta 60-100' if t < 100 else 'theta>=100')

        for s, gg, th, sc in zip(gens, G, TH, scaffolds):
            if not s or Chem.MolFromSmiles(s) is None:
                errs['invalid'] += 1
                band_hits[band(th)].append(False)
                outcome.append(False)
                continue
            h, e = reaches(s, sc, gg[0], th)
            if h is None:
                errs[e] += 1
                band_hits[band(th)].append(False)   # unscorable == failed to produce a candidate
                outcome.append(False)
            else:
                hits.append(h)
                band_hits[band(th)].append(h)
                outcome.append(bool(h))
        # See baselines.py for why there are two denominators. reach_rate_all counts EVERY
        # generation, treating 'no warhead' and 'unparseable' as failures to produce a covalent
        # candidate rather than quietly dropping them from the sample. Within this file both arms
        # come from the same policy so the scorable fraction should be similar -- if it is NOT,
        # shuffling g changed how often the model emits a usable warhead at all, which is itself a
        # geometry effect and must not be hidden by conditioning on scorability.
        n_all = max(len(gens), 1)
        res[arm] = dict(n=len(gens), n_valid=len(valid),
                        validity=len(valid) / n_all,
                        n_scored=len(hits),
                        reach_rate=(float(np.mean(hits)) if hits else None),
                        reach_rate_all=float(sum(hits)) / n_all,
                        scorable_frac=len(hits) / n_all,
                        errors=dict(errs), gens=gens[:50],
                        # PER-EXAMPLE OUTCOMES, in anchor order, so the arms can be compared PAIRED.
                        # The two arms share the same anchor list by construction, but the first
                        # version of this file saved only aggregates -- which forced an UNPAIRED
                        # two-proportion test on the night's single most important measurement
                        # (+4.3 points, SE 2.8, z=1.54, p=0.123). McNemar on the discordant pairs is
                        # materially more powerful because it conditions away between-anchor
                        # variance, which here is large: anchors differ wildly in how reachable
                        # their gap is. Losing that power on the primary endpoint was a real cost.
                        per_example=[(None if h is None else bool(h)) for h in outcome],
                        by_theta={k: dict(n=len(v), reach=float(np.mean(v)))
                                  for k, v in sorted(band_hits.items())})
        print('  %-11s validity %5.1f%%  scorable %4d/%d (%.0f%%)  reach|scorable %s  REACH|ALL %.1f%%'
              % (arm, 100 * res[arm]['validity'], len(hits), n_all,
                 100 * res[arm]['scorable_frac'],
                 ('%.1f%%' % (100 * res[arm]['reach_rate'])) if hits else 'n/a',
                 100 * res[arm]['reach_rate_all']))
        for k, v in res[arm]['by_theta'].items():
            print('       %-13s reach %5.1f%%  (n=%d)' % (k, 100 * v['reach'], v['n']))

    print('\nDESCRIPTOR NULL (true vs shuffled generations) -- must be ~0.500:')
    nb = descriptor_null(res['true_g']['gens'], res['shuffled_g']['gens'])
    for k, v in sorted(nb.items(), key=lambda kv: -kv[1]):
        print('   %-8s %.3f%s' % (k, v, '  <-- LEAK' if v > 0.60 else ''))
    res['descriptor_null'] = nb

    t, s = res['true_g']['reach_rate'], res['shuffled_g']['reach_rate']
    n_t, n_s = res['true_g']['n_scored'], res['shuffled_g']['n_scored']
    print('\n' + '=' * 64)
    # MINIMUM SAMPLE GUARD. An early run rendered a confident "FAIL -- geometry is decorative" on
    # 2 versus 5 scorable molecules. A verdict at that n is noise wearing a conclusion's clothes,
    # and this campaign has already been burned four times by results that looked decisive and were
    # artifacts. Refuse to render a verdict below MIN_SCORED rather than print one with a caveat.
    MIN_SCORED = 60
    if t is None or s is None or min(n_t, n_s) < MIN_SCORED:
        print('INCONCLUSIVE: scored only %s/%s molecules per arm; need >= %d.'
              % (n_t, n_s, MIN_SCORED))
        print('  NO VERDICT IS RENDERED. Raise --n or fix the re-cut failure modes above.')
        verdict = 'inconclusive'
    else:
        ta, sa = res['true_g']['reach_rate_all'], res['shuffled_g']['reach_rate_all']
        drop = ta - sa
        print('H1 SHUFFLE  (primary endpoint = reach over ALL generations):')
        print('   true g %.1f%%  vs  shuffled g %.1f%%   DROP %+.1f points' % (100 * ta, 100 * sa,
                                                                              100 * drop))
        print('   secondary (over scorable only): %.1f%% vs %.1f%%   [scorable %.0f%% vs %.0f%%]'
              % (100 * t, 100 * s, 100 * res['true_g']['scorable_frac'],
                 100 * res['shuffled_g']['scorable_frac']))
        # PAIRED TEST. Same anchors in the same order, so only the DISCORDANT pairs carry
        # information; between-anchor variance (large here -- anchors differ a lot in how
        # reachable their gap is) is conditioned away.
        pa, pb = res['true_g']['per_example'], res['shuffled_g']['per_example']
        b = sum(1 for x, y in zip(pa, pb) if x and not y)
        c = sum(1 for x, y in zip(pa, pb) if y and not x)
        if b + c:
            chi = (abs(b - c) - 1) ** 2 / (b + c)
            pv = math.erfc(math.sqrt(chi / 2.0))
            print('   McNemar (PAIRED): true-only %d, shuffled-only %d, chi2 %.2f, p = %.4f'
                  % (b, c, chi, pv))
            res['mcnemar'] = dict(b=b, c=c, chi2=chi, p=pv)
        if drop >= 0.10:
            verdict = 'PASS'
            print('  PASS. The decoder uses the geometry channel.')
        else:
            verdict = 'FAIL'
            print('  FAIL. Geometry is decorative -- this is LibInvent with extra inputs.')
            print('  Do NOT scale to 10^6. Fix the conditioning before spending more compute.')
    res['verdict'] = verdict
    json.dump(res, open(a.out, 'w'), indent=1)
    print('  wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
