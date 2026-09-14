"""PHASE B: pocket-conditioned generation, with an episodic (Reptile/MAML-style) arm.

WHAT PHASE B ADDS. Every Phase A row is pocket-free: d and cos(theta) are sampled from fragment
conformer envelopes and v_free is pinned at 1.0. Phase B measures the same three channels from REAL
deposited structures -- 1,332 rows over 87 cysteine targets -- so v_free finally varies (median 0.750,
39.6% clear, 3% fully blocked) instead of being a constant.

THE TASK IS INVERTED RELATIVE TO PHASE A, and this matters for the warm start. In Phase A the
electrophile is in the GENERATED half; in Phase B it is RETAINED and the model regrows everything
else. So cos(theta) does not mean the same thing in the two phases even with matching sign
conventions, and a B-from-A transfer must be treated as a hypothesis to test, not an assumption.
That is exactly why B-scratch is run alongside it.

ARMS
  B-scratch   from the Mol2Mol prior                      baseline
  B-from-A    warm start from the Phase A role checkpoint does the pocket-free prior transfer?
  B-maml      episodic, one task per TARGET               the few-shot arm

WHY REPTILE RATHER THAN FULL MAML. Second-order MAML needs to differentiate through the inner loop,
which is expensive and fiddly on a 30M-param decoder. Reptile is first-order: adapt on a task for k
steps, then move the meta-parameters toward the adapted ones. It optimises the same objective to
first order and is the standard practical choice at this scale.

WHY EPISODIC IS THE RIGHT SHAPE HERE. 87 targets, median a handful of structures each -- this is
inherently few-shot. The deployment question is "given a NEW pocket, adapt and generate", so the
train/test split is over TARGETS, not rows: held-out targets are adapted on a small support set and
scored on their query set. A row-level split would leak the pocket and answer a different question.
"""
import os, sys, csv, json, time, math, copy, random, argparse, collections
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from train_phaseA import PhaseA, subsequent_mask
from build_roles import join

PRIOR = 'models/reinvent4_mol2mol_warhead_tokens.prior'

# WHICH DE-LEAK FILTER BUILT THE SPLIT. v1 = target + exact keep (2 clauses). v2 = + isotope-stripped
# keep + parent (4 clauses), from 14:02 today. v3 = + canonicalised isotope-stripped REGROW,
# the only half the model must generate; v1/v2 policed `keep` three ways and it not at all. The two produce DIFFERENT held-out sets -- 167 vs 141
# rows at seed 404 -- so a replay harness that applies the wrong one computes a loss that will not
# match the checkpoint's own recorded value. That is not hypothetical: nulls/replay.py still applies
# v1, and per_channel_permutation.py ASSERTS the replayed loss equals the recorded one, so repointing
# replay.py blindly would fire that assert on every pre-14:02 checkpoint while silently mis-scoring
# every post-14:02 one. Stamping the version is what lets a replay match the checkpoint it is
# replaying instead of guessing. Absence of the key means v1, which is correct for every checkpoint
# written before this line existed.
DELEAK_VERSION = 3


def _strip_iso(s):
    """Drop dummy-atom isotope labels: '[4*]CCO' and '[7*]CCO' are the SAME retained half."""
    import re as _re
    return _re.sub(r'\[\d+\*\]', '[*]', s or '')


_CANON_CACHE = {}
_CANON_FALLBACKS = set()


def _canon_iso(s):
    """Isotope-stripped AND canonicalised fragment key. Canonicalisation adds exactly nothing on
    `keep` (measured 0/141, 0/131, 0/66, 0/162 -- the isotope relabelling is the whole effect
    there), but `regrow` fragments come from a pool written by several different producers, so the
    same fragment can arrive in two different SMILES spellings. Falls back to the stripped string
    when RDKit cannot parse, rather than dropping the row from the guard -- a fragment the parser
    rejects must still be compared, or it slips through the filter unchecked."""
    if s in _CANON_CACHE:
        return _CANON_CACHE[s]
    t = _strip_iso(s)
    m = Chem.MolFromSmiles(t) if t else None
    if m is None:
        # FAIL-OPEN, AND COUNTED. An unparseable fragment falls back to the STRIPPED RAW string,
        # which lives in a different key space from every canonical key -- so the same molecule
        # spelled differently, one parsing and one not, compares UNEQUAL and the leak survives the
        # guard. The docstring used to call this "compared anyway"; it is compared, but against
        # nothing it can match. Measured today: 0 of 1004 regrow and 0 of 1004 keep fragments fail
        # to parse, so the branch never fires and no filed number is affected. It is a trap for the
        # next data file, and a guard whose strictness depends on a data property that nothing
        # reports is the pattern this project keeps being bitten by -- so it reports.
        _CANON_FALLBACKS.add(t)
    out = Chem.MolToSmiles(m) if m is not None else t
    _CANON_CACHE[s] = out
    return out


def canon_fallback_count():
    """How many distinct fragments took the unparseable fallback. Non-zero means the de-leak guard
    is comparing across two key spaces and its strictness is NOT what the clause claims."""
    return len(_CANON_FALLBACKS)


def de_leak_holdout(tr_rows, va_rows, version=DELEAK_VERSION):
    """THE de-leak filter. Imported by every consumer; retyped by none.

    This function exists because the sentence 'the filter now lives in ONE place:
    train_phaseB.de_leak_holdout, imported rather than retyped' was written into a comment in
    maml_adapt_eval.py while the function did not exist and the filter was retyped three lines
    below the sentence denying it. There were FOUR independent copies at that moment
    (train_phaseB, maml_adapt_eval, nulls/replay, nulls/deltageom_t3_deleak). Writing the remedy
    down is not performing it; the comment was the same failure class as hardcoding a conclusion
    into a print statement before seeing the output.

    `version` MUST come from the checkpoint's own `deleak_version` stamp (absent => 1), never be
    hardcoded by the caller: v1 and v2 produce different held-out SETS (167 vs 141 rows at seed
    404), so applying the wrong one yields a loss that cannot be compared with the checkpoint's
    recorded value. Measured cost of getting it wrong: +0.0056 at s404, +0.0041 at s505 -- 13-21%
    of the +0.027..+0.033 transfer effect it would be used to judge.

    Note the asymmetry: this touches ONLY the held-out set. tr_rows is untouched, so two arms that
    differ only in filter version are BIT-IDENTICAL in weights and differ only in scoring. That is
    why a version straddle is fixed by RE-SCORING, never by retraining.
    """
    if version not in (1, 2, 3):
        raise ValueError('unknown deleak version %r (expected 1, 2 or 3)' % (version,))
    tr_tgt = {r['target'] for r in tr_rows}
    tr_keep = {r.get('keep', '') for r in tr_rows}
    out = [r for r in va_rows
           if r['target'] not in tr_tgt
           and r.get('keep', '') not in tr_keep]
    if version >= 2:
        tr_keep_iso = {_strip_iso(r.get('keep', '')) for r in tr_rows}
        tr_parent = {r.get('parent', '') for r in tr_rows if r.get('parent')}
        out = [r for r in out
               if _strip_iso(r.get('keep', '')) not in tr_keep_iso
               and (not r.get('parent') or r.get('parent') not in tr_parent)]
    if version >= 3:
        # v3 POLICES THE GENERATED HALF. v1/v2 have four clauses and THREE of them guard `keep`,
        # which the model receives verbatim inside the anchor and can simply copy. `regrow` -- the
        # only half it must actually invent -- was guarded by nothing, because build_rows dropped
        # the field before the filter ever saw it. Measured on the v2 held-out sets: 7.2% of rows
        # share a `regrow` with training as an exact string, and 28.0% once dummy isotopes are
        # stripped and the fragment is canonicalised (46.6% at seed 505). Same 3.9x raw-string
        # understatement as #125 found for `keep`, one field over.
        # WHAT THIS DOES AND DOES NOT FIX: both arms train on identical rows at a matched seed, so
        # the memorisable fragment is available to both and the leak is expected to cancel in the
        # PAIRED difference. It does not cancel in LEVELS: a held-out loss computed under v1/v2 is,
        # for roughly a quarter of its rows, partly a retrieval score rather than a measure of
        # generalisation to new chemistry.
        tr_regrow_iso = {_canon_iso(r.get('regrow', '')) for r in tr_rows if r.get('regrow')}
        out = [r for r in out
               if not r.get('regrow') or _canon_iso(r['regrow']) not in tr_regrow_iso]
    return out


def resolve_deleak_version(ck):
    """Read the filter version OFF the checkpoint. Absence means v1 -- correct for every checkpoint
    written before the stamp existed."""
    return int(ck.get('deleak_version', 1) or 1)


class GeomEncoder(nn.Module):
    """Encodes g=(d, cos_theta, v_free) into ONE cross-attention memory row.

    WHY THIS EXISTS (#165). #138 measured that the model extracts ~nothing from this channel:
    permuting all three inputs costs -0.000188, i.e. NEGATIVE. #165 then measured the ceiling and
    found the channel is weak but NOT empty -- g predicts the regrow fragment's rotatable-bond
    count at held-out R2 +0.171 (shuffled control +0.025) and its size at ~+0.08, on a
    target-disjoint split. So there is real information that the current encoder does not use.

    THE SUSPECT IS THE ENCODING, NOT THE INJECTION SITE, and the evidence is an asymmetry inside
    this very codebase: Phase A injects a DISCRETE role token via nn.Embedding as one prepended
    memory row and it works (deterministic GAP +0.6564, DEPENDENCE +0.1701). Phase B injects a
    CONTINUOUS 3-vector through Linear(3,64)->ReLU->Linear(64,d) at the SAME site, same shape, same
    prepend -- and it is ignored. Three raw scalars through a linear layer is the textbook
    spectral-bias failure; a learned lookup is not. That also explains why z-scoring helped
    OPTIMISATION but not INFORMATIVENESS (#134, and #164's demotion of it): it was a scaling
    problem, which is what you would expect if the encoder could barely represent the function.

    EVERY MODE'S STATE IS A PARAMETER OR A BUFFER, DELIBERATELY. Bin edges and the RFF projection
    are not learned, so the lazy choice is to keep them in Python and stamp them into the
    checkpoint. Tonight has four instances of that going wrong (cond_repeat evaluated k=8 as k=1;
    gzscore/gmu/gsd evaluated a z-scored model on raw inputs, inflating loss by 43-120x the effect
    under study; the ROLES order; DELEAK_VERSION). register_buffer puts them in state_dict, so
    load_state_dict(strict=True) FAILS LOUDLY on a mismatch instead of silently scoring a different
    architecture. Structural beats conventional: a value that cannot be separated from the weights
    cannot be lost.
    """

    def __init__(self, enc, d_model, n_bins=16, n_rff=32, edges=None, seed=20260914):
        super().__init__()
        self.enc = enc
        self.n_bins = n_bins
        if enc == 'mlp':
            self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, d_model))
        elif enc == 'bin':
            # One embedding table per channel, summed. Structurally the Phase A mechanism.
            self.emb = nn.Embedding(3 * n_bins, d_model)
            if edges is None:
                raise ValueError('bin encoder needs TRAIN quantile edges; passing None would let '
                                 'train and eval bin the same value differently')
            self.register_buffer('edges', torch.as_tensor(edges, dtype=torch.float))
        elif enc == 'rff':
            g = torch.Generator().manual_seed(seed)
            # Fixed random projection -> sin/cos features. The standard spectral-bias fix, keeps
            # the input continuous rather than quantising it.
            self.register_buffer('W', torch.randn(3, n_rff, generator=g) * 1.0)
            self.net = nn.Linear(2 * n_rff, d_model)
        else:
            raise ValueError('unknown cond encoder %r' % (enc,))

    def forward(self, g):
        g = g.float()
        if self.enc == 'mlp':
            return self.net(g)
        if self.enc == 'bin':
            # bucketize per channel, offset into a shared table
            idx = []
            for c in range(3):
                b = torch.bucketize(g[:, c].contiguous(), self.edges[c].contiguous())
                idx.append(b.clamp(0, self.n_bins - 1) + c * self.n_bins)
            return self.emb(torch.stack(idx, 1)).sum(1)
        proj = g @ self.W
        return self.net(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1))


def train_quantile_edges(tr_rows, n_bins=16):
    """Bin edges from TRAIN rows only. Computing them over all rows would leak the held-out
    distribution into the encoder, which is the gstats mistake (#PhaseBData) one module over."""
    g = np.array([[float(r['d']), float(r['cos_theta']), float(r['v_free'])] for r in tr_rows])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.stack([np.quantile(g[:, c], qs) for c in range(3)])


class PhaseBData(torch.utils.data.Dataset):
    """Rows of (anchor, target, geometry). The anchor is the retained half carrying the warhead,
    joined to a DIFFERENT regrow fragment; the target is the real molecule. So the edit is always
    'rewrite the non-reactive half', and the warhead is retained by construction."""

    def __init__(self, rows, vocab, tok, max_len=128, gstats=None):
        self.rows, self.v, self.t, self.max_len = rows, vocab, tok, max_len
        # gstats = (mu, sd) as 3-vectors, or None to feed the raw values (the original behaviour).
        # MUST be computed on TRAIN rows only and passed in -- computing it here over self.rows would
        # silently z-score the held-out set with its OWN statistics, which is a leak and would also
        # make train and eval features live on different scales.
        self.gstats = gstats

    def __len__(self):
        return len(self.rows)

    def enc(self, s):
        return torch.tensor(np.asarray(self.v.encode(self.t.tokenize(s)))
                            .astype(np.int64)[:self.max_len], dtype=torch.long)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            s, t = self.enc(r['anchor']), self.enc(r['target'])
        except Exception:
            return None
        g = torch.tensor([float(r['d']), float(r['cos_theta']), float(r['v_free'])],
                         dtype=torch.float)
        # WHY THIS OPTION EXISTS. The three channels are fed RAW, and their natural scales differ by
        # ~6-8x in standard deviation (measured on the phaseB file: sd d 2.7095, cos_theta 0.4518,
        # v_free 0.3260). The first Linear's input-column norms are equal to within 2% and move 0.1%
        # between ep0 and ep2 -- the network never compensated -- so the per-SD drive is d 7.1428,
        # cos_theta 1.1650, v_free 0.8505. d carries 6.1x the drive of cos_theta and 8.4x of v_free,
        # and the emitted conditioning row collapses to rank 1 (PC1 95.4%, corr(|row|, raw d) 0.9998,
        # mean pairwise cosine 0.984 -- one ray rescaled by d). The decoder receives approximately
        # one scalar, at 3.1x the norm of every encoder memory row.
        # That matters because v_free is the entire stated reason Phase B exists ("v_free finally
        # varies") and it is the channel most suppressed. So "the Phase B channel is inert/redundant"
        # (#83/#91) may be measuring a broken channel rather than a useless one. This flag is how we
        # find out, and it is OFF by default so the contrast is a clean A/B against every number
        # already on the books.
        if self.gstats is not None:
            mu, sd = self.gstats
            g = (g - mu) / sd
        return s, t, g


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    ss, ts, gs = zip(*batch)
    B, Ks, Kt = len(ss), max(len(x) for x in ss), max(len(x) for x in ts)
    src = torch.zeros(B, Ks, dtype=torch.long)
    tgt = torch.zeros(B, Kt, dtype=torch.long)
    sm = torch.zeros(B, 1, Ks, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(ss, ts)):
        src[i, :len(a)] = a
        sm[i, 0, :len(a)] = True
        tgt[i, :len(b)] = b
    return src, sm, tgt, torch.stack(gs)


def build_rows(path, rng):
    """Join each retained half to a foreign regrow fragment to make the anchor."""
    raw = list(csv.DictReader(open(path)))
    pool = sorted({r['regrow'] for r in raw if r.get('regrow')})
    out, fail = [], 0
    for r in raw:
        keep, regrow = r.get('keep'), r.get('regrow')
        if not keep or not regrow:
            continue
        tgt = join(keep, regrow)
        alt = pool[rng.randrange(len(pool))]
        if alt == regrow:
            alt = pool[(pool.index(regrow) + 1) % len(pool)]
        anc = join(keep, alt)
        if not tgt or not anc or anc == tgt:
            fail += 1
            continue
        # MW CAP, as build_roles enforces on Phase A products. Random foreign regrow makes anchors
        # far larger than anything real: p95 792 Da, max 1,919, and 24.4% over 600 -- off-manifold
        # for the Mol2Mol prior that every arm starts from.
        ma = Chem.MolFromSmiles(anc)
        if ma is None or Descriptors.MolWt(ma) > 600.0:
            fail += 1
            continue
        out.append(dict(anchor=anc, target=tgt, d=r['d'], cos_theta=r['cos_theta'],
                        v_free=r['v_free'], target_id=r.get('target', 'UNK'),
                        pid=r.get('pid', ''),
                        # keep AND parent must survive: the de-leak filter needs `keep`, and the
                        # episode rule needs the PARENT LIGAND, not the PDB entry id.
                        keep=keep, parent=r.get('parent', r.get('keep', '')),
                        # CARRY `regrow`. It was dropped here, which is why no de-leak clause ever
                        # checked it: the filter cannot police a field it is never handed. `keep`
                        # is copyable (it sits verbatim in the anchor); `regrow` is the ONLY half
                        # the model must generate, and it was the unguarded one. Adding a key is
                        # inert for the split -- build_rows' rng consumption is unchanged and
                        # PhaseBData reads only anchor/target/d/cos_theta/v_free.
                        regrow=regrow))
    return out, fail


def loss_on(model, batch, dev, ce):
    src, sm, tgt, g = [x.to(dev) for x in batch]
    ti, to = tgt[:, :-1], tgt[:, 1:]
    lg = model.generator(model.forward_cond(src, sm, ti, subsequent_mask(ti.size(1), dev), g))
    m = (to != 0).float()
    per = ce(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).view(to.shape)
    return (per * m).sum() / m.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='experiments/covalentformer/data/phaseB/phaseB.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', choices=['scratch', 'from_a', 'maml'], default='scratch')
    ap.add_argument('--init', default='', help='Phase A checkpoint for the from_a / maml arms')
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    # Reptile
    ap.add_argument('--inner-steps', type=int, default=5)
    ap.add_argument('--inner-lr', type=float, default=1e-4)
    ap.add_argument('--meta-eps', type=float, default=0.3)
    ap.add_argument('--episodes', type=int, default=600)
    ap.add_argument('--support', type=int, default=10)
    ap.add_argument('--inner-bs', type=int, default=3)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=20260914)
    # GEOMETRY ENCODER (#165). mlp = the current one, measured IGNORED by #138. bin = discretise
    # to embeddings, structurally the Phase A mechanism that demonstrably works. rff = random
    # Fourier features, the standard spectral-bias fix that keeps the input continuous.
    ap.add_argument('--cond-enc', choices=['mlp', 'bin', 'rff'], default='mlp')
    ap.add_argument('--n-bins', type=int, default=16)
    ap.add_argument('--n-rff', type=int, default=32)
    ap.add_argument('--zscore', action='store_true',
                    help='z-score (d, cos_theta, v_free) using TRAIN-row statistics. OFF by '
                         'default so the contrast against every filed number is a clean A/B.')
    a = ap.parse_args()
    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = random.Random(a.seed)

    ck = torch.load(a.prior, map_location='cpu', weights_only=False)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    tok = SMILESTokenizer()
    npm = dict(ck['network_parameter'])
    model = PhaseA(mode="geom", **npm)
    # Phase B conditions on THREE channels; PhaseA's 'geom' mode takes two. Widen the first layer.
    d_model = npm['model_dimension']
    # CONDITIONING ENCODER: constructed HERE for mlp/rff, which need nothing from the data.
    # `bin` needs TRAIN-ONLY quantile edges and therefore cannot be built until after the split
    # exists -- it is constructed below, immediately after tr_rows. Building it here referenced
    # tr_rows 69 lines before its definition and raised NameError on every --cond-enc bin launch,
    # i.e. the one arm the GeomEncoder docstring argues is most likely to work could not start.
    # A crash rather than a wrong number, and it announced itself the first time it ran.
    if a.cond_enc == 'mlp':
        model.geom_mlp = GeomEncoder('mlp', d_model)
    elif a.cond_enc == 'rff':
        model.geom_mlp = GeomEncoder('rff', d_model, n_rff=a.n_rff, seed=a.seed)
    print('  cond encoder: %s' % a.cond_enc)
    missing, unexpected = model.load_state_dict(ck['network_state'], strict=False)
    bad = [k for k in missing if not k.startswith('geom_mlp')]
    if bad:
        print('FATAL: prior did not load for %s' % bad[:5])
        return 2
    # Default must be explicit, not absent: a scratch/zerocond arm should SAY it had no init rather
    # than leave the reader to infer it from a missing key (the same reason cond_dim is stamped).
    _init_meta = {'init': None, 'init_mode': None, 'init_epoch': None, 'init_seed': None,
                  'init_train_file': None, 'init_sha256': None}
    if a.arm in ('from_a', 'maml') and a.init:
        src_ck = torch.load(a.init, map_location='cpu', weights_only=False)
        sd = {k: v for k, v in src_ck['model_state'].items()
              if not k.startswith('role_emb') and not k.startswith('geom_mlp')}
        src_mode = src_ck.get('mode', 'UNRECORDED')
        if src_mode != 'role':
            print('  WARNING: warm-start checkpoint reports mode=%r, expected "role". A geom or '
                  'none checkpoint would be accepted silently and is NOT the intended init.'
                  % src_mode)
        m2, _u2 = model.load_state_dict(sd, strict=False)
        bad2 = [k for k in m2 if not k.startswith('geom_mlp')]
        if bad2:
            print('  FATAL: warm start left %d unexpected tensors uninitialised: %s'
                  % (len(bad2), bad2[:5]))
            return 2
        print('  warm-started decoder+encoder from %s (role_emb/geom_mlp NOT transferred: the '
              'Phase A conditioning has a different meaning and a different shape)' % a.init)
        # STAMP THE INIT. Every from_a-family checkpoint written before this line carried BYTE-
        # IDENTICAL metadata -- ckpt_B_from_a/ep0, ckpt_B_froma0_s101/ep0 and ckpt_B_fromA3_s101/ep0
        # all say {'arm':'from_a','mode':'geom','cond_dim':3,'seed':101} and nothing else. So the
        # ONE fact that distinguishes those arms, which Phase A run and which EPOCH of it they
        # started from, survived only in directory names and /tmp logs. That is the whole content of
        # the from_a-vs-froma0 contrast behind #49/#51/#55; an artifact-only reader cannot rebuild
        # it, and per the rule that an unrecomputable metric is not validated, it was not validated.
        # The source epoch in particular is load-bearing: the recovered mapping shows from_a starts
        # from ckpt_A_role/ep2 while froma0 starts from ckpt_A0/ep0, so that pairing is init-EPOCH-
        # mismatched as well as init-MODE-mismatched.
        import hashlib as _hl
        _sha = _hl.sha256(open(a.init, 'rb').read()).hexdigest()
        _init_meta = {'init': os.path.abspath(a.init), 'init_mode': src_mode,
                      'init_epoch': src_ck.get('epoch'), 'init_seed': src_ck.get('seed'),
                      'init_train_file': src_ck.get('train_file'), 'init_sha256': _sha}
        print('  INIT STAMPED: mode=%s epoch=%s seed=%s sha=%s'
              % (src_mode, src_ck.get('epoch'), src_ck.get('seed'), _sha[:16]))
    model.to(dev)

    rows, fail = build_rows(a.data, rng)
    # "join failures" WAS A LIE ABOUT OUR OWN DATA. Measured decomposition of the counter across
    # six seeds: join() returned empty ZERO times; 99.7% is the DELIBERATE MW>600 anchor cap and the
    # remaining 1-6 are anc==tgt (the random foreign fragment reproduced the target). A reader of the
    # old string concludes the join logic fails ~28% of the time. It never fails.
    # Also: the row count is SEED-DEPENDENT (920 at s101 .. 970 at s505, a 5.4% range) because the
    # foreign-fragment draw and the target-disjoint split both consume the rng. "Phase B trains on
    # 960 rows" was the seed-404 value quoted as if it were fixed. Paired arms at a MATCHED seed do
    # share a row set, so within-seed contrasts are safe; cross-seed pooling is over different
    # datasets and loss LEVELS must not be pooled across seeds.
    print('arm=%s device=%s | rows %d (dropped %d: MW>600 cap + degenerate anchor; NOT join errors)'
          % (a.arm, dev, len(rows), fail))
    by_t = collections.defaultdict(list)
    for r in rows:
        by_t[r['target_id']].append(r)
    targets = sorted(by_t)
    rng.shuffle(targets)
    # SPLIT ON TARGET, not on rows: the deployment question is a NEW pocket.
    n_val = max(2, int(0.2 * len(targets)))
    val_t, tr_t = set(targets[:n_val]), set(targets[n_val:])
    tr_rows = [r for r in rows if r['target_id'] in tr_t]
    if a.cond_enc == 'bin':
        # TRAIN-ONLY edges, and this is the first point at which tr_rows exists. Computing
        # them over all rows would leak the held-out geometry distribution into the encoder
        # -- the gstats mistake one module over. The edges travel as a registered buffer, so
        # a checkpoint cannot be scored against different bin boundaries than it was trained
        # with.
        model.geom_mlp = GeomEncoder('bin', d_model, n_bins=a.n_bins,
                                     edges=train_quantile_edges(tr_rows, a.n_bins))
        model.to(dev)
        print('  cond encoder: bin (%d quantile bins/channel, TRAIN-only edges)' % a.n_bins)
    va_rows = [r for r in rows if r['target_id'] in val_t]
    # A TARGET-disjoint split is NOT a molecule-disjoint split. 14 ligands are deposited against
    # more than one protein, so 4 of them straddle the split: 13.9% of held-out rows had their
    # exact target SMILES in training, and 26.4% shared a retained half. Drop those rows -- with
    # only 144 val rows, 20 leaked ones move the held-out loss visibly.
    # THE FILTER COMPARED RAW STRINGS AND SO MISSED 11% OF ITS OWN TARGET. `keep` halves carry dummy
    # attachment isotopes, so a chemically IDENTICAL retained half written [4*]... on one side and
    # [7*]... on the other compares unequal and passes straight through. Measured across six seeds:
    # exact-string `keep` leak 0.00%, but 10.99% (84/764 pooled held-out rows; per seed
    # 26/21/7/3/4/23) once isotopes are stripped. Canonicalising on top adds exactly nothing -- the
    # isotope relabelling IS the entire effect. This is #125's blind spot one level down, in the
    # Phase B split rather than the Phase A one.
    # IT WAS NOT INFLATING ANY RESULT: re-evaluating both arms with those 84 rows removed moves
    # #134 from +0.0099805 to +0.0103585, still 6/6 positive, still p=0.03125 -- the leak was mildly
    # DILUTING the effect, not creating it. Fixed anyway, because it would recur in every future
    # split built by this code.
    # `parent` IS ALSO CHECKED NOW. Parent-ligand disjointness currently measures 0.00%, but it held
    # INCIDENTALLY -- target-disjointness plus the exact-target rule happened to remove it, and
    # nothing in the filter ever looked at `parent`. Incidental correctness is not correctness.
    # NOTE FOR COMPARABILITY: splits cut AFTER this change are not identical to splits cut before it.
    # Every existing ckpt_B* was trained on the pre-fix split.
    # ONE definition, at module scope. See de_leak_holdout's docstring for why this is not inlined.
    n0 = len(va_rows)
    va_rows = de_leak_holdout(tr_rows, va_rows, version=DELEAK_VERSION)
    # Report the drop count as SEED-SPECIFIC, because it is. "the de-leak filter drops 41 rows" was
    # the seed-404 value quoted as if fixed; across six seeds it was 41/47/13/14/20/18 PRE-fix and
    # 67/68/20/17/24/41 POST-fix.
    # AND THE ISOTOPE INCREMENT IS SEED-SPECIFIC TOO. I wrote "+26" into two findings as though it
    # were the size of the isotope leak. It is the seed-404 value; the increments are
    # +26/+21/+7/+3/+4/+23 over those six seeds and reach +30 at seed 101. The arithmetic
    # pre + isotope == post holds at EVERY seed -- that is the real result -- but the magnitude does
    # not generalise, and quoting +26 as "the" isotope leak repeats the mistake this comment exists
    # to warn about, one paragraph above itself.
    # Held-out size after THIS filter ranges 59 (s707) to 147 (s909), a 2.49x spread on 16-17
    # targets. (An earlier version of this comment quoted 62-170 / 2.74x -- those were the PRE-fix
    # sizes, i.e. it described code that no longer existed two lines below it.) So within-seed
    # pairing is safe and cross-seed pooling of held-out LEVELS is not.
    print('  de-leaked held-out: %d -> %d rows (dropped %d at THIS seed: shared target, keep half, '
          'isotope-equivalent keep half, or parent ligand)'
          % (n0, len(va_rows), n0 - len(va_rows)))
    if not va_rows:
        print('  FATAL: de-leak removed EVERY held-out row. That is a filter bug, not a data '
              'property -- every held-out loss would be computed on an empty list.')
        return 2
    print('  TARGET-disjoint split: %d train targets / %d held-out | %d / %d rows'
          % (len(tr_t), len(val_t), len(tr_rows), len(va_rows)))

    # GEOMETRY NORMALISATION STATS -- computed on TRAIN ROWS ONLY, then applied to every split.
    # Fitting these on all rows would leak held-out distribution into training; fitting them
    # per-split would put train and eval features on different scales, which is worse than not
    # normalising at all. Both failure modes are silent, so the stats are stamped into the
    # checkpoint below and any loader must replay them -- an unreplayed normalisation is exactly
    # the cond_repeat class of bug, where the state_dict loads cleanly and the eval is simply wrong.
    gstats = None
    if a.zscore:
        _g = torch.tensor([[float(r['d']), float(r['cos_theta']), float(r['v_free'])]
                           for r in tr_rows], dtype=torch.float)
        _mu, _sd = _g.mean(0), _g.std(0)
        _sd = torch.where(_sd < 1e-6, torch.ones_like(_sd), _sd)  # a constant channel stays constant
        gstats = (_mu, _sd)
        print('  GEOM Z-SCORE ON (fitted on %d TRAIN rows): mu=[%.4f %.4f %.4f] sd=[%.4f %.4f %.4f]'
              % (len(tr_rows), *_mu.tolist(), *_sd.tolist()))
        print('    raw per-SD drive ratio d:cos:v_free was ~6.1 : 1.2 : 0.9 -- after scaling, 1:1:1')
    else:
        print('  GEOM Z-SCORE OFF (raw features) -- baseline arm, matches every number already filed')

    ce = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    os.makedirs(a.out, exist_ok=True)
    t0, hist = time.time(), []

    def eval_loss(rws):
        if not rws:
            return float('nan')
        dl = torch.utils.data.DataLoader(PhaseBData(rws, vocab, tok, gstats=gstats), batch_size=a.bs,
                                         shuffle=False, collate_fn=collate)
        model.eval()
        # TOKEN-WEIGHTED, not a mean of batch means. The old form was `tot += batch_mean;
        # tot/n_batches`, which gives a 1-row final batch the same weight as a 16-row one. On a
        # 106-234 row held-out set that is 7-15 batches, so the tail batch carried up to 14% of the
        # estimate. Measured impact on the from_a-vs-scratch comparison: 0.0003, because both arms
        # share the rows and the batch boundaries so the bias is common-mode and cancels in the
        # pairing. It does NOT cancel for absolute losses, which moved by up to 0.018.
        num, den = 0.0, 0.0
        with torch.no_grad():
            for b in dl:
                if b is None:
                    continue
                src, sm, tgt, g = [x.to(dev) for x in b[:4]]
                ti, to = tgt[:, :-1], tgt[:, 1:]
                lg = model.generator(model.forward_cond(src, sm, ti,
                                                        subsequent_mask(ti.size(1), dev), g))
                per = ce(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).view(to.shape)
                m = (to != 0).float()
                num += float((per * m).sum())
                den += float(m.sum())
        return num / max(den, 1.0)

    if a.arm in ('scratch', 'from_a'):
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        dl = torch.utils.data.DataLoader(PhaseBData(tr_rows, vocab, tok, gstats=gstats), batch_size=a.bs,
                                         shuffle=True, collate_fn=collate, drop_last=True)
        for ep in range(a.epochs):
            model.train()
            tot, n = 0.0, 0
            for b in dl:
                if b is None:
                    continue
                loss = loss_on(model, b, dev, ce)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach())
                n += 1
                # PER-STEP PROGRESS. Without it a slow epoch and a hung process look identical --
                # this cost real time twice tonight before I started checking log mtimes.
                if n % 50 == 0:
                    print('  ep%d %4d/%d loss %.4f (%.1f min)'
                          % (ep, n, len(dl), tot / n, (time.time() - t0) / 60), flush=True)
            v = eval_loss(va_rows)
            print('EPOCH %d  train %.4f | HELD-OUT TARGETS %.4f  [%.1f min]'
                  % (ep, tot / max(n, 1), v, (time.time() - t0) / 60), flush=True)
            hist.append(dict(epoch=ep, train=tot / max(n, 1), val=v))
            torch.save({'model_state': model.state_dict(), 'network_parameter': npm,
                        'vocabulary': ck['vocabulary'], 'epoch': ep, 'arm': a.arm,
                        # mode/cond_dim MUST be recorded. Without them a loader doing the natural
                        # PhaseA(**npm) gets the DEFAULT mode='role' -- a random role embedding and
                        # no geometry channel -- and the guard used in both trainers accepts it.
                        'mode': 'geom', 'cond_dim': 3, 'seed': a.seed,
                        'deleak_version': DELEAK_VERSION,
                        # STAMP THE OPTIMISER SETTINGS. bs and lr change what the artifact
                        # MEANS and were travelling nowhere: two arms launched with different
                        # --bs would look like a matched contrast on disk and nothing could
                        # reveal otherwise. That is the cond_repeat class exactly, and it was
                        # live in the one comparison this file exists to support.
                        'bs': a.bs, 'lr': a.lr, 'epochs': a.epochs,
                        'cond_enc': a.cond_enc, 'n_bins': a.n_bins, 'n_rff': a.n_rff,
                        # THE MAML ARM'S ENTIRE DEFINITION. A maml checkpoint MEANS
                        # inner_steps x episodes x inner_lr x meta_eps x support x inner_bs,
                        # and not one of them travelled. #80 records that the MAML verdict
                        # 'now rests on STEP-COUNT INVARIANCE, not a p-value' -- the step
                        # count is precisely the value that did not travel inside the
                        # artifact the verdict is about, so the invariance claim could not be
                        # re-verified from the checkpoints alone. Two maml checkpoints
                        # trained at different --inner-steps were indistinguishable on disk.
                        'episodes': a.episodes, 'inner_steps': a.inner_steps,
                        'inner_lr': a.inner_lr, 'meta_eps': a.meta_eps,
                        'support': a.support, 'inner_bs': a.inner_bs,
                        'device': str(dev), 'prior': a.prior,
                        # THIS STAMP DOES NOT ENABLE THE LEAK GUARD, AND AN EARLIER VERSION OF
                        # THIS COMMENT CLAIMED IT DID. Stamping train_file moves the scorers from
                        # one fail-open branch ('records no train_file') to a DIFFERENT fail-open
                        # branch ('training file not on disk'), because assert_eval_split resolves
                        # the name relative to the VALID file's directory -- data/roles/ -- where a
                        # phaseB.csv never lives. Verified: the guard emits UNVERIFIABLE either way.
                        # Two further reasons it can never fire on a Phase B checkpoint, both
                        # measured: (i) if the file WERE found, the guard does
                        # {r['anchor'] for r in DictReader(...)} and phaseB.csv has no 'anchor'
                        # column -- it is pid,parent,keep,regrow,warhead_class,d,cos_theta,v_free,
                        # cys,chain,target -- so it raises KeyError; (ii) role_compliance.load
                        # cannot even OPEN a ckpt_B*, failing on geom_mlp.0.weight shape [64,3] vs
                        # PhaseA geom mode's [64,2]. Neither scorer reaches assert_eval_split at all.
                        # The stamp is worth keeping as PROVENANCE -- it records which dataset built
                        # the checkpoint, which nothing else did -- but it is not a guard fix, and
                        # saying it was is the same 'correct change, false sentence in the same
                        # breath' pattern this file already documents twice.
                        'train_file': os.path.basename(a.data), 'data': a.data,
                        'gzscore': bool(a.zscore),
                        'gmu': (gstats[0].tolist() if gstats else None),
                        'gsd': (gstats[1].tolist() if gstats else None),
                        **_init_meta,
                        'history': hist}, os.path.join(a.out, 'ep%d.ckpt' % ep))
    else:
        # REPTILE. One episode = one target. Adapt k steps on that target, then pull the meta
        # weights toward the adapted ones.
        # EPISODES MUST BE PARENT-DISJOINT, NOT JUST TARGET-DISJOINT. Rows inside a target are
        # BRICS cuts of the SAME ligand -- median distinct parent ligands per target is 1 -- so
        # ~49 of the 73 "episodes" are one molecule cut several ways, and the inner loop would
        # memorise a compound rather than adapt to a pocket. Only 24 targets have >=3 distinct
        # parents. Require >=2 distinct parent ligands for a task to count.
        def _parents(t):
            return {r.get('parent', '') for r in by_t[t]}
        tr_list = [t for t in tr_t if len(by_t[t]) >= 2 and len(_parents(t)) >= 2]
        dropped = [t for t in tr_t if len(by_t[t]) >= 2 and len(_parents(t)) < 2]
        print('  episodic tasks: %d usable (>=2 rows AND >=2 distinct parent ligands)' % len(tr_list))
        print('  dropped %d single-ligand pseudo-tasks -- support and query would be the same '
              'compound cut differently' % len(dropped))
        if len(tr_list) < 5:
            print('  FATAL: too few genuine episodes for meta-learning.')
            return 2
        for it in range(a.episodes):
            t = tr_list[rng.randrange(len(tr_list))]
            rws = by_t[t]
            sup = rws if len(rws) <= a.support else [rws[i] for i in
                                                     rng.sample(range(len(rws)), a.support)]
            before = copy.deepcopy(model.state_dict())
            inner = torch.optim.Adam(model.parameters(), lr=a.inner_lr)
            # FRESH MINIBATCH PER INNER STEP. With support=4 and batch_size=min(bs,4) the loader
            # yielded exactly ONE batch and the inner loop re-used it 5 times -- which collapses
            # Reptile's cross-minibatch term E[d(g_i . g_j)] to the gradient of ||g||^2 on a single
            # batch, i.e. ordinary joint training with a 0.3 damping factor. The meta-learning was
            # not happening at all.
            dl = torch.utils.data.DataLoader(PhaseBData(sup, vocab, tok, gstats=gstats),
                                             batch_size=min(a.inner_bs, len(sup)), shuffle=True,
                                             collate_fn=collate)
            model.train()
            it_dl = iter(dl)
            for _s in range(a.inner_steps):
                try:
                    b = next(it_dl)
                except StopIteration:
                    it_dl = iter(dl)
                    b = next(it_dl)
                for b in ([b] if b is not None else []):
                    if b is None:
                        continue
                    loss = loss_on(model, b, dev, ce)
                    inner.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    inner.step()
            after = model.state_dict()
            merged = {k: before[k] + a.meta_eps * (after[k].to(before[k].dtype) - before[k])
                      if before[k].is_floating_point() else after[k] for k in before}
            model.load_state_dict(merged)
            if (it + 1) % 100 == 0:
                v = eval_loss(va_rows)
                print('EPISODE %d  HELD-OUT TARGETS %.4f  [%.1f min]'
                      % (it + 1, v, (time.time() - t0) / 60), flush=True)
                hist.append(dict(episode=it + 1, val=v))
                torch.save({'model_state': model.state_dict(), 'network_parameter': npm,
                            'vocabulary': ck['vocabulary'], 'episode': it + 1, 'arm': a.arm,
                            'mode': 'geom', 'cond_dim': 3, 'seed': a.seed,
                        'deleak_version': DELEAK_VERSION,
                        # STAMP THE OPTIMISER SETTINGS. bs and lr change what the artifact
                        # MEANS and were travelling nowhere: two arms launched with different
                        # --bs would look like a matched contrast on disk and nothing could
                        # reveal otherwise. That is the cond_repeat class exactly, and it was
                        # live in the one comparison this file exists to support.
                        'bs': a.bs, 'lr': a.lr, 'epochs': a.epochs,
                        'cond_enc': a.cond_enc, 'n_bins': a.n_bins, 'n_rff': a.n_rff,
                        # THE MAML ARM'S ENTIRE DEFINITION. A maml checkpoint MEANS
                        # inner_steps x episodes x inner_lr x meta_eps x support x inner_bs,
                        # and not one of them travelled. #80 records that the MAML verdict
                        # 'now rests on STEP-COUNT INVARIANCE, not a p-value' -- the step
                        # count is precisely the value that did not travel inside the
                        # artifact the verdict is about, so the invariance claim could not be
                        # re-verified from the checkpoints alone. Two maml checkpoints
                        # trained at different --inner-steps were indistinguishable on disk.
                        'episodes': a.episodes, 'inner_steps': a.inner_steps,
                        'inner_lr': a.inner_lr, 'meta_eps': a.meta_eps,
                        'support': a.support, 'inner_bs': a.inner_bs,
                        'device': str(dev), 'prior': a.prior,
                        # THIS STAMP DOES NOT ENABLE THE LEAK GUARD, AND AN EARLIER VERSION OF
                        # THIS COMMENT CLAIMED IT DID. Stamping train_file moves the scorers from
                        # one fail-open branch ('records no train_file') to a DIFFERENT fail-open
                        # branch ('training file not on disk'), because assert_eval_split resolves
                        # the name relative to the VALID file's directory -- data/roles/ -- where a
                        # phaseB.csv never lives. Verified: the guard emits UNVERIFIABLE either way.
                        # Two further reasons it can never fire on a Phase B checkpoint, both
                        # measured: (i) if the file WERE found, the guard does
                        # {r['anchor'] for r in DictReader(...)} and phaseB.csv has no 'anchor'
                        # column -- it is pid,parent,keep,regrow,warhead_class,d,cos_theta,v_free,
                        # cys,chain,target -- so it raises KeyError; (ii) role_compliance.load
                        # cannot even OPEN a ckpt_B*, failing on geom_mlp.0.weight shape [64,3] vs
                        # PhaseA geom mode's [64,2]. Neither scorer reaches assert_eval_split at all.
                        # The stamp is worth keeping as PROVENANCE -- it records which dataset built
                        # the checkpoint, which nothing else did -- but it is not a guard fix, and
                        # saying it was is the same 'correct change, false sentence in the same
                        # breath' pattern this file already documents twice.
                        'train_file': os.path.basename(a.data), 'data': a.data,
                            'gzscore': bool(a.zscore),
                            'gmu': (gstats[0].tolist() if gstats else None),
                            'gsd': (gstats[1].tolist() if gstats else None),
                            **_init_meta,
                            'history': hist}, os.path.join(a.out, 'ep%d.ckpt' % (it + 1)))
    json.dump(hist, open(os.path.join(a.out, 'history.json'), 'w'), indent=1)
    print('done in %.1f min' % ((time.time() - t0) / 60))
    return 0


if __name__ == '__main__':
    sys.exit(main())
