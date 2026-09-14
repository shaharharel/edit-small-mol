"""*** THIS FILE'S FALSIFIER WAS UNREACHABLE AND THE RUN WAS KILLED. DO NOT USE AS WRITTEN. ***

The falsifier below fires when (non-warhead retention) - (EDIT_WARHEAD retention) < 10 pts, on the
premise that an EDIT_WARHEAD row destroys the acrylamide. IT DOES NOT. EDIT_WARHEAD in this corpus
is a BRICS fragment SWAP -- "edit the fragment that happens to contain the warhead" -- and the
replacement fragment carries its OWN acrylamide. Measured on the ground-truth targets:

    valid_clean  EDIT_WARHEAD n=3417  target still contains an acrylamide:  98.4%
                 EDIT_SCAFFOLD n=464                                       100.0%
                 EDIT_DECORATION n=791                                      93.2%

So a model reproducing the ground truth PERFECTLY scores rr - wr = about -1.0 and takes the
FALSIFIED branch. The printed conclusion was guaranteed by the data structure regardless of the
model -- a hardcoded conclusion in falsifier form, which is the exact failure pattern this project
has been bitten by twice before. The n=16 result (100% retention for all four roles) was therefore
NOT the harness bug it looked like; it is approximately what correct behaviour looks like here.

TO MAKE THIS FILE VALID, the comparison arm must be one where the ground truth actually removes the
electrophile. Two candidates, neither yet built:
  - restrict to EDIT_WARHEAD rows whose frag_to lacks an acrylamide (count first: possibly ~1.6% of
    3,417 rows, i.e. ~55, which would be the entire supply and likely unpowered);
  - change the metric from "acrylamide present" to "the specific frag_from was replaced", which
    measures the edit that was actually requested rather than a property that survives it.

Also fix before rerunning: line `res[req] = dict(agg)` drops Counter's default-0, so the unguarded
res[r]['acryl_kept'] lookups raise KeyError precisely when a role yields zero retaining generations
-- i.e. it crashes exactly in the case the falsifier was watching for.

=== ORIGINAL DOCSTRING BELOW, RETAINED FOR THE RECORD ===

DOES THE ROLE TOKEN CONTROL WARHEAD PRESERVATION? The metric the 6-way eval lost on.

THE STANDING RESULT. In the full manuscript comparison our generator retained the acrylamide in
52.2% of outputs against 94-99% for the baselines. That was measured on the warhead_DIRECTED model,
whose whole job is to rewrite the reactive end -- so it was being scored on the inverse of its
objective. The deployment use case is the opposite: hold the warhead fixed and rewrite everything
else.

THE PREDICTION. Ask for EDIT_SCAFFOLD / EDIT_LINKER / EDIT_DECORATION and the electrophile sits in
the RETAINED half, so retention should approach 100% by construction. That alone proves nothing --
a model that simply never touches an electrophile would score the same.

THE FALSIFIER, WHICH IS THE POINT OF THIS FILE. The SAME anchors are also run with EDIT_WARHEAD
requested. If the token controls warhead preservation, retention must DROP sharply there. If
retention is ~100% whichever role is requested, then the model is not preserving the warhead on
instruction -- it is just never editing warheads, the token is irrelevant to this behaviour, and the
retention number is not evidence of control. Both arms are required; the non-warhead arm on its own
is uninterpretable.

Reported per requested role, with validity and unchanged-rate alongside, because a model that
returns the input verbatim scores 100% retention and is worthless.
"""
import os, sys, csv, json, argparse, collections
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from train_phaseA import PhaseA, subsequent_mask, ROLES, R2I
from reachability import CLASSIFIER_VERSION, CORPUS_LABELLED_WITH
from reachability import electrophile_index

# the manuscript's retention metric is specifically the acrylamide; the 5-class electrophile
# vocabulary is reported next to it so a swap WITHIN the reactive class is not scored as retention
ACRYLAMIDE = Chem.MolFromSmarts('[CX3]=[CX3][CX3](=O)[NX3]')


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


@torch.no_grad()
def sample(model, vocab, tok, anchors, cond, device, max_len=128, temp=0.8):
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
        nxt = (torch.multinomial(torch.softmax(lg / temp, -1), 1).squeeze(-1) if temp > 0
               else lg.argmax(-1))
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
        done = done | (nxt == eos)
        if bool(done.all()):
            break
    out, decode_fail = [], 0
    for i in range(B):
        ids = ys[i, 1:].tolist()
        if eos in ids:
            ids = ids[:ids.index(eos)]
        try:
            out.append(tok.untokenize(vocab.decode(np.array(ids))))
        except Exception:
            # NOT a bare pass emitting '': that exact pattern reported 0% validity as a model
            # failure when vocab.decode was simply returning a token list. Counted and surfaced.
            out.append('')
            decode_fail += 1
    return out, decode_fail


def profile(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    return {'canon': Chem.MolToSmiles(m),
            'acryl': len(m.GetSubstructMatches(ACRYLAMIDE)),
            'elec': electrophile_index(m)[0] is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/covalentformer/ckpt_A_role/ep2.ckpt')
    ap.add_argument('--valid', default='experiments/covalentformer/data/roles/valid_clean.csv')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--bs', type=int, default=40)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    torch.manual_seed(a.seed)                      # unseeded sampling cost this project two headlines
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    # THE SAME TWO GUARDS role_compliance HAS. I added both to role_compliance and to NEITHER of
    # them here -- in the file my own train_phaseA comment explicitly names as the OTHER importer of
    # the mutable ROLES global. Fixing one of two files named in the same sentence is the same shape
    # as sanitising `pooled` and leaving the identical NaN in `within_stratum` one field over.
    # (1) ROLE ORDER. role_emb is a 4x256 Embedding, so reordering ROLES permutes what every index
    #     MEANS while the shape is unchanged: loads clean under strict=True, no warning, and this
    #     file writes per_role / retained_roles_pct straight into its artifact.
    _r = ck.get('roles')
    if _r is None:
        print('  ROLE ORDER UNVERIFIABLE: %s predates the roles stamp. per_role numbers below '
              'assume the CURRENT order %s and cannot be checked.'
              % (os.path.basename(a.ckpt), list(ROLES)))
    elif list(_r) != list(ROLES):
        raise SystemExit('FATAL role-order mismatch: checkpoint trained with %s but this process '
                         'uses %s. Every per-role number would be silently permuted.'
                         % (list(_r), list(ROLES)))
    else:
        # Announce the PASS, same reason as role_compliance: silence on match makes a verified
        # guard indistinguishable from one that never ran.
        print('  role order VERIFIED against checkpoint stamp: %s' % (list(_r),))
    # (2) CLASSIFIER STRADDLE, same reason as role_compliance: this file's retention test calls
    #     electrophile_index, and every corpus on disk was labelled by v1.
    # DELEGATES. This was the copy that made reachability's "retyped by nobody" a false claim:
    # role_compliance was converted and this one was not, so the count went 4 -> 2, not 4 -> 1,
    # while the docstring asserted the refactor was complete. Same shape as #157, three hours
    # later, in the remedy for #157. The text is semantically equivalent so no number moved --
    # but the next edit to the canonical message would silently have skipped this file.
    from reachability import announce_classifier_version
    announce_classifier_version('warhead_retention.py')
    model = PhaseA(mode=ck.get('mode', 'role'), **dict(ck['network_parameter']))
    model.load_state_dict(ck['model_state'])
    model.to(a.device).eval()
    assert_eval_split(ck, a.valid)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(_v['tokens'])
    tok = SMILESTokenizer()
    print('WARHEAD RETENTION BY REQUESTED ROLE  ckpt=%s ep=%s mode=%s seed=%d temp=%.2f'
          % (os.path.basename(os.path.dirname(a.ckpt)), ck.get('epoch'), ck.get('mode'),
             a.seed, a.temp), flush=True)

    rows = [r for r in csv.DictReader(open(a.valid))]
    # only anchors that ALREADY carry an acrylamide can be scored on retaining one
    keep = []
    for r in rows:
        p = profile(r['anchor'])
        if p and p['acryl'] > 0:
            keep.append(r)
    if a.n and len(keep) > a.n:
        keep = [keep[i] for i in rng.choice(len(keep), a.n, replace=False)]
    print('  anchors carrying an acrylamide: %d of %d rows' % (len(keep), len(rows)), flush=True)
    if not keep:
        print('  NOTHING TO SCORE')
        return 1

    anchors = [r['anchor'] for r in keep]
    res = {}
    for req in ROLES:
        agg = collections.Counter()
        fails = 0
        for i in range(0, len(anchors), a.bs):
            ch = anchors[i:i + a.bs]
            c = torch.full((len(ch),), R2I[req], dtype=torch.long)
            gens, df = sample(model, vocab, tok, ch, c, a.device, temp=a.temp)
            fails += df
            for an, g in zip(ch, gens):
                pa, pg = profile(an), profile(g)
                agg['n'] += 1
                if pg is None:
                    agg['invalid'] += 1
                    continue
                agg['valid'] += 1
                if pg['canon'] == pa['canon']:
                    agg['unchanged'] += 1
                if pg['acryl'] > 0:
                    agg['acryl_kept'] += 1
                if pg['elec']:
                    agg['elec_kept'] += 1
        n, v = agg['n'], max(agg['valid'], 1)
        res[req] = dict(agg)
        print('  %-16s n=%4d valid %5.1f%%  unchanged %5.1f%%  ACRYLAMIDE %5.1f%%  '
              'any-electrophile %5.1f%%  decode_fail %d'
              % (req, n, 100 * agg['valid'] / max(n, 1), 100 * agg['unchanged'] / v,
                 100 * agg['acryl_kept'] / v, 100 * agg['elec_kept'] / v, fails), flush=True)

    nonwh = [r for r in ROLES if r != 'EDIT_WARHEAD']
    kept = sum(res[r]['acryl_kept'] for r in nonwh)
    tot = sum(max(res[r]['valid'], 0) for r in nonwh)
    wh = res['EDIT_WARHEAD']
    rr = 100 * kept / max(tot, 1)
    wr = 100 * wh['acryl_kept'] / max(wh['valid'], 1)
    print()
    print('  warhead-RETAINED roles : %.1f%% acrylamide retention (n=%d)' % (rr, tot))
    print('  EDIT_WARHEAD requested : %.1f%% acrylamide retention (n=%d)' % (wr, wh['valid']))
    print('  manuscript baselines   : 94-99%% | our warhead-directed model scored 52.2%%')
    print()
    if rr - wr < 10.0:
        print('  FALSIFIED: retention barely moves with the requested role (%.1f pts). The model is '
              'not preserving the warhead ON INSTRUCTION -- it is either never editing warheads or '
              'never respecting the request. This retention number is NOT evidence of control.'
              % (rr - wr))
    else:
        print('  The token moves retention by %.1f pts, so preservation tracks the INSTRUCTION '
              'rather than being a fixed property of the model.' % (rr - wr))
    print('  NOTE unchanged-rate above: a model echoing its input scores 100%% retention and is '
          'useless. Read retention and unchanged together, never retention alone.')
    if a.out:
        json.dump({'ckpt': a.ckpt, 'epoch': ck.get('epoch'), 'seed': a.seed, 'temp': a.temp,
                   'per_role': res, 'retained_roles_pct': rr, 'warhead_pct': wr},
                  open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
