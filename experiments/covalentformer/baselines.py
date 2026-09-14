"""BASELINE ARMS: unconditioned contextual generation, scored on the same endpoint.

THE COMPARISON THE PROJECT ACTUALLY RESTS ON. "Ours beats mol2mol" is table stakes and proves
nothing about geometry: our model sees a pocket requirement and covalent fine-tuning, mol2mol sees
neither, so a win confounds three advantages at once. The arm that isolates geometry is OUR OWN
MODEL WITH SHUFFLED g -- same parameters, same fine-tuning, same information budget, wrong numbers.
That one lives in shuffle_test.py. This file supplies the other two rungs so the ladder is complete:

    mol2mol_base   no covalent FT, no geometry        the floor
    v1_covaFT      covalent FT, no geometry           isolates the fine-tuning contribution
    ours shuffled  covalent FT, WRONG geometry        isolates the geometry contribution
    ours           covalent FT, right geometry        the claim

Read as differences: (v1_covaFT - mol2mol_base) is what covalent fine-tuning buys, and
(ours - ours_shuffled) is what geometry buys. Only the second is the paper's claim.

WHY THESE PRIORS CANNOT BE CONDITIONED. Both are plain Mol2Mol with no geometry channel, so they are
handed the anchor alone. They are then scored against the SAME requirement the conditioned arm was
given. That is deliberately unfair in our favour on the endpoint -- and it is exactly why a win here
is not evidence by itself. The baselines exist to calibrate the scale of the endpoint, not to be
beaten.

SAME ENDPOINT, SAME SCORER, SAME ANCHORS. Every arm is re-cut and scored by the identical envelope
routine, on the identical anchor list, so nothing differs but the policy that wrote the SMILES.
"""
import os, sys, csv, json, argparse, collections
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from train_geom import subsequent_mask
from shuffle_test import reaches, descriptor_null

PRIORS = {
    'mol2mol_base': 'paper/reproducibility/checkpoints/mol2mol_base.prior',
    'v1_covaFT': 'paper/reproducibility/checkpoints/v1_covaFT.prior',
}


def load_prior(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    npm = dict(ck['network_parameter'])
    m = EncoderDecoder(**npm)
    m.load_state_dict(ck['network_state'])
    m.to(device).eval()
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    return m, vocab, SMILESTokenizer()


@torch.no_grad()
def sample_plain(model, vocab, tok, anchors, device, max_len=128, temp=0.8):
    """Unconditioned sampling: memory is the anchor encoding alone, no extra rows."""
    B = len(anchors)
    enc = [np.asarray(vocab.encode(tok.tokenize(s))).astype(np.int64) for s in anchors]
    Ks = max(len(e) for e in enc)
    src = torch.zeros(B, Ks, dtype=torch.long, device=device)
    src_mask = torch.zeros(B, 1, Ks, dtype=torch.bool, device=device)
    for i, e in enumerate(enc):
        src[i, :len(e)] = torch.from_numpy(e).to(device)
        src_mask[i, 0, :len(e)] = True
    mem = model.encode(src, src_mask)

    toks = vocab.tokens()
    bos = vocab['^'] if '^' in toks else 1
    eos = vocab['$'] if '$' in toks else 2
    ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        out = model.decode(mem, src_mask, ys, subsequent_mask(ys.size(1), device))
        logits = model.generator(out)[:, -1, :]
        nxt = (logits.argmax(-1) if temp <= 0 else
               torch.multinomial(torch.softmax(logits / temp, dim=-1), 1).squeeze(-1))
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
        try:
            out.append(tok.untokenize(vocab.decode(np.array(ids))))
        except Exception as exc:
            fails[type(exc).__name__] += 1
            out.append('')
    if fails:
        print('    WARNING: %d/%d failed to DECODE (%s) -- harness bug, not a model result'
              % (sum(fails.values()), B, dict(fails)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--valid', default='experiments/covalentformer/data/tierA/valid.csv')
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--out', default='experiments/covalentformer/data/baselines.json')
    # DEVICE IS EXPLICIT because MPS is a SHARED, SERIALISING resource on this machine. Running
    # this alongside train_geom.py with both on 'mps' stalled training from 143 steps/min to under
    # 50 -- the two processes contend for one GPU queue. Baselines are small and latency-tolerant,
    # so they belong on CPU whenever a training job owns MPS.
    ap.add_argument('--device', default='auto')
    # Default is the literal that used to be hardcoded at the rng line, so the anchor SET is
    # unchanged and prior baseline numbers stay comparable. Exposing it is what makes the
    # REPLICATION possible: a single draw from this script is not a result (#24).
    ap.add_argument('--seed', type=int, default=20260913)
    a = ap.parse_args()
    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)

    rows = [r for r in csv.DictReader(open(a.valid)) if r['label'] == '1']
    # SEED TORCH TOO. numpy was seeded here and torch was not, which is the WORST of the two
    # possible mistakes rather than half of one: the anchor set is reproducible, so two runs print
    # the same "anchors N (seed matched)" line and select identical molecules, while sample_plain's
    # torch.multinomial at temp=0.8 draws a different completion for every one of them. The run
    # LOOKS seeded at exactly the point a reader would check. This script mints the floor rung of
    # the ladder every other number is compared against, and an unseeded floor is why #24's
    # best-of-N effect came out +16.5 on one draw and +4.1 on the next.
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(rows), min(a.n, len(rows)), replace=False)
    rows = [rows[i] for i in idx]
    anchors = [r['anchor'] for r in rows]
    scaffolds = [r['scaffold'] for r in rows]
    ds = [float(r['d']) for r in rows]
    thetas = [float(r['theta']) for r in rows]
    # THIS LINE USED TO ASSERT "seed matched to shuffle_test" UNCONDITIONALLY, at every seed.
    # shuffle_test.py:273 and sweep_test.py:113 still hardcode default_rng(20260913), so the moment
    # --seed is used for the replication it was added to enable, the anchor sets DIVERGE -- measured
    # at ~24% overlap (141/600 at seed 1, 155/600 at 42, 149/600 at 20260914) -- while the log kept
    # claiming they matched. Cross-file rungs of the ladder would then be computed on different
    # molecules under a header promising SAME ANCHORS. Say which case we are actually in.
    _ANCHOR_SEED_PEERS = 20260913  # the literal still hardcoded in shuffle_test/sweep_test
    print('anchors %d on %s | seed %d -- anchor set %s'
          % (len(rows), dev, a.seed,
             'MATCHES shuffle_test/sweep_test' if a.seed == _ANCHOR_SEED_PEERS else
             'DIVERGES from shuffle_test/sweep_test (they hardcode %d); do NOT compare rungs '
             'across files at this seed' % _ANCHOR_SEED_PEERS))

    res = {}
    for name, path in PRIORS.items():
        if not os.path.exists(path):
            print('  SKIP %s -- %s not found' % (name, path))
            continue
        model, vocab, tok = load_prior(path, dev)
        gens = []
        for i in range(0, len(rows), a.bs):
            gens += sample_plain(model, vocab, tok, anchors[i:i + a.bs], dev, temp=a.temp)
            print('  %s %d/%d' % (name, min(i + a.bs, len(rows)), len(rows)), flush=True)
        valid = [s for s in gens if s and Chem.MolFromSmiles(s) is not None]
        hits, errs = [], collections.Counter()
        for s, sc, d, th in zip(gens, scaffolds, ds, thetas):
            if not s or Chem.MolFromSmiles(s) is None:
                errs['invalid'] += 1
                continue
            h, e = reaches(s, sc, d, th)
            if h is None:
                errs[e] += 1
            else:
                hits.append(h)
        # TWO DENOMINATORS, AND THE SECOND IS THE ONE THAT COMPARES ACROSS ARMS.
        # reach_rate divides by the SCORABLE generations, which differs wildly between arms:
        # mol2mol_base is a non-covalent prior (0.4% acrylamide retention in the source paper), so
        # most of its output has no warhead and is unscorable, leaving a denominator of 3 while
        # v1_covaFT leaves 17. Comparing those two rates compares two different populations and
        # would let an arm look good precisely BECAUSE it rarely emits a warhead.
        # reach_rate_all counts every generation, scoring 'no warhead' / 'unparseable' as a failure
        # to produce a covalent candidate -- which is exactly what it is. That is the primary.
        n_all = max(len(gens), 1)
        res[name] = dict(n=len(gens), validity=len(valid) / n_all,
                         n_scored=len(hits),
                         reach_rate=(float(np.mean(hits)) if hits else None),
                         reach_rate_all=float(sum(hits)) / n_all,
                         scorable_frac=len(hits) / n_all,
                         errors=dict(errs), gens=gens[:50])
        print('  %-14s validity %5.1f%%  scorable %4d/%d (%.0f%%)  reach|scorable %s  '
              'REACH|ALL %.1f%%'
              % (name, 100 * res[name]['validity'], len(hits), n_all,
                 100 * res[name]['scorable_frac'],
                 ('%.1f%%' % (100 * res[name]['reach_rate'])) if hits else 'n/a',
                 100 * res[name]['reach_rate_all']))

    if len(res) == 2:
        ks = list(res)
        nb = descriptor_null(res[ks[0]]['gens'], res[ks[1]]['gens'])
        print('\nDESCRIPTOR NULL between baseline arms (context only):')
        for k, v in sorted(nb.items(), key=lambda kv: -kv[1])[:4]:
            print('   %-8s %.3f' % (k, v))
        res['descriptor_null_between_baselines'] = nb

    # STAMP THE RUN CONDITIONS INTO THE ARTIFACT. Without these the JSON cannot be reproduced from
    # its own contents -- which is the same defect I fixed in train_phaseB an hour ago and left
    # standing here. temp is not cosmetic: at 0.8 it materially sets validity (95.0% -> 75.0% was
    # measured across two seeds), so a validity number without its temperature is uninterpretable.
    # anchor_seed_matches_peers records whether this run's anchors coincide with shuffle_test's
    # hardcoded draw, so a reader can tell from the FILE whether cross-file rungs are comparable.
    res['_run'] = dict(seed=a.seed, temp=a.temp, n=a.n, valid=a.valid, device=dev,
                       anchor_seed_matches_peers=bool(a.seed == _ANCHOR_SEED_PEERS))
    json.dump(res, open(a.out, 'w'), indent=1)
    print('\n  wrote %s' % a.out)
    print('  NOTE: these are the floor and the fine-tuning rung. The geometry claim is decided by')
    print('  ours vs ours-with-shuffled-g, not by either of these.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
