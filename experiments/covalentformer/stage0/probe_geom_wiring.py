"""IS THE geom CONDITIONING CHANNEL ACTUALLY REACHING THE LOSS?

WHY THIS EXISTS. Two independent data sources -- the v1 chembl36 splits and the fresh v3 splits --
both returned GAP_perm of +0.0000/+0.0001 from the mode=geom smoke arm. A dead-flat GAP across two
different corpora is the signature of a channel that never influences the forward pass, NOT of a
corpus with no signal. Training longer on a disconnected wire produces the same +0.0000 forever, so
this must be settled BEFORE the arm sweep spends hours on it.

THE MEASUREMENT SEPARATES TWO HYPOTHESES THAT LOOK IDENTICAL IN A TRAINING LOG:
  H-WIRE   the channel is disconnected / degenerate -- an UNTRAINED model's loss is BIT-IDENTICAL
           under d, -d, permuted d and zeroed d. No amount of training can then move GAP.
  H-IGNORE the channel is connected -- the untrained model IS sensitive (random geom_mlp -> random
           prepended memory row -> different logits) -- but training drove the decoder to ignore
           the prepended row, so the TRAINED model's sensitivity collapses toward zero.
The discriminator is the UNTRAINED model. That is the whole point of probing both.

This is a forward-pass measurement only: no optimiser, no sampling, nothing seeded-and-single-draw.
The four conditions are evaluated on THE SAME BATCHES in the same order, so the comparison is exact
rather than noisy -- differences are the channel, not batch composition.

Deltas are printed at 8 decimal places. The training log prints 4, which is why a real-but-tiny
sensitivity and a hard zero were indistinguishable there.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_phaseA import PhaseA, RoleData, collate, PRIOR  # noqa: E402
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer  # noqa: E402


def batch_loss(model, bt, dev, transform):
    """Mean per-token NLL over one batch, with `cond` passed through `transform`."""
    src, sm, tgt, cond, _roles = bt
    src, sm, tgt, cond = src.to(dev), sm.to(dev), tgt.to(dev), cond.to(dev)
    cond = transform(cond)
    tin, tout = tgt[:, :-1], tgt[:, 1:]
    tm = (tout != 0).unsqueeze(1)
    K = tin.size(1)
    causal = torch.triu(torch.ones(K, K, dtype=torch.bool, device=dev), 1).logical_not()
    tmask = (tin != 0).unsqueeze(1) & causal
    with torch.no_grad():
        out = model.forward_cond(src, sm, tin, tmask, cond)
        lp = model.generator(out)
        per = torch.nn.functional.nll_loss(
            lp.reshape(-1, lp.size(-1)), tout.reshape(-1), reduction='none'
        ).view(tout.shape)
        m = (tout != 0).float()
        return float((per * m).sum() / m.sum().clamp(min=1))


def run(model, loader, dev, label, gen):
    """Evaluate all four conditions on the SAME batches, in the same order."""
    ident = lambda c: c                                      # noqa: E731
    neg = lambda c: c * -1.0                                 # noqa: E731
    zero = lambda c: torch.zeros_like(c)                     # noqa: E731

    tot = {k: 0.0 for k in ('base', 'neg', 'perm', 'zero')}
    nb = 0
    for bt in loader:
        if bt is None:
            continue
        # the permutation is drawn from a SEEDED generator so this script replicates exactly
        p = torch.randperm(bt[3].size(0), generator=gen)
        tot['base'] += batch_loss(model, bt, dev, ident)
        tot['neg'] += batch_loss(model, bt, dev, neg)
        tot['perm'] += batch_loss(model, bt, dev, lambda c: c[p.to(c.device)])
        tot['zero'] += batch_loss(model, bt, dev, zero)
        nb += 1
        if nb >= 40:
            break
    for k in tot:
        tot[k] /= max(nb, 1)

    print('\n--- %s  (%d batches) ---' % (label, nb))
    print('  loss(d)          %.8f' % tot['base'])
    print('  loss(-d)         %.8f   GAP_neg  %+.8f' % (tot['neg'], tot['neg'] - tot['base']))
    print('  loss(perm d)     %.8f   GAP_perm %+.8f' % (tot['perm'], tot['perm'] - tot['base']))
    print('  loss(0)          %.8f   GAP_zero %+.8f' % (tot['zero'], tot['zero'] - tot['base']))
    dead = (tot['neg'] == tot['base'] and tot['perm'] == tot['base'] and tot['zero'] == tot['base'])
    print('  BIT-IDENTICAL under all three perturbations: %s' % ('YES' if dead else 'NO'))
    return dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--valid-file', default='valid.csv')
    ap.add_argument('--ckpt', default='', help='trained arm checkpoint; probed in addition to untrained')
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260915)
    a = ap.parse_args()

    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ck = torch.load(a.prior, map_location='cpu', weights_only=False)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    tok = SMILESTokenizer()
    npm = dict(ck['network_parameter'])

    va = RoleData(os.path.join(a.data, a.valid_file), vocab, tok, 'geom')
    print('valid rows: %d   device: %s' % (len(va), dev))

    # WHAT THE INPUT ACTUALLY LOOKS LIKE. cos_theta is a constant 0.0 in this corpus by
    # construction (the adapter has no angle to supply), so the channel varies in ONE dimension.
    # If d were also near-constant the probe would read as dead for a data reason, not a wiring
    # one -- so the spread is printed rather than assumed.
    ds = np.array([float(r.get('d', 0.0)) for r in va.rows])
    cs = np.array([float(r.get('cos_theta', 0.0)) for r in va.rows])
    print('  d:        mean %+.4f  sd %.4f  distinct %d' % (ds.mean(), ds.std(), len(set(ds))))
    print('  cos_theta: mean %+.4f  sd %.4f  distinct %d' % (cs.mean(), cs.std(), len(set(cs))))
    if ds.std() == 0:
        print('  FATAL: d is constant -- the probe cannot distinguish wiring from data.')
        return 2

    def loader():
        return torch.utils.data.DataLoader(va, batch_size=a.bs, shuffle=False, collate_fn=collate)

    m0 = PhaseA(mode='geom', **npm)
    m0.load_state_dict(ck['network_state'], strict=False)
    m0.to(dev).eval()
    g = torch.Generator().manual_seed(a.seed)
    dead0 = run(m0, loader(), dev, 'UNTRAINED (prior + random geom_mlp) -- THE DISCRIMINATOR', g)

    dead1 = None
    if a.ckpt and os.path.exists(a.ckpt):
        ck2 = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        # THE KEY IS `model_state`. My first version tried network_state -> state_dict -> ck2 and
        # fell all the way through to the RAW CHECKPOINT DICT, whose keys are 'epoch', 'seed',
        # 'history'... none of which are parameter names. With strict=False that loads ZERO
        # weights and reports no error, so the probe measured a RANDOMLY INITIALISED model and
        # printed loss 5.60 where the trainer's own log said 0.1363. It still produced a
        # confident-looking four-line table. A silent-no-op load is the exact shape of a bug that
        # yields a plausible wrong number rather than a crash, so the load is now CHECKED.
        st = None
        for k in ('model_state', 'network_state', 'state_dict'):
            if isinstance(ck2.get(k), dict):
                st = ck2[k]
                break
        if st is None:
            print('\n  FATAL: no parameter dict in %s (keys: %s)' % (a.ckpt, list(ck2)[:8]))
            return 2
        # READ THE MODE FROM THE CHECKPOINT. Hardcoding mode='geom' here was the SECOND half of the
        # same bug, and my first fix did not catch it because the guard counted the wrong side.
        # A mode='none' state_dict is a strict SUBSET of a geom model's parameters, so loading one
        # into PhaseA(mode='geom') gives 262/262 matched, 0 unexpected -- a clean bill of health --
        # while leaving geom_mlp.{0,2}.{weight,bias} MISSING and therefore RANDOMLY INITIALISED.
        # A random geom_mlp is demonstrably sensitive to d, so the probe would have reported a live
        # conditioning channel on a model that has none. run_v3_arms.sh is queued to produce exactly
        # such a checkpoint (A0_attach_path, mode=none), so this was about to fire on real data.
        # The guard now counts MISSING on the MODEL side, which is the side that can be random.
        ck_mode = ck2.get('mode')
        if ck_mode not in ('none', 'role', 'geom'):
            print('\n  FATAL: checkpoint does not stamp a usable mode (got %r). A probe that has to'
                  ' GUESS the architecture is how the k=8-evaluated-as-k=1 bug happened.' % ck_mode)
            return 2
        m1 = PhaseA(mode=ck_mode, **npm)
        missing, unexpected = m1.load_state_dict(st, strict=False)
        print('\n  ckpt mode=%s | %d in ckpt, %d MISSING (random), %d unexpected'
              % (ck_mode, len(st), len(missing), len(unexpected)))
        if missing:
            print('  FATAL: %d model tensors are NOT in the checkpoint and would be RANDOM: %s'
                  % (len(missing), list(missing)[:6]))
            return 2
        if ck_mode != 'geom':
            print('  ckpt mode is %r, not geom -- there is no d channel to perturb. A geom-style'
                  ' GAP table on this checkpoint would be meaningless, so it is not computed.'
                  % ck_mode)
            return 0
        m1.to(dev).eval()
        g = torch.Generator().manual_seed(a.seed)
        dead1 = run(m1, loader(), dev, 'TRAINED arm checkpoint', g)

    print('\n=== VERDICT ===')
    if dead0:
        print('H-WIRE: the UNTRAINED model is bit-identical under d/-d/perm/zero. The channel does')
        print('  NOT reach the loss. Every geom GAP on disk is structurally zero and no amount of')
        print('  training or data can move it. Fix the wiring before running any arm.')
    else:
        print('H-WIRE REJECTED: the untrained model IS sensitive to d, so the channel reaches the')
        print('  loss and the architecture is sound.')
        if dead1 is True:
            print('H-IGNORE: the TRAINED model is bit-identical -- training collapsed the channel.')
        elif dead1 is False:
            print('  The trained model is also sensitive; compare the two magnitudes above. A GAP')
            print('  that is small but non-zero is a WEAK channel, which is a result about the')
            print('  data, not a bug. Quote it against GAP_perm, never against zero (#193).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
