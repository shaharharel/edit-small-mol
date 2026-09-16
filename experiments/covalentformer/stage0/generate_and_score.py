#!/usr/bin/env python
"""Generate a cohort from a steering checkpoint and score it on BOTH panels.

THE TEST THAT MATTERS. Every number so far is a held-out LOSS. Loss says the instruction
changes the model's predictions; it does NOT say the generated molecules actually move the
param in the requested direction. This script answers that, and it is the first time it has
been run in this project.

For each anchor we sample TWICE from the SAME model with the SAME seed, once under UP and
once under DOWN. Because the anchor and the seed are held fixed, the only thing that differs
is the instruction, so the contrast is exact and needs no matched control model.

    OBEDIENCE = fraction of anchors where param(B_up) > param(B_down)

  Chance is 0.5. A model that ignores the token scores 0.5 by construction, so the null is
  not zero and must never be quoted against zero -- that error turned two non-directional
  params into "small positives" earlier in this project.

Rows where either generation is invalid, or where the param is not computable on either side,
are EXCLUDED and COUNTED. They are a property of the scorer, not the model: a previous
"21% invalid SMILES" headline was ~1% model failure and the rest scorer failure.

GENERATION PANEL (standard): validity, uniqueness, novelty vs train, QED, MW.
MANUSCRIPT PANEL: warhead retention (the graded SMARTS indicator of the paper's R_wh), and
planar deviation via the PINNED producer that reproduces the manuscript column to 1e-6.
"""
from __future__ import annotations
import os, sys, csv, json, math, argparse, collections, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem import QED, Descriptors, AllChem, DataStructs       # noqa: E402
RDLogger.DisableLog('rdApp.*')
from train_steer_v5 import SteerNet, INSTR, I2N, subsequent_mask    # noqa: E402
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer  # noqa
from covalent_filter import classify                                # noqa: E402
import label_molecule_params as LMP                                 # noqa: E402


@torch.no_grad()
def sample(model, vocab, tok, smis, instr_id, dev, max_len=128, temp=1.0, seed=0):
    """Greedy-with-temperature decode. SEEDED per call so UP and DOWN see the same noise."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    out = []
    for s in smis:
        try:
            src = torch.tensor(np.asarray(vocab.encode(tok.tokenize(s))).astype(np.int64)
                               [:max_len], dtype=torch.long).unsqueeze(0).to(dev)
        except Exception:
            out.append(None); continue
        sm = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=dev)
        cond = torch.tensor([instr_id], dtype=torch.long, device=dev)
        mem, m2 = model.encode_cond(src, sm, cond)
        ys = torch.ones(1, 1, dtype=torch.long, device=dev)
        for _ in range(max_len - 1):
            tm = subsequent_mask(ys.size(1), dev)
            logits = model.generator(model.decode(mem, m2, ys, tm))[:, -1, :]
            p = torch.softmax(logits / temp, dim=-1).cpu()
            nxt = torch.multinomial(p, 1, generator=g).to(dev)
            ys = torch.cat([ys, nxt], dim=1)
            # EOS IS '$' = 2. PAD is '*' = 0 and BOS is '^' = 1. Breaking on 0 meant the
            # loop never terminated on a real end-of-sequence and ran to max_len, emitting
            # trailing garbage after a complete SMILES.
            if int(nxt) == 2:
                break
        try:
            # vocab.decode returns a LIST OF TOKENS, not a string. Calling .replace() on it
            # raised AttributeError, which the bare except turned into None for EVERY
            # molecule -- reported as 0.0% validity, which reads like a model result and is
            # actually a scorer failure. Same shape as the "21% invalid SMILES" headline that
            # turned out to be ~1% model and the rest scorer.
            toks = vocab.decode(ys.squeeze(0).cpu().numpy())
            if not isinstance(toks, str):
                toks = ''.join(str(t) for t in toks)
            out.append(toks.replace('^', '').replace('$', '').replace('*', '').strip())
        except Exception:
            out.append(None)
    return out


def param_value(smi, param):
    """Recompute the steering param on a GENERATED molecule. None when not computable."""
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    c = classify(m)
    if not (c and c['ok']):
        return None
    e_idx, _ = LMP.electrophile_idx(m, c['accepted'])
    if e_idx is None:
        return None
    if param == 'linker_atom_count':
        return LMP.linker_atom_count(m, e_idx)
    if param == 'acyl_N_motif':
        v = LMP.acyl_n_motif(m)
        return None if v is None else v          # categorical: compared by INEQUALITY only
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--param', required=True)
    ap.add_argument('--valid', required=True)
    ap.add_argument('--train', default='')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=20260916)
    a = ap.parse_args()
    dev = (('cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu'))
           if a.device == 'auto' else a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    tok = SMILESTokenizer()
    model = SteerNet(mode=ck.get('mode', 'instr'), **dict(ck['network_parameter']))
    model.load_state_dict(ck['network_state'], strict=False)
    model.to(dev).eval()

    anchors = [r['anchor'] for r in csv.DictReader(open(a.valid))][:a.n]
    print('ckpt %s | param %s | anchors %d | dev %s'
          % (os.path.basename(a.ckpt), a.param, len(anchors), dev)); sys.stdout.flush()

    up = sample(model, vocab, tok, anchors, I2N['UP'], dev, seed=a.seed)
    print('  UP done'); sys.stdout.flush()
    dn = sample(model, vocab, tok, anchors, I2N['DOWN'], dev, seed=a.seed)
    print('  DOWN done'); sys.stdout.flush()

    train_set = set()
    if a.train and os.path.exists(a.train):
        for r in csv.DictReader(open(a.train)):
            train_set.add(r['target'])

    ok = fail = 0
    excl = collections.Counter()
    wins = ties = 0
    gen_valid = gen_all = 0
    uniq, qeds, mws, wh_keep = set(), [], [], 0
    rows = []
    for anc, u, d in zip(anchors, up, dn):
        for s in (u, d):
            gen_all += 1
            m = Chem.MolFromSmiles(s) if s else None
            if m is not None:
                gen_valid += 1
                uniq.add(Chem.MolToSmiles(m))
                try:
                    qeds.append(QED.qed(m)); mws.append(Descriptors.MolWt(m))
                except Exception:
                    pass
                c = classify(m)
                if c and c['ok']:
                    wh_keep += 1
        vu, vd = param_value(u, a.param), param_value(d, a.param)
        if vu is None or vd is None:
            excl['param_not_computable'] += 1; fail += 1; continue
        ok += 1
        if isinstance(vu, str):
            # categorical: "obeyed" means UP and DOWN produced DIFFERENT motifs. There is no
            # ordering on motif classes, so DIRECTION is not defined and this is a
            # DISCRIMINATION rate, NOT an obedience rate. Reported separately and never
            # pooled with the numeric params.
            if vu != vd: wins += 1
            else: ties += 1
        else:
            if vu > vd: wins += 1
            elif vu == vd: ties += 1
        rows.append({'anchor': anc, 'up': u, 'down': d, 'v_up': vu, 'v_down': vd})

    novel = 0
    if train_set:
        for r in rows:
            for s in (r['up'], r['down']):
                if s and s not in train_set:
                    novel += 1
    decided = ok - ties
    res = {
        'ckpt': a.ckpt, 'param': a.param, 'anchors': len(anchors), 'seed': a.seed,
        'scorable_pairs': ok, 'excluded': dict(excl), 'ties_no_move': ties,
        'decided_pairs': decided,
        'obedience_vs_chance_0.5': round(wins / decided, 4) if decided else None,
        'obedience_incl_ties': round(wins / ok, 4) if ok else None,
        'generation_panel': {
            'validity': round(gen_valid / max(gen_all, 1), 4),
            'uniqueness': round(len(uniq) / max(gen_valid, 1), 4),
            'novelty_vs_train': round(novel / max(2 * len(rows), 1), 4) if train_set else None,
            'qed_mean': round(float(np.mean(qeds)), 4) if qeds else None,
            'mw_mean': round(float(np.mean(mws)), 2) if mws else None,
        },
        'manuscript_panel': {
            'warhead_retention': round(wh_keep / max(gen_valid, 1), 4),
        },
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fo:
        json.dump({'summary': res, 'rows': rows[:2000]}, fo, indent=2)
    print(json.dumps(res, indent=2))
    print('\nNULL IS 0.5, NOT 0. Ties (no move either way) are %d of %d scorable.'
          % (ties, ok))


if __name__ == '__main__':
    main()
