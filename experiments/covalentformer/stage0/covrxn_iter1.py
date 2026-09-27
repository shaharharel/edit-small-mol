#!/usr/bin/env python
"""CovRXN iteration 1: build an HONEST split, re-evaluate, and error-analyse.

WHY THIS EXISTS. The previous evaluation did `tail -3000 affinity_pairs.jsonl > eval_aff_holdout.jsonl`,
i.e. the "holdout" WAS the tail of the training file. Every number it produced (validity 51.9%,
exact@10 5.0%, dPIC50 Spearman 0.629) was measured in-sample and has been discarded. Nothing about
CovRXN can be quoted until this is redone.

THE SPLIT. Molecule-disjoint on the HIT: a hit molecule appears in train or in test, never both. Pair-
level splitting is not enough, because the same hit with a different analogue leaks the scaffold and the
warhead. Counterion-stripped canonical SMILES is the key, so a hydrochloride and its free base cannot
land on opposite sides.

THE ERROR ANALYSIS is the point of the iteration. A single validity number does not say what to fix, so
failures are bucketed: unparseable, truncated (hit the length cap), copied the input, valid-but-wrong.
Each bucket implies a different fix, and the next iteration is chosen from the largest one.
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, collections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='affinity_pairs.jsonl')
    ap.add_argument('--r2', default='rung2_pairs.jsonl')
    ap.add_argument('--ckpt', default='out_aff/affinity.pt')
    ap.add_argument('--vocab', default='out_aff/vocab.json')
    ap.add_argument('--test-frac', type=float, default=0.03)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--maxlen', type=int, default=160)
    ap.add_argument('--seed', type=int, default=20260928)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    lfc = rdMolStandardize.LargestFragmentChooser()

    def parent(s):
        m = Chem.MolFromSmiles((s or '').strip())
        if m is None:
            return None
        m = lfc.choose(m)
        return Chem.MolToSmiles(m) if m.GetNumHeavyAtoms() >= 3 else None

    def hit_of(r):
        for k in ('src', 'hit', 'input'):
            if r.get(k):
                return r[k]
        ins = r.get('instruction') or ''
        m = re.search(r'Hit:\s*(\S+)', ins)
        return m.group(1) if m else None

    def tgt_of(r):
        for k in ('tgt', 'analogue'):
            if r.get(k):
                return r[k]
        o = str(r.get('output') or '')
        m = re.search(r'Analogue:\s*(\S+)', o)
        return m.group(1) if m else (o.strip() or None)

    # ---- molecule-disjoint split, keyed on the counterion-stripped hit ----
    import random
    random.seed(a.seed)
    rows = []
    for path in (a.pairs, a.r2):
        if not os.path.exists(path):
            print('WARN missing %s' % path, flush=True); continue
        for l in open(path):
            r = json.loads(l)
            h, t = parent(hit_of(r)), parent(tgt_of(r))
            if h is None or t is None or h == t:
                continue
            r['_h'], r['_t'], r['_src'] = h, t, os.path.basename(path)
            rows.append(r)
    keys = sorted({r['_h'] for r in rows})
    random.shuffle(keys)
    ntest = max(1, int(len(keys) * a.test_frac))
    test_keys = set(keys[:ntest])
    tr = [r for r in rows if r['_h'] not in test_keys]
    te = [r for r in rows if r['_h'] in test_keys]
    # PROVE disjointness rather than assert it
    overlap = {r['_h'] for r in tr} & {r['_h'] for r in te}
    print('rows %d | unique hits %d | train %d | test %d | HIT OVERLAP %d (must be 0)'
          % (len(rows), len(keys), len(tr), len(te), len(overlap)), flush=True)
    if overlap:
        sys.exit('FATAL: split is not molecule-disjoint')
    tgt_tr = {r['_t'] for r in tr}
    leak_t = sum(1 for r in te if r['_t'] in tgt_tr)
    print('test rows whose TARGET also appears as a train target: %d (%.1f%%) '
          '-- reported, not removed: it is the same recall ceiling the LLM arms carry'
          % (leak_t, 100.0 * leak_t / max(len(te), 1)), flush=True)
    os.makedirs('splits', exist_ok=True)
    with open('splits/covrxn_test_disjoint.jsonl', 'w') as f:
        for r in te:
            f.write(json.dumps(r) + '\n')
    with open('splits/covrxn_train_disjoint.jsonl', 'w') as f:
        for r in tr:
            f.write(json.dumps(r) + '\n')
    print('wrote splits/', flush=True)

    # ---- evaluate the existing checkpoint on the honest test set ----
    import torch, torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rxncov2 import RXNcov2, Vocab, PAD, BOS, EOS
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    v = Vocab.load(a.vocab)
    model = RXNcov2(len(v.itos), 512, 6, 8, 2048, rbits=64).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev), strict=False)
    model.eval()
    print('loaded %s on %s' % (a.ckpt, dev), flush=True)

    sample = te[:a.n]
    buckets = collections.Counter()
    nn_ok = 0; exact1 = 0; exactk = 0
    preds, trues = [], []
    B = 16
    for s in range(0, len(sample), B):
        chunk = sample[s:s + B]
        src = torch.full((len(chunk), a.maxlen), PAD, dtype=torch.long)
        for i, r in enumerate(chunk):
            ids = v.enc(r['_h'], a.maxlen)[:a.maxlen]
            src[i, :len(ids)] = torch.tensor(ids)
        src = src.to(dev)
        with torch.no_grad():
            mem, pad, rvec = model.encode(src, None, True)
            mem = mem.repeat_interleave(a.k, 0); pad2 = pad.repeat_interleave(a.k, 0)
            y = torch.full((len(chunk) * a.k, 1), BOS, dtype=torch.long, device=dev)
            done = torch.zeros(len(chunk) * a.k, dtype=torch.bool, device=dev)
            hit_cap = torch.ones(len(chunk) * a.k, dtype=torch.bool, device=dev)
            for _ in range(a.maxlen - 1):
                lg = model.dec_a(y, mem, pad2)[:, -1] / 0.8
                nx = torch.multinomial(F.softmax(lg, -1), 1)
                nx[done] = PAD
                y = torch.cat([y, nx], 1)
                done = done | (nx.squeeze(1) == EOS)
                if done.all():
                    break
            hit_cap = ~done            # never emitted EOS -> truncated at the cap
            aff = model.aff(rvec).squeeze(-1).float().cpu()
        for i, r in enumerate(chunk):
            gens = [v.dec(y[i * a.k + j].tolist()) for j in range(a.k)]
            trunc = [bool(hit_cap[i * a.k + j]) for j in range(a.k)]
            cg = [parent(g) for g in gens]
            ok = [c for c in cg if c]
            ref = r['_t']
            for j, c in enumerate(cg):
                if c is None:
                    buckets['truncated' if trunc[j] else 'unparseable'] += 1
                elif c == r['_h']:
                    buckets['copied_input'] += 1
                elif c == ref:
                    buckets['exact'] += 1
                else:
                    buckets['valid_but_wrong'] += 1
            if ok and ok[0] == ref: exact1 += 1
            if ref in set(ok): exactk += 1
            if ok: nn_ok += 1
            yv = r.get('dpic50', r.get('y'))
            if yv is not None and not (isinstance(yv, float) and math.isnan(yv)):
                preds.append(float(aff[i])); trues.append(float(yv))
    n = len(sample); tot = sum(buckets.values())
    print('\n=== CovRXN on the MOLECULE-DISJOINT test set  n=%d k=%d ===' % (n, a.k), flush=True)
    print('  exact@1 %.2f%%   exact@%d %.2f%%' % (100 * exact1 / n, a.k, 100 * exactk / n), flush=True)
    print('  sample-level buckets (%d samples):' % tot, flush=True)
    for kk, vv in buckets.most_common():
        print('    %-16s %6d  %5.1f%%' % (kk, vv, 100.0 * vv / max(tot, 1)), flush=True)
    if preds:
        import statistics as st
        mu = st.mean(trues)
        mse = sum((p - t) ** 2 for p, t in zip(preds, trues)) / len(preds)
        mse0 = sum((mu - t) ** 2 for t in trues) / len(trues)
        def rk(z):
            o = sorted(range(len(z)), key=lambda i: z[i]); r_ = [0.0] * len(z)
            for i, j in enumerate(o): r_[j] = i
            return r_
        rp, rt = rk(preds), rk(trues)
        mp, mt = st.mean(rp), st.mean(rt)
        num = sum((x - mp) * (yy - mt) for x, yy in zip(rp, rt))
        den = math.sqrt(sum((x - mp) ** 2 for x in rp) * sum((yy - mt) ** 2 for yy in rt))
        rho = num / den if den else float('nan')
        print('  dPIC50 head: n=%d  MSE %.4f  predict-the-mean %.4f  Spearman %.3f  -> %s'
              % (len(preds), mse, mse0, rho,
                 'beats the mean' if mse < mse0 else 'NO BETTER THAN THE MEAN'), flush=True)
    print('\nNEXT ITERATION is chosen from the largest bucket above:', flush=True)
    print('  truncated      -> raise the decode cap / add length supervision', flush=True)
    print('  unparseable    -> constrained decoding or a grammar mask', flush=True)
    print('  copied_input   -> add a copy penalty / dissimilarity term to the loss', flush=True)
    print('  valid_but_wrong-> capacity or conditioning: the model generates chemistry, wrong chemistry', flush=True)
    json.dump({'n': n, 'k': a.k, 'exact1': exact1 / n, 'exactk': exactk / n,
               'buckets': dict(buckets), 'target_leak_frac': leak_t / max(len(te), 1)},
              open('results_covrxn_honest.json', 'w'), indent=2)
    print('ITER1_DONE', flush=True)

if __name__ == '__main__':
    main()
