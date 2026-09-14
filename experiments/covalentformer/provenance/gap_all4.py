"""READ-ONLY reproduction of Finding #85's per-role DETERMINISTIC fragment GAP,
plus the same statistic recomputed under v2 role labels.

Mask/atom-mapping logic is taken verbatim from /tmp/frag_gap_det.py (that IS the definition of
the fragment mask). Difference: this dumps the per-row loss under ALL FOUR role tokens, so both
the v1-label grouping (Finding #85) and the v2-label grouping are derivable from one model pass
at identical cost (frag_gap_det.py already did 4 forwards per batch: true + 3 wrong).

Writes ONLY to /tmp/qa29_t4/.
"""
import sys, os, csv, json, collections
import numpy as np
import torch, torch.nn as nn

REPO = '/Users/shaharharel/Documents/github/edit-small-mol'
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, 'experiments/covalentformer'))
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from train_phaseA import PhaseA, subsequent_mask, ROLES, R2I
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from build_roles import classify

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'experiments/covalentformer/ckpt_A_role_strat/ep2.ckpt'
SRC = 'experiments/covalentformer/data/roles/valid_strat.csv'
OUT = '/tmp/qa29_t4/gap_all4_rows.json'
NONATOM = set('()[]=#-+/\\@.%0123456789')


def atom_token(t):
    if t in ('^', '$') or t == '':
        return False
    if t[0] in NONATOM and not t.startswith('['):
        return False
    return True


def frag_positions(target, keep, tok):
    T = Chem.MolFromSmiles(target); K = Chem.MolFromSmiles(keep)
    if T is None or K is None:
        return None
    K = Chem.DeleteSubstructs(K, Chem.MolFromSmarts('[#0]'))
    if K.GetNumAtoms() < 1:
        return None
    try:
        Chem.SanitizeMol(K)
    except Exception:
        return None
    mt = T.GetSubstructMatch(K)
    if not mt:
        return None
    keepatoms = set(mt)
    smi = Chem.MolToSmiles(T)
    if not T.HasProp('_smilesAtomOutputOrder'):
        return None
    order = list(map(int, T.GetProp('_smilesAtomOutputOrder')[1:-1].split(',')))
    toks = tok.tokenize(smi); pos = set(); ai = 0
    for i, t in enumerate(toks):
        if atom_token(t):
            if ai < len(order) and order[ai] not in keepatoms:
                pos.add(i)
            ai += 1
    return (smi, toks, pos) if pos else None


_cache = {}
def v2role(smi):
    if smi not in _cache:
        m = Chem.MolFromSmiles(smi)
        _cache[smi] = None if m is None else classify(m, set(), set())
    return _cache[smi]


torch.set_num_threads(4)
torch.manual_seed(101)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
m = PhaseA(mode=ck.get('mode', 'role'), **dict(ck['network_parameter']))
m.load_state_dict(ck['model_state']); m.eval()
_v = ck['vocabulary']
vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(_v['tokens'])
tok = SMILESTokenizer()
ce = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

rows = [r for r in csv.DictReader(open(SRC)) if r.get('keep')]
buf, skipped = [], 0
for r in rows:
    fp = frag_positions(r['target'], r['keep'], tok)
    if fp is None:
        skipped += 1; continue
    buf.append((r, fp))
print('ckpt=%s ep=%s mode=%s | %d rows mapped (skipped %d) src=%s'
      % (CKPT, ck.get('epoch'), ck.get('mode'), len(buf), skipped, SRC), flush=True)

out = []
B = 24
for i in range(0, len(buf), B):
    ch = buf[i:i + B]
    tt = [c[1][1] for c in ch]
    at = [tok.tokenize(c[0]['anchor']) for c in ch]
    ids = [np.asarray(vocab.encode(t)).astype(np.int64) for t in tt]
    aid = [np.asarray(vocab.encode(t)).astype(np.int64) for t in at]
    K = max(len(x) for x in aid); T = max(len(x) for x in ids)
    src_t = torch.zeros(len(ch), K, dtype=torch.long)
    sm = torch.zeros(len(ch), 1, K, dtype=torch.bool)
    tgt = torch.zeros(len(ch), T, dtype=torch.long)
    fmask = torch.zeros(len(ch), T - 1)
    for j, (a_, t_) in enumerate(zip(aid, ids)):
        src_t[j, :len(a_)] = torch.from_numpy(a_)
        sm[j, 0, :len(a_)] = True
        tgt[j, :len(t_)] = torch.from_numpy(t_)
        for p in ch[j][1][2]:
            if 1 <= p < T:
                fmask[j, p - 1] = 1.0
    ti, to = tgt[:, :-1], tgt[:, 1:]
    msk = subsequent_mask(ti.size(1), 'cpu')
    pad = (to != 0).float(); fm = fmask * pad

    def L(cidx):
        c = torch.full((len(ch),), cidx, dtype=torch.long)
        with torch.no_grad():
            lg = m.generator(m.forward_cond(src_t, sm, ti, msk, c))
        per = ce(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).view(to.shape)
        return (per * fm).sum(1) / fm.sum(1).clamp(min=1)

    allL = [L(k) for k in range(4)]     # loss under EVERY role token
    for j, c in enumerate(ch):
        if float(fm[j].sum()) < 1:
            continue
        r = c[0]
        out.append(dict(role_v1=r['role'],
                        role_v2=v2role(r['frag_to']),
                        role_v2_from=v2role(r['frag_from']),
                        anchor=r['anchor'],
                        fragshare=float(fm[j].sum()) / max(float(pad[j].sum()), 1),
                        loss=[float(allL[k][j]) for k in range(4)]))
    if i % 480 == 0:
        print('  %d/%d' % (i, len(buf)), flush=True)

json.dump(dict(ckpt=CKPT, epoch=ck.get('epoch'), src=SRC, roles=ROLES, r2i=R2I,
               n=len(out), rows=out), open(OUT, 'w'))
print('WROTE %s  n=%d' % (OUT, len(out)), flush=True)


def table(keyname, title):
    acc = collections.defaultdict(lambda: [0., 0., 0, 0.])
    for r in out:
        k = r[keyname]
        if k is None:
            continue
        idx = R2I[k]
        a = acc[k]
        a[0] += r['loss'][idx]
        a[1] += sum(r['loss'][(idx + d) % 4] for d in (1, 2, 3)) / 3.0
        a[2] += 1
        a[3] += r['fragshare']
    print('\n%s' % title)
    print('  %-16s %7s %9s %11s %11s %10s' % ('role', 'n', 'fragshare', 'true', 'ctrl', 'GAP'))
    tot = [0., 0., 0, 0.]
    for r in ROLES:
        a = acc.get(r)
        if not a or a[2] == 0:
            continue
        print('  %-16s %7d %9.3f %11.4f %11.4f %+10.4f'
              % (r, a[2], a[3] / a[2], a[0] / a[2], a[1] / a[2], (a[1] - a[0]) / a[2]))
        for k in range(4):
            tot[k] += a[k]
    print('  %-16s %7d %9.3f %11.4f %11.4f %+10.4f'
          % ('ALL', tot[2], tot[3] / tot[2], tot[0] / tot[2], tot[1] / tot[2],
             (tot[1] - tot[0]) / tot[2]))


table('role_v1', 'A) v1 LABELS (stored `role` column) -- this is Finding #85')
table('role_v2', 'B) v2 LABELS (classify() re-run today on frag_to)')
table('role_v2_from', 'C) v2 LABELS (classify() re-run today on frag_from)')
