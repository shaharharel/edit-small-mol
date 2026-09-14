"""PHASE A TRAINING: role-conditioned covalent editing. Three arms, one script.

    A0       no conditioning at all              CONTROL
    A-role   + role token                        PRIMARY -- matches the deployment use case
    A-geom   + (d, cos theta) reach conditioning TRANSFER TEST only

All three get the covalent chemistry LM for free, because every target is a covalent molecule and
plain token cross-entropy on those strings is the chemistry lesson -- that alone took v1_covaFT from
0.4% to 94.9% acrylamide retention. So any DIFFERENCE between arms is attributable to the
conditioning, not to the chemistry. That is the whole point of running A0.

WHY ROLE AND NOT GEOMETRY IS PRIMARY. In the deployment mode the warhead is RETAINED, so both the
attachment atom and the electrophile sit in the kept half and d(A->E) is fixed before generation
starts. Conditioning on d is meaningless there. The geometry that matters once the warhead is fixed
is pocket-dependent and belongs in Phase B.

CLASS-WEIGHTED LOSS. Role frequencies are naturally skewed (measured 34x between EDIT_WARHEAD and
EDIT_LINKER). Unweighted, the model would learn to ignore the token for rare roles and the channel
would look dead for exactly the reason the geometry channel did. Weights are inverse-frequency,
computed from the training file, and reported.

TORCH IS SEEDED. Unseeded sampling cost this project two headline numbers that did not replicate
(+16.5 -> +4.1, and +4.3 -> +1.5 null), so every run here fixes both numpy and torch seeds.
"""
import os, sys, csv, json, time, math, random, argparse, collections
import numpy as np
import torch
import torch.nn as nn

REINVENT = '/Users/shaharharel/Documents/github/REINVENT4'
sys.path.insert(0, REINVENT)
from reinvent.models.transformer.core.network.encode_decode.model import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer

PRIOR = 'models/reinvent4_mol2mol_warhead_tokens.prior'
ROLES = ['EDIT_WARHEAD', 'EDIT_LINKER', 'EDIT_SCAFFOLD', 'EDIT_DECORATION']
R2I = {r: i for i, r in enumerate(ROLES)}


def subsequent_mask(size, device):
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0)


class PhaseA(EncoderDecoder):
    """Mol2Mol + optional conditioning rows prepended to the cross-attention memory.

    mode='none'  -> identical to the prior; the control arm
    mode='role'  -> one row, a 4-way role embedding
    mode='geom'  -> one row, MLP over (d, cos theta)
    """

    def __init__(self, *a, mode='role', **kw):
        super().__init__(*a, **kw)
        d = self.model_dimension
        self.mode = mode
        if mode == 'role':
            self.role_emb = nn.Embedding(len(ROLES), d)
        elif mode == 'geom':
            self.geom_mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, d))

    def encode_cond(self, src, src_mask, cond):
        mem = self.encode(src, src_mask)
        if self.mode == 'none':
            return mem, src_mask
        if self.mode == 'role':
            row = self.role_emb(cond.long()).unsqueeze(1)
        else:
            row = self.geom_mlp(cond.float()).unsqueeze(1)
        mem = torch.cat([row, mem], dim=1)
        pad = torch.ones(mem.size(0), 1, 1, dtype=src_mask.dtype, device=src_mask.device)
        return mem, torch.cat([pad, src_mask], dim=-1)

    def forward_cond(self, src, src_mask, tgt, tgt_mask, cond):
        mem, m2 = self.encode_cond(src, src_mask, cond)
        return self.decode(mem, m2, tgt, tgt_mask)


class RoleData(torch.utils.data.Dataset):
    def __init__(self, path, vocab, tok, mode, max_len=128, limit=0):
        self.rows = list(csv.DictReader(open(path)))
        if limit:
            self.rows = self.rows[:limit]
        self.v, self.t, self.mode, self.max_len = vocab, tok, mode, max_len

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
        if self.mode == 'geom':
            c = torch.tensor([float(r.get('d', 0.0)), float(r.get('cos_theta', 0.0))],
                             dtype=torch.float)
        else:
            c = torch.tensor(R2I.get(r.get('role', 'EDIT_SCAFFOLD'), 0), dtype=torch.long)
        return s, t, c, R2I.get(r.get('role', 'EDIT_SCAFFOLD'), 0)


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    ss, ts, cs, rs = zip(*batch)
    B, Ks, Kt = len(ss), max(len(x) for x in ss), max(len(x) for x in ts)
    src = torch.zeros(B, Ks, dtype=torch.long)
    tgt = torch.zeros(B, Kt, dtype=torch.long)
    sm = torch.zeros(B, 1, Ks, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(ss, ts)):
        src[i, :len(a)] = a
        sm[i, 0, :len(a)] = True
        tgt[i, :len(b)] = b
    return src, sm, tgt, torch.stack(cs), torch.tensor(rs, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='experiments/covalentformer/data/roles')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['none', 'role', 'geom'], default='role')
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=20260914)
    # Defaults reproduce the previous hardcoded behaviour EXACTLY, so this is additive and every
    # existing launch is unaffected. Needed for the role-stratified split (build_split.py), whose
    # held-out set is the only one that can power EDIT_LINKER: the role-blind split leaves 13 clean
    # LINKER rows, which is the entire supply, not a small sample.
    ap.add_argument('--train-file', default='train.csv')
    ap.add_argument('--valid-file', default='valid.csv')
    a = ap.parse_args()
    dev = (('mps' if torch.backends.mps.is_available() else
            ('cuda' if torch.cuda.is_available() else 'cpu')) if a.device == 'auto' else a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    ck = torch.load(a.prior, map_location='cpu', weights_only=False)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    tok = SMILESTokenizer()
    npm = dict(ck['network_parameter'])
    model = PhaseA(mode=a.mode, **npm)
    missing, unexpected = model.load_state_dict(ck['network_state'], strict=False)
    bad = [k for k in missing if not (k.startswith('role_emb') or k.startswith('geom_mlp'))]
    # SEED IN THE LOG HEADER. ckpt_A_role was trained before the checkpoint recorded its seed and
    # the header did not print one, so its seed is now UNRECOVERABLE from any artifact on disk --
    # which makes the A0-vs-A-role control comparison provisional rather than clean. A run whose
    # seed cannot be read back cannot be paired with its own control.
    print('mode=%s device=%s seed=%d | new tensors %d | unexpected %d'
          % (a.mode, dev, a.seed, len(missing), len(unexpected)))
    if bad:
        print('  FATAL: prior did not load for %s' % bad[:5])
        return 2
    model.to(dev)

    tr = RoleData(os.path.join(a.data, a.train_file), vocab, tok, a.mode, limit=a.limit)
    va = RoleData(os.path.join(a.data, a.valid_file), vocab, tok, a.mode,
                  limit=(a.limit // 10 if a.limit else 0))
    print('  train %d | valid %d  [%s / %s]' % (len(tr), len(va), a.train_file, a.valid_file))
    # PROVENANCE: a checkpoint must record which split produced it. valid_strat.csv is 99.7%
    # anchor-leaked against a model trained on the OLD train.csv, so a mismatched pair yields a
    # spectacular fake result. Stamping it here makes the pairing checkable after the fact.
    # THIS GUARD USED TO BE A FILENAME SUBSTRING TEST, AND IT PASSED THE WORST PAIR IN THE PROJECT.
    #   if ('strat' in a.train_file) != ('strat' in a.valid_file): FATAL
    # For the DEFAULT pair (train.csv, valid.csv) that is False != False -> no FATAL, so the trainer
    # would train and validate on a pair measured at 82.4% ANCHOR OVERLAP and print a flatteringly
    # low valid loss. It only caught a MIXED pair, never a uniformly-leaked one. That is the same
    # defect already fixed in role_compliance.assert_eval_split and warhead_retention; this is the
    # mint-side script, where it matters most, and it was the one place still unfixed.
    # MEASURED FROM THE FULL FILES, NOT FROM tr.rows/va.rows.
    # My first version of this guard read the already-loaded rows to avoid a file read. That is
    # WRONG under --limit: RoleData truncates, so with --limit 2000 the guard compared 2000 train
    # anchors against 200 valid ones and reported the (train.csv, valid.csv) pair -- measured at
    # 82.4% overlap on the FULL files -- as 1.0%, comfortably PASSING its own 5% threshold. A guard
    # whose strictness silently depends on an unrelated debugging flag is worse than no guard,
    # because it reads as verification. Always measure the whole files.
    # TWO FURTHER FALSE-PASS ROUTES, both found by hunting rather than by a crash, both LATENT on
    # today's files but neither excluded by anything the guard did:
    #   (A) It compares RAW STRINGS. The same molecule written two ways -- 'C=CC(=O)Nc1ccccc1' and
    #       'c1ccccc1NC(=O)C=C' -- is unequal as text and identical as chemistry, so a valid set
    #       written in a different SMILES convention than train would report 0% overlap and PASS.
    #   (B) Whitespace. A leading space makes an anchor unequal to its train twin AND still truthy,
    #       so the fail-closed empty-column check does not fire either.
    # .strip() closes (B) outright. (A) cannot be closed cheaply -- canonicalising 263k anchors
    # costs minutes on every training launch -- so instead the guard MEASURES whether the files are
    # canonical on a sample and refuses to certify a split it cannot actually compare. Verified
    # today: train_strat and valid_strat are 0 whitespace-dirty and 0 non-canonical, so this changes
    # no result; it stops the guard from silently depending on that remaining true.
    # THE FILTER'S OWN DROP RATE MUST BE REPORTED, or the guard measures a subset and calls it the
    # file. `if r.get('anchor','').strip()` silently removes blank-anchor rows from BOTH sides, but
    # RoleData applies NO such filter and validates on ALL rows. So the guard's population and the
    # training population can diverge, and the divergence is invisible in the output. Demonstrated
    # by QA: a valid file of 2000 leaked + 200 clean rows FATALs at 90.9% with anchors intact, and
    # PASSES at 0.00% once the 2000 leaked anchors are BLANKED -- while the trainer still validates
    # on all 2200. My earlier FATAL covered only the TOTAL-emptiness endpoint, not the interior.
    # This is the third instance tonight of the same shape: a guard whose strictness depends on a
    # property of the data that nothing reports. Count the drops and refuse above 2%.
    def _anchors(fn):
        # Returns (unique anchors, raw row count, NON-BLANK ROW count). The third value is not the
        # size of the set: train_strat repeats anchors by construction (only 7% are singletons), so
        # comparing set size against row count would read as a 90%+ drop rate on a perfectly clean
        # file. The drop rate has to be measured in ROWS, which is the unit the filter works in.
        raw = kept = 0
        out = set()
        with open(os.path.join(a.data, fn)) as fh:
            for r in csv.DictReader(fh):
                raw += 1
                v = r.get('anchor', '').strip()
                if v:
                    kept += 1
                    out.add(v)
        return out, raw, kept

    def _noncanon_rate(fn, k=300, _rng=None):
        """Fraction of a RANDOM sample whose anchor is not already canonical SMILES.

        This gate is what licenses comparing anchors as RAW STRINGS below, so every way it can
        return a low number without having looked at anything is a way to certify a leaked split.
        It had three, and QA drove a 90.9%-leaked file with ZERO blank rows straight through all
        of them:
          (i)   `m is None` was not counted. An unparseable anchor is truthy, so it survives the
                blank filter AND the drop-count I just added, and then `m is not None` skips it.
                300 junk rows at the head of a file scored 0/300 = 0.0% non-canonical.
          (ii)  `bad / max(len(smis), 1)` returns 0.0 for an EMPTY sample -- the same
                zero-means-unmeasured construction I wrote six lines of capitals about thirty
                lines further down, reproduced here without noticing.
          (iii) the sample was `zip(range(300), reader)`, i.e. the FIRST 300 rows. Fixed-size and
                positional, so it is defeatable by scale alone with no adversary: 300 bad rows at
                the head of a 30,000-row file is 1% dropped -- under any threshold worth setting --
                while the sample is still 100% bad. A merely sorted or blocked file is sampled
                unrepresentatively for the same reason.
        Unparseable now counts as non-canonical (it is strictly worse: a string RDKit cannot read
        cannot be compared to anything), the sample is drawn at random, and too small a sample is
        a refusal rather than a 0.0.
        """
        import random as _random
        from rdkit import Chem as _C
        from rdkit import RDLogger as _RDL
        _RDL.DisableLog('rdApp.*')
        with open(os.path.join(a.data, fn)) as fh:
            rows = [r.get('anchor', '').strip() for r in csv.DictReader(fh)]
        smis = [s for s in rows if s]
        if len(smis) < min(k, 50):
            return None  # too little to measure; the caller must refuse, not read this as clean
        rng = _rng or _random.Random(20260914)
        smis = rng.sample(smis, min(k, len(smis)))
        bad = 0
        for s in smis:
            m = _C.MolFromSmiles(s)
            if m is None or _C.MolToSmiles(m) != s:
                bad += 1
        return bad / len(smis)
    _tr_anchors, _tr_raw, _tr_kept = _anchors(a.train_file)
    with open(os.path.join(a.data, a.valid_file)) as _fh:
        _all_va = list(csv.DictReader(_fh))
    _va = [r for r in _all_va if r.get('anchor', '').strip()]
    # BOTH sides are load-bearing and both fail the same way. Blank the VALID anchors and the leaked
    # rows leave the numerator; blank the TRAIN anchors and they leave the set being matched against.
    # Either one drives the measured overlap toward zero while the trainer still sees every row.
    # _tr_kept, NOT len(_tr_anchors). The set is DEDUPLICATED; the drop rate must be in ROWS, the
    # unit the filter works in. Passing the set size measured the DUPLICATE rate instead and made
    # this guard fire on every clean file in the project (train_strat.csv: 0 blank anchors, 263,248
    # rows, 127,752 unique -> reported "135,496 of 263,248 rows have a blank anchor", i.e. 51.5%).
    # Worse than a false alarm: this FATAL sits BEFORE the leak measurement below, so it made the
    # 82.4% leak check unreachable dead code for any train file with >2% repeated anchors -- which
    # is every train file here. I wrote the docstring warning about exactly this two edits ago and
    # then passed the set anyway.
    for _fn, _kept, _raw in ((a.valid_file, len(_va), len(_all_va)),
                             (a.train_file, _tr_kept, _tr_raw)):
        if _raw and (_raw - _kept) / _raw > 0.02:
            print('  FATAL: %d of %d rows in %s have a blank anchor. They are excluded from the '
                  'overlap measurement but NOT from training, so the reading below would describe '
                  '%d rows and not the file. Refusing to certify a split I am only partly reading.'
                  % (_raw - _kept, _raw, _fn, _kept))
            return 2
    # FAIL CLOSED ON AN ABSENT ANCHOR COLUMN. Without this the guard has the SAME failure mode as
    # the filename test it replaced: if the valid CSV has no 'anchor' column (or all blank), _va is
    # EMPTY, the ratio is 0/max(0,1) = 0.0, and it prints "eval split MEASURED: 0.0% overlap" --
    # certifying a split it never checked. A guard that reports 0% when it measured NOTHING is
    # worse than no guard. Refuse instead, and say which of the two files is at fault.
    if not _tr_anchors or not _va:
        print('  FATAL: cannot measure split overlap -- %s contributed %d train anchors and %s '
              'contributed %d valid rows with a non-empty "anchor" column. A 0%% reading here would '
              'mean UNMEASURED, not clean. Refusing to train.'
              % (a.train_file, len(_tr_anchors), a.valid_file, len(_va)))
        return 2
    # Refuse to certify if either file is not canonical: a raw-string comparison between
    # differently-written SMILES reports 0% overlap regardless of the true overlap, so a 0% reading
    # on a non-canonical pair means UNCOMPARABLE, not clean -- the same distinction as the
    # empty-column case above.
    _nc_tr, _nc_va = _noncanon_rate(a.train_file), _noncanon_rate(a.valid_file)
    # None means the sample was too small to measure. That is a REFUSAL, not a pass -- `max(None, x)`
    # would raise anyway, but the point is that it must never silently compare as 0.
    if _nc_tr is None or _nc_va is None:
        print('  FATAL: too few non-blank anchors to sample for canonicality (%s -> %s, %s -> %s). '
              'Cannot certify that a raw-string overlap comparison is meaningful on these files.'
              % (a.train_file, _nc_tr, a.valid_file, _nc_va))
        return 2
    if max(_nc_tr, _nc_va) > 0.02:
        print('  FATAL: anchors are not canonical or not parseable (%s %.0f%%, %s %.0f%% of a random '
              '300-row sample; unparseable counts as non-canonical). This guard compares raw strings, '
              'so it CANNOT measure overlap between differently-written SMILES and a 0%% reading would '
              'be meaningless. Canonicalise both files first.'
              % (a.train_file, 100 * _nc_tr, a.valid_file, 100 * _nc_va))
        return 2
    _leak = sum(1 for r in _va if r['anchor'].strip() in _tr_anchors) / len(_va)
    if _leak > 0.05:
        print('  FATAL: %.1f%% of %s rows have an anchor that also appears in %s (threshold 5%%). '
              'These sets are not mutually held out -- refusing to train.'
              % (100 * _leak, a.valid_file, a.train_file))
        return 2
    print('  eval split MEASURED: %.1f%% anchor overlap between %s and %s (threshold 5%%)'
          % (100 * _leak, a.valid_file, a.train_file))
    cnt = collections.Counter(R2I.get(r.get('role', 'EDIT_SCAFFOLD'), 0) for r in tr.rows)
    tot = sum(cnt.values())
    w = torch.tensor([tot / (len(ROLES) * max(cnt.get(i, 0), 1)) for i in range(len(ROLES))],
                     dtype=torch.float, device=dev)
    print('  role counts %s' % {ROLES[i]: cnt.get(i, 0) for i in range(len(ROLES))})
    print('  class weights %s' % [round(float(x), 2) for x in w])

    dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=collate,
                                     drop_last=True)
    dlv = torch.utils.data.DataLoader(va, batch_size=a.bs, shuffle=False, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    os.makedirs(a.out, exist_ok=True)
    t0, hist = time.time(), []

    for ep in range(a.epochs):
        model.train()
        tot_l, n = 0.0, 0
        for step, b in enumerate(dl):
            if b is None:
                continue
            src, sm, tgt, cond, roles = [x.to(dev) for x in b]
            ti, to = tgt[:, :-1], tgt[:, 1:]
            out = model.forward_cond(src, sm, ti, subsequent_mask(ti.size(1), dev), cond)
            lg = model.generator(out)
            per = ce(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).view(to.shape)
            # CLASS-WEIGHTED so rare roles are not ignored
            wts = w[roles].unsqueeze(1)
            mask = (to != 0).float()
            loss = (per * mask * wts).sum() / (mask * wts).sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_l += loss.item()
            n += 1
            if step % 200 == 0:
                print('  ep%d %5d/%d loss %.4f (%.1f min)'
                      % (ep, step, len(dl), tot_l / max(n, 1), (time.time() - t0) / 60), flush=True)

        model.eval()
        vt, vn, vs, vsn = 0.0, 0, 0.0, 0
        vd, vdn = 0.0, 0   # deterministic all-wrong-token control
        with torch.no_grad():
            for b in dlv:
                if b is None:
                    continue
                src, sm, tgt, cond, roles = [x.to(dev) for x in b]
                ti, to = tgt[:, :-1], tgt[:, 1:]
                msk = subsequent_mask(ti.size(1), dev)
                lg = model.generator(model.forward_cond(src, sm, ti, msk, cond))
                m2 = (to != 0).float()
                vt += float((ce(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).view(to.shape)
                             * m2).sum() / m2.sum().clamp(min=1))
                vn += 1
                if a.mode != 'none':
                    # KEPT for comparability with every history.json written before this change --
                    # but it is a DILUTED and BATCH-ORDER-DEPENDENT control. randperm within a
                    # batch returns a row its OWN role with probability sum_r p_r^2; on this role
                    # mix that is ~25-57% of rows contributing exactly zero, so the reported GAP
                    # understates the true wrong-token penalty by roughly 1.7-2.5x and is not
                    # reproducible across batch orders. Do not quote `gap` as an effect size.
                    perm = torch.randperm(cond.size(0), device=dev)
                    lg2 = model.generator(model.forward_cond(src, sm, ti, msk, cond[perm]))
                    vs += float((ce(lg2.reshape(-1, lg2.size(-1)), to.reshape(-1)).view(to.shape)
                                 * m2).sum() / m2.sum().clamp(min=1))
                    vsn += 1
                    # DETERMINISTIC ALL-WRONG CONTROL, added alongside rather than replacing so no
                    # version boundary is created (mixing two estimators across a code change is
                    # exactly what contaminated the Phase B conditioning contrast tonight). Every
                    # row is scored under all roles it is NOT, and averaged: zero fixed points, no
                    # RNG, reproducible. This is the number to quote.
                    # *** ROLE MODE ONLY. In geom mode this control was a NO-OP that printed a
                    # FABRICATED GAP_det of ~0. cond is (B,2) there, so `ck_ = cond` handed the
                    # model the CORRECT conditioning and the loop just recomputed the clean valid
                    # loss four times over four row subsets. A geom arm would have printed a real
                    # permutation GAP beside GAP_det~0, and the natural reading -- "the
                    # deterministic control shows the GAP was an artifact" -- is exactly backwards.
                    # A control that cannot detect the thing it exists to detect is worse than no
                    # control, which is the lesson of the unreachable falsifier earlier tonight.
                    # Refusing is correct: "wrong token" has no meaning for a continuous channel,
                    # and a geom control needs a different construction (e.g. resampled (d,theta)
                    # from other rows), not a relabelling.
                    tot_w, n_w = 0.0, 0
                    if a.mode == 'role':
                        for k in range(len(ROLES)):
                            sel = (roles != k)
                            if not bool(sel.any()):
                                continue
                            ck_ = torch.full_like(cond, k)
                            lg3 = model.generator(model.forward_cond(src, sm, ti, msk, ck_))
                            per3 = ce(lg3.reshape(-1, lg3.size(-1)),
                                      to.reshape(-1)).view(to.shape)
                            mm = m2 * sel.float().unsqueeze(1)
                            if float(mm.sum()) > 0:
                                tot_w += float((per3 * mm).sum() / mm.sum())
                                n_w += 1
                    elif ep == 0 and vdn == 0:
                        print('  NOTE: the deterministic all-wrong control is ROLE-ONLY. mode=%s '
                              'reports GAP_det=nan rather than a fabricated zero.' % a.mode)
                    if n_w:
                        vd += tot_w / n_w
                        vdn += 1
        v = vt / max(vn, 1)
        sh = vs / max(vsn, 1) if vsn else float('nan')
        gap = sh - v
        det = vd / max(vdn, 1) if vdn else float('nan')
        gap_det = det - v
        print('EPOCH %d  train %.4f | valid %.4f | perm-ctrl %.4f GAP %+.4f | ALL-WRONG %.4f '
              'GAP_det %+.4f  <- quote this one [%.1f min]'
              % (ep, tot_l / max(n, 1), v, sh, gap, det, gap_det, (time.time() - t0) / 60),
              flush=True)
        hist.append(dict(epoch=ep, train=tot_l / max(n, 1), valid=v, shuffled=sh, gap=gap,
                         all_wrong=det, gap_det=gap_det))
        torch.save({'model_state': model.state_dict(), 'network_parameter': npm,
                    'vocabulary': ck['vocabulary'], 'epoch': ep, 'mode': a.mode,
                    'train_file': a.train_file, 'valid_file': a.valid_file, 'seed': a.seed,
                    # STAMP THE ROLE ORDER. This is the last genuinely SILENT architecture-replay
                    # hazard in the project and it is the worst-shaped one: role_emb is a 4x256
                    # Embedding, so reordering ROLES permutes what every index MEANS while the
                    # tensor shape is unchanged. The checkpoint then loads cleanly under
                    # strict=True -- no error, no warning, and nothing in any artifact from which
                    # the permutation could be detected afterwards. role_compliance and
                    # warhead_retention both import ROLES from this module rather than reading it
                    # from the checkpoint, so trainer and scorer share a MUTABLE GLOBAL instead of
                    # a saved value. That is exactly the cond_repeat failure mode (a k=8 model
                    # silently evaluated as k=1), except cond_repeat at least had a key to read.
                    # Every per-role number on the books (#85, #97, #114, #124, #126) would be
                    # silently permuted by a one-line edit to line 36, with no way to notice.
                    'roles': list(ROLES),
                    # STAMP THE DEVICE. It was recoverable for runs 1 and 2 ONLY because their
                    # /tmp logs happened to survive; delete those and which backend produced
                    # them is gone. Not academic: run 3 exists to estimate Phase-A-RUN-level
                    # variance at sd ~0.003, and a cpu-vs-mps difference would be confounded
                    # with the run difference at that scale. I launched run 3 on cpu, measured
                    # 25.6 steps/min against mps's 148, and had to INFER runs 1/2 were mps from
                    # a throughput comment in an unrelated file before relaunching. A one-word
                    # stamp removes the inference.
                    #
                    # THE STAMP BEGINS WITH THE NEXT RUN, NOT WITH RUN 3. This file's mtime is
                    # 14:40; run 3's ep0 was written 14:38, so run 3's interpreter held pre-edit
                    # code for its entire life and its checkpoints carry device=None like every
                    # earlier run. The paragraph above reasoned specifically about run 3 and read
                    # as though the inference problem were solved for it; it is not. This is the
                    # same import-staleness that hit the B4c deleak stamp, and I annotated it
                    # there and not here. Run 3's device remains an INFERENCE (throughput: epochs
                    # at 34.5 and 28.6 min, versus the ~3h/epoch that 25.6 steps/min on cpu would
                    # give -- consistent with mps, and with my having relaunched it on mps), and
                    # an inference is what the stamp exists to eliminate. It is recorded as such
                    # in the checkpoints rather than asserted as a measurement.
                    # str(dev), THE RESOLVED DEVICE -- NOT str(a.device), THE ARGUMENT. The
                    # default is --device auto, so the argument spelling wrote the literal string
                    # "auto": a device field that is populated, non-null, and carries no
                    # information. That is the exact inference problem the paragraph above exists
                    # to eliminate, reintroduced one line below it. A stamp that cannot be wrong
                    # is the point; a stamp that says "auto" is worse than an absent one, because
                    # absent announces itself and "auto" reads as a measurement.
                    'device': str(dev),
                    # THESE SIX CHANGE WHAT THE ARTIFACT MEANS AND TRAVELLED NOWHERE.
                    # The bs/lr/epochs fix went into train_phaseB at both save sites and not
                    # into this file at all -- the same exposure, left live in the trainer
                    # whose A0-vs-A-role contrast is exactly the kind it breaks. `limit`
                    # truncates train AND (via limit//10) valid, so a limited run is
                    # indistinguishable from a full one. `prior` selects the initialisation,
                    # so every "from the Mol2Mol prior" claim rested on a value no artifact
                    # recorded. And `epochs` is the ONE declared difference between
                    # ckpt_A_role_conv and ckpt_A_role_strat_s3 -- without it those two
                    # checkpoints carry byte-identical provenance at ep2 while being the two
                    # arms of a deliberate contrast, distinguished only by directory name.
                    'bs': a.bs, 'lr': a.lr, 'epochs': a.epochs, 'limit': a.limit,
                    'prior': a.prior, 'data': a.data,
                    'history': hist}, os.path.join(a.out, 'ep%d.ckpt' % ep))
        json.dump(hist, open(os.path.join(a.out, 'history.json'), 'w'), indent=1)
    print('done in %.1f min' % ((time.time() - t0) / 60))
    return 0


if __name__ == '__main__':
    sys.exit(main())
