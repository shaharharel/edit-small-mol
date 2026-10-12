#!/usr/bin/env python3
"""Score a generator run on the four things we actually care about, from saved generations alone.

    python eval_axes.py sft_runs/pocket/eval/generations.jsonl
    python eval_axes.py sft_runs/{pocket,blind}/eval/generations.jsonl --compare
    python eval_axes.py <gen.jsonl> --json out.json

No GPU, no model load, no tokenizer: it reads the `generations.jsonl` that eval_sft.py writes and
recomputes everything. That separation is deliberate -- generation is the expensive, non-reproducible
step, and scoring is the part we will change our minds about. Every metric here can be re-derived,
disputed or added to months later without spending an A100-hour, and a metric nobody can recompute
is a metric nobody can check.

THE FOUR AXES

1. SMILES GENERATION -- can it emit a parseable molecule at all.
   Format compliance is reported separately from chemical validity, because they are different
   failures with different fixes: a model that knows the chemistry but not our output format is one
   SFT epoch away from working, while a model emitting confident nonsense is not. Base Llama-3.1
   scored 1/5 as-emitted and 4/5 once the SMILES was extracted from its prose -- almost entirely a
   format failure. Collapsing the two into one rate hides which one you have.
   DISTINCT COUNT is here because validity is gameable: a model that returns the same valid molecule
   every time scores 100% validity and is useless. Base Llama returned sulfanilamide in 3 of 4 hits.

2. ANALOGUE GENERATION -- is the molecule it proposes the right one.
   Scored against the ECHO BASELINE, not against zero. Returning the input unchanged is always
   valid, always well-formed, and always wrong, yet it scores a high Tanimoto to the true analogue
   because an analogue pair is by construction a small edit. So the only honest question is whether
   the model beats copying its input, which is what `beats_echo` measures. Absolute Tanimoto without
   that comparison is the single easiest number to fool yourself with here.

3. REASONING QUALITY -- is the sentence true, and does it obey the rules the chemists wrote.
   GROUNDEDNESS (do the residues it names exist in the pocket it was shown) is the headline, and it
   is reported against the REFERENCE CEILING measured on the same rows -- about 0.89, not 1.0,
   because the chairs' own sentences name residues the input crystal never touched (the analogue's
   new group reaches them; 100% of such mentions trace to the analogue's contact table, 0% are
   invented). A model at 0.89 is at ceiling, not 11% hallucinatory.
   The BANNED-CLAIM rates come from REVIEWER.md's hard rules: no burial or solvent-facing claims
   (rule 4 -- chemists found "solvent-facing" false for 40% of site atoms), no room or channel
   claims, no potency claims. These are rejection criteria for a human labeller, so they are
   rejection criteria for the model.

4. POCKET AWARENESS -- is the pocket doing any work.
   Two independent readings, because the cheap one is confoundable:
     (a) ARM DELTA (--compare): the same metric on a pocket-aware and a pocket-blind run. Clean,
         but it costs a second training run and only speaks about the arm, not about an example.
     (b) POCKET FOLLOWING: of the residues a sentence names, how many are in the pocket table that
         THIS prompt carried. A model ignoring the pocket can still name plausible residues from
         parametric memory -- LEU, VAL, PHE with common numbers -- and those will mostly miss. This
         is per-example, needs no ablation run, and is the metric to carry forward to the 7k.
   The blind arm is scored against the pocket table it was never shown, on purpose: that is what
   makes the comparison mean "reading the table is why the citations land".

Reference rows are scored by the identical code path and reported as the CEILING line. Any metric
where the reference does not look near-perfect is a bug in the metric, not a triumph of the model --
that line exists to catch exactly that.
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

AA = frozenset('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL '
               'MSE SEC PYL HID HIE HIP'.split())
_RES = re.compile(r'\b([A-Z]{3})\s?(\d+)')
_DIST = re.compile(r'\d\.\d{1,2}')
# REVIEWER.md's hard rules, as patterns. Rule 4 bans burial and solvent-facing outright; the room
# and channel claims were removed from the brief when the probe was shown to contradict the edit;
# potency is not in evidence anywhere in this corpus, so any mention of it is invention.
BANNED = (('burial_or_solvent', re.compile(r'\bburi(ed|al)|solvent[- ]facing|solvent[- ]exposed', re.I)),
          ('room_or_channel', re.compile(r'\bchannel\b|\bsub-?pocket\b|\broom\b|\bcavity\b', re.I)),
          ('potency', re.compile(r'potenc|\bIC50\b|\bKi\b|affinit|more active|improv(e|es|ed)\b', re.I)))
STOP = frozenset('a an the of to in at by for with and or but from its it is are was were be been '
                 'as that this which while on into onto over under than then both no not'.split())


def residues(text):
    """Residue tokens a sentence claims, filtered to the amino-acid vocabulary.

    The filter is load-bearing. A bare `[A-Z]{2,4}\\d+` scores the `CD1` of `LEU168.CD1` as a
    separate residue called CD1, which read as a hallucination and put the first version of the
    groundedness metric at 0.66 instead of 0.89.
    """
    return {'%s%s' % (m.group(1), m.group(2)) for m in _RES.finditer(text or '') if m.group(1) in AA}


def content_f1(gen, ref):
    """Token F1 on content words. A weak proxy for 'says the same thing', and labelled as such.

    It cannot tell a correct sentence from a fluent one that happens to reuse the reference's
    vocabulary, so it is never the headline -- it is here to catch a run that has gone off topic
    entirely, which the chemistry metrics would not notice.
    """
    g = collections.Counter(w for w in re.findall(r'[a-z0-9.]+', (gen or '').lower()) if w not in STOP)
    r = collections.Counter(w for w in re.findall(r'[a-z0-9.]+', (ref or '').lower()) if w not in STOP)
    if not g or not r:
        return 0.0
    ov = sum((g & r).values())
    p, rc = ov / sum(g.values()), ov / sum(r.values())
    return round(2 * p * rc / (p + rc), 4) if ov else 0.0


def _fp(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None, None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(m), m.GetNumHeavyAtoms()


def tanimoto(a, b):
    from rdkit import DataStructs
    fa, _ = _fp(a)
    fb, _ = _fp(b)
    return round(DataStructs.TanimotoSimilarity(fa, fb), 4) if (fa is not None and fb is not None) else None


def heavy(smi):
    _, n = _fp(smi)
    return n


def mean(xs, nd=4):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), nd) if xs else None


def frac(num, den, nd=4):
    return round(num / den, nd) if den else None


def score_rows(recs, what='generated'):
    """All four axes over one run's rows. `what` picks the generated or the reference column, so the
    reference ceiling is computed by this exact code and cannot drift from the thing it bounds."""
    sent_key = 'gen_sentence' if what == 'generated' else 'reference_sentence'
    smi_key = 'gen_smiles' if what == 'generated' else 'reference_smiles'
    n = len(recs)
    S = collections.Counter()
    tan_gen, tan_echo, dh_err, f1s, words, gmac = [], [], [], [], [], []
    rr, rp, rf = [], [], []   # recall/precision/F1 of residues against the reference's own set
    rr_chance = []            # the same recall if residues were drawn uniformly from the pocket
    g_hit = g_tot = 0
    canon_seen = collections.Counter()
    beats = ties = 0

    for r in recs:
        sent = r.get(sent_key) or ''
        smi = r.get(smi_key)
        ref_smi = r.get('reference_smiles')
        inp = r.get('input_smiles')

        # --- axis 1: SMILES generation
        if smi:
            S['format_ok'] += 1
        hg = heavy(smi) if smi else None
        if hg is not None:
            S['valid'] += 1
            canon_seen[smi] += 1
        if r.get('echo'):
            S['echo'] += 1

        # --- axis 2: analogue accuracy, always against the echo baseline
        if hg is not None and ref_smi:
            t_gen = tanimoto(smi, ref_smi)
            tan_gen.append(t_gen)
            if inp:
                t_echo = tanimoto(inp, ref_smi)
                tan_echo.append(t_echo)
                if t_gen is not None and t_echo is not None:
                    beats += t_gen > t_echo
                    ties += t_gen == t_echo
            if heavy(ref_smi) is not None and inp and heavy(inp) is not None:
                true_d = heavy(ref_smi) - heavy(inp)
                got_d = hg - heavy(inp)
                dh_err.append(abs(got_d - true_d))
                if true_d != 0:
                    S['edit_direction_right'] += (got_d > 0) == (true_d > 0)
                    S['edit_direction_n'] += 1
            if tanimoto(smi, ref_smi) == 1.0:
                S['exact_match'] += 1

        # --- axis 3: reasoning quality
        nm = residues(sent)
        tbl = set(r.get('table_residues') or r.get('pocket_residues') or [])
        if not tbl:  # eval_sft stores the split lists rather than the table itself
            tbl = set(r.get('grounded_residues') or []) | set()
        if nm:
            S['names_residue'] += 1
            h = len(nm & tbl)
            S['follows_pocket'] += h > 0
            g_hit += h
            g_tot += len(nm)
            gmac.append(h / len(nm))
            S['fully_grounded'] += (h == len(nm))
        # Did it name the residues the REFERENCE names, not merely residues that exist?
        # Groundedness is necessary and trivially gameable: every residue in the prompt is grounded,
        # so a model that copies three names out of the table scores 1.0 while saying nothing true.
        # This is the sharp reasoning metric -- it asks whether the model picked the contacts the
        # chemist picked, which is the actual claim the sentence is making.
        refn = set(r.get('reference_residues') or [])
        if refn and nm:
            S['ref_res_pairs'] += 1
            ov = len(nm & refn)
            rr.append(ov / len(refn))
            rp.append(ov / len(nm))
            rf.append(0.0 if not ov else 2 * (ov / len(nm)) * (ov / len(refn))
                      / ((ov / len(nm)) + (ov / len(refn))))
            S['names_a_reference_residue'] += ov > 0
            # The null: pick the same NUMBER of residues uniformly from this pocket's table. The
            # hypergeometric mean overlap is |named| * |ref| / |table|, so chance recall is simply
            # |named| / |table| -- analytic, no sampling, no seed. Without this line a recall of
            # 0.39 is uninterpretable: a small pocket makes guessing look like skill.
            if tbl:
                rr_chance.append(min(1.0, len(nm) / len(tbl)))
        if _DIST.search(sent):
            S['cites_distance'] += 1
        w = len(sent.split())
        if w:
            words.append(w)
            S['over_35_words'] += w > 35
        for name, pat in BANNED:
            if pat.search(sent):
                S['banned_' + name] += 1
        f1s.append(content_f1(sent, r.get('reference_sentence')))

    out = {
        'n': n,
        # axis 1
        'format_compliance': frac(S['format_ok'], n),
        'validity_overall': frac(S['valid'], n),
        'validity_of_emitted': frac(S['valid'], S['format_ok']),
        'distinct_smiles': len(canon_seen),
        'distinct_frac': frac(len(canon_seen), S['valid']),
        'most_common_count': canon_seen.most_common(1)[0][1] if canon_seen else 0,
        'echo_rate': frac(S['echo'], n),
        # axis 2
        'exact_match': frac(S['exact_match'], n),
        'tanimoto_to_true_analogue': mean(tan_gen),
        'tanimoto_echo_baseline': mean(tan_echo),
        'beats_echo_frac': frac(beats, len(tan_echo)),
        'ties_echo_frac': frac(ties, len(tan_echo)),
        'heavy_atom_delta_abs_err': mean(dh_err, 2),
        'edit_direction_accuracy': frac(S['edit_direction_right'], S['edit_direction_n']),
        # axis 3
        'names_residue_frac': frac(S['names_residue'], n),
        'cites_distance_frac': frac(S['cites_distance'], n),
        'groundedness_micro': frac(g_hit, g_tot),
        'groundedness_macro': mean(gmac),
        'fully_grounded_frac': frac(S['fully_grounded'], S['names_residue']),
        'mean_words': mean(words, 1),
        'over_35_words_frac': frac(S['over_35_words'], n),
        'ref_residue_recall': mean(rr),
        'ref_residue_recall_chance': mean(rr_chance),
        'ref_residue_recall_lift_over_chance': (
            round(mean(rr) / mean(rr_chance), 2) if mean(rr) and mean(rr_chance) else None),
        'ref_residue_precision': mean(rp),
        'ref_residue_f1': mean(rf),
        'names_a_reference_residue_frac': frac(S['names_a_reference_residue'], S['ref_res_pairs']),
        'content_f1_vs_reference': mean(f1s),
        'banned_burial_or_solvent': frac(S['banned_burial_or_solvent'], n),
        'banned_room_or_channel': frac(S['banned_room_or_channel'], n),
        'banned_potency': frac(S['banned_potency'], n),
        # axis 4 (per-example half; the arm-delta half needs --compare).
        # Of the sentences that name any residue, how many name at least one that is actually in the
        # pocket this prompt carried. A model working from parametric memory names plausible
        # residues that mostly miss, so this separates "read the table" from "guessed a LEU".
        'follows_pocket_frac': frac(S['follows_pocket'], S['names_residue']),
    }
    return out


def load(path, data_path):
    """Join generations to ground truth by pair id.

    The pocket table and the input SMILES live in the sft_out data file, not in generations.jsonl.
    Reconstructing the table from the generated sentence's grounded/ungrounded split would make the
    reference ceiling depend on the model's own output, which would silently move the yardstick with
    the thing being measured.
    """
    truth = {}
    for line in open(data_path):
        d = json.loads(line)
        truth[d['pair']] = d
    recs = [json.loads(l) for l in open(path)]
    missing = 0
    for r in recs:
        d = truth.get(r['pair'])
        if not d:
            missing += 1
            continue
        r['table_residues'] = d['table_residues']
        r['input_smiles'] = d['input_smiles']
        r['reference_smiles'] = d['analogue_smiles']
        r.setdefault('reference_sentence', d['reference_sentence'])
    if missing:
        print('WARNING: %d of %d generation rows have no ground-truth row in %s'
              % (missing, len(recs), data_path), file=sys.stderr)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('generations', nargs='+', help='one or more generations.jsonl')
    ap.add_argument('--data', default='sft_out/sft_pocket.jsonl',
                    help='sft data file supplying the pocket table, input and reference per pair')
    ap.add_argument('--compare', action='store_true',
                    help='print the arm-to-arm delta (axis 4a); expects pocket first, blind second')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    runs = {}
    for p in a.generations:
        recs = load(p, a.data)
        label = os.path.basename(os.path.dirname(os.path.dirname(p.rstrip('/')))) or p
        runs[label] = {'generated': score_rows(recs, 'generated'),
                       'reference_ceiling': score_rows(recs, 'reference')}
        print('\n%s  (%s)  n=%d' % ('=' * 78, label, len(recs)))
        g, c = runs[label]['generated'], runs[label]['reference_ceiling']
        for k in g:
            if k == 'n':
                continue
            print('  %-30s %10s      ceiling %s' % (k, g[k], c.get(k)))

    if a.compare and len(a.generations) >= 2:
        ks = list(runs)
        A, B = runs[ks[0]]['generated'], runs[ks[1]]['generated']
        print('\n%s\nAXIS 4a -- ARM DELTA  (%s minus %s)' % ('=' * 78, ks[0], ks[1]))
        for k in A:
            if k == 'n' or not isinstance(A[k], (int, float)) or not isinstance(B.get(k), (int, float)):
                continue
            print('  %-30s %10s  -  %-10s = %+.4f' % (k, A[k], B[k], A[k] - B[k]))

    if a.json:
        json.dump(runs, open(a.json, 'w'), indent=2)
        print('\nwrote %s' % a.json)


if __name__ == '__main__':
    main()
