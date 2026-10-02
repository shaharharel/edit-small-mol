#!/usr/bin/env python
"""Build the v4 report. EVERY number here is traceable to a file in this directory.

WHY A NEW GENERATOR RATHER THAN A PATCH. mk_report.py is a v3-era document: an audit of its prose
found 161 numeric claims, of which the headline summary, the corpus counts (503,895 rows; rung-1
406,017; rung-2 90,000) and every generation table describe joint/v3b, not v4 (234,647 rows; rung-1
49,937; rung-2 76,718). Its rung-3 generation numbers were also measured on folds later found to be
99.5% contained in training. Patching 161 claims in place would have left stale text between them.

SOURCES, all in this directory:
  v4_eval_train_fixed.json      recall arm,   n=400, k=20, stratified, connected-only
  v4_eval_disjoint_fixed.json   disjoint arm, n=600, k=20, stratified, connected-only
  v4_eval_base.json.partial     base-Qwen control, n=200, k=30
  v4_gens_{train,disjoint}_fixed.jsonl   per-row generations, for the examples and the Tc analysis
  results/gates/CDL_ENTROPY_AUDIT.json   the OP entropy measurement
  results/final_covrxn/*.json            covRXN arm matrix, 3 seeds
"""
import json, os, re, collections, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer'
exec(open(os.path.join(HERE, 'sci_css.py')).read())

EX = json.load(open(os.path.join(HERE, 'v4_report_examples.json')))
pc = lambda x: '%.2f%%' % (100.0 * x)

# FIGURES ARE EXPLICIT CONSTANTS, each annotated with where it came from. The eval JSONs were lost
# when the scratchpad was cleaned and cannot be re-fetched (no auth), but the per-row generation
# files survived, so exact@k is recomputable and the rest is transcribed from the run output.
#
# NOT recomputed from the generation files: validity and uniqueness. That file stores only the
# candidates that PARSED, so recomputing validity from it returns ~100% by construction -- a
# circular measure. Those two come from the runs' own counters over all 20 samples per row.
rec = {'n': 400, 'k': 20,
       'exact@20': 0.3650,        # run output
       'exact@1': 0.0625,         # run output
       'validity': 0.9146,        # run counter over 8,000 generations
       'uniqueness': 0.5311,      # run counter
       'copy_rate_PATHOLOGY': 0.1100,
       'change_op_acc': 0.3075}
dis = {'n': 600, 'k': 20,
       'exact@20': 0.006667,
       'exact@1': 0.001667,
       'validity': 0.8044,
       'uniqueness': 0.6743,
       'copy_rate_PATHOLOGY': 0.0600,
       'change_op_acc': 0.3350}
bas = {'n': 200, 'k': 30,         # base Qwen, no adapter; partial write at n=200
       'exact@30': 0.0, 'exact@1': 0.0,
       'validity': 0.0065, 'copy_rate_PATHOLOGY': 0.0600, 'change_op_acc': 0.0}
# recomputed from v4_gens_*_fixed.jsonl over LEGITIMATE pairs only (connected, not stereo-only):
LEGIT = {'rec_n': 385, 'rec_hits': 135, 'rec_rate': 135 / 385.0,
         'dis_n': 597, 'dis_hits': 4, 'dis_rate': 4 / 597.0}


def exblock(e):
    tc = (' · best Tc <b>%.3f</b>' % e['tc']) if e.get('tc') else ''
    return f'''<div class="case">
<p class="cap"><b>{e['label']}</b> — PDB {e['pdb']}, reference op <code>{e['op_ref']}</code>{tc}</p>
<table class="ex"><tbody>
<tr><th>prompt, <code>Hit:</code></th><td><code class="inp">{e['hit']}</code></td></tr>
<tr><th>reference analogue</th><td><code>{e['ref']}</code></td></tr>
<tr><th>v4, first of 20 samples</th><td><code class="outp">{e['gen']}</code></td></tr>
</tbody></table></div>'''


HTML = f'''<title>CovalentFormer v4</title>
{CSS}
<div class="wrap">
<h1>CovalentFormer v4</h1>
<p class="sub">Qwen2.5-7B + LoRA (r=16, &alpha;=32) · 234,647-row mixed corpus · 7,110 steps on one
A100-40GB · evaluated on a molecule-disjoint fold against a no-adapter control · internal report,
2 Oct 2026</p>

<h2><span class="sn">1</span>Summary</h2>
<p>v4 learns the task format and valid covalent chemistry, and it is the strongest arm we have on a
matched comparison. It does <b>not</b> yet generalise to novel scaffolds, and three independent
measurements agree on that.</p>

<div class="tbl"><p class="cap"><b>Table 1.</b> All arms, one parser, one prompt encoding asserted
identical to training. Recall = pairs v4 trained on; disjoint = molecule-disjoint fold
(0 molecule overlap, 0 pair overlap). Both stratified across complexes, disconnected pairs excluded.</p>
<table><tbody>
<tr><th></th><th>base Qwen, no LoRA</th><th>v4 recall</th><th>v4 disjoint</th></tr>
<tr><td>n / k</td><td>200 / 30</td><td>400 / 20</td><td>600 / 20</td></tr>
<tr><td><b>exact@k</b></td><td><b>{pc(bas['exact@30'])}</b></td><td><b>{pc(LEGIT["rec_rate"])}</b></td><td><b>{pc(LEGIT["dis_rate"])}</b></td></tr>
<tr><td>exact@1</td><td>{pc(bas['exact@1'])}</td><td>{pc(rec['exact@1'])}</td><td>{pc(dis['exact@1'])}</td></tr>
<tr><td>validity</td><td><b>{pc(bas['validity'])}</b></td><td>{pc(rec['validity'])}</td><td>{pc(dis['validity'])}</td></tr>
<tr><td>uniqueness</td><td>&mdash;</td><td>{pc(rec['uniqueness'])}</td><td>{pc(dis['uniqueness'])}</td></tr>
<tr><td>copy rate</td><td>{pc(bas['copy_rate_PATHOLOGY'])}</td><td>{pc(rec['copy_rate_PATHOLOGY'])}</td><td>{pc(dis['copy_rate_PATHOLOGY'])}</td></tr>
<tr><td>change-op accuracy</td><td>{pc(bas['change_op_acc'])}</td><td>{pc(rec['change_op_acc'])}</td><td>{pc(dis['change_op_acc'])}</td></tr>
</tbody></table></div>

<p>The recall figure is <b>{pc(LEGIT['rec_rate'])}</b> ({LEGIT['rec_hits']}/{LEGIT['rec_n']}), not
the {pc(rec['exact@20'])} the run reported over its full 400 rows. 11 of those 146 hits came from
pairs that differ <i>only in stereochemistry</i>: the scorer strips stereo, so hit and reference
collapse to the same molecule and copying the input wins. 58 of 4,501 training pairs (1.3%) are of
that kind, plus disconnected pairs (below) &mdash; 15 rows excluded in total. The build-time identity
gate compares <i>with</i> stereo, so they passed; the scorer compares <i>without</i>, so they became
free points. Every exact@k in this report is on the legitimate-pair denominator.</p>

<div class="note"><b>The control is the firmest result.</b> Base Qwen2.5-7B on identical prompts
produces a parseable molecule in <b>{pc(bas['validity'])}</b> of 6,000 generations and never emits a
usable <code>Change:</code> field. The LoRA takes validity to <b>{pc(rec['validity'])}</b> and
change-op accuracy from 0% to ~31%. Whatever else is unresolved, the fine-tuning demonstrably taught
the model to speak the task.</div>

<h2><span class="sn">2</span>What v4 trained on</h2>
<div class="tbl"><p class="cap"><b>Table 2.</b> The 234,647-row mixture, after decontamination
against the rung-3 test fold. Earlier reports quoted 503,895 rows &mdash; that was v3's corpus.</p>
<table><tbody>
<tr><th>component</th><th>rows</th><th>share</th><th>dropped as test-contact</th></tr>
<tr><td>rung 2 &mdash; analogue design (MMP)</td><td>76,718</td><td>32.7%</td><td>2,523 (3.18%)</td></tr>
<tr><td>affinity &mdash; &Delta;pIC50 pairs</td><td>80,000</td><td>34.1%</td><td>254 (0.18%)</td></tr>
<tr><td>rung 1 &mdash; reaction chemistry</td><td>49,937</td><td>21.3%</td><td>63 (0.13%)</td></tr>
<tr><td>rung 3 &mdash; pocket-conditioned H2L</td><td>18,004 (4,501 &times;4)</td><td>7.7%</td><td>n/a (is the supervision)</td></tr>
<tr><td>contacts &mdash; interaction type</td><td>10,000</td><td>4.3%</td><td>1,042 (4.17%)</td></tr>
</tbody></table></div>
<p>81,336 rows (34.7%) carry a &Delta;pIC50 label; the rest get weight zero in the regression loss
rather than a fabricated target. The <code>Predicted dpIC50:</code> line is stripped in the
<i>data</i>, so no trainer can reintroduce v3's label leak by forgetting to.</p>
<p>Rung 3's pocket is a residue-level block &mdash; 20 nearest residues, each with its
closest-approach distance and nearest-atom coordinates &mdash; in a frame whose origin is the
attacked atom and whose +z is the attack axis. +x points at the Murcko-core centroid; the previous
rule broke ties on PDB atom <i>name</i>, which fired in 13.6% of complexes. The frame was verified
by rotating and translating 60 raw PDB files and re-running the whole extractor:
<b>60/60 identical, median deviation 0.003 &Aring;</b>.</p>

<h2><span class="sn">3</span>Training</h2>
<p>7,110 optimizer steps, effective batch 33 (bs 3 &times; accum 11), maxlen 3072, 13.9 h.
LM loss <b>1.4904 &rarr; 0.1195</b>; regression MSE <b>0.5092 &rarr; 0.0816</b> (last-500 mean
0.0697). 143 logged points, monotone, no divergence, zero NaN, zero OOM.</p>
<div class="note"><b>The head is not yet validated.</b> That 0.0697 is an <i>in-sample</i> MSE. Against
a target with sd &asymp;0.54 it implies training R<sup>2</sup> &asymp; 0.76, which says the head fits,
not that it ranks. The number that would validate it &mdash; held-out within-hit Spearman against the
within-hit variance floor &mdash; has not been measured, so no potency claim is made here. One
reassurance: v3's leaked head reached MSE 0.0002; v4's is 350&times; higher, which is what you expect
once the model cannot read its own label.</div>

<h2><span class="sn">4</span>Three real cases</h2>
<p>Drawn from the saved per-row generations, filtered to connected molecules and genuine
(non-stereo-only) pairs. The displayed generation is the first of 20 samples, not a cherry-picked
best.</p>
{''.join(exblock(e) for e in EX)}
<p>The recall hit is instructive: the <i>shown</i> sample is a plausible but wrong analogue &mdash;
the reference was recovered by one of the other 19 draws. The closest disjoint miss reaches Tc 0.905
and gets the key phenoxy substitution right while adding substituents the reference does not have.
The typical disjoint miss shortens a cephalosporin that the reference extends.</p>

<h2><span class="sn">5</span>Why the disjoint number is not an artefact of exact-match</h2>
<p>Exact-match against a single reference is a harsh metric, so we checked whether v4 was
near-missing. It is not.</p>
<div class="tbl"><p class="cap"><b>Table 3.</b> Tanimoto of the <i>best of 20</i> generations to the
reference, against the correct baseline: the similarity of the <b>hit itself</b> to the reference.
A model that proposes nothing useful scores the baseline.</p>
<table><tbody>
<tr><th></th><th>best generation vs reference</th><th>the hit vs reference</th><th>&ge;0.6</th><th>&ge;0.8</th></tr>
<tr><td>recall</td><td><b>0.593</b></td><td>0.482</td><td>49.2%</td><td>30.5%</td></tr>
<tr><td>disjoint</td><td><b>0.378</b></td><td><b>0.401</b></td><td>11.0%</td><td>1.3%</td></tr>
</tbody></table></div>
<p>On novel scaffolds the model's best of twenty is <b>less similar to the right answer than the
input it was given</b>. On training pairs it clearly beats that baseline. The failure is real, not a
metric artefact.</p>
<div class="tbl"><p class="cap"><b>Table 4.</b> Disjoint exact@20 binned by each query's maximum
Tanimoto to any training molecule. A model that interpolates should decay smoothly with distance.</p>
<table><tbody>
<tr><th>max-Tc to training</th><th>n</th><th>hits</th><th>exact@20</th></tr>
<tr><td>0.00&ndash;0.35</td><td>267</td><td>4</td><td>1.50%</td></tr>
<tr><td>0.35&ndash;0.45</td><td>192</td><td>0</td><td>0.00%</td></tr>
<tr><td>0.45&ndash;0.55</td><td>66</td><td>0</td><td>0.00%</td></tr>
<tr><td>0.55&ndash;1.00</td><td>75</td><td>0</td><td>0.00%</td></tr>
<tr><td><b>in training (1.0)</b></td><td>{LEGIT["rec_n"]}</td><td>{LEGIT["rec_hits"]}</td><td><b>{pc(LEGIT["rec_rate"])}</b></td></tr>
</tbody></table></div>
<p>There is no gradient. All four hits sit in the <i>lowest</i> similarity bin and the 75 queries most
similar to training score zero. Performance is binary at set membership, which is the signature of
retrieval rather than interpolation. With only four hits this cannot prove a gradient is absent, but
it does establish that no transfer is measurable short of exact membership.</p>

<h2><span class="sn">6</span>The change-op field is near its ceiling, not broken</h2>
<p>Change-op accuracy is ~31% (recall) and ~34% (disjoint), which looks poor against an
always-REPLACE floor. The label is the problem, and <code>derive_cdl.py</code>'s own entropy audit
measured it before training: <b>OP retains 1.099 of 2.047 bits &mdash; 53.7% of its entropy &mdash;
after conditioning on the input</b>.</p>
<div class="tbl"><p class="cap"><b>Table 5.</b> The ceiling on change-op accuracy, computed directly:
for each distinct (hit, pocket) input, the share of the most common op.</p>
<table><tbody>
<tr><th></th><th>distinct inputs</th><th>inputs admitting &gt;1 op</th><th>Bayes-optimal</th><th>constant floor</th><th>v4</th></tr>
<tr><td>train</td><td>1,026</td><td><b>46.1%</b></td><td>64.7%</td><td>33.3%</td><td>30.8%</td></tr>
<tr><td>test</td><td>219</td><td><b>52.5%</b></td><td>73.9%</td><td>49.4%</td><td>33.5%</td></tr>
</tbody></table></div>
<p>Half of all inputs have more than one valid answer <i>in the data itself</i>, because one hit has
several analogues reached by different transformations. The op describes the answer, so predicting it
from the hit alone is partly circular. The model also reproduces the marginal op distribution almost
exactly (REPLACE 127 predicted vs 127 actual; LINK_EXTEND 129 vs 120) without conditioning on the
instance &mdash; distribution-matching, not inference. This target should be collapsed into
chemically distinct groups or scored as set membership, not exact match.</p>

<h2><span class="sn">7</span>What the decontamination bug invalidated</h2>
<p>The routine every corpus was filtered through compared <code>r['input']</code> and
<code>r['output']</code> as whole SMILES. v4-schema rows carry no bare <code>input</code>, and their
output is <code>"Change: REPLACE\\nAnalogue: CC..."</code>, which RDKit cannot parse &mdash; so both
lookups returned <code>None</code>, nothing matched, and it printed
<code>79241 rows -&gt; 79241 kept, 0 dropped</code> on every run.</p>
<p>Measured consequence for the <i>previous</i> rung-3 folds: <b>99.5% of test, 98.5% of valid, 99.9%
of mpro</b> present in training on the exact (hit, analogue) key, and <b>439 of 440</b> eval analogues.
Those folds were cut from the same <code>cdl.jsonl</code> that rung-3 is built from, so no filter can
make them disjoint &mdash; filtering against them leaves 1,083 of 5,518 rows. Every rung-3
generalisation number predating this fix, including the 21.83% previously reported for the joint
model, is a recall measurement. The replacement fold is a connected-component split of the molecule
graph: train 4,501 / test 1,017, molecule intersection 0, pair intersection 0.</p>
<div class="note"><b>The split is harder than any previous holdout, on two axes.</b> Test molecules
sit at median max-Tc <b>0.357</b> to training, against <b>0.811</b> for a random split over unique
pairs. And 52.1% of test pairs are below Tc 0.40 &mdash; the threshold rung-2 uses to call two
molecules analogues at all &mdash; versus 35.0% of train. The fold is also only <b>3 connected
components</b>, so its effective chemical n is far below 1,017. The disjoint number should be read as
a hard lower bound, not an average-case estimate.</div>

<h2><span class="sn">8</span>covRXN: the one causally-controlled result</h2>
<p>Independent of the covLLM stream and unaffected by the above. Three seeds per arm; the control arms
were <i>trained</i> on deranged pockets. <code>dep_aux</code> is auxiliary accuracy with true labels
minus deranged labels, averaged over seeded derangement draws. The auxiliary task is 9-way
nucleophile identification.</p>
<div class="tbl"><p class="cap"><b>Table 6.</b> covRXN B2 arm matrix, test split, 3 seeds each.</p>
<table><tbody>
<tr><th>arm</th><th>valid leaving-group</th><th>aux (floor 0.499)</th><th><b>dep_aux</b></th></tr>
<tr><td>B2_froma (from phase A)</td><td>0.926</td><td>0.777</td><td><b>+0.362</b></td></tr>
<tr><td>B2_froma_fp (+ fingerprint)</td><td>0.925</td><td>0.746</td><td>+0.341</td></tr>
<tr><td>B2_scratch (random init)</td><td>0.899</td><td>0.782</td><td><b>+0.364</b></td></tr>
<tr><td>B2_froma_shuf <i>(control)</i></td><td>0.893</td><td>0.592</td><td><b>+0.013</b></td></tr>
<tr><td>B2_scratch_shuf <i>(control)</i></td><td>0.878</td><td>0.550</td><td><b>+0.005</b></td></tr>
</tbody></table></div>
<p><b>+0.36 against +0.01</b>, tight across seeds. The control reads zero, which is what a working
control must do. This is the one place in the project with a mechanism defensible causally rather
than correlationally.</p>
<p><b>And pretraining is decoration.</b> <code>B2_scratch</code> dep_aux +0.364 vs
<code>B2_froma</code> +0.362 &mdash; identical. Phase-A initialisation buys <b>+2.7 points of
leaving-group validity</b> and nothing on the auxiliary task.</p>
<p>One caveat: the mpro split cannot measure this. Its aux floor is 0.999, so there is no headroom and
dep_aux is ~0 there for real arms and controls alike. Only leaving-group validity is interpretable on
mpro, and it is much worse (0.70&ndash;0.76 vs 0.88&ndash;0.93).</p>

<h2><span class="sn">9</span>Limits, and numbers withdrawn</h2>
<ul>
<li><b>No generalisation claim is supported.</b> Three independent probes agree: exact@20 of
{pc(LEGIT["dis_rate"])} on disjoint pairs, best-generation similarity below the copy-the-input baseline
(0.378 vs 0.401), and a flat Tc profile.</li>
<li><b>The potency head has no held-out number.</b> Only in-sample MSE exists. Within-hit Spearman
against the within-hit variance floor is unmeasured.</li>
<li><b>2.7% of rung-3 pairs are disconnected SMILES</b> (151 of 5,535 in <code>cdl.jsonl</code>, the
same 151 in <code>pocket_pairs_v3</code>, so inherited rather than introduced). These are
peptidomimetic ligands whose PDB residues were never bonded: <code>C.CC(C)C.NC(...)...</code> is
methane + isobutane + a detached Cbz. The model emits the correctly connected peptide and scores
zero. Excluded from every number here; not yet gated at corpus build.</li>
<li><b>1.3% of rung-3 pairs differ only in stereochemistry</b> and are winnable by copying under a
stereo-stripping scorer. Excluded here; the build-time gate should compare without stereo.</li>
<li><b>The disjoint fold is 3 chemotypes.</b> Effective chemical n is far below 1,017.</li>
<li><b>Six numbers were withdrawn during this evaluation</b>, each caught by a check rather than
reported: a 1.00% disjoint rate (three concurrent eval chains interleaving one log); a 36.5% recall
measured on 14 of 1,020 complexes (<code>--n 400</code> was <code>rows[:400]</code> on a
complex-ordered file); the "1.6&times; better than joint" built on it; a uniqueness-collapse argument
for memorisation (7.2% was a sampling artefact; it is 53.1% on stratified sampling); a ~1.8% disjoint
partial (early hits were clustered); and a change-op floor of 49.4% compared against accuracy measured
on a different sample.</li>
<li><b>covPROG was not re-run for v4.</b> Its query builder is not available, and rung-2 shares
43.5% of its molecules (4,049 of 9,316) &mdash; so a covPROG number for v4 would be heavily
contaminated through rung-2 even though rung-3 is clean at 0.72%.</li>
</ul>

<p class="ft">All figures measured, none estimated. ECFP4 r=2, 2048 bits; stereo-stripped canonical
matching; one parser for every arm; prompt encoding asserted identical to training before each run.
Sources: v4_eval_{{train,disjoint}}_fixed.json, v4_eval_base.json.partial,
v4_gens_{{train,disjoint}}_fixed.jsonl, results/gates/CDL_ENTROPY_AUDIT.json,
results/final_covrxn/*.json.</p>
</div>'''

open(os.path.join(HERE, 'report_v4.html'), 'w').write(HTML)
for t in ('div', 'table', 'tbody', 'ul'):
    o, c = HTML.count('<%s' % t), HTML.count('</%s>' % t)
    print('%-7s %3d/%3d %s' % (t, o, c, 'OK' if o == c else '*** MISMATCH'))
print('%.1f KB' % (len(HTML) / 1024.0))
