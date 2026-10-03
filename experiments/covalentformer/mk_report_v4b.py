#!/usr/bin/env python
"""CovalentFormer v4b report -- STRUCTURE FIRST, numbers later.

HOW THIS IS MEANT TO BE USED. Every data-side fact in this report is measured and final: corpus
composition, integrity gates, split construction, label distributions, the real training rows, and
the live training dynamics. Every MODEL-side result is a slot in RESULTS below, rendered as an
explicit PENDING chip until a number is dropped in. When v4b finishes, filling this report in is a
data entry job against `RESULTS`, not a rewrite -- which is the whole point of building it now.

THE FLOORS ARE PRE-REGISTERED. Each metrics table declares its majority-class (or copy-the-input)
floor from the measured label distribution BEFORE any result exists. That matters because several
v4-era numbers looked impressive until compared against the right floor -- the change-op field read
as a 49.4% success until the conditional-entropy audit put the Bayes-optimal ceiling at 64.7% and
the majority-class floor at 37.2%, which turned "the model is weak here" into "the field is nearly
saturated". Declaring the floor first makes that mistake impossible to repeat.
"""
from __future__ import annotations
import html, json, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(os.path.join(SP, 'v4b_examples.json')))

# ============================================================================================
# RESULTS -- every None renders as a PENDING chip. Drop numbers in here when training finishes.
# ============================================================================================
PEND = None
RESULTS = {
    # ---- rung 1: four classification/generation subtasks, v4b vs base Qwen2.5-7B ----
    'rung1': {
        'ADDUCT':      {'base_exact': PEND, 'v4b_exact': PEND, 'base_valid': PEND, 'v4b_valid': PEND, 'loss': PEND},
        'RETRO':       {'base_exact': PEND, 'v4b_exact': PEND, 'base_valid': PEND, 'v4b_valid': PEND, 'loss': PEND},
        'SELECTIVITY': {'base_acc': PEND, 'v4b_acc': PEND, 'base_macro_f1': PEND, 'v4b_macro_f1': PEND, 'loss': PEND},
        'WARHEAD_ID':  {'base_acc': PEND, 'v4b_acc': PEND, 'base_macro_f1': PEND, 'v4b_macro_f1': PEND, 'loss': PEND},
        'gen_example': None,
    },
    'rung2': {'base_valid': PEND, 'v4b_valid': PEND, 'base_warhead_kept': PEND, 'v4b_warhead_kept': PEND,
              'base_scaffold_kept': PEND, 'v4b_scaffold_kept': PEND, 'base_novel': PEND, 'v4b_novel': PEND,
              'exact_at_1': PEND, 'exact_at_20': PEND, 'loss': PEND, 'gen_example': None},
    'rung3': {'valid': PEND, 'warhead_kept': PEND, 'exact_at_1': PEND, 'exact_at_20': PEND,
              'op_acc': PEND, 'intent_acc': PEND, 'warhead_move_acc': PEND,
              'contact_p': PEND, 'contact_r': PEND, 'contact_f1': PEND,
              'tc_to_ref': PEND, 'tc_copy_floor': PEND, 'loss': PEND, 'gen_example': None},
    'affinity': {'mse': PEND, 'mae': PEND, 'r2': PEND, 'spearman_within_hit': PEND,
                 'variance_floor': PEND, 'n_groups': PEND, 'gen_example': None},
}

# ============================================================================================
# MEASURED FACTS -- all verified 2026-10-03 against the files on ai-gpu-a100.
# ============================================================================================
CORPUS = [  # (rung, rows, pct, unique, mean_tok, tok_share, heldout)
    ('rung 1 &middot; reaction chemistry', 373935, 63.9, '373,935', 53, 26, 32082),
    ('affinity &middot; potency head', 130667, 22.3, '130,667', 207, 36, 12408),
    ('rung 2 &middot; analogue design', 68868, 11.8, '68,868', 161, 15, 5988),
    ('rung 3 &middot; covalent H2L', 12060, 2.1, '3,015 &times;4', 1495, 24, 334),
]
TOTAL_TRAIN, TOTAL_VALID, TOTAL_TOK = 585530, 3034, 75.8

GATES = [  # (label, n, note)
    ('rows offered by the covalent-program miner', 5535, 'pairs with a hit, an analogue and a co-crystal'),
    ('LIGAND block disagreed with the Hit SMILES', -1527, 'pocket geometry described a different molecule'),
    ('disconnected SMILES (salts, fragments)', -150, 'multi-fragment hit or analogue'),
    ('stereo-only difference', -61, 'winnable by copying the input'),
    ('identical after canonicalisation', -17, 'no edit at all'),
    ('duplicate pairs removed', -431, 'same (hit, analogue) seen twice'),
    ('<b>rung 3 train rows surviving</b>', 3015, '<b>plus 334 held out</b>'),
]

TC3 = [('&lt; 0.30', 0, 'excluded by the miner'), ('0.30 &ndash; 0.40', 1192, ''),
       ('0.40 &ndash; 0.55', 953, ''), ('0.55 &ndash; 0.70', 513, ''), ('&ge; 0.70', 357, '')]
TC3 = [(k, n) for k, n, _ in TC3]

TRAIN_CFG = [
    ('base model', 'Qwen2.5-7B (frozen, bf16)'),
    ('adapter', 'LoRA r=16, &alpha;=32, dropout 0.05 on q/k/v/o/gate/up/down &mdash; 196 adapters'),
    ('potency head', 'Siamese MLP on [hit, ana, ana&minus;hit], 2,774,529 params, input 3&times;3584'),
    ('head input', 'forward hook on the final Qwen2RMSNorm (not output_hidden_states)'),
    ('batch', 'bs 3 &times; accum 11 = 33 examples per optimizer step'),
    ('max length', '3,072 tokens; length-grouped batching, p50 198 chars / p95 517 / max 2,969'),
    ('schedule', '1.00 epoch = 17,743 steps; pre-tokenised before step 0'),
    ('validation', 'every 500 steps on 3,000 held-out rows, patience 4'),
    ('insurance', 'checkpoint every 500 steps, full snapshot every 2,000, --auto-resume'),
]

LIVE = [
    ('step', '6,950 / 17,743', '39.2% of epoch 1'),
    ('train LM loss', '~0.143', 'running mean over the last 10 logged steps'),
    ('train potency MSE', '~0.10', '~165 labelled rows per step of 33'),
    ('validation LM loss', '0.1782', 'monotone across all 5 evaluations'),
    ('validation potency MSE', '0.1107', 'R&sup2; &asymp; 0.62 against the label variance'),
    ('errors / NaN / OOM', '0 / 0 / 0', 'across 2,900 logged steps'),
    ('watchdog relaunches', '0', 'armed 06:22, every check since has logged ok'),
    ('throughput', '6.2 s/step', '689 tokens/s &mdash; see &sect;11'),
    ('remaining', '18.6 h', 'for epoch 1 at the measured rate'),
]
VALID_CURVE = [(4500, 0.2057, 0.1324), (5000, 0.1971, 0.1226), (5500, 0.1912, 0.1248),
               (6000, 0.1837, 0.1286), (6500, 0.1782, 0.1107)]

GPU = [
    ('measured throughput', '689 tokens/s', 'the same trainer reached 3,437 tok/s on v4'),
    ('GPU utilization', '48% mean, 0&ndash;81% swing', '12 samples over 24 s'),
    ('power draw', '~181 W of a 400 W cap', 'SM clock pinned at max 1,410 MHz'),
    ('CPU', '1.01 load average on 12 cores', 'python at 103% of one core &mdash; not dataloader-bound'),
    ('padded tokens per micro-batch', '~389', 'bs 3 &times; 130-token corpus mean'),
    ('effective compute', '42 of 312 TFLOPS', '13.5% of A100 bf16 peak'),
    ('micro-batches per epoch', '195,173', '~9,300 under an 8,192-token budget &mdash; 21&times; fewer'),
]


def esc(s):
    return html.escape(str(s if s is not None else ''))


def chip(v, unit='', kind=None):
    """A result slot. None renders as an explicit PENDING chip."""
    if v is None:
        return '<span class="pend">pending</span>'
    k = ' ' + kind if kind else ''
    return '<span class="num%s">%s%s</span>' % (k, esc(v), unit)


def row_example(title, instruction, output, note=None, extra=None):
    e = '<div class="exnote">%s</div>' % note if note else ''
    x = extra or ''
    return """<div class="ex">
  <div class="exh">%s</div>
  %s
  <div class="exg">
    <div class="exlab">INPUT &mdash; what the model is shown</div>
    <pre class="exin">%s</pre>
    <div class="exlab out">LABEL &mdash; what it must produce</div>
    <pre class="exout">%s</pre>
  </div>
  %s
</div>""" % (title, e, esc(instruction), esc(output), x)


def gen_example(slot, note):
    """A v4b generation. Renders a placeholder frame until the checkpoint exists."""
    if slot is None:
        return """<div class="ex pendex">
  <div class="exh">CovalentFormer v4b &mdash; input and output <span class="pend">pending</span></div>
  <div class="exnote">%s</div>
  <div class="exg">
    <div class="exlab">INPUT &mdash; held-out row, fed verbatim</div>
    <pre class="exin ph">awaiting the trained checkpoint</pre>
    <div class="exlab out">v4b OUTPUT &mdash; generated, not retrieved</div>
    <pre class="exout ph">awaiting the trained checkpoint</pre>
  </div>
</div>""" % note
    return row_example('CovalentFormer v4b &mdash; input and output', slot['input'], slot['output'], note)


def table(headers, rows, cls=''):
    h = ''.join('<th>%s</th>' % x for x in headers)
    b = ''.join('<tr>%s</tr>' % ''.join('<td>%s</td>' % c for c in r) for r in rows)
    return '<div class="tw"><table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (cls, h, b)


# ------------------------------------------------------------------------------------------
CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#15181d; --ink2:#454c57; --ink3:#6b7280;
  --line:#dfe3e9; --line2:#eef1f5;
  --accent:#0f6f74; --accent2:#0a4f53; --accent-bg:#e6f2f2;
  --good:#1a7f4b; --good-bg:#e7f5ec;
  --warn:#9a6212; --warn-bg:#fdf3e2;
  --bad:#a3302a; --bad-bg:#fbeceb;
  --code:#f3f5f8; --codeink:#1d2229;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --serif:ui-serif,Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12151a; --panel:#181c23; --ink:#e8eaee; --ink2:#b3bac6; --ink3:#8b93a1;
  --line:#2a313b; --line2:#222831;
  --accent:#4fb3b8; --accent2:#7fd0d4; --accent-bg:#16292b;
  --good:#5ec98a; --good-bg:#14281d;
  --warn:#e0a84e; --warn-bg:#2b2113;
  --bad:#e4756d; --bad-bg:#2b1716;
  --code:#1e232b; --codeink:#dfe3ea;
}}
:root[data-theme="dark"]{
  --bg:#12151a; --panel:#181c23; --ink:#e8eaee; --ink2:#b3bac6; --ink3:#8b93a1;
  --line:#2a313b; --line2:#222831;
  --accent:#4fb3b8; --accent2:#7fd0d4; --accent-bg:#16292b;
  --good:#5ec98a; --good-bg:#14281d;
  --warn:#e0a84e; --warn-bg:#2b2113;
  --bad:#e4756d; --bad-bg:#2b1716;
  --code:#1e232b; --codeink:#dfe3ea;
}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.page{max-width:980px;margin:0 auto;padding:44px 22px 110px}
h1{font-family:var(--serif);font-size:clamp(28px,5vw,40px);line-height:1.14;margin:0 0 6px;
  letter-spacing:-.018em;text-wrap:balance;font-weight:600}
.sub{color:var(--ink3);font-size:15px;margin:0 0 4px}
.kicker{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin:0 0 14px;font-weight:600}
h2{font-family:var(--serif);font-size:23px;line-height:1.24;margin:52px 0 4px;font-weight:600;
  letter-spacing:-.01em;display:flex;align-items:baseline;gap:12px;text-wrap:balance}
h2 .sn{font-family:var(--mono);font-size:12px;color:var(--accent);border:1px solid var(--line);
  background:var(--accent-bg);border-radius:4px;padding:2px 7px;flex:none;font-weight:600}
h3{font-size:16px;margin:30px 0 6px;font-weight:650;letter-spacing:-.005em}
h2+.lede,h3+.lede{color:var(--ink2);margin:8px 0 14px}
p{margin:0 0 13px}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
hr{border:0;border-top:1px solid var(--line);margin:40px 0}
code{font-family:var(--mono);font-size:.885em;background:var(--code);color:var(--codeink);
  padding:1.5px 5px;border-radius:4px;border:1px solid var(--line2)}

/* ---- status banner ---- */
.banner{background:var(--warn-bg);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:8px;padding:16px 18px;margin:22px 0 10px}
.banner .bt{font-weight:700;color:var(--warn);font-size:13px;letter-spacing:.04em;
  text-transform:uppercase;font-family:var(--mono);margin-bottom:6px}
.banner p{margin:0;color:var(--ink2);font-size:14.5px}

/* ---- panels ---- */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:18px 20px;margin:18px 0}
.panel>h3:first-child{margin-top:0}

/* ---- tables ---- */
.tw{overflow-x:auto;margin:14px 0;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:min(100%,520px)}
th{text-align:left;font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink3);font-weight:600;padding:11px 13px;border-bottom:1px solid var(--line);
  background:var(--line2);white-space:nowrap}
td{padding:10px 13px;border-bottom:1px solid var(--line2);vertical-align:top;color:var(--ink2)}
tbody tr:last-child td{border-bottom:0}
td:first-child{color:var(--ink);font-weight:500}
table.nums td+td,table.nums th+th{text-align:right;font-variant-numeric:tabular-nums}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;font-weight:600;color:var(--ink)}
.num.good{color:var(--good)}
.num.bad{color:var(--bad)}
.num.dim{color:var(--ink3);font-weight:500}
.pend{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--warn);background:var(--warn-bg);border:1px dashed var(--warn);border-radius:4px;
  padding:2px 7px;font-weight:600;white-space:nowrap}
.floor{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink3);font-size:13px}

/* ---- labeled row examples ---- */
.ex{border:1px solid var(--line);border-radius:9px;margin:16px 0;overflow:hidden;background:var(--panel)}
.ex.pendex{border-style:dashed}
.exh{font-family:var(--mono);font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;
  font-weight:600;color:var(--ink);background:var(--line2);padding:10px 14px;
  border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.exnote{padding:11px 14px 0;color:var(--ink3);font-size:13.5px}
.exg{padding:12px 14px 14px}
.exlab{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);font-weight:600;margin:8px 0 5px}
.exlab.out{color:var(--accent)}
pre{margin:0;font-family:var(--mono);font-size:12.5px;line-height:1.5;white-space:pre-wrap;
  word-break:break-word;overflow-wrap:anywhere;background:var(--code);color:var(--codeink);
  padding:11px 13px;border-radius:6px;border:1px solid var(--line2);overflow-x:auto}
pre.exin{border-left:3px solid var(--line);max-height:420px;overflow-y:auto}
pre.exout{border-left:3px solid var(--accent)}
pre.ph{color:var(--ink3);font-style:italic;background:transparent;border-style:dashed;
  max-height:none;overflow-y:visible}
pre.scroll{max-height:380px;overflow-y:auto}

/* ---- key/value strip ---- */
.kv{display:grid;grid-template-columns:minmax(150px,auto) 1fr;gap:0;border:1px solid var(--line);
  border-radius:8px;overflow:hidden;background:var(--panel);margin:14px 0}
.kv>div{padding:9px 13px;border-bottom:1px solid var(--line2);font-size:14px}
.kv>div:nth-child(2n+1){font-family:var(--mono);font-size:12px;color:var(--ink3);
  background:var(--line2);letter-spacing:.02em}
.kv>div:nth-child(2n){color:var(--ink2)}
.kv>div:nth-last-child(1),.kv>div:nth-last-child(2){border-bottom:0}

/* ---- callouts ---- */
.note{border-left:3px solid var(--accent);background:var(--accent-bg);padding:13px 16px;
  border-radius:0 7px 7px 0;margin:16px 0;font-size:14.5px;color:var(--ink2)}
.note b{color:var(--ink)}
.defect{border-left:3px solid var(--bad);background:var(--bad-bg);padding:13px 16px;
  border-radius:0 7px 7px 0;margin:16px 0;font-size:14.5px;color:var(--ink2)}
.defect b{color:var(--bad)}
.ok{border-left:3px solid var(--good);background:var(--good-bg);padding:13px 16px;
  border-radius:0 7px 7px 0;margin:16px 0;font-size:14.5px;color:var(--ink2)}
.ok b{color:var(--good)}

/* ---- phase cards ---- */
.phase{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:0 9px 9px 0;
  background:var(--panel);padding:18px 20px;margin:20px 0}
.phase .ph-h{display:flex;gap:11px;align-items:baseline;flex-wrap:wrap;margin-bottom:3px}
.phase .ph-n{font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:.1em;
  color:var(--accent);text-transform:uppercase}
.phase .ph-t{font-family:var(--serif);font-size:19px;font-weight:600;letter-spacing:-.008em}
.phase .ph-m{font-family:var(--mono);font-size:11.5px;color:var(--ink3);margin-left:auto;
  font-variant-numeric:tabular-nums}
.phase>p{color:var(--ink2);margin:9px 0 4px}
.footer{margin-top:70px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--ink3);font-size:12.5px;font-family:var(--mono)}
@media(max-width:620px){
  .kv{grid-template-columns:1fr}
  .kv>div:nth-child(2n+1){border-bottom:0;padding-bottom:0;background:transparent}
  .kv>div:nth-child(2n){padding-top:3px}
  .phase .ph-m{margin-left:0;width:100%}
}
"""


def build():
    o = []
    A = o.append

    # =========================== HEAD ===========================
    A('<!doctype html>')
    A('<html lang="en"><head><meta charset="utf-8">')
    A('<meta name="viewport" content="width=device-width,initial-scale=1">')
    A('<title>CovalentFormer v4b &mdash; report structure</title>')
    A('<style>%s</style>' % CSS)
    A('</head><body>')
    A('<div class="page">')
    A('<p class="kicker">CovalentFormer &middot; v4b &middot; report structure</p>')
    A('<h1>A covalent medicinal chemist, trained in three phases</h1>')
    A('<p class="sub">Qwen2.5-7B + LoRA over 585,530 rows: reaction chemistry, analogue design, '
      'and pocket-conditioned covalent hit-to-lead, with a Siamese potency head.</p>')
    A('<p class="sub">Compiled 2026-10-03 &middot; <code>rlhf-system</code> @ <code>16040d4</code></p>')

    A("""<div class="banner">
  <div class="bt">Structure final &middot; model results pending</div>
  <p>v4b is <b>39.2% through its first epoch</b> (step 6,950 of 17,743) with ~18.6 h remaining.
  Everything on the data side of this report is measured and final &mdash; corpus, integrity gates,
  splits, label distributions, the real training rows, and the live loss curve. Every
  <span class="pend">pending</span> chip is a model result that needs the finished checkpoint.
  Each metrics table below already declares its <b>pre-registered floor</b>, so the results can be
  read the moment they land.</p>
</div>""")

    # =========================== 1. THE THREE PHASES ===========================
    A('<h2><span class="sn">1</span>The three training phases</h2>')
    A('<p class="lede">The model is taught covalent medicinal chemistry in three rungs, each one '
      'presupposing the last. Rung 1 teaches what a covalent reaction <i>is</i>. Rung 2 teaches how '
      'to edit a molecule without breaking it. Rung 3 asks for the real job: given a pocket and a '
      'covalent hit, propose a better covalent analogue and say why. A fourth channel, the potency '
      'head, runs across rungs 2 and 3 and is the only part of the system that outputs a number '
      'rather than text.</p>')

    A(table(
        ['phase', 'rows trained', 'share', 'unique', 'mean tokens', 'share of tokens', 'held out'],
        [[n, '{:,}'.format(r), '%.1f%%' % p, u, str(mt), '%d%%' % ts, '{:,}'.format(h)]
         for n, r, p, u, mt, ts, h in CORPUS]
        + [['<b>total</b>', '<b>%s</b>' % '{:,}'.format(TOTAL_TRAIN), '<b>100%</b>',
            '&mdash;', '<b>130</b>', '<b>100%</b>', '<b>50,812</b>']],
        'nums'))
    A('<p style="color:var(--ink3);font-size:13.5px;margin-top:-2px">Rung 3 is repeated 4&times; in '
      'the mix (3,015 unique pairs &rarr; 12,060 rows) because it is the target task and is three '
      'orders of magnitude rarer than rung 1. The token-share column is why: at 1,495 tokens a row '
      'it is 2.1% of the rows but 24% of the compute.</p>')

    # ---- Phase 1 ----
    A("""<div class="phase">
  <div class="ph-h"><span class="ph-n">Phase 1</span><span class="ph-t">Reaction chemistry</span>
  <span class="ph-m">373,935 rows &middot; 63.9%</span></div>
  <p>Four subtasks that force the model to represent the covalent bond-forming event itself, not
  just the molecules around it. <b>ADDUCT</b> applies the reaction forward; <b>RETRO</b> inverts it;
  <b>SELECTIVITY</b> names the residue attacked; <b>WARHEAD_ID</b> names the electrophile class.
  Together they are the vocabulary the later rungs assume. No pocket is shown at this rung &mdash;
  these are properties of the ligand alone.</p>
</div>""")
    r1 = EX['rung1']
    sub_notes = {
        'ADDUCT': 'Forward direction. The acrylamide C=C is attacked by a cysteine thiol, so the '
                  'vinyl becomes a thioether &mdash; note the product keeps every stereocentre.',
        'RETRO': 'Inverse direction, and the harder of the two: the model must recognise that '
                 '<code>C(=N)SC</code> is a nitrile that has already reacted, and undo it back to '
                 '<code>C#N</code>.',
        'SELECTIVITY': 'Six residue classes. A sulfonyl fluoride is the giveaway here &mdash; it is '
                       'the one warhead that routinely attacks histidine and tyrosine rather than '
                       'cysteine.',
        'WARHEAD_ID': 'Eight electrophile classes. Pure recognition, and the cheapest signal in the '
                      'corpus to learn.',
    }
    for st in ('ADDUCT', 'RETRO', 'SELECTIVITY', 'WARHEAD_ID'):
        d = r1.get(st)
        if not d:
            continue
        instr = d['instruction'] + (('\n\nInput: ' + d['input']) if d.get('input') else '')
        A(row_example('Rung 1 &middot; %s &mdash; a real labelled row' % st, instr, d['output'],
                      sub_notes.get(st)))

    d1 = EX['rung1_dist']
    A('<h3>What the rung-1 labels actually look like</h3>')
    A('<p class="lede">These distributions set the majority-class floors used in &sect;5. Both '
      'label sets are heavily skewed, so accuracy alone would be misleading &mdash; macro-F1 is the '
      'honest metric.</p>')
    A(table(['warhead class', 'rows', 'share'],
            [[w, '{:,}'.format(n), '%.1f%%' % (100.0 * n / d1['n_warhead'])]
             for w, n in d1['warhead'][:8]], 'nums'))
    A(table(['residue attacked', 'rows', 'share'],
            [[s, '{:,}'.format(n), '%.1f%%' % (100.0 * n / d1['n_sel'])]
             for s, n in d1['selectivity']], 'nums'))

    # ---- Phase 2 ----
    A("""<div class="phase">
  <div class="ph-h"><span class="ph-n">Phase 2</span><span class="ph-t">Analogue design &middot; hit-to-lead</span>
  <span class="ph-m">68,868 rows &middot; 11.8%</span></div>
  <p>Matched molecular pairs over covalent ligands. Given a hit, propose a close analogue that
  <b>keeps the warhead and the binding scaffold</b> &mdash; the two constraints that make an edit a
  medicinal-chemistry move rather than an arbitrary new molecule. The label names the transform
  explicitly before giving the SMILES, which turns an open-ended generation into a decision the
  model can be scored on. Still no pocket: this rung teaches editing, not placement.</p>
</div>""")
    r2 = EX['rung2']
    if r2:
        A(row_example('Rung 2 &mdash; a real labelled row', r2['instruction'], r2['output'],
                      'A textbook medicinal-chemistry move: <b>N-methylpiperazine &rarr; '
                      'morpholine</b>, swapping a basic amine for an ether to shed basicity and the '
                      'hERG and hepatic-clearance liabilities that come with it. The acrylamide '
                      'warhead and the entire aminothiazole-benzamide scaffold are untouched, which '
                      'is exactly the constraint the instruction imposes.'))
    A("""<div class="defect"><b>A real defect in this rung, found while compiling this report.</b>
  1,088 of 68,868 rung-2 rows (1.58%) carry a degenerate transform label of the form
  <code>substitute X -&gt; X</code> &mdash; for instance <code>substitute Cl -&gt; Cl</code>, emitted
  when the chloro <i>moves position</i> on a ring: the labeller names the atom swapped but not the
  position moved, so the label reads as a no-op. The SMILES pair is correct and the edit is real;
  only the natural-language label is uninformative. It is small enough not to justify restarting a
  run that is 39% complete, and it is logged here rather than quietly fixed.</div>""")

    # ---- Phase 3 ----
    A("""<div class="phase">
  <div class="ph-h"><span class="ph-n">Phase 3</span><span class="ph-t">Covalent hit-to-lead, pocket-conditioned</span>
  <span class="ph-m">3,015 unique &middot; 2.1%</span></div>
  <p>The target task. The model is given the hit <i>and its own co-crystal pocket</i>, expressed in
  the <b>covalent reaction frame</b>: the attacked electrophilic atom sits at the origin, the attack
  axis is +z, and +x points toward the Murcko-scaffold centroid. That frame is the reason the
  geometry generalises across complexes &mdash; every pocket is expressed relative to the chemistry
  that is about to happen, not to an arbitrary crystallographic axis.</p>
  <p>The output is a full medicinal-chemistry rationale, not a bare SMILES: the structural
  <b>Change</b>, the <b>Intent</b> (is this edit for recognition, for reactivity, or both), the
  <b>Warhead</b> move, the contacts it <b>Gains</b> and <b>Loses</b>, and only then the
  <b>Analogue</b>.</p>
</div>""")
    r3 = EX['rung3']
    if r3:
        A(row_example(
            'Rung 3 &mdash; a real labelled row (%s, %s)' % (esc(r3.get('pdb_a') or ''), esc(r3.get('protein') or '')),
            r3['instruction'], r3['output'],
            'HIV-1 reverse transcriptase. The hit carries a <b>fluorosulfate</b> that attacks '
            '<b>TYR181</b> at a 135&deg; approach angle &mdash; so the ELEC atom at the origin is '
            'sulfur, not carbon. The label swaps that warhead to a haloacetamide and extends the '
            'linker, trading four contacts away for four others including a new TYR188 hydrogen '
            'bond. The LIGAND block lists every heavy atom in frame coordinates; the POCKET block '
            'gives the 20 nearest of 43 residues by minimum distance.',
            extra='<div class="exg"><div class="exlab">Why this row survived the integrity gate</div>'
                  + table(['check', 'value'],
                          [['heavy atoms in the Hit SMILES', chip(r3.get('hit_heavy'))],
                           ['atoms in the LIGAND block', chip(r3.get('n_lig_block'))],
                           ['ratio (gate accepts 0.80&ndash;1.25)', chip(
                               '%.2f' % (r3['n_lig_block'] / r3['hit_heavy'])
                               if r3.get('hit_heavy') else None)],
                           ['Tanimoto, hit to analogue', chip(r3.get('tc'))],
                           ['nucleophile', chip(r3.get('nucleophile'))]], 'nums')
                  + '</div>'))

    d3 = EX['rung3_dist']
    A('<h3>What the rung-3 labels actually look like</h3>')
    A('<p class="lede">These four distributions are the pre-registered floors for &sect;7. Two of '
      'them are badly skewed, which is exactly the trap the v4 change-op number fell into.</p>')
    for key, title in (('op', 'Change'), ('intent', 'Intent'), ('warhead', 'Warhead move')):
        rows = d3[key]
        A('<h3 style="margin:18px 0 4px;font-size:14px">%s</h3>' % title)
        A(table(['label', 'rows', 'share'],
                [[k, '{:,}'.format(n), '%.1f%%' % (100.0 * n / d3['n'])] for k, n in rows[:6]],
                'nums'))
    A('<h3 style="margin:26px 0 4px;font-size:14px">How far apart a hit and its analogue actually are</h3>')
    A('<p class="lede">This is the single most important number for calibrating expectations on rung 3, '
      'and it is a property of the data rather than the model.</p>')
    A(table(['hit &rarr; analogue Tanimoto', 'pairs', 'share'],
            [[k, '{:,}'.format(n), '%.1f%%' % (100.0 * n / 3015)] for k, n in TC3]
            + [['<b>mean 0.482 &middot; median 0.436 &middot; p10 0.320</b>', '<b>3,015</b>', '<b>100%</b>']],
            'nums'))
    A("""<div class="note"><b>The median rung-3 "analogue" shares only Tc 0.436 with its hit.</b>
  39.5% of pairs sit between 0.30 and 0.40, and the miner enforces a hard floor at 0.30 &mdash; no
  pair falls below it. These are <i>programme-level</i> jumps across a real optimisation campaign,
  not tight matched molecular pairs, which is why exact recovery of the reference analogue is a very
  demanding metric here. It also reframes v4's headline failure: a best-of-20 output at Tc 0.378 to
  the reference, when the reference is itself only ~0.44 from the hit, means the model landed about
  as far from the target as its own input did.</div>""")
    A("""<div class="defect"><b>A second real defect, worth stating before any result is read.</b>
  308 of 3,015 rung-3 rows (10.2%) have <code>Warhead: SWAP_TO:none</code> &mdash; the "improved"
  analogue has <i>no electrophile at all</i>. These are mined from real covalent programs where a
  team abandoned the warhead, so they are genuine medicinal chemistry, but they contradict the
  instruction the model is given ("keep the warhead"). They will inflate any warhead-retention
  metric's apparent error. &sect;7 therefore reports warhead retention on the
  <code>KEEP</code> subset separately.</div>""")

    # ---- Potency head ----
    A("""<div class="phase">
  <div class="ph-h"><span class="ph-n">Channel 4</span><span class="ph-t">The potency head</span>
  <span class="ph-m">130,667 rows &middot; 22.3%</span></div>
  <p>A Siamese MLP over <code>[hit_vec, ana_vec, ana_vec &minus; hit_vec]</code>, reading the final
  hidden state through a forward hook, trained with a masked regression loss so unlabelled rows
  contribute nothing. It predicts <b>&Delta;pIC50</b> &mdash; the potency gain of the analogue over
  its hit &mdash; and it is the piece of v3 that had to be rebuilt from scratch.</p>
</div>""")
    aff = EX['affinity']
    if aff:
        A(row_example('Potency channel &mdash; a real labelled row', aff['instruction'],
                      aff['output'] + '\n\n[regression target, not generated: dpic50 = %s]' % aff.get('dpic50'),
                      'The structural edit is acrylamide &rarr; propiolamide &mdash; same heavy-atom '
                      'count, a more reactive Michael acceptor. The &Delta;pIC50 of +1.32 is the '
                      'regression target and is read by the head, <b>not</b> emitted as text.'))
    A("""<div class="ok"><b>The v3 leak is dead in the data, not just at train time.</b> Every v3
  potency number was void because the response span contained the literal line
  <code>Predicted dpIC50: +1.32</code> &mdash; inside the very span a response-pooled head reads, so
  the head could read its own answer. v3b stripped it in the trainer, which left the defect one
  forgotten flag away from returning. v4b strips it in the corpus: a scan of all 585,530 training
  rows finds <b>0</b> whose output mentions <code>dpIC50</code>, and the build refuses to write a
  file that fails that check.</div>""")

    # =========================== 2. INTEGRITY ===========================
    A('<h2><span class="sn">2</span>What was thrown away, and why</h2>')
    A('<p class="lede">Rung 3 is the task that matters and the task with the least data, which makes '
      'it the easiest place to accept bad rows. 31.7% of what the miner offered was rejected.</p>')
    A(table(['gate', 'rows', 'what it catches'],
            [[g, '{:+,}'.format(n) if n < 0 else '{:,}'.format(n), note] for g, n, note in GATES], 'nums'))
    A("""<div class="note"><b>The biggest single gate is the one nobody would think to write.</b>
  1,527 rows &mdash; 27.6% of everything offered &mdash; had a LIGAND coordinate block whose atom
  count disagreed with the Hit SMILES by more than &plusmn;25%. The pocket geometry in those rows
  describes a <i>different molecule</i> than the SMILES the model is asked to edit, which would have
  taught the model to ignore the geometry. The gate compares ligand-block atoms against the hit's
  heavy-atom count and demands a ratio in 0.80&ndash;1.25.</div>""")
    A('<div class="ok"><b>Post-gate integrity checks, run against the actual training file.</b> '
      'All 12,060 rung-3 analogue SMILES parse under RDKit (0 failures). 0 rows carry the dpIC50 '
      'leak. <code>Change: NONE</code> survives on only 16 of 3,015 rows (0.5%).</div>')

    # =========================== 3. HOW WE MEASURE ===========================
    A('<h2><span class="sn">3</span>How the held-out sets are built</h2>')
    A('<p class="lede">v4 had a held-out split for rung-3 generation only, which is why it has no '
      'validation loss, no early stopping and no per-rung evaluation. v4b splits all four channels, '
      'each on the key that makes memorisation useless.</p>')
    A(table(['channel', 'split key', 'held out', 'why this key'],
            [['rung 1', 'input molecule', '32,082',
              'all four subtasks key off the same molecule, so a row-level split would let a test '
              'molecule reappear under another subtask'],
             ['rung 2', 'deduplicated pair', '5,988',
              'pair-disjoint; 4,385 duplicate pairs (5.5%) were removed first, after a random split '
              'put 600 test pairs&rsquo; twins into training'],
             ['rung 3', 'deduplicated pair', '334', 'same construction; 431 duplicates removed'],
             ['affinity', 'hit group', '12,408',
              'the metric is within-hit Spearman, so every analogue of a hit must land on one side '
              'or the metric is undefined']]))
    A("""<div class="note"><b>Deliberately not a component split.</b> v4's rung-3 fold assigned whole
  connected components of the molecule graph, which put test molecules at median max-Tc <b>0.357</b>
  to training &mdash; below the Tc&nbsp;&ge;&nbsp;0.40 that rung 2 itself uses to call two molecules
  analogues at all &mdash; and collapsed the fold into ~3 chemotypes, so the effective chemical
  <i>n</i> was 3, not 1,017. That is a worst-case probe, not a measurement. v4b uses pair-level
  disjointness and instead <b>tags every test row with its max ECFP4 Tanimoto to training</b>, so
  difficulty tiers (<code>all</code>, <code>&le;0.8</code>, <code>&le;0.7</code>) become a reporting
  choice from one checkpoint rather than a corpus decision requiring a retrain.</div>""")

    # =========================== 4. TRAINING ===========================
    A('<h2><span class="sn">4</span>Training setup</h2>')
    A('<div class="kv">%s</div>' % ''.join('<div>%s</div><div>%s</div>' % (k, v) for k, v in TRAIN_CFG))
    A('<p style="color:var(--ink3);font-size:13.5px">Qwen2.5-7B has zero free reserved token slots '
      'and <code>resize_token_embeddings</code> is forbidden here, so every structured field '
      '(<code>Change:</code>, <code>Intent:</code>, &hellip;) is ordinary text rather than a special '
      'token &mdash; which also means the potency head cannot key off a sentinel and must pool the '
      'response span.</p>')

    # =========================== 5. RUNG 1 RESULTS ===========================
    A('<h2><span class="sn">5</span>Rung 1 evaluation &mdash; vs base Qwen2.5-7B</h2>')
    A('<p class="lede">All four subtasks on the 32,082-row held-out fold, against the untuned base '
      'model given the identical prompt. The base model is the honest control for "did fine-tuning '
      'teach chemistry, or was it already there?" For v4, base-model SMILES validity on the '
      'generative rungs was <b>0.65%</b> &mdash; the floor is near zero, and any generative number '
      'above a few percent is real.</p>')
    A(table(['subtask', 'metric', 'floor (pre-registered)', 'base Qwen', 'v4b', 'held-out loss'],
            [['ADDUCT', 'exact match / validity',
              '<span class="floor">~0% exact</span>', chip(RESULTS['rung1']['ADDUCT']['base_exact'], '%'),
              chip(RESULTS['rung1']['ADDUCT']['v4b_exact'], '%'), chip(RESULTS['rung1']['ADDUCT']['loss'])],
             ['RETRO', 'exact match / validity',
              '<span class="floor">~0% exact</span>', chip(RESULTS['rung1']['RETRO']['base_exact'], '%'),
              chip(RESULTS['rung1']['RETRO']['v4b_exact'], '%'), chip(RESULTS['rung1']['RETRO']['loss'])],
             ['SELECTIVITY', 'accuracy / macro-F1',
              '<span class="floor">36.3% (always CYS)</span>', chip(RESULTS['rung1']['SELECTIVITY']['base_acc'], '%'),
              chip(RESULTS['rung1']['SELECTIVITY']['v4b_acc'], '%'), chip(RESULTS['rung1']['SELECTIVITY']['loss'])],
             ['WARHEAD_ID', 'accuracy / macro-F1',
              '<span class="floor">35.1% (always nitrile)</span>', chip(RESULTS['rung1']['WARHEAD_ID']['base_acc'], '%'),
              chip(RESULTS['rung1']['WARHEAD_ID']['v4b_acc'], '%'), chip(RESULTS['rung1']['WARHEAD_ID']['loss'])]],
            'nums'))
    A('<p style="color:var(--ink3);font-size:13.5px">The two classification floors come from the '
      'measured label distributions in &sect;1: CYS is 36.3% of SELECTIVITY and nitrile is 35.1% of '
      'WARHEAD_ID. An accuracy of 60% on either would therefore be a <i>weak</i> result, not a '
      'strong one &mdash; which is why macro-F1 is reported alongside.</p>')
    A(gen_example(RESULTS['rung1']['gen_example'],
                  'One held-out rung-1 row per subtask, fed verbatim, with v4b&rsquo;s generation '
                  'beside the base model&rsquo;s on the same prompt.'))

    # =========================== 6. RUNG 2 RESULTS ===========================
    A('<h2><span class="sn">6</span>Rung 2 evaluation &mdash; vs base Qwen2.5-7B</h2>')
    A('<p class="lede">5,988 held-out pairs. Rung 2 has no single right answer &mdash; many analogues '
      'are reasonable &mdash; so exact match is reported but is not the headline. The constraint '
      'metrics are: did the proposal parse, did it keep the warhead, did it keep the scaffold, and '
      'is it actually different from the input.</p>')
    A(table(['metric', 'floor (pre-registered)', 'base Qwen', 'v4b', 'note'],
            [['SMILES validity', '<span class="floor">0.65% (v4-era base)</span>',
              chip(RESULTS['rung2']['base_valid'], '%'), chip(RESULTS['rung2']['v4b_valid'], '%'),
              'parses under RDKit'],
             ['warhead retained', '<span class="floor">100% by copying</span>',
              chip(RESULTS['rung2']['base_warhead_kept'], '%'), chip(RESULTS['rung2']['v4b_warhead_kept'], '%'),
              'trivially satisfiable &mdash; read with the next row'],
             ['scaffold retained', '<span class="floor">100% by copying</span>',
              chip(RESULTS['rung2']['base_scaffold_kept'], '%'), chip(RESULTS['rung2']['v4b_scaffold_kept'], '%'),
              'Murcko scaffold identity'],
             ['non-trivial (&ne; input)', '<span class="floor">0% by copying</span>',
              chip(RESULTS['rung2']['base_novel'], '%'), chip(RESULTS['rung2']['v4b_novel'], '%'),
              '<b>the metric copying cannot win</b>'],
             ['exact match @1', '&mdash;', '&mdash;', chip(RESULTS['rung2']['exact_at_1'], '%'),
              'recovers the reference analogue'],
             ['exact match @20', '&mdash;', '&mdash;', chip(RESULTS['rung2']['exact_at_20'], '%'),
              'best of 20 samples'],
             ['held-out LM loss', '&mdash;', '&mdash;', chip(RESULTS['rung2']['loss']), 'rung-2 rows only']]))
    A("""<div class="note"><b>Why "non-trivial" is the row to read first.</b> Warhead and scaffold
  retention are both 100% satisfiable by echoing the input, and v4's evaluation was nearly fooled by
  exactly this: 11 of 146 apparent rung-3 recall hits were stereo-only pairs the model won by
  copying. Retention and novelty are only meaningful <i>as a pair</i> &mdash; high on both is the
  only good outcome.</div>""")
    A(gen_example(RESULTS['rung2']['gen_example'],
                  'One held-out rung-2 hit, fed verbatim, with v4b&rsquo;s proposed analogue and the '
                  'base model&rsquo;s on the same prompt.'))

    # =========================== 7. RUNG 3 RESULTS ===========================
    A('<h2><span class="sn">7</span>Rung 3 evaluation &mdash; vs the held-out test set</h2>')
    A('<p class="lede">334 held-out pocket-conditioned pairs, pair-disjoint from training and tagged '
      'by Tanimoto tier. This is the task the system exists for, and the one where v4 produced the '
      'result worth being honest about.</p>')
    A(table(['metric', 'floor (pre-registered)', 'v4b', 'what it tests'],
            [['SMILES validity', '<span class="floor">0.65% base</span>', chip(RESULTS['rung3']['valid'], '%'),
              'the proposal is a molecule'],
             ['warhead retained (KEEP subset)', '<span class="floor">100% by copying</span>',
              chip(RESULTS['rung3']['warhead_kept'], '%'),
              'restricted to the 2,323 <code>KEEP</code> rows &mdash; see the &sect;1 defect'],
             ['exact match @1', '&mdash;', chip(RESULTS['rung3']['exact_at_1'], '%'),
              'recovers the real programme analogue'],
             ['exact match @20', '&mdash;', chip(RESULTS['rung3']['exact_at_20'], '%'), 'best of 20'],
             ['Change accuracy', '<span class="floor">37.2% (always REPLACE)</span>',
              chip(RESULTS['rung3']['op_acc'], '%'), 'the structural move'],
             ['Intent accuracy', '<span class="floor">75.2% (always RECOGNITION)</span>',
              chip(RESULTS['rung3']['intent_acc'], '%'), '<b>a nearly-saturated field</b>'],
             ['Warhead-move accuracy', '<span class="floor">77.0% (always KEEP)</span>',
              chip(RESULTS['rung3']['warhead_move_acc'], '%'), 'also nearly saturated'],
             ['contact Gains/Loses P / R / F1', '&mdash;',
              '%s / %s / %s' % (chip(RESULTS['rung3']['contact_p']), chip(RESULTS['rung3']['contact_r']),
                                chip(RESULTS['rung3']['contact_f1'])),
              'does it know which contacts the edit buys'],
             ['best-of-20 Tc to reference', '<span class="floor">0.401 = the hit itself</span>',
              chip(RESULTS['rung3']['tc_to_ref']), '<b>must beat copying the input</b>'],
             ['held-out LM loss', '&mdash;', chip(RESULTS['rung3']['loss']), 'rung-3 rows only']]))
    A("""<div class="note"><b>Read the three label fields against their floors, not against 100%.</b>
  Intent is 75.2% RECOGNITION and Warhead-move is 77.0% KEEP, so a 78% accuracy on either is
  roughly <i>nothing learned</i>. This is the exact error v4 nearly shipped: a 49.4% change-op score
  read as failure until a conditional-entropy audit showed only 1.099 of 2.047 bits of the field are
  recoverable from the input at all, putting the Bayes-optimal ceiling at 64.7% &mdash; so 49.4%
  against a 37.2% floor was most of the available signal, not a shortfall.</div>""")
    A("""<div class="defect"><b>The bar v4 failed, stated in advance.</b> On novel scaffolds, v4's
  best-of-20 analogue reached Tanimoto <b>0.378</b> to the reference while the hit it was handed
  already sat at <b>0.401</b> &mdash; the model did worse than copying its own input. Its Tc profile
  was flat: <b>0%</b> of hits landed in the 0.55&ndash;1.0 band, and the headline 35.06% exact@20
  came entirely from exact membership. v4b's rung-3 result is only meaningful if
  <code>Tc to reference &gt; 0.401</code> and the 0.55&ndash;1.0 band is non-empty.</div>""")
    A(gen_example(RESULTS['rung3']['gen_example'],
                  'One held-out complex, the full pocket block fed verbatim, with v4b&rsquo;s complete '
                  'Change / Intent / Warhead / Gains / Loses / Analogue output.'))

    # =========================== 8. POTENCY ===========================
    A('<h2><span class="sn">8</span>Potency head evaluation</h2>')
    A('<p class="lede">12,408 held-out rows across hit groups held out whole. The headline is '
      '<b>within-hit Spearman</b>, not global R&sup2;: ranking analogues <i>of the same hit</i> is the '
      'decision a chemist actually makes, and a global metric can look strong purely by separating '
      'easy hits from hard ones.</p>')
    A(table(['metric', 'floor / reference', 'v4b', 'note'],
            [['MSE', '<span class="floor">label variance</span>', chip(RESULTS['affinity']['mse']),
              'validation MSE is already live at <b>0.1107</b> (R&sup2; &asymp; 0.62)'],
             ['MAE', '&mdash;', chip(RESULTS['affinity']['mae']), '&Delta;pIC50 units'],
             ['global R&sup2;', '<span class="floor">0.0</span>', chip(RESULTS['affinity']['r2']),
              'across all held-out pairs'],
             ['within-hit Spearman', '<span class="floor">within-hit variance floor</span>',
              chip(RESULTS['affinity']['spearman_within_hit']),
              '<b>the metric that matters</b>'],
             ['variance floor', '&mdash;', chip(RESULTS['affinity']['variance_floor']),
              'how much within-hit spread exists to be ranked at all'],
             ['hit groups scored', '&mdash;', chip(RESULTS['affinity']['n_groups']),
              'groups with &ge;3 analogues']]))
    A('<div class="note"><b>Why the variance floor is reported next to the Spearman.</b> If the '
      'analogues of a given hit barely differ in potency, a high within-hit Spearman is noise and a '
      'low one is meaningless. The floor says how much orderable spread the held-out groups actually '
      'contain, so the correlation can be read as a fraction of what was rankable.</div>')
    A(gen_example(RESULTS['affinity']['gen_example'],
                  'One held-out hit group: the hit, its analogues, the true &Delta;pIC50 ordering and '
                  'v4b&rsquo;s predicted ordering side by side.'))

    # =========================== 9. LIVE DYNAMICS ===========================
    A('<h2><span class="sn">9</span>Training dynamics &mdash; live</h2>')
    A('<p class="lede">These are real, current numbers, not placeholders. v4b is the first run in '
      'this project with a validation loss at all.</p>')
    A(table(['quantity', 'value', 'note'], [[k, '<span class="num">%s</span>' % v, n] for k, v, n in LIVE]))
    A('<h3>Validation curve</h3>')
    A(table(['step', 'LM loss', 'potency MSE', '&Delta; LM'],
            [[str(s), '<span class="num">%.4f</span>' % lm, '%.4f' % mse,
              ('<span class="num good">%.4f</span>' % (lm - VALID_CURVE[i - 1][1])) if i else '&mdash;']
             for i, (s, lm, mse) in enumerate(VALID_CURVE)], 'nums'))
    A('<div class="ok"><b>Five consecutive improvements, so early stopping is nowhere near firing.</b> '
      'Patience is 4 evaluations; the LM loss has fallen at every single one. The run resumed cleanly '
      'from step 4,000 after a validation-loop bug was fixed, and has logged 0 errors, 0 NaN and 0 '
      'OOM across 2,900 steps since.</div>')
    A("""<div class="defect"><b>The bug that cost 500 steps, recorded.</b> v4b crashed at step 500 with
  <code>KeyError: 'hm'</code>: the new validation loop popped local variable names
  (<code>vb.pop('hm')</code>) instead of the collate dictionary's keys
  (<code>vb.pop('hitmask')</code>). Because the crash landed before the first checkpoint, all 500
  steps were lost. The fix is also why <code>--ckpt-every 500</code> and
  <code>--snapshot-every 2000</code> now exist.</div>""")

    # =========================== 10. LIMITS ===========================
    A('<h2><span class="sn">10</span>What v4b deliberately does not do</h2>')
    A(table(['omitted', 'why'],
            [['the contacts task', 'Predicting a contact <i>type</i> given the geometry was measured '
              '<b>96.1% solvable by arithmetic</b> on the distance columns &mdash; it teaches labelling, '
              'not editing. Uncapped it would also have cost ~93M tokens, more than every other '
              'channel combined. The generative masked-restore version is the right task and is not '
              'built yet.'],
             ['the non-covalent PDB half', 'The 76,840-structure mirror needs re-downloading. '
              '<code>extract_noncov.py</code> is written and tested &mdash; 4,141 ligand records and '
              '82,562 interactions from 1,749 structures in 53 s &mdash; but this run is '
              'covalent-only.'],
             ['any change to the pocket encoding', 'Held fixed at the atom+residue format on purpose. '
              'The distance-only "invariant" encoding is a separate experiment, not a variable to '
              'change in the same run that changes the corpus.'],
             ['a second epoch', 'The schedule is 1.00 epoch = 17,743 steps. Whether to continue is a '
              'decision for the validation curve, not a pre-commitment.']]))
    A("""<div class="defect"><b>Six v4-era numbers were withdrawn on audit, and are not reused here.</b>
  A 1.00% disjoint score (produced by three concurrent evaluation chains interleaving one log); a
  36.5% recall measured on 14 of 1,020 complexes because <code>rows[:n]</code> was mistaken for a
  sample; a "1.6&times; better than joint" comparison against a different sample; a
  uniqueness-collapse memorisation claim (the 7.2% was a sampling artifact &mdash; 53.1% was correct);
  a ~1.8% disjoint partial; and a 49.4% change-op floor compared against the wrong cohort. The
  stratified sampling, prompt-signature assertions and pre-registered floors in this report exist
  because of those six.</div>""")

    # =========================== 11. COMPUTE ===========================
    A('<h2><span class="sn">11</span>Compute note &mdash; the run is launch-bound, not compute-bound</h2>')
    A('<p class="lede">Worth recording because it is a config error, not a hardware limit, and '
      'because it changes what the next run should do.</p>')
    A(table(['quantity', 'measured', 'note'], [[k, '<span class="num">%s</span>' % v, n] for k, v, n in GPU]))
    A("""<div class="note"><b>The cause is a batch size carried over from a corpus of the opposite
  shape.</b> <code>--bs 3</code> was correct for v4, where rung 3 dominated at ~2,200 tokens a row,
  so a micro-batch was ~6,600 tokens and the A100 was well fed. v4b inverted the mix: rung 1 is now
  63.9% of rows at <b>53 tokens</b> each, so the same <code>bs 3</code> ships ~389-token
  micro-batches. 63.9% of all wall-clock is spent on rows that are 1/28th the size of a rung-3 row
  but cost nearly as much, because the cost is per-op CPU dispatch &mdash; roughly 6,000 dispatches
  per micro-batch once LoRA's 196 adapters and the gradient-checkpoint re-forward are counted, at
  ~94&nbsp;&micro;s each &mdash; and not FLOPs. Hence 48% utilization, 181 W of 400 W, and one CPU
  core pinned while 11 sit idle.</div>""")
    A('<p><b>The fix preserves the optimization exactly.</b> Batching to a token budget rather than a '
      'fixed sequence count keeps the effective batch at 33 examples per step while collapsing 11 '
      'micro-batches into 1, giving an identical gradient with 21&times; fewer launches across the '
      'epoch. Long rung-3 rows overflow the budget and simply accumulate more steps. Validation gets '
      'the same win for free: it currently runs 1,000 forward passes of 3 rows every 500 steps, about '
      '6% of all wall-clock, and would drop to ~50. Gradient checkpointing should then stay on, '
      'because it is only wasteful while batches are this thin.</p>')

    A('<div class="footer">CovalentFormer v4b &middot; structure compiled 2026-10-03 from the live run on '
      'ai-gpu-a100 &middot; all data-side figures verified against the corpus on disk &middot; '
      'model-side results pending the finished checkpoint</div>')
    A('</div>')
    A('</body></html>')
    return '\n'.join(o)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SP, 'covalentformer_report_v4b.html')
    h = build()
    with open(out, 'w') as f:
        f.write(h)
    n_pend = h.count('class="pend"')
    print('wrote %s (%.1f KB)' % (out, len(h) / 1024.0))
    print('pending slots rendered: %d' % n_pend)
