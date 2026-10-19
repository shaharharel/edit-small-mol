#!/usr/bin/env python3
"""Add two walkthrough sections to covalentformer_report.html: labelling, and pocket encoding.

Both are built from REAL artefacts in the repo, not prose written from memory:

  * the labelling walkthrough uses pair 7BAA T5Z->T6N end to end -- the packet text the reviewers
    actually saw (batches/batch_0001.txt), the three sentences they actually wrote
    (label_out/raw/batch_0001_lab{A,B,C}.json), the chair's verdict and stated reasoning
    (label_out/batch_0001.json), and the training row that resulted (sft_out_g3/sft_pocket.jsonl);
  * the encoding walkthrough quotes that same pair's MODEL_INPUT block and annotates every field,
    so a reader can see exactly what the model is and is not given.

Using one pair throughout is deliberate: the reader sees the same molecule travel from pocket
geometry to labelled training target, rather than four disconnected fragments.

The sections are inserted after section 4 (Generation sample) and the later ones renumbered.
Idempotent: re-running replaces the two sections rather than appending duplicates.
"""
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(HERE, 'v6')
REPORT = os.path.join(HERE, 'covalentformer_report.html')
PAIR = '7BAA T5Z->T6N'

MARK_LABEL = '<div id="walkthrough-labelling"></div>'
MARK_ENC = '<div id="walkthrough-encoding"></div>'


def esc(s):
    return html.escape(str(s), quote=False)


def packet_block(path, pair):
    """The packet text for one pair, verbatim, up to the next pair banner."""
    txt = open(path).read()
    i = txt.index('PAIR %s' % pair)
    i = txt.rindex('=' * 20, 0, i)
    j = txt.find('=' * 20, txt.index('PAIR %s' % pair) + 10)
    j = txt.find('\n', txt.find('PAIR ', j)) if j != -1 else len(txt)
    return txt[i:j].rstrip()


def model_input_only(packet):
    """Just the MODEL_INPUT portion -- what the model sees at inference."""
    a = packet.index('--- MODEL_INPUT')
    for marker in ('--- LABEL_ONLY', '--- label-side', '--- ANSWER'):
        b = packet.find(marker, a)
        if b != -1:
            return packet[a:b].rstrip()
    return packet[a:].rstrip()


def build_labelling_section(sn):
    packet = packet_block(os.path.join(V6, 'batches/batch_0001.txt'), PAIR)
    revs = {}
    for lab in 'ABC':
        # label_out_g4/, not label_out/: the packets in batches/ are regenerated per pass, and
        # label_out/ holds the OLD g1 retrospective labels. Mixing them showed a chair verdict from
        # one pass beside a training row from another, whose sentences differ -- caught by checking
        # that the row's target equals the chair's final string, which it did not.
        d = json.load(open(os.path.join(V6, 'label_out_g4/raw/batch_0001_lab%s.json' % lab)))
        revs[lab] = d.get(PAIR) or {}
    chair = json.load(open(os.path.join(V6, 'label_out_g4/batch_0001.json')))[PAIR]
    # batches/batch_0001 fed the g4 pass, not g3, so search the corpora rather than assuming one.
    row = corpus = None
    for cand in ('sft_out_g4', 'sft_out_g3g4', 'sft_out_g3', 'sft_out_g2', 'sft_out'):
        fp = os.path.join(V6, cand, 'sft_pocket.jsonl')
        if not os.path.exists(fp):
            continue
        for line in open(fp):
            r = json.loads(line)
            if r['pair'] == PAIR:
                row, corpus = r, cand
                break
        if row:
            break
    assert row is not None, 'training row for %s not found in any corpus' % PAIR
    # the walkthrough is only honest if all four artefacts come from the SAME pass
    assert (row.get('reference_sentence') or '').strip() == (chair.get('final') or '').strip(), (
        'training target does not match the chair verdict -- the artefacts are from different '
        'labelling passes, which would make this walkthrough fiction')

    rev_rows = ''.join(
        '<tr><td><b>reviewer %s</b></td><td><div class="lbl-rsn">%s</div>'
        '<div class="lbl-meta">basis cited: <code>%s</code></div></td></tr>'
        % (lab, esc(revs[lab].get('sentence') or '(none)'),
           esc((revs[lab].get('basis') or '')[:300]))
        for lab in 'ABC')

    return """%s
<h2><span class="sn">%d</span>How a reasoning label is made</h2>
<p class="lede">Every training target in this project is one sentence written by a medicinal chemist
about one matched molecular pair, then adjudicated by a chair. This section follows a single pair,
<code>%s</code>, all the way through: the packet three reviewers saw, the three sentences they
wrote, the chair's verdict and its stated reasoning, and the training row that came out. Nothing
here is reconstructed &mdash; it is the actual text from the actual run.</p>

<h3>1. What the reviewer is asked to do</h3>
<p>One sentence, <b>maximum 35 words</b>, naming what the edit gives up and what it keeps, anchored
on measured geometry. The instruction set carries <b>eleven hard rules</b>; the ones that do most of
the work are: name only residues that appear on the <i>input</i> side of this pair's evidence; say
&ldquo;hydrogen bond&rdquo; only where the typing supports it and &ldquo;covalent&rdquo; only where a
covalent row supports it; never describe an atom as buried, enclosed or solvent-facing; make no
claim about what the analogue's <i>new</i> atoms touch, because at inference the model has no
analogue structure; claim no potency improvement, since there is no potency data; never build a
claim on a contact whose typing contains <code>uncertain</code>; and <b>disclose a multi-site
edit</b>, referring to groups rather than raw atom names.</p>

<h3>2. The packet &mdash; exactly what three reviewers saw</h3>
<p class="tnote">Reviewers see more than the model does: the model-input block below, plus
label-side tables giving the analogue's own crystal contacts, which input contacts survived the
edit, and where the edit sits. That asymmetry is the point &mdash; the label may be written from
knowledge the model will not have at inference, which is why rule 1 forbids attributing a gain to
anything but the analogue's crystal.</p>
<pre class="packet">%s</pre>

<h3>3. The three sentences, written independently</h3>
<div class="tw"><table class="lbl-t"><tbody>%s</tbody></table></div>
<p class="tnote">All three converge on the same chemistry &mdash; bromine's 3.53 &Aring; packing on
LYS122.CE traded for a carbon the analogue's crystal places 3.45 &Aring; from the same atom, with
the CYS42 covalent bond holding. They differ in what they disclose: only reviewer A flags that
<i>both</i> ring substituents change, which rule 11 requires.</p>

<h3>4. The chair adjudicates</h3>
<p>The chair sees the same evidence plus the three sentences, and returns exactly one of
<code>ADOPT n</code>, <code>REWRITE</code> or <code>REJECT</code>. <b>REJECT is an expected
outcome, not a failure</b> &mdash; a pair whose edited atoms sit in solvent touching nothing yields
no design rationale, and the guidance is explicit that writing &ldquo;the new atoms touch nothing
within 4.0 &Aring;&rdquo; is a rejection with extra words rather than a label.</p>
<div class="tw"><table class="lbl-t"><tbody>
<tr><td><b>verdict</b></td><td><code>%s</code></td></tr>
<tr><td><b>final sentence</b></td><td><div class="lbl-rsn">%s</div></td></tr>
<tr><td><b>chair's reasoning</b></td><td><div class="lbl-rsn">%s</div></td></tr>
</tbody></table></div>

<h3>5. What becomes the training row</h3>
<p>The adopted sentence is the completion the model is fine-tuned to emit, given the model-input
block from step 2 and nothing else. The analogue SMILES is appended on its own line.</p>
<div class="tw"><table class="lbl-t"><tbody>
<tr><td><b>corpus</b></td><td><code>%s</code> &middot; labelling pass
<code>%s</code></td></tr>
<tr><td><b>split / chair verdict</b></td><td><code>%s</code> / <code>%s</code></td></tr>
<tr><td><b>analogue SMILES</b></td><td><code>%s</code></td></tr>
<tr><td><b>training target</b></td><td><div class="lbl-rsn">%s</div></td></tr>
</tbody></table></div>
<p class="tnote">Scale: three independent reviewers plus one chair per pair. Across the g3 pass this
produced 2,326 training rows from the pairs that survived adjudication. Two measured caveats on this
process belong here rather than in a footnote. <b>A chair ADOPT is not a guarantee of correctness</b>:
a three-chemist re-audit of 18 ADOPT rows found 6 that described their edit incorrectly and 2 that
cited a residue absent from the table. And separately, <b>159 of the 2,326 rows (6.8%%) contain a
stereocentre inversion at constant centre count that the sentence never mentions</b> &mdash; when such
an inversion occurs, the sentence misses it 89%% of the time. Both numbers are reasons the labelling
gate is being tightened, not reasons to discount the corpus, but a reader should know them.</p>
""" % (MARK_LABEL, sn, esc(PAIR), esc(packet), rev_rows,
       esc(chair.get('verdict') or ''), esc(chair.get('final') or ''),
       esc(chair.get('why') or ''), esc(corpus),
       esc(row.get('label_generation') or '?'), esc(row['split']),
       esc(row.get('verdict') or ''),
       esc(row.get('analogue_smiles') or ''), esc(row.get('reference_sentence') or ''))


def build_encoding_section(sn):
    packet = packet_block(os.path.join(V6, 'batches/batch_0001.txt'), PAIR)
    mi = model_input_only(packet)
    return """%s
<h2><span class="sn">%d</span>What the pocket encoding actually contains</h2>
<p class="lede">The model is never given coordinates, a grid, a graph or a learned structural
embedding. It is given <b>text</b>: the input ligand's SMILES, and one row per heavy atom describing
that atom's measured relationship to the protein. Everything the model knows about the pocket is in
the block below &mdash; the same pair as the previous section, quoted verbatim.</p>
<pre class="packet">%s</pre>

<h3>Field by field</h3>
<div class="tw"><table class="lbl-t"><thead><tr><th>field</th><th>what it is, and the trap in it</th>
</tr></thead><tbody>
<tr><td><code>ATOM TABLE COVERAGE</code></td><td>How many input heavy atoms got a row, and how many
were dropped as unmodelled or as clashes. It exists so an incomplete table cannot be mistaken for a
complete one.</td></tr>
<tr><td><code>atom</code></td><td>The chemical-component (CCD) atom name, e.g. <code>BR</code>,
<code>C12</code>, <code>S</code>. <b>Rows are sorted alphabetically, not in molecule order</b>, and
the table omits atoms with no contact &mdash; so atom names cannot be mapped to molecule positions by
counting. Recovering ownership needs the CCD definition; assuming positional correspondence produces
confident wrong answers.</td></tr>
<tr><td><code>faces</code></td><td>The single <i>nearest</i> protein atom, as
<code>RES123.ATOM</code>, plus its chain. One row shows one neighbour, so a group with several
contacts appears only through its own several atoms.</td></tr>
<tr><td><code>dist</code></td><td>The measured distance in &Aring;ngstr&ouml;ms from the deposited
crystal. This is the number models reproduce almost perfectly and attribute to the wrong atom about
half the time.</td></tr>
<tr><td><code>ang</code></td><td>Donor&ndash;H&ndash;acceptor angle where one is defined, otherwise
<code>-</code>. Its absence is why many polar rows are typed as packing rather than as hydrogen
bonds.</td></tr>
<tr><td><code>interaction</code></td><td>The typing, never truncated:
<code>hydrophobic contact</code>, <code>polar/apolar packing</code>, <code>H-bond, good geometry</code>,
<code>SALT BRIDGE</code>, <code>METAL COORDINATION</code>, <code>COVALENT BOND to CYS42.SG</code>, or
an explicit refusal such as <code>donor/acceptor INCOMPATIBLE &hellip; not a hydrogen bond</code>.
Covalent calls are made against the ideal bond length for that element pair, not a flat cutoff, and
backbone atoms are excluded as nucleophiles.</td></tr>
<tr><td>second line under a row</td><td>A <code>polar partner</code>: a polar contact that is not the
nearest atom overall, so a polar interaction is not hidden by a closer apolar one.</td></tr>
<tr><td><code>NOTHING WITHIN 4.0 A</code></td><td>Named atoms with no contact at all. <b>This is
positive information</b>: a claim about an atom listed here is invention, and the list is printed so
that completeness is checkable.</td></tr>
<tr><td><code>pocket envelope</code></td><td>Every residue with an atom within 8.0 &Aring;, banded
rather than given exact distances, and labelled <b>REACHABLE SPACE, NOT CONTACTS</b>. A residue at
7.5 &Aring; touches nothing; the banding exists so an envelope distance cannot be quoted as a measured
contact.</td></tr>
<tr><td>caveat tags</td><td>Per-row and per-pair flags: <code>[INDEP, CYS42 excl]</code> marks a
nearest <i>independent</i> contact with the covalently bonded residue excluded;
<code>uncertain</code> marks geometry the crystal does not support; suspect residues and water
clashes are flagged. The rules forbid building a claim on anything flagged uncertain &mdash; an
instruction the models routinely ignore.</td></tr>
</tbody></table></div>
<p class="tnote">Two consequences worth stating. First, <b>the encoding is lossless about what it
reports and silent about everything else</b>: no coordinates, no second-nearest neighbour, no
solvent-accessible surface, no electrostatics. A model cannot infer that two cited atoms are
adjacent, which is part of why attribution is hard. Second, <b>every field above is machine-checkable
against the model's output</b>, which is what makes a verifier-driven training signal possible: a
cited number either is or is not a row of this table, and a claimed interaction type either does or
does not match the typing.</p>
""" % (MARK_ENC, sn, esc(mi))


def main():
    s = open(REPORT).read()

    # idempotent: drop previously inserted copies
    # The marker sits immediately BEFORE its own <h2>, so searching for "the next h2 after the
    # marker" finds that same heading at offset ~0 and deletes only the marker, leaving a duplicate
    # section behind -- which is exactly what happened on the first re-run. Skip past the section's
    # own heading before looking for the one that ends it.
    # REVERSE document order. Removing the labelling section cuts from its marker to the next
    # <h2> after its own -- which is the ENCODING section's heading -- so going forwards eats
    # the encoding marker and heading and orphans its body, leaving one extra section behind.
    for mark in (MARK_ENC, MARK_LABEL):
        while mark in s:
            a = s.index(mark)
            own = re.search(r'<h2><span class="sn">\d+</span>', s[a + len(mark):])
            after = a + len(mark) + (own.end() if own else 0)
            nxt = re.search(r'<h2><span class="sn">\d+</span>', s[after:])
            b = after + nxt.start() if nxt else len(s)
            s = s[:a] + s[b:]

    # Anchor on the TITLE, not the number. Section numbers are rewritten on every run, so after a
    # removal pass the remaining headings are stale (1,2,3,4,7,8...) and an sn-based anchor fails --
    # which is how the second run died.
    anchor = re.search(r'<h2><span class="sn">\d+</span>The three training phases', s)
    assert anchor, 'anchor section "The three training phases" not found'
    at = anchor.start()
    block = build_labelling_section(5) + '\n' + build_encoding_section(6) + '\n'
    s = s[:at] + block + s[at:]

    seq = iter(range(1, 99))
    s = re.sub(r'(<h2><span class="sn">)\d+(</span>)',
               lambda m: '%s%d%s' % (m.group(1), next(seq), m.group(2)), s)

    # a <pre> style for the verbatim packets, added once
    if '.packet{' not in s:
        s = s.replace('.tbd{', '.packet{white-space:pre-wrap;font-family:var(--mono);font-size:11px;'
                               'line-height:1.45;background:var(--line2);border:1px solid var(--line);'
                               'border-radius:4px;padding:12px 14px;overflow-x:auto;margin:12px 0}\n.tbd{', 1)

    open(REPORT, 'w').write(s)
    for m in re.finditer(r'<h2><span class="sn">(\d+)</span>(.*?)</h2>', s):
        print(' %2s  %s' % (m.group(1), re.sub('<[^>]+>', '', m.group(2))[:58]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
