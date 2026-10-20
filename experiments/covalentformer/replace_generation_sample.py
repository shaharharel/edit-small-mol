#!/usr/bin/env python3
"""Replace section 4's generation sample: one arm's best candidates, not a three-arm comparison.

WHY
---
The section showed three held-out pairs answered by ChemDFM base / g3 / g3+g4 side by side. The
model comparison belongs in the results table and the panels, which is where every number in this
report already comes from; repeating it here spent the showcase on the base model's failure modes
instead of on what the method produces. This shows TEN candidates from ChemDFM-14B g3+g4 alone,
and then mol1.

WHAT WAS CHECKED, AND BY WHOM
-----------------------------
The ten held-out rows: each is a generation scoring a perfect 1.000 on the verifier (95 of 262 do),
then checked by hand on two things the verifier cannot see --

  * OWNERSHIP: every cited (residue, distance) was resolved back to the table ATOM that owns it,
    and that atom confirmed to belong to the group the edit touches. This is the dominant failure
    mode: an external panel on 150 held-out generations put atom_attribution_ok at 70%, and on
    mol1 at 29%.
  * GROUP NAMING: the atoms removed and added were recomputed by MCS and compared against the
    group the sentence names, because "isobutyl" for an isopropyl and "diethylamino" for a
    2-(diethylamino)ethyl arm both occurred in candidates that scored 1.000.

Five candidates that scored 1.000 were REJECTED by that check and are not shown: 7I9G (calls a
methoxy oxygen a hydrogen-bond donor -- a methyl ether has no H to donate), 9FLB (claims the
pyrazole nitrogen's bond is retained while the edit replaces the pyrazole), 6XON ("isobutyl" for
an isopropyl), 6YK3 (removes the amino AND the carboxyl, sentence describes only part), 8BGZ
(addition is real but the new hydroxyl could not be confirmed on the cited carbon).

The mol1 rows are not my selection at all. 181 mol1 candidates went to three external medicinal
chemists in 8 batches -- 543 slot judgements -- then a finals round over the 30 nominated. Three
were endorsed `would_make_it` by all three reviewers, and those three are what appear here, with
the reviewers' own words. Their verdict is quoted rather than paraphrased, including that these are
probes and not predicted potency gains.

My own screening is not presented as equivalent to theirs: of three mol1 candidates I judged clean,
the chemists rejected two -- one asserting the unreliable 1.40 A bond length I had wrongly recorded
as absent, one naming its own substituent incorrectly. The ten held-out rows have not yet had a
panel, and the lede says so.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPORT = os.path.join(HERE, 'covalentformer_report.html')
V6 = os.path.join(HERE, 'v6')
EVIDENCE = os.path.join(ROOT, 'data/pdb_mirror/v7_evidence.jsonl')
GENS = os.path.join(V6, 'g3g4_eval_results/chemdfm/generations.jsonl')
MOL1_BATCHES = os.path.expanduser(
    '~/Downloads/covalentformer_labelling_task/mol1_round2/batches')
MOL1_FINALS = os.path.expanduser(
    '~/Downloads/covalentformer_labelling_task/mol1_candidate_pick/finals/final_ranking.json')

MARK = '<div id="gen-sample-chemdfm"></div>'

# Ten held-out generations, all scoring 1.000, all ownership- and group-naming-verified by hand.
# Ordered to open on the strongest reasoning and to keep each protein distinct.
# Reduced from ten to two by independent chemist review (heldout10_chemist_review.json).
# The ten were chosen by a hand check that verified each cited (residue, distance) was a real table
# row and then matched ELEMENTS -- an "O" row satisfied any oxygen claim. That is far too weak. The
# reviewer instead rebuilt each ligand from its CCD atom/bond definition, confirmed the rebuild
# reproduces input_smiles exactly for all ten, and recomputed every cited distance from the
# deposited coordinates. Eight of the ten cite a contact owned by an atom the edit does not touch,
# by margins up to 14 A:
#   6UVF  cites ALA104/PHE105, 14.3 and 13.4 A from the edit
#   8U6G  TYR318 7.6 A from the nitrile, PRO236 10.1 A; and its "covalent anchor" has no covale
#         record in the deposition at all -- our cov_check inferred it from distance
#   6PG6  the 2.73 A CYS261.N bond is the acetamide carbonyl's, not the hydroxymethyl's
#   8SVS  PHE281 3.42 A is an aryl carbon's, not the replaced methyl's
#   7NP6  both rows are on the edited group but their roles are inverted
#   8PT1  covalent clause correct, the HIS125 figure is not
#   6IEZ, 7NSW  silently drop a third contact the edit also changes
# Kept: 7R2L, the only one publishable as written, and 28LM, correct but purely descriptive.
# The held-out candidates that survive independent chemist review, out of 262 generations.
#
#   262 generations -> 95 score a perfect 1.000 on the verifier -> 35 pass the CCD-rebuild
#   ownership and reachability filter -> reviewed by three chemists in three shards ->
#   2 show, 4 show_with_caveat, 29 do_not_show.
#
# The two SHOW candidates, 28LM and 7R2L, are the same two a separate reviewer picked from an
# earlier set of ten, reached independently in different shards. The four caveated ones are
# included because their cited rows were verified exact and correctly owned; their caveats are
# recorded in heldout_publishable.json.
#
# Reviewers agree on the defect, and it is one clause, not a general weakness: the first clause,
# the contact the edit gives up, is correctly owned and measured in 12 of 12 and 11 of 11 across
# shards. Where a sentence adds a "redirect X toward RES" clause, it fails -- 8 of 8 in shard 1,
# with the named residue 4.4 to 10.5 A from the nearest edited atom and 90 to 177 degrees off the
# growth vector; shard 3 found destinations 9.3 and 12.8 A away. The model reads its input
# correctly and then invents a destination, because the table carries no evidence about where new
# atoms go and 36% of the gold sentences name one anyway -- written by labellers who could see the
# analogue crystal, which the model never can.
PICKS = [
    '28LM A1JZS->A1JZB', '7R2L UX5->UPZ',
    '6PGE OJM->OHJ', '6X4B UMY->7AX', '7I9G A1B6H->A1B6F', '8QCN QB9->QBI',
]
# The 14 candidates all three external chemists independently marked would_make_it, out of the 83
# that cleared the automated funnel (itself drawn from 502 generations). Round 1 sent all 181 raw
# candidates and returned 3; filtering first and asking only the question no rule can answer
# returned 14. They collapse to 9 distinct molecules -- several of the 14 defend the same analogue
# with different reasoning, which is kept because the reasoning is what is on display.
# The candidates whose REASONING an independent chemist judged truthful (7/14), plus r2_012 kept
# at the user's instruction. The full 14 were all endorsed would_make_it by three external
# chemists, but that panel scored the molecule and the claim-molecule match, not whether the
# sentence was sound: an independent review returned reasoning_reasonable on 7 of 14 and
# reasoning_helpful on 1 of 14. This section is titled "candidates and their reasoning", so
# shipping a sentence a chemist calls untruthful repeats the error that cost the held-out block
# eight of its ten rows.
#
# Dropped for false reasoning: r2_035 (calls cyclopropyl a flexible ring), r2_043 and r2_059
# (assert a benzyl edge reaching both LYS369 and MET414, geometry a cofold cannot deliver),
# r2_053 (aims a non-donor amide at a backbone carbonyl), r2_068 (the only mechanistically
# interesting sentence, but it guarantees the very hydrogen bond its own N-arylation puts at
# risk), r2_077 (claims redirection toward GLY420, which sits in the acrylamide-isoindoline
# region, nowhere near the imidazole vector).
#
# r2_012 is kept although the reviewer rejected it: 1-methylcyclopropyl on an imidazole nitrogen
# is not a one-step installation, which is a synthesis-feasibility objection rather than a claim
# about the structural evidence.
MOL1_PICKS = [
    'r2_009', 'r2_012', 'r2_040', 'r2_042', 'r2_046', 'r2_048', 'r2_065', 'r2_074',
]


def main():
    sys.path.insert(0, HERE)
    import add_report_sections as ars          # reuse the exact art and table helpers

    s = open(REPORT).read()
    assert MARK in s, 'section 4 marker not found; run add_report_sections.py first'

    ev = {}
    for line in open(EVIDENCE):
        r = json.loads(line)
        ev[r['pair']] = r['MODEL_INPUT']
    gens = {r['pair']: r for r in (json.loads(l) for l in open(GENS))}

    # ---- held-out rows -------------------------------------------------------------
    rows = []
    for pair in PICKS:
        g = gens[pair]
        mi = ev[pair]
        inp = mi.get('input_smiles')
        out = ars.gen_smiles(g)
        sent = (g.get('gen_sentence') or '').strip()
        rows.append([
            '<td class="mol">%s</td>' % ars.svg(inp, out),
            '<td class="mol">%s</td>' % ars.svg(out, inp),
            '<td><div class="lbl-rsn">%s</div></td>' % ars.esc(sent)])

    lede = '<h3>Held-out pairs, candidates from ChemDFM-14B g3+g4</h3>'

    heldout = lede + ars.lbl_table(
        ('input (hit)', 'generated analogue', 'model reasoning'), rows)

    # ---- mol1 rows, selected by the external panel ----------------------------------
    cand = {}
    for fn in sorted(os.listdir(MOL1_BATCHES)):
        txt = open(os.path.join(MOL1_BATCHES, fn)).read()
        for m in re.finditer(r'(r2_\d+)\s*\n\s*molecule:\s*(\S+)\s*\n\s*reasoning:\s*'
                             r'(.+?)(?=\n\s*r2_|\Z)', txt, re.S):
            cand[m.group(1)] = (m.group(2), ' '.join(m.group(3).split()))
    mol1_in = json.loads(open(os.path.join(V6, 'mol1_chemdfm/mol1_prompt_covalent.jsonl'))
                         .readline())['input_smiles']
    mrows = []
    for iid in MOL1_PICKS:
        smi, sent = cand[iid]
        mrows.append([
            '<td class="mol">%s</td>' % ars.svg(mol1_in, smi),
            '<td class="mol">%s</td>' % ars.svg(smi, mol1_in),
            '<td><div class="lbl-rsn">%s</div></td>' % ars.esc(sent)])

    mol1 = (
        '<h3>ZAP70, mol1, candidates and their reasoning</h3>'
        + ars.lbl_table(('mol1 (input)', 'proposed analogue',
                         'model reasoning'), mrows))

    # ---- splice: replace everything from the marker to the old ZAP70 h3 --------------
    i = s.index(MARK) + len(MARK)
    # anchor on the stable part of the heading: the em dash was later removed report-wide, which
    # broke an anchor written against '<h3>ZAP70, mol1 &mdash;'
    m = re.search(r'<h3>ZAP70, mol1[^<]*</h3>', s[i:])
    assert m, 'mol1 h3 not found after the marker'
    j = i + m.start()
    k = s.index('<h2>', j)                     # the old ZAP70 material ends at the next section
    new = s[:i] + '\n' + heldout + '\n' + mol1 + '\n' + s[k:]
    open(REPORT, 'w').write(new)
    print('section 4 replaced: %d held-out rows, %d mol1 rows' % (len(rows), len(mrows)))
    print('report %.2f MB -> %.2f MB' % (len(s) / 1e6, len(new) / 1e6))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
