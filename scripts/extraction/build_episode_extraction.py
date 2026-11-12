#!/usr/bin/env python
"""v5.3 extraction: design EPISODES from rationale-bearing sections, then linked to compounds.

WHAT v1 GOT WRONG, measured. v1 extracted SENTENCES THAT PAIR COMPOUNDS and asked for the
purpose "in their words". Both halves were wrong, and the metrics say how wrong:

    old `purpose`:  median 7 words, 46% a complete sentence, **0% name a structural change**,
                    0% briefable, 23% are a bare comparative result
                    ("limited activity", "its potency", "Even better activity")

The sentence that compares two compounds is the sentence LEAST likely to carry a hypothesis --
it is SAR bookkeeping. The rationale lives in the introduction, the design section and the
discussion, often pages from the compound numbers. So v1 mined the wrong passages and then
compressed what it did find into a fragment: quotes are a median 3.8x longer than the purposes
drawn from them, and only 27% of records carry a recoverable design intent at all.

WHAT v2 CHANGES. Three things, each with a number behind it.

1. THE UNIT IS A DESIGN EPISODE, not a sentence. An episode is hypothesis -> change ->
   outcome, a discourse unit that may span a paragraph or a whole section. This is why v1's
   intents were "super short": it was never given a story-sized window. Independent evidence
   that width is the binding constraint: 38.3% of pocket sentences only become linkable to a
   compound at a +/-5,500-character window.

2. RATIONALE FIRST, COMPOUNDS SECOND. Find where the authors argue, then resolve which
   compounds the argument produced -- the inverse of v1. The linking step is known to be
   tractable (38.3% LINKED with a verbatim tie in the pocket test).

3. TWO SENTENCES, NOT ONE, and the first one must be a SENTENCE.
       reason_verbatim    the authors' own reason, their words, but COMPLETE -- never a
                          fragment. Measured at 100% complete, 0.96 grounded when done this
                          way, and it halves the outcome-only rate (23.1% -> 12.5%).
       design_rationale   the extractor's rephrasing, using the quote and the paper context,
                          written so a chemist could act on it. Measured at 92% names a
                          change, 85% carries a hypothesis, 0% outcome-only, 69% briefable.

THE CAP, AND WHY IT IS NOT COSMETIC. `design_rationale` may run to a few sentences, capped at
3 sentences / 80 words. The cap exists because the refined sentence is the most briefable AND
the least grounded field we have measured -- 0.50, against 0.96 for the authors' own words.
Only half its content words appear in the passage. Length buys drift, so the cap is paired
with a grounding floor: a rationale whose content words are not mostly in the supplied text is
rejected, not kept. The floor is set at 0.6 here and should be TUNED on this 20-paper test
rather than trusted.

WHY `none` MUST STAY CHEAP TO SAY. ~27-33% of SAR statements in these papers carry an
authorial rationale at all; the rest is bookkeeping. An extractor that cannot return zero
episodes for a paper will manufacture them, and a corpus of invented hypotheses is worse than
a small corpus of real ones. So the brief states the expected hit rate, and a paper with no
design reasoning is a correct answer.

Usage:
  python scripts/extraction/build_episode_extraction.py \
      --packets ~/Downloads/covalentformer_labelling_task/lit_intent_100/packets \
      --out <scratch>/episode_v2 --n-papers 20
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

BRIEF = """You are extracting MEDICINAL-CHEMISTRY DESIGN EPISODES from one paper.

A design episode is not a sentence. It is the unit a chemist would recognise as one piece of
design thinking:

    a HYPOTHESIS or objective the authors held
      -> a STRUCTURAL CHANGE they made because of it
        -> the OUTCOME they measured

The three parts are usually in different places. The hypothesis is typically in the
introduction or a "design"/"rationale" paragraph. The change is in the chemistry or a table.
The outcome is in the SAR discussion. Your job is to put them back together.

HOW TO WORK

PASS 1 -- find where the authors ARGUE. Read for passages that give a reason: an introduction
stating what the series needed, a design paragraph, a structure-based argument about the
binding site, a paragraph opening "In order to...", "We reasoned that...", "To address
this...", "Based on the crystal structure...", "This prompted us to...". These passages are
where design intent lives. Do NOT start from the SAR tables.

PASS 2 -- for each such passage, find which COMPOUNDS the argument produced. Use the paper's
own labels ("12", "5a", "(R)-23"). The labels may be a paragraph or a table away. If the text
does not let you tie specific labels to the argument, say so -- do not attach the nearest
number you can see.

WHAT TO RETURN PER EPISODE

  reason_verbatim    The authors' stated reason IN THEIR WORDS -- but as a COMPLETE SENTENCE.
                     Quote them. If their sentence is long, quote the clause that carries the
                     reason, completed into a grammatical sentence. NEVER a fragment: "limited
                     activity", "its potency", "bulky substituents" are all WRONG. If the
                     authors state no reason, this is null and the episode is not an episode.

  design_rationale   YOUR rephrasing of the same intent. AT MOST 3 SENTENCES AND 80 WORDS. It
                     must say what was changed, what property or interaction the change was
                     meant to affect, and the hypothesis connecting them. Write it so a
                     chemist who never read the paper could act on it.

                     **EVERY CHEMICAL SPECIFIC IN IT MUST BE SUPPORTED BY THE
                     `evidence_quote` YOU RECORD -- not by the rest of the paper.** The rest
                     of the paper tells you WHICH passage is a rationale and WHICH compounds
                     it concerns. It must not supply chemistry. Do not import from elsewhere
                     in the paper, or from your own knowledge:
                       - a group, substituent, ring or position the quote does not name
                       - a property or assay the quote does not mention
                       - a measured value, potency, fold-change or half-life
                       - an outcome, when the quote states only an intention
                       - a named protein, residue or target the quote does not name
                     If the quote states an intent but not the group it was applied to, say so
                     at that level of detail rather than filling the gap. A rationale that is
                     vaguer than you would like but fully supported is worth far more to us
                     than a precise one that adds a detail the quote does not carry.

                     This is the most common defect of an earlier round: a chemist panel
                     found 96% of rationales well-formed but 44% OVERSTATED -- correct in
                     direction while asserting specificity the passage never gave. Those are
                     unusable, so err toward the quote every time.

                     **BUT THE RATIONALE MUST STILL BE AN INSTRUCTION, NOT A PARAPHRASE.**
                     Staying inside the quote is a constraint on CONTENT, not a licence to
                     restate the sentence. Write it as something a chemist could act on.
                     Where the quote supports it, say what was changed AND what property or
                     interaction the change targets. Where the quote gives only one of those,
                     give that one -- do not invent the other, and do not retreat to
                     restating the measured result.

                         WANTED:    "Replace the alkyl ether at R1 with an oxidation-resistant
                                     group to improve microsomal stability, the hypothesis
                                     being that oxidation of that group limits stability."
                         NOT THIS:  "The alkyl ethers were less stable in human microsomes."
                                     -- true, and a RESULT, not an intent.
                         NOT THIS:  "Improve metabolic stability." -- an aim with no change.

                     A previous round grounded faithfully (97% of its chemistry was correct)
                     but 23% of its rationales were bare outcome statements and only 54%
                     named a structural change, because they hugged the quote's wording
                     instead of stating the design decision. Both failures matter: an
                     overstated rationale teaches wrong chemistry, a paraphrased one teaches
                     nothing.

  change_from / change_to    The structural change, as specific as the text supports.

  reference          The compound THIS PASSAGE compares against, as the paper labels it.
                     Usually one. It is NOT the series parent and NOT the paper's lead
                     compound. If the passage says a new series was built on the preferred
                     groups from an earlier table, those earlier compounds are NOT references
                     here -- they were never compared one-to-one with these. Give the label
                     the passage itself names, or return no episode. If the passage genuinely
                     works from several references (different series, different parents), give
                     each its own episode; do not pool them.

                     **`null` IS PERMITTED AND IS NOT A FAILURE.** A passage can state a real
                     hypothesis and a real structural move without naming a comparator -- "we
                     wished to find out whether the pocket would tolerate polar substituents at
                     R2" is design reasoning whether or not a label is attached. Emit it with
                     `reference: null` rather than discarding it. In trials, the mandatory
                     version silently threw away roughly half of the usable rationales.

  secondhand         true when THIS PAPER is narrating someone else's design decision -- a
                     review, a perspective, a retrospective. The reasoning is still real, but
                     the compound numbering is the REVIEW's, not the original paper's, so these
                     resolve to a molecule far less often (75% of labels against 91% from
                     primary papers, measured). Flag them; we decide later what to do with them.

                     A review episode has a higher bar: the passage must name a SPECIFIC
                     mechanism or property AND name compounds. "X is a known anticancer
                     pharmacophore, thus hybridisation with Y may provide novel agents" is a
                     section opener, not a design episode -- one review in our trial carried 18
                     such passages and only 5 real ones. Count the openers in
                     `n_rationale_passages_found`; do not emit them.

  reference_source   "stated"             the passage names the comparator label
                     "implied_by_passage" the comparator is unambiguous from the surrounding
                                          text (the series parent just introduced, the entry
                                          immediately above in the same table)
                     "none"               no comparator; `reference` is null

  analogues          A LIST of the compounds the passage puts on the other side of this
                     change. ONE RATIONALE LEGITIMATELY COVERS SEVERAL ANALOGUES -- "we tried
                     these four ways to get the effect" is one episode, not four. Each entry:

                       {"label": "39",
                        "what_changed": "the connecting oxygen replaced by sulfide",
                        "outcome": "reduced activity",
                        "outcome_direction": "worse"}

                     what_changed       describes THAT analogue against the reference, in the
                                        paper's terms. It must DIFFER between entries wherever
                                        the paper distinguishes them. One shared sentence
                                        across four analogues is a table, not four described
                                        analogues.
                                        IF THE PAPER DOES NOT DISTINGUISH THEM, DO NOT LIST
                                        THEM SEPARATELY. Give the analogues you can describe
                                        individually and leave the rest out; a list of
                                        identical descriptions tells us nothing and 10.5% of
                                        the previous round carried one.
                     outcome_direction  "better" | "worse" | "neutral" | "not_stated", for
                                        THAT analogue, AND ALWAYS ABOUT THE STATED `endpoint`.
                                        A change that improves the endpoint while costing
                                        something else is "better", with the cost written into
                                        `outcome` ("stability improved; broad-spectrum potency
                                        lost"). This case has been scored inconsistently in
                                        trials, so the rule is stated rather than left to
                                        judgement. A passage often lists compounds that
                                        support the trend together with ones that refute it;
                                        they must not share a direction.

                     AN ANALOGUE BELONGS IN THIS LIST ONLY IF it differs from the reference by
                     the change this episode describes. If it also carries a different
                     R-group, a different series letter, or a different core, leave it out.
                     We attach this rationale to the structural difference, so a difference at
                     two positions makes the rationale false.

  pairs              OPTIONAL, and the right answer when one rationale covers SEVERAL
                     MATCHED PAIRS at once: "the iso-Leucettinibs 5, 20 and 35 were compared
                     with 2, 17 and 32" is one argument over three pairs. Give
                     [["2","5"],["17","20"],["32","35"]] and leave `reference`/`analogues`
                     empty. Use this instead of picking one pair and dropping the rest -- an
                     extractor following the earlier wording kept 32/35 and discarded two
                     perfectly good comparisons.
                     ONE PAIR SET BELONGS TO ONE EPISODE. If two passages argue for the same
                     set of compounds, they are one episode with the better-supported
                     rationale, not two. A previous paper returned the identical 200-pair list
                     twice under two near-identical reasons, which doubles the apparent data
                     and teaches the same thing twice.

  endpoint           What the CHANGE was meant to affect -- not what happened to be measured:
                     "potency" | "selectivity" | "solubility" | "metabolic_stability" |
                     "permeability" | "pk" | "toxicity" | "herg" | "physchem" | "synthetic" |
                     "other" | "not_stated".
                     "herg" is its own value because papers write "hERG profile" or "hERG
                     binding" and never "toxicity", and the rule below forbids promoting their
                     word to a nearest fit -- five episodes in our trial ended up in "other"
                     for want of this one entry.
                     Take it from the authors' stated aim, not from the assay table. A paper
                     that reports IC50s for every compound is not thereby a potency episode:
                     "fluorine substitution consistently imparted decreased stability against
                     all plasma species" is metabolic_stability with a potency outcome
                     attached. If the passage names two aims, give the one the change TARGETS
                     and put the other in `outcome`. "physchem" covers pKa, logP/logD and
                     basicity when they are the stated target.
                     "not_stated" is a correct answer and is preferred to a guess: we use this
                     field to decide whether a measured potency may confirm direction, and a
                     wrong value here silently flips a pair.
                     THE PAPER'S OWN WORD GOVERNS. If the text says only "instability", do not
                     promote it to "metabolic_stability" because that is the nearest item on
                     the list -- give "other" and put the paper's term in `endpoint_verbatim`.
                     A nearest-fit is a guess wearing a controlled vocabulary.

  endpoint_verbatim  The paper's own phrase for the aim, exactly as printed, or null.
  evidence_quote     The verbatim passage carrying the reason.
  evidence_section   Where it came from: "introduction" | "design" | "results" |
                     "discussion" | "conclusion" | "unclear".
  basis              "experimental_structure" (they had a crystal/cocrystal/NMR structure) |
                     "model" (docking, homology, pharmacophore) | "sar_only" | "unclear".

  residues_verbatim  Any protein residues the quote names, EXACTLY as printed, one per entry:
                     ["Leu123", "Arg143"]. Keep the paper's own numbering and spelling -- do
                     NOT renumber, do not convert three-letter to one-letter, do not infer a
                     residue that is only implied. [] if none.
  target_named       The protein the quote is about, as the paper names it ("falcipain-2",
                     "KAT6A", "PPARdelta"), or null.
  target_identifier  Any UniProt accession, PDB code, or construct/isoform note stated
                     ANYWHERE in the paper for that protein ("P12345", "4K2R", "residues
                     1-289 construct", "numbering follows the mature protein"), or null.
                     This one field may draw on the whole paper, not just the quote -- it is
                     bookkeeping, not chemistry.

                     WHY WE ASK: we fold these proteins and try to match the residue the
                     authors name onto our own structure. A paper's numbering often differs
                     from the canonical sequence by a construct offset, and without the
                     identifier and the verbatim spelling we cannot detect the offset, so a
                     correctly extracted residue silently lands on the wrong amino acid.
                     Record what the paper says and let us resolve it.
  intent_class       "design_reason"    a hypothesis held BEFORE the change was made
                     "post_hoc_explanation"  the authors explain a result they did not design
                                        for -- real reasoning, but not an intent
                     (anything that is only a measured outcome is NOT an episode; omit it)
  outcome            What they measured, one short clause, or null.

WHAT NOT TO DO

Do not emit an episode for a bare SAR observation. "The 4-pyridyl isostere showed lower
affinity (7.55 vs 8.50)" is a result, not a design episode, however much it looks like one.
Roughly a quarter to a third of the design-relevant statements in a paper of this kind carry
an authorial rationale; the rest is bookkeeping. **A paper that yields two or three real
episodes, or none, is a correct and expected answer.** We would far rather have three true
episodes than thirty manufactured ones -- invented hypotheses are worse to us than no data,
because we train on them.

Do not supply chemistry from your own knowledge. If the text says "inversion of the amino acid
centre", do not write "L to D". If it says "Ca-substitution", do not write "alpha-methyl".

Do not write about the extraction. The rationale is the paper's design decision in your words;
it must not mention us, the quote, the authors' hindsight or this task. Clauses like
"following the authors' suggestion", "based on the post-hoc interpretation", "consistent with
the finding that" are OUR vocabulary leaking into the label -- 17% of the previous round
carried one. State the decision; the provenance already lives in `evidence_quote`.

  WANTED:   reference "1",
            analogues [{"label":"39","what_changed":"connecting oxygen replaced by sulfide",
                        "outcome":"reduced activity","outcome_direction":"worse"},
                       {"label":"31","what_changed":"connecting oxygen replaced by carbon",
                        "outcome":"slightly less potent than the sulfide",
                        "outcome_direction":"worse"}]
  NOT THIS: reference "1", analogues [{"label":"39"},{"label":"40"},{"label":"31"}] sharing one
            what_changed -- that is a table, not three described analogues.
  NOT THIS: pooling the compounds that SUPPORTED a trend with the ones that REFUTED it under a
            single rationale. Each analogue carries its own outcome_direction.

Return ONE json object, no prose, no code fence:
{"paper": "<doc id as given>",
 "n_rationale_passages_found": <int>,
 "episodes": [
   {"reason_verbatim": "...", "design_rationale": "...", "evidence_quote": "...",
    "evidence_section": "...", "intent_class": "...", "endpoint": "...", "basis": "...",
    "change_from": "...", "change_to": "...",
    "residues_verbatim": [...], "target_named": "...", "target_identifier": "...",
    "outcome": "...",
    "reference": "12", "reference_source": "stated", "secondhand": false,
    "endpoint_verbatim": "...",
    "analogues": [
      {"label": "39", "what_changed": "...", "outcome": "...", "outcome_direction": "worse"}
    ]}
 ]}
An empty episodes list is a valid answer."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-papers', type=int, default=20)
    ap.add_argument('--max-body', type=int, default=110000)
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--only-primary', action='store_true',
                    help='restrict to paper_kind==primary. OFF by default: see the note in '
                         'main(). Production has never used it.')
    ap.add_argument('--extractors', nargs='+', default=['e1'],
                    help='one task file per paper PER extractor, for agreement')
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'returned').mkdir(exist_ok=True)

    papers = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.packets), '*.json'))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        body = d.get('body_text') or ''
        if len(body) < 8000:
            continue
        papers.append(d)

    # EVERY PAPER IS EXTRACTED. This used to keep only `paper_kind == primary` and silently
    # discard the rest -- 12 of 20 papers in one trial set, with nothing in the output saying
    # so. The classifier is a heuristic over titles and sections, it is wrong often enough to
    # matter, and a review that explains someone else's SAR still carries design reasoning we
    # can use. The 869-paper production run never did this (it shipped 658 primary + 215
    # review/qsar); only this trial builder did, which made trials unrepresentative of
    # production. `paper_kind` rides along on every task so yield can be analysed by kind
    # AFTERWARDS instead of being decided beforehand.
    primary = [d for d in papers if (d.get('paper_kind') or '').lower() == 'primary']
    pool = [d for d in papers if (d.get('paper_kind') or '').lower() == 'primary'] \
        if a.only_primary else papers
    rnd = random.Random(a.seed)
    rnd.shuffle(pool)
    sel = pool[:a.n_papers]

    for d in sel:
      for ex in a.extractors:
        doc = f"doc_{d['doc_id']}"
        body = (d.get('body_text') or '')[:a.max_body]
        caps = (d.get('caption_text') or '')[:8000]
        (out / f'ep__{doc}__{ex}.json').write_text(json.dumps({
            'extractor': ex,
            'instructions': BRIEF,
            'paper': doc,
            'title': d.get('title'),
            'journal': d.get('journal'),
            'year': d.get('year'),
            'paper_kind': d.get('paper_kind'),
            'body_text': body,
            'caption_text': caps,
        }, indent=1))

    print(f'{len(papers)} usable packets, {len(primary)} primary, '
          f'{len(papers) - len(primary)} review/qsar -- all extracted; '
          f'built {len(sel)*len(a.extractors)} episode tasks '
          f'({len(sel)} papers x {len(a.extractors)} extractors) in {out}')
    print(f'  body text capped at {a.max_body:,} chars; '
          f'median supplied {sorted(len((d.get("body_text") or "")[:a.max_body]) for d in sel)[len(sel)//2]:,}')
    for d in sel[:5]:
        print(f'   doc_{d["doc_id"]:<8} {(d.get("paper_kind") or "?"):8} '
              f'{(d.get("title") or "")[:64]}')
    print('\nPREREGISTERED COMPARISON against v1 on these same papers:')
    print('  v1 `purpose`:  median 7 words, 0% name a change, 0% briefable, 23% outcome-only')
    print('  v2 passes if `design_rationale` is briefable on >=60% of episodes AND grounded')
    print('  >=0.6 AND episodes-per-paper is not inflated (a paper yielding 0-3 is expected).')
    print('  The failure mode to watch is OVER-production: many episodes per paper with high')
    print('  briefable scores would mean the extractor is manufacturing hypotheses.')


if __name__ == '__main__':
    main()
