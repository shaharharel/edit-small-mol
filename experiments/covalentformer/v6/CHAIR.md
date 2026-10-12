# Chair — adjudicate three reviewer sentences for one pair

You see the same evidence the three reviewers saw, plus their three sentences.

Choose ONE of:
  ADOPT <n>   — reviewer n's sentence is correct and useful as written
  REWRITE     — supply your own sentence (same hard rules), when all three are partly right
  REJECT      — the evidence cannot support any grounded sentence for this pair

REJECT is a legitimate and expected outcome, and it is the INFORMATIVE answer when a pair has no
pocket consequence -- not the lazy one. Being asked for a more informative sentence never means
manufacturing one: a pair whose edited atoms sit in solvent touching nothing yields no design
rationale, and writing "the new atoms touch nothing within 4.0 A" is not a rationale, it is a
rejection with extra words. In a 20-pair trial the two pairs correctly rejected on this ground were
both talked into labels once the panel was pushed to be more informative; that is a worse outcome
than either rejection. Reject when: the changed atoms have no measured
contacts, every atom is solvent-facing with nothing to say, or the pair is not a medicinal-chemistry
analogue (e.g. a tagged probe, a crystallography construct, a cofactor).

Prefer a sentence that names what the ANALOGUE gains or loses in the pocket over one that only
describes the INPUT. Apply the same nine hard rules from REVIEWER.md -- including rule 9: a sentence quoting an
`uncertain` analogue contact is rejected.

Return exactly:
VERDICT: ADOPT 2 | REWRITE | REJECT
SENTENCE: <the final sentence, or empty if REJECT>
WHY: <one short clause>