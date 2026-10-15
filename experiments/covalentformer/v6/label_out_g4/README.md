# g4 — second labelling pass over the full 7,069-pair corpus

Safe copy of the deliverable from `~/Downloads/covalentformer_labelling_task`, taken 2026-10-15.
That folder is outside the repo and on a Downloads path, so this is the authoritative copy.

## What this is

A second, independent human labelling of the same corpus g1–g3 were drawn from: same pipeline,
same briefs (`briefs/REVIEWER.md` and `briefs/CHAIR_FULL.md` are **byte-identical** to the copies
in this directory's parent, verified by diff), but **different labellers and different chairs**.
Three independent labellers per batch, then one chair adjudicating, as in g3.

## Contents

```
batch_0001.json .. batch_0071.json   chair deliverables, one record per pair
raw/batch_NNNN_lab{A,B,C}.json       the three independent labeller readings per batch
provenance/                          rule_report.json, chunk_manifest.json, versions/,
                                     chemist_review_1100/, runner.stdout.log,
                                     missing_only_preservation_snapshot.json
```

Not copied, because it is bulky and regenerable from `data/pdb_mirror/v7_evidence.jsonl`:
`label_work/jobs` (365 MB), `label_work/evidence` (56 MB), `derived_rule_evidence.jsonl` (33 MB),
and `batches/` (55 MB).

## State: COMPLETE

- **71 of 71 chair files, 213 labeller files, 7,069 of 7,069 pairs.** `tools/verify_output.py`
  passes: every manifest pair has exactly one non-empty label with a valid verdict, every batch has
  its three labeller readings, and no key is a bare PDB code.
- 6,178 usable (ADOPT+REWRITE), 87.4%. ADOPT 5,102 / REWRITE 1,076 / REJECT 891.
- `batch_0018` arrived in a second delivery. Before adding it, the 6,969 records copied earlier
  were compared against the new source field by field: **0 changed, 0 lost, exactly +100 new**. So
  the second delivery was additive and did not quietly revise anything.
- Stamped `label_generation = g4_v2_input_anchored_envelope_pass2` by `stamp_g4.py`, which exists
  because dataset_manifest.py resolves a generation from the batch NUMBER -- and g4 re-labels
  batches 0001-0071, the same numbers as g1/g2/g3, in a separate directory. Stamping was held
  until verify_output.py passed so the field could not assert a completeness that did not exist.

## Measured against g3 (identical scoring code, `compare_g3_g4.py` and `g34_attrib.py`)

| | g3 (2,644 usable) | g4 (6,178 usable) |
|---|---|---|
| usable | 86.2% | 87.4% |
| all six hard rules | 91.5% | 91.1% |
| r2, no analogue distance cited | 98.7% | 100.0% |
| every cited distance exists in the input table | 92.4% | **100.0%** |
| (residue, distance) co-occur in one table row | 55.2% | **78.6%** |
| r1c, residues from the contact table alone | 89.0% | 80.9% |
| words per sentence | 32.9 | 26.0 |
| distances per sentence | 2.65 | 1.44 |
| duplicate sentences | 0 | 0 |
| lexical type/token ratio, size-matched | 0.0234 | 0.0315 |

g4 is terser and cites about half as many distances, but the ones it cites are attributed
correctly far more often — +23.4 points on the exact failure class three medicinal chemists found
dominant in model output. Its lower r1c means it leans more on the pocket envelope, which the
brief permits.

The (residue, distance) check pairs each residue mention with the nearest following number, so it
undercounts when a sentence separates the two across clauses. That floor applies equally to both
generations.

2,600 pairs carry a usable label in **both** g3 and g4, and the sentences differ on 100% of them,
which supports a properly paired comparison; a blind three-chemist panel over 30 of those pairs is
in `../panel_g3_vs_g4/`.
