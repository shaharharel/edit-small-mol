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

## State at the time of copy

- **70 of 71 chair files.** `batch_0018` is absent and has **no labeller files either**, so those
  100 pairs need a full three-labeller plus chair pass, not a cheap chair re-run. The labelling
  team is sending that batch separately.
- 6,969 of 7,069 pairs labelled; 6,092 usable (ADOPT+REWRITE), 87.4%.
- Records carry no `label_generation` stamp yet. Stamping is deliberately left until the corpus is
  complete, so the manifest cannot record a generation that is still missing a batch.

## Measured against g3 (identical scoring code, `compare_g3_g4.py` and `g34_attrib.py`)

| | g3 | g4 |
|---|---|---|
| usable | 86.2% | 87.4% |
| all six hard rules | 91.5% | 91.0% |
| r2, no analogue distance cited | 98.7% | 100.0% |
| every cited distance exists in the input table | 92.4% | **100.0%** |
| (residue, distance) co-occur in one table row | 55.2% | **78.7%** |
| r1c, residues from the contact table alone | 89.0% | 80.9% |
| words per sentence | 32.9 | 26.0 |
| distances per sentence | 2.65 | 1.44 |
| duplicate sentences | 0 | 0 |
| lexical type/token ratio, size-matched | 0.0234 | 0.0317 |

g4 is terser and cites about half as many distances, but the ones it cites are attributed
correctly far more often — +23.5 points on the exact failure class three medicinal chemists found
dominant in model output. Its lower r1c means it leans more on the pocket envelope, which the
brief permits.

The (residue, distance) check pairs each residue mention with the nearest following number, so it
undercounts when a sentence separates the two across clauses. That floor applies equally to both
generations.

2,600 pairs carry a usable label in **both** g3 and g4, and the sentences differ on 100% of them,
which supports a properly paired comparison; a blind three-chemist panel over 30 of those pairs is
in `../panel_g3_vs_g4/`.
