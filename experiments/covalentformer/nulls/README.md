# The length-matched shape-blind null — DERIVATION, not a constant

`sweep_test.py` used to hardcode `LOOKUP_NULL_BINNED = 0.2917`. No script produced it and no
artifact recorded it, and it was WRONG: the binned within-stratum metric is MONOTONE in the null
policy's own length response, so a null is uninterpretable unless it is LENGTH-MATCHED to the model
it judges. 0.2917 implies a policy with rho(d,bonds) ~ 0.84-0.88 — far more length-obedient than the
model's measured +0.4992 — so it compared the model against a strictly stronger policy and read the
difference as failure.

**These files exist so that never happens again.** The replacement number must be re-derivable, so
the derivation and its inputs live in the repo rather than in /tmp.

    build_length_null_pool.py   builds the fragment pool (bond count, median reach) from
                                data/envelopes.jsonl and data/envelopes_20k.jsonl
    derive_length_matched_null.py  sweeps the lookup tolerance tau, calibrates to the model's own
                                rho(d,bonds), and reports the binned null at that calibration
    _pool_envelopes.json        persisted pool, 3,390 fragments
    _pool_envelopes_20k.json    persisted pool, 12,680 fragments

## The honest interval is WIDER than one implementation suggests

These scripts vary POOL and n. They do NOT vary the POLICY — what "shape-blind length-matched
lookup" means as a sampler. A second, independent implementation of the same six words (Laplacian
rather than Gaussian kernel; bond counts weighted by natural abundance rather than by kernel alone),
calibrated to the same rho(d,bonds)=+0.4992, returns +0.0971 / +0.1153 / +0.1495 where this one
returns +0.0992 / +0.1174 / +0.1198 — and on the SAME envelopes_20k pool the two disagree by 0.05,
more than twice the model's margin.

    this implementation:        +0.0992 .. +0.1198
    independent implementation: +0.0971 .. +0.1495
    HONEST UNION:               +0.0971 .. +0.1495
    model binned pooled rho:    +0.1389 (sd 0.0175)

So the model sits INSIDE the union. The right word for that is UNRESOLVABLE, not INDISTINGUISHABLE:
"indistinguishable" names a measurement that came back null, and this one never had the resolution to
come back either way. Two good-faith readings of the same definition disagree by more than the effect,
so the verdict is a property of whose null you use, not of the model. This implementation's own
per-replicate sd is ~0.021, WIDER than the 0.0206 range it produces.

TWO FURTHER REASONS NOT TO TREAT THIS AS A SETTLED NULL:
  - IT IS A CROSS-POPULATION LEVELS COMPARISON. The null is built from the envelope fragment pools
    (data/envelopes.jsonl, envelopes_20k.jsonl); the model's rho is measured on ITS OWN GENERATIONS.
    Those are different populations, so the two sides are not commensurable to begin with.
  - ONE OF THE TWO POOLS IS ALREADY FILED AS UNSOURCED. #84 records envelopes_20k as an ORPHAN
    artifact, 56% of it untraceable to any input on disk. A verdict that settles the shape question
    should not rest on an artifact that has itself been filed as unreproducible.

History of this one comparison: BELOW (#94) -> ABOVE (#129) -> WITHIN (#154) -> UNRESOLVABLE. Four
verdicts on one number is the signal that the measurement, not the model, is what needs rebuilding.

A CEILING THAT BOUNDS ANY SHAPE STATISTIC HERE: over the fragment population rho(bond count, median
reach) = +0.9665 and only 5.8% of reach variance is WITHIN bond count (4.9% on the 20k pool). A small
partial rho is largely that ceiling asserting itself, not necessarily a model failure.
