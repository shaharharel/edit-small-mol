# Phase B warm-start provenance — RECOVERED, not stamped

Every `ckpt_B_*` checkpoint written before 2026-09-14 records **byte-identical** metadata:

    {'epoch': 0, 'arm': 'from_a', 'mode': 'geom', 'cond_dim': 3, 'seed': 101}

`ckpt_B_from_a/ep0`, `ckpt_B_froma0_s101/ep0` and `ckpt_B_fromA3_s101/ep0` are indistinguishable
from each other by their own contents. The one fact that separates the arms — **which Phase A run,
and which epoch of it, supplied the warm start** — survived only in directory names and `/tmp/*.log`.

`train_phaseB.py` now stamps `init`, `init_mode`, `init_epoch`, `init_seed`, `init_train_file` and
`init_sha256` into every saved checkpoint. That fixes future runs. It does **not** retro-label the
checkpoints below, which is why this file exists: without it, the mapping is lost on the next
`/tmp` clear and none of the Phase B contrasts can be rebuilt from artifacts on disk.

## Mapping (recovered by QA-code-23 from launch logs)

| Phase B arm | warm-started from | init epoch |
|---|---|---|
| `from_a`, `from_a_s202`, `from_a_s303`, `maml` | `ckpt_A_role/ep2` | 2 |
| `fromaE0_s101` / `_s202` / `_s303` | `ckpt_A_role/ep0` | 0 |
| `froma0_s101` / `_s202` / `_s303` | `ckpt_A0/ep0` | 0 |
| `fromA2_s101` / `_s202` / `_s303` | `ckpt_A_role_strat/ep2` (sha `b65ef670fad37a10…`) | 2 |
| `fromA3_s101` / `_s202` / `_s303` | `ckpt_A_role_strat_s2/ep2` (sha `07a4d17739209dfb…`) | 2 |
| `scratch*`, `zerocond*` | no init | — |

Only the `fromA2`/`fromA3` families carry an `INIT= sha256` line at all, and that came from the
launcher shell, not from the trainer.

## Two consequences for results already filed

1. **The `from_a` vs `froma0` contrast behind #49/#51/#55 is init-EPOCH-mismatched**, not merely
   init-MODE-mismatched: `from_a` starts from `ckpt_A_role/ep2` while `froma0` starts from
   `ckpt_A0/ep0`. `fromaE0` (← `ckpt_A_role/ep0`) is `froma0`'s epoch-matched partner and is the
   comparison that should be quoted. This needs reconciling with what #55 and #57 already say
   about that pairing.

2. **`ckpt_A_role/*` has no `train_file`, `valid_file` or `seed` key at all.** Verified on disk:
   `ckpt_A_role/ep0.ckpt` and `ep2.ckpt` are exactly `{'epoch': N, 'mode': 'role'}`, whereas
   `ckpt_A0/ep0.ckpt` is `{'epoch':0,'mode':'none','train_file':'train.csv','valid_file':'valid.csv','seed':101}`.
   An earlier version of this line said those keys "record None". That was wrong, and it is the
   same absent-versus-measured-zero conflation this whole file exists to complain about. The
   conclusion is unaffected — `src_ck.get('seed')` returns None either way, so #30's unrecoverable
   Phase A seed still propagates into the entire `from_a` family and the split guard still cannot
   be applied retroactively — but the stated fact about disk contents was false.

## Evidence class of each row — read this before quoting the table

| rows | backing |
|---|---|
| the two sha256 values | **artifact-backed.** Independently recomputed from the Phase A files; both match. |
| every arm→init ROW in the table above | **log-backed only.** Nothing on disk links any `ckpt_B_*` to any Phase A file — that absence is the entire reason this file exists. The hashes are correct hashes *of those Phase A checkpoints*; no artifact connects them to `fromA2`/`fromA3`. |

The one exception, established independently: **`ckpt_Bconv_from_a_s{101,202,303}` ← `ckpt_A_role_strat/ep2.ckpt`**, recovered by
encoder-weight L2 distance (1.1947, against 7.35 for the next-nearest candidate and 14.30 for a
random-init control) and then confirmed when a fresh run stamped `sha=b65ef670fad37a10` — matching
the hash in the table above by a third, independent route.

## The `ckpt_Bconv_*` family is NOT stamped either

The six 12-epoch convergence runs (`ckpt_Bconv_{scratch,from_a}_s{101,202,303}`) launched at 11:50,
*before* the stamping edit landed at 12:06, and held the pre-edit module for their whole life. Their
checkpoints carry no `init_*` keys despite being written after the edit. So the freshest Phase B
evidence on disk has the same provenance hole as the runs this document was written about — covered
only by the weight-distance recovery noted above. The later `_s{404,505,606}` runs *are* stamped.
