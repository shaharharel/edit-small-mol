#!/bin/bash
# Drive both arms of the pocket-dependence ablation end to end, on the box, in one place.
#
# Sequential on purpose: one A100, and two 8B trainings sharing it would contend for memory and make
# the wall-clock numbers meaningless. Order is pocket -> blind -> eval pocket -> eval blind -> eval
# untuned base, so the two trained arms exist before anything is scored and a crash in eval cannot
# cost a training run. Every stage appends to its own log and is skipped if its output already
# exists, so a preempted VM can re-run this script and resume rather than start over.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/quris/bin/python
export HF_HUB_OFFLINE=1          # the snapshot is complete and cached; no hub dependency mid-run
mkdir -p sft_runs

stage () {  # stage <sentinel> <log> <cmd...>
  local sentinel="$1"; shift; local log="$1"; shift
  if [ -e "$sentinel" ]; then echo "SKIP  $sentinel exists"; return 0; fi
  echo "START $(date -Is)  -> $log"
  "$@" >>"$log" 2>&1
  local rc=$?
  echo "END   $(date -Is) rc=$rc  $log"
  return $rc
}

stage sft_runs/pocket/adapter        sft_runs/train_pocket.log \
      $PY train_sft.py --arm pocket --out sft_runs/pocket
stage sft_runs/blind/adapter         sft_runs/train_blind.log \
      $PY train_sft.py --arm blind  --out sft_runs/blind
stage sft_runs/pocket/eval/eval_summary.json sft_runs/eval_pocket.log \
      $PY eval_sft.py --arm pocket --run sft_runs/pocket --out sft_runs/pocket/eval
stage sft_runs/blind/eval/eval_summary.json  sft_runs/eval_blind.log \
      $PY eval_sft.py --arm blind  --run sft_runs/blind  --out sft_runs/blind/eval
# The untuned base on the pocket prompts: the zero-shot floor both arms are measured against.
stage sft_runs/base_pocket/eval_summary.json sft_runs/eval_base.log \
      $PY eval_sft.py --arm pocket --base-only --out sft_runs/base_pocket
echo "ALL DONE $(date -Is)"
