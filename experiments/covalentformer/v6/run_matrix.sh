#!/bin/bash
# Drive the whole 2x2 pocket-dependence matrix plus the zero-shot baselines, on the box.
#
#                           pocket-aware        pocket-blind
#   SmileyLlama-1.1         run 1 (x2 seeds)    run 2 (x2 seeds)
#   Llama-3.1-8B-Instruct   run 3 (x2 seeds)    run 4 (x2 seeds)
#
# ORDER IS THE PRIORITY ORDER, deliberately. The zero-shot baselines run first because they are
# generation-only, cost minutes, and tell us what each checkpoint brings BEFORE any LoRA -- which is
# what decides whether SmileyLlama is the right start at all. Then run 1 and run 2, the ablation
# that can invalidate the entire experiment; then runs 3 and 4. A preemption part-way therefore
# leaves the most important comparisons already done rather than a half-filled matrix.
#
# Sequential on one A100 on purpose: two 8B trainings sharing it would contend for memory and make
# the wall-clock numbers meaningless. Every stage is skipped if its output exists, so re-running
# after a preemption resumes instead of starting over.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/quris/bin/python
export HF_HUB_OFFLINE=1          # both snapshots are complete and local; no hub dependency mid-run
# DATA_DIR / RUNS_DIR make the label-set size a parameter instead of a fork of this script. The
# 1,000-pair matrix and the 2,000-pair matrix are the same experiment at two training-set sizes, and
# that comparison is only readable if literally nothing else differs. Crucially a new RUNS_DIR also
# means a rebuild can never overwrite the data a live run is training on -- rebuilding sft_out in
# place while cells were running would have swapped the split underneath them.
DATA_DIR=${DATA_DIR:-sft_out}
RUNS_DIR=${RUNS_DIR:-sft_runs}
mkdir -p "$RUNS_DIR"

stage () {  # stage <sentinel> <log> <cmd...>
  local sentinel="$1"; shift; local log="$1"; shift
  if [ -e "$sentinel" ]; then echo "SKIP  $log"; return 0; fi
  # The log lives INSIDE the run directory, which only train_sft.py creates -- but the shell opens
  # the redirect before the command runs, so without this mkdir every stage died instantly with
  # "No such file or directory" and never executed. All eight cells failed this way.
  mkdir -p "$(dirname "$log")"
  echo "START $(date -Is)  $log"
  "$@" >>"$log" 2>&1
  # Capture rc IMMEDIATELY, into a variable, before any other command runs. Written inline as
  # `echo "... rc=$? ..."` the $(date -Is) substitution executes first and resets $?, so every
  # stage reported rc=0 whatever happened -- which is exactly how eight instant failures printed
  # "ALL DONE" and looked like a clean run.
  local rc=$?
  echo "END   $(date -Is) rc=$rc  $log"
  return $rc
}

# --- zero-shot baselines: what each checkpoint brings before training -----------------------
for M in smiley base; do
  stage $RUNS_DIR/zeroshot_${M}/eval_summary.json $RUNS_DIR/zeroshot_${M}.log \
        $PY eval_sft.py --model $M --arm pocket --split val --data-dir $DATA_DIR --out $RUNS_DIR/zeroshot_${M}
done

# --- the 2x2, two seeds each, priority order ------------------------------------------------
for CELL in "smiley pocket" "smiley blind" "base pocket" "base blind"; do
  set -- $CELL; M=$1; A=$2
  for S in 0 1; do
    R=$RUNS_DIR/${M}_${A}_s${S}
    stage $R/adapter      $R/train.log  $PY train_sft.py --model $M --arm $A --seed $S --out $R --data-dir $DATA_DIR
    stage $R/eval/eval_summary.json $R/eval.log \
          $PY eval_sft.py --model $M --arm $A --adapter $R/adapter --split val --seed $S \
                          --data-dir $DATA_DIR --out $R/eval
  done
done
echo "ALL DONE $(date -Is)"
