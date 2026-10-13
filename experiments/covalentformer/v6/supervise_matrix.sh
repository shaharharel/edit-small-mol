#!/bin/bash
# Keep the 2x2 matrix making forward progress across preemptions. Runs LOCALLY, not on the box.
#
#   ./supervise_matrix.sh            # until the matrix finishes or MAX_CYCLES is reached
#
# ai-gpu-a100 is preemptible and this zone is actively reclaiming it: preempted 2026-10-08 and again
# 2026-10-12 eight minutes into the first training run. The matrix is ~3.5 h sequential, so the
# expected number of interruptions is greater than zero and babysitting it by hand wastes a person.
#
# WHAT MAKES THIS SAFE TO LOOP
#   run_matrix.sh is idempotent. Each stage is guarded by a sentinel (the adapter directory, or the
#   eval summary), so a finished stage is skipped, and train_sft.py --resume auto picks up the last
#   saved checkpoint of a partially-trained cell. So relaunching after a preemption resumes; it does
#   not restart from scratch, and it cannot double-write a finished cell.
#
# WHAT IT COSTS WHEN PREEMPTED
#   Checkpoints are per-epoch, so one preemption loses at most the epoch in flight -- about 7 min
#   for a pocket cell. The 2026-10-12 preemption landed mid-epoch-1 with no checkpoint yet, which is
#   the worst case and still only cost 8 minutes.
#
# IT NEVER STOPS THE VM. It only starts one that preemption has TERMINATED. It exits 0 when all
# WANT_CELLS cells have an eval summary on disk, and exits 1 when STALL_LIMIT consecutive relaunches
# add no new cell -- a genuinely broken stage should surface to a human rather than be retried
# forever, while a merely preempted one should not. MAX_CYCLES bounds a pathological loop.
set -u
ZONE=us-central1-a
VM=ai-gpu-a100
REMOTE=~/covf_v6
RUNS_DIR=${RUNS_DIR:-sft_runs}
DATA_DIR=${DATA_DIR:-sft_out}
LOG=${LOG:-matrix_driver.log}
MAX_CYCLES=${MAX_CYCLES:-200}
SLEEP_POLL=${SLEEP_POLL:-120}
WANT_CELLS=${WANT_CELLS:-8}      # 4 cells x 2 seeds
STALL_LIMIT=${STALL_LIMIT:-3}    # relaunches with no new cell before giving up
last_done=-1
relaunches=0

say () { echo "[supervise $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

for ((cycle = 1; cycle <= MAX_CYCLES; cycle++)); do
  status=$(gcloud compute instances describe "$VM" --zone "$ZONE" \
           --format='value(status)' 2>/dev/null)
  say "cycle $cycle/$MAX_CYCLES, vm=$status"

  if [ "$status" != "RUNNING" ]; then
    say "starting $VM (preemption leaves it TERMINATED; this never stops a running VM)"
    gcloud compute instances start "$VM" --zone "$ZONE" >/dev/null 2>&1
    until gcloud compute ssh "$VM" --zone "$ZONE" --command 'nvidia-smi -L' 2>/dev/null \
          | grep -q A100; do
      sleep 15
    done
    say "box back up"
  fi

  # COMPLETION IS MEASURED IN ARTIFACTS, NOT IN THE DRIVER'S LOG. run_matrix.sh prints "ALL DONE"
  # when it reaches the end of its stage list even if individual stages failed -- on 2026-10-12 it
  # printed ALL DONE with smiley_pocket_s0 unbuilt, because that cell's resume had died on a
  # corrupt checkpoint and the driver simply moved on. Counting the eval summaries that actually
  # exist is the only completion test that cannot lie, and it is also what makes a relaunch
  # worthwhile: the driver's sentinels skip finished cells, so re-running it retries exactly the
  # ones still missing.
  done_cells=$(gcloud compute ssh "$VM" --zone "$ZONE" \
    --command "ls $REMOTE/$RUNS_DIR/*_s[01]/eval/eval_summary.json 2>/dev/null | wc -l" \
    2>/dev/null | tr -d '[:space:]')
  done_cells=${done_cells:-0}
  say "cells complete: $done_cells/$WANT_CELLS"
  if [ "$done_cells" -ge "$WANT_CELLS" ]; then
    say "all $WANT_CELLS cells have an eval summary -- matrix complete"
    exit 0
  fi
  if [ "$done_cells" -le "$last_done" ] && [ "$relaunches" -ge "$STALL_LIMIT" ]; then
    say "no new cell completed across $STALL_LIMIT relaunches -- stopping for a human"
    gcloud compute ssh "$VM" --zone "$ZONE" \
      --command "grep -E 'rc=[1-9]' $REMOTE/$LOG | tail -6" 2>/dev/null
    exit 1
  fi
  [ "$done_cells" -gt "$last_done" ] && relaunches=0
  last_done=$done_cells

  # Relaunch only when nothing is already running, so a healthy run is never disturbed.
  if gcloud compute ssh "$VM" --zone "$ZONE" \
       --command 'pgrep -f run_matrix.sh >/dev/null' 2>/dev/null; then
    say "a driver is already running (one GPU, matrices are sequential); waiting"
  else
    say "no driver running; relaunching (finished stages skip, partial cells resume)"
    relaunches=$((relaunches + 1))
    gcloud compute ssh "$VM" --zone "$ZONE" \
      --command "cd $REMOTE && DATA_DIR=$DATA_DIR RUNS_DIR=$RUNS_DIR nohup ./run_matrix.sh >> $LOG 2>&1 & sleep 3" \
      >/dev/null 2>&1
  fi
  sleep "$SLEEP_POLL"
done
say "MAX_CYCLES reached without ALL DONE -- not retrying further"
exit 2
