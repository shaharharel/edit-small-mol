#!/bin/bash
# v4 WATCHDOG for a100-a (SPOT -- already preempted once during this build).
#
# LIVENESS COMES FROM nvidia-smi, NOT pgrep. A pgrep pattern matching the trainer's name also
# matches this watchdog's own argv and any ssh --command carrying that string. That exact false
# positive already fired once tonight: a check reported TRAINER_ALIVE with 302 s "elapsed" while
# nothing was on the GPU at all, which hid a dead run for ~25 minutes. The compute-app list cannot
# self-match because only real CUDA contexts appear in it.
#
# Every 10 min:
#   no GPU process        -> relaunch with --auto-resume (restores adapter + head, fast-forwards LR)
#   step counter frozen   -> after 3 consecutive checks (30 min) kill and resume from checkpoint
#   'saved adapter' in log-> the run finished; exit
# Writes v4_watchdog.log only; it never generates prompts.
HOME_DIR=/home/shaharh_quris_ai
L=$HOME_DIR/v4_watchdog.log
TOTAL=15745
say() { echo "$(date -u +'%m-%d %H:%M:%S') $*" >> "$L"; }

last=-1
stall=0
for i in $(seq 1 3000); do
  # SLEEP FIRST, THEN CHECK. Checking immediately on startup raced the trainer: a 7B model takes
  # ~90 s to load plus ~150 s to tokenise the length filter, during which nvidia-smi shows no compute
  # app. The watchdog read that as "dead", launched a SECOND trainer, and the second one then hit
  # "REFUSING TO START: only 24794 MiB free" because the first had meanwhile claimed the card. Three
  # restarts tonight logged that refusal for exactly this reason.
  sleep 600
  step=$(grep -oE 'step [0-9]+/' "$HOME_DIR/v4_train.log" 2>/dev/null | tail -1 | grep -oE '^step [0-9]+' | grep -oE '[0-9]+')
  step=${step:-0}

  if grep -q 'saved adapter' "$HOME_DIR/v4_train.log" 2>/dev/null; then
    say "training COMPLETE at step $step -- watchdog exiting"
    exit 0
  fi

  if [ "$napps" -eq 0 ]; then
    say "NO GPU PROCESS at step $step -- relaunching with --auto-resume"
    cd "$HOME_DIR" && setsid bash -c 'nohup ./a_v4.sh >/dev/null 2>&1 </dev/null &'
    stall=0
  elif [ "$step" -eq "$last" ] && [ "$step" -gt 0 ]; then
    stall=$((stall + 1))
    say "step frozen at $step ($stall consecutive checks)"
    if [ "$stall" -ge 3 ]; then
      say "WEDGED for 30 min -- killing and resuming from checkpoint"
      pkill -f train_v4_head
      sleep 20
      cd "$HOME_DIR" && setsid bash -c 'nohup ./a_v4.sh >/dev/null 2>&1 </dev/null &'
      stall=0
    fi
  else
    say "ok step $step/$TOTAL ($((step * 100 / TOTAL))%) gpu_apps=$napps"
    stall=0
  fi
  last=$step
done
