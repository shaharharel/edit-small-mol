#!/bin/bash
# Restart the v4 run cleanly. THIS MUST BE A SCRIPT, NOT AN ssh --command STRING.
#
# `pkill -f train_v4_head` sent as an ssh command matches the remote sshd child process whose argv
# IS that command string, so the command kills its own session and gcloud reports exit 255. Four
# consecutive "relaunches" failed that way tonight while scp to the same host kept working, which is
# what made it look like a flaky IAP tunnel. Keeping the pattern inside a file means the ssh command
# line only ever contains "v4_restart.sh".
set -u
HOME_DIR=/home/shaharh_quris_ai
cd "$HOME_DIR"

# kill by pidfile-free but self-safe means: match the python invocation, excluding this script's pid
for pid in $(pgrep -f 'train_v4_head\.py' | grep -v "^$$\$"); do
  kill "$pid" 2>/dev/null
done
pkill -f 'v4_watchdog\.sh' 2>/dev/null

# WAIT FOR THE GPU TO ACTUALLY FREE. A 7B model does not release its ~16 GB the instant its process
# is signalled, and train_v4_head.py refuses to start below 30 GB free (correctly -- accelerate would
# otherwise silently place the model on CPU and run ~100x slower). A fixed `sleep 5` lost one restart
# to exactly that race: "REFUSING TO START: only 24796 MiB free", after which nothing was training.
for _ in $(seq 1 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" -eq 0 ] && break
  sleep 5
done
sleep 5

rm -rf "$HOME_DIR/v4_out" "$HOME_DIR/v4_train.log"
setsid bash -c "nohup $HOME_DIR/a_v4.sh >/dev/null 2>&1 </dev/null &"
sleep 3
setsid bash -c "nohup $HOME_DIR/v4_watchdog.sh >/dev/null 2>&1 </dev/null &"
sleep 20
echo "gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
echo "RESTART_DONE"
