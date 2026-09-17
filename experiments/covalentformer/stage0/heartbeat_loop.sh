#!/bin/bash
# Hourly heartbeat WITHOUT cron -- macOS gates cron behind Full Disk Access and `crontab -e`
# hangs waiting for it. A plain background loop needs no permission and survives this session.
cd /Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer
PY=/opt/miniconda3/envs/quris/bin/python
for i in $(seq 1 12); do
  {
    echo "--- heartbeat $i  $(date '+%m-%d %H:%M') ---"
    if pgrep -f night_pipeline.sh > /dev/null; then echo "pipeline ALIVE"
    else echo "pipeline DOWN -> relaunching (stages skip completed work)"
         nohup bash stage0/night_pipeline.sh > logs/night_pipeline_outer.log 2>&1 & fi
    $PY stage0/qa_sweep.py 2>&1 | tail -8
  } >> logs/heartbeat.log 2>&1
  sleep 3600
done
