#!/bin/bash
# Resurrection watchdog — runs from cron every 2 min.
# Ensures orchestrator + driver + sampler are alive as needed.
# Uses lock file to avoid stampede.
LOCK=/tmp/boltz_dpo_watchdog.lock
if [ -f $LOCK ]; then
  ppid=$(cat $LOCK 2>/dev/null)
  if [ -n "$ppid" ] && kill -0 $ppid 2>/dev/null; then
    exit 0
  fi
fi
echo $$ > $LOCK

ROOT=/home/shaharh_quris_ai/edit-small-mol
CAMPAIGN=$ROOT/data/paper_pair_training/boltz_dpo_campaign
LOGS=$CAMPAIGN/logs

log() { echo "[$(date -u +%FT%TZ)] $*" >> $LOGS/watchdog.log; }

# If DONE flag exists, nothing to do
if [ -f $ROOT/data/agent_coord/from_boltz_dpo_campaign_DONE.txt ]; then
  rm -f $LOCK
  exit 0
fi

# Ensure orchestrator running
if ! pgrep -f "boltz_dpo_campaign_orchestrator.sh" > /dev/null; then
  log "orchestrator DOWN — restarting"
  cd $ROOT
  nohup bash experiments/boltz_dpo_campaign_orchestrator.sh >> $LOGS/orchestrator_main.log 2>&1 &
fi

rm -f $LOCK
