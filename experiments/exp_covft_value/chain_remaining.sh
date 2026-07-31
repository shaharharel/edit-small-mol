#!/bin/bash
# Wait for base sampling PID then run covft and warhead_tokens serially
while ps -p 72243 > /dev/null 2>&1; do sleep 10; done
echo "[$(date +%H:%M:%S)] base done — starting covft" >> sample_run.log
/opt/miniconda3/envs/quris/bin/reinvent -d mps sample_covft_prior.toml >> sample_run.log 2>&1
echo "[$(date +%H:%M:%S)] covft done — starting warhead_tokens" >> sample_run.log
/opt/miniconda3/envs/quris/bin/reinvent -d mps sample_warhead_tokens_prior.toml >> sample_run.log 2>&1
echo "[$(date +%H:%M:%S)] all 3 sampling jobs done" >> sample_run.log
