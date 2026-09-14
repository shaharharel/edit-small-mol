#!/bin/bash
# Launch the Phase B from_a arms from Phase A RUN 3's ep2, the matched pair #146 requires.
#
# WAITS ON THE ARTIFACT, NOT ON A PID. `pgrep -f train_phaseA` matches this watcher's own argv
# (the pattern appears in the command line that greps for it), so a pgrep-based wait can see a
# process that is only itself and either spin forever or fire instantly. Polling for the ckpt is
# immune to that.
#
# AND IT WAITS FOR THE WRITE TO FINISH. torch.save on an 80MB checkpoint is not atomic: the file
# exists, and is truncated, for some seconds. Loading it mid-write gives a corrupt-read crash at
# best and a silently partial state_dict at worst. Require the size to match ep1's and to hold
# steady across two polls before touching it.
set -u
CF=/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer
PY=/opt/miniconda3/envs/quris/bin/python
EP2=$CF/ckpt_A_role_strat_s3/ep2.ckpt
REF=$CF/ckpt_A_role_strat_s3/ep1.ckpt
DEADLINE=$(( $(date +%s) + 3600 ))

REFSZ=$(stat -f%z "$REF")
echo "$(date '+%H:%M:%S') waiting for $EP2 (expect ~$REFSZ bytes, ref=ep1)"
while true; do
  if [ -f "$EP2" ]; then
    s1=$(stat -f%z "$EP2"); sleep 20; s2=$(stat -f%z "$EP2")
    if [ "$s1" = "$s2" ] && [ "$s1" -ge "$REFSZ" ]; then
      echo "$(date '+%H:%M:%S') ep2 complete and stable at $s2 bytes"; break
    fi
    echo "$(date '+%H:%M:%S') ep2 present but still growing ($s1 -> $s2), waiting"
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "$(date '+%H:%M:%S') GIVING UP: ep2 did not appear within 60 min. Run 3 has stalled or died."
    echo "NOT launching the from_a arms against ep0/ep1 -- runs 1 and 2 supplied ep2 inits and"
    echo "matching that is the entire point of a run-level replication."
    exit 1
  fi
  sleep 60
done

# Verify the init BEFORE spending 25 minutes on it: it must load, and it must carry the roles
# stamp (run 3 is the first checkpoint that does, so this is the first time the guard is reachable).
$PY - <<PYEOF || { echo "INIT REJECTED -- not launching"; exit 1; }
import torch, sys
sys.path.insert(0, '$CF')
from train_phaseA import ROLES
ck = torch.load('$EP2', map_location='cpu', weights_only=False)
r = ck.get('roles')
print('  init epoch=%s mode=%s device=%s roles=%s' % (ck.get('epoch'), ck.get('mode'), ck.get('device'), r))
if r is None:
    raise SystemExit('FATAL: run 3 ep2 carries no roles stamp -- it was supposed to be the first that does')
if list(r) != list(ROLES):
    raise SystemExit('FATAL role-order mismatch: %s vs %s' % (list(r), list(ROLES)))
print('  roles VERIFIED, init accepted')
PYEOF

# Matched to the B4c scratch arms, whose stamps read: arm=scratch mode=geom cond_dim=3
# gzscore=False data=phaseB.csv, 3 epochs, seeds 404/505/606. Everything except --arm/--init/--out
# must be identical or the contrast is not a contrast.
cd /Users/shaharharel/Documents/github/edit-small-mol
for S in 404 505 606; do
  echo "$(date '+%H:%M:%S') launching from_a seed $S"
  $PY -u experiments/covalentformer/train_phaseB.py \
    --data experiments/covalentformer/data/phaseB/phaseB.csv \
    --out experiments/covalentformer/ckpt_B4c_from_a_s$S \
    --arm from_a --init "$EP2" --seed $S --epochs 3 \
    > experiments/covalentformer/logs_B4c_from_a_s$S.log 2>&1 &
done
wait
echo "$(date '+%H:%M:%S') all three from_a arms finished"
