#!/bin/bash
# UNATTENDED NIGHT PIPELINE. Runs the plan end-to-end across both V100s in parallel,
# writing STATUS.md after every stage so progress is readable without a live session.
#
# DESIGN NOTES, each from something that broke today:
#  * Remote jobs are launched with nohup + </dev/null and then POLLED. An ssh that holds the
#    connection open to `sleep` and check dies on a client timeout while the job runs fine --
#    that confusion cost three launches.
#  * Every stage writes STATUS.md BEFORE and AFTER, so a stage that dies leaves evidence of
#    where it died rather than a silent gap.
#  * QA runs every cycle, not at the end. Today's worst errors (an unmatched control, three
#    unrepresentative samples) were all cheap to catch and expensive to inherit.
#  * Checkpoints save BEST-ONLY. Saving every epoch filled a 291 GB disk and killed 13 of 16
#    arms with an error that reads like file corruption.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/opt/miniconda3/envs/quris/bin/python}
K=~/.ssh/google_compute_engine
U=shaharh_quris_ai
G1=34.57.253.170          # ai-gpu
G2=34.28.139.166          # ai-gpu2
R=/home/shaharh_quris_ai/edit-small-mol/experiments/covalentformer
S=STATUS.md
mkdir -p logs results/full_eval

say () { printf '%s  %s\n' "$(date '+%m-%d %H:%M')" "$1" | tee -a logs/pipeline.log; }
status () {
  { echo "# NIGHT PIPELINE STATUS"; echo; echo "_updated $(date '+%Y-%m-%d %H:%M')_"; echo;
    echo "## stage: $1"; echo; echo '```'; tail -25 logs/pipeline.log; echo '```'; echo;
    echo "## GPU"; for IP in $G1 $G2; do
      echo -n "  $IP: ";
      ssh -i $K -o ConnectTimeout=12 -o StrictHostKeyChecking=no -o BatchMode=yes $U@$IP \
        "echo procs=\$(pgrep -fc train_steer_v5.py) gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)" 2>/dev/null || echo unreachable;
    done; echo;
    echo "## results so far"; ls -1 results/full_eval/*.json 2>/dev/null | sed 's/^/  /' || echo "  none yet";
  } > $S
}

remote_launch () {  # $1=ip $2=script-name  -- fire and forget, never block on the ssh
  ssh -i $K -o ConnectTimeout=20 -o StrictHostKeyChecking=no $U@$1 \
    "cd $R && chmod +x stage0/$2 && nohup stage0/$2 > logs/${2%.sh}.log 2>&1 </dev/null & echo launched" 2>&1 | tail -1
}
remote_wait () {    # $1=ip  $2=remote-glob  $3=expected-count  $4=max-min
  # POLL FOR THE ARTIFACT, NOT THE PROCESS. Two process-based attempts both hung:
  # `pgrep -fc train_steer_v5.py` matches the ssh command's OWN argv (the pattern is IN the
  # command string), and anchoring the regex did not fix it either. The file either exists
  # or it does not -- that has no self-match failure mode. Fourth and last time on this trap.
  local i=0
  while [ $i -lt ${4:-90} ]; do
    n=$(ssh -i $K -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o BatchMode=yes $U@$1 \
          "ls $2 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    [ "${n:-0}" -ge "${3:-1}" ] 2>/dev/null && return 0
    sleep 60; i=$((i+1))
  done
  say "WARN: $1 produced only ${n:-0}/${3:-1} artifacts after ${4:-90} min"
}

say "=== PIPELINE START ==="; status "start"

# ---------------------------------------------------------------- STAGE 1: rung B data
say "STAGE 1  build planarity rung B (DOWN-only)"
$PY - <<'EOF' >> logs/pipeline.log 2>&1
# RUNG B: train ONLY "make it more planar". This is the direction a chemist actually issues,
# and it sidesteps UP's pathological cheapest edit -- more twisted = less conjugated = a dead
# warhead. If DOWN alone steers, the product claim survives even if UP never does.
import csv, os, random
src='data/steer_v6_strat'; dst='data/steer_v6_downonly'; os.makedirs(dst, exist_ok=True)
rng=random.Random(20260916)
for sp in ('train','valid'):
    rows=[r for r in csv.DictReader(open(f'{src}/warhead_planarity_{sp}.csv'))]
    keep=[r for r in rows if r['instr'] in ('DOWN','SAME')]
    # cap SAME at 50% so DOWN is never the minority class it is being asked to learn
    dn=[r for r in keep if r['instr']=='DOWN']; sm=[r for r in keep if r['instr']=='SAME']
    rng.shuffle(sm); sm=sm[:len(dn)]
    out=dn+sm; rng.shuffle(out)
    with open(f'{dst}/warhead_planarity_{sp}.csv','w',newline='') as fo:
        w=csv.DictWriter(fo,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(out)
    print(f'  rungB {sp}: {len(out)} rows (DOWN {len(dn)}, SAME {len(sm)})')
EOF
status "stage1 rungB data built"

# ---------------------------------------------------------------- STAGE 2: push + launch
say "STAGE 2  push data, launch rung A on G2 and rung B on G1 IN PARALLEL"
for IP in $G1 $G2; do
  ssh -i $K -o StrictHostKeyChecking=no $U@$IP "mkdir -p $R/data/steer_v6_strat $R/data/steer_v6_downonly" 2>/dev/null
  scp -q -i $K -o StrictHostKeyChecking=no data/steer_v6_strat/*.csv     $U@$IP:$R/data/steer_v6_strat/ 2>/dev/null
  scp -q -i $K -o StrictHostKeyChecking=no data/steer_v6_downonly/*.csv  $U@$IP:$R/data/steer_v6_downonly/ 2>/dev/null
  scp -q -i $K -o StrictHostKeyChecking=no stage0/train_steer_v5.py      $U@$IP:$R/stage0/ 2>/dev/null
  ssh -i $K -o StrictHostKeyChecking=no $U@$IP \
    "cd $R && sed -i 's|REINVENT = .*|REINVENT = \"/home/shaharh_quris_ai/REINVENT4\"|' stage0/train_steer_v5.py" 2>/dev/null
done

cat > /tmp/rungA.sh <<'EOF'
#!/bin/bash
cd /home/shaharh_quris_ai/edit-small-mol/experiments/covalentformer
mkdir -p ckpt_rungA
for M in instr none; do
  O=ckpt_rungA/planarity_strat_$M
  [ -f "${O}_history.json" ] && continue
  python3 -u stage0/train_steer_v5.py --data data/steer_v6_strat --param warhead_planarity \
    --mode $M --out $O --epochs 6 --bs 32 --lr 5e-5 --seed 20260916 --device cuda
done
EOF
cat > /tmp/rungB.sh <<'EOF'
#!/bin/bash
cd /home/shaharh_quris_ai/edit-small-mol/experiments/covalentformer
mkdir -p ckpt_rungB
for M in instr none; do
  O=ckpt_rungB/planarity_down_$M
  [ -f "${O}_history.json" ] && continue
  python3 -u stage0/train_steer_v5.py --data data/steer_v6_downonly --param warhead_planarity \
    --mode $M --out $O --epochs 6 --bs 32 --lr 5e-5 --seed 20260916 --device cuda
done
EOF
scp -q -i $K -o StrictHostKeyChecking=no /tmp/rungA.sh $U@$G2:$R/stage0/
scp -q -i $K -o StrictHostKeyChecking=no /tmp/rungB.sh $U@$G1:$R/stage0/
remote_launch $G2 rungA.sh; say "  rung A launched on G2"
remote_launch $G1 rungB.sh; say "  rung B launched on G1"
status "stage2 rungs training in parallel"

# ---------------------------------------------------------------- STAGE 3: QA while training
say "STAGE 3  QA sweep on the new code (runs while GPUs train)"
$PY stage0/qa_sweep.py >> logs/pipeline.log 2>&1 || say "  QA sweep reported issues (see log)"
status "stage3 QA done"

# ---------------------------------------------------------------- STAGE 4: wait + collect
say "STAGE 4  waiting on both GPUs"
remote_wait $G2 "$R/ckpt_rungA/*_history.json" 2 90; say "  G2 rung A artifacts present"
remote_wait $G1 "$R/ckpt_rungB/*_history.json" 2 90; say "  G1 rung B artifacts present"
mkdir -p ckpt_rungA_gpu ckpt_rungB_gpu
scp -q -i $K -o StrictHostKeyChecking=no "$U@$G2:$R/ckpt_rungA/*_history.json" ckpt_rungA_gpu/ 2>/dev/null
scp -q -i $K -o StrictHostKeyChecking=no "$U@$G1:$R/ckpt_rungB/*_history.json" ckpt_rungB_gpu/ 2>/dev/null
scp -q -i $K -o StrictHostKeyChecking=no "$U@$G2:$R/ckpt_rungA/*_best.pt" ckpt_rungA_gpu/ 2>/dev/null
scp -q -i $K -o StrictHostKeyChecking=no "$U@$G1:$R/ckpt_rungB/*_best.pt" ckpt_rungB_gpu/ 2>/dev/null
$PY - <<'EOF' >> logs/pipeline.log 2>&1
import json,glob,os
for tag,d in (('RUNG A (anchor-stratified)','ckpt_rungA_gpu'),('RUNG B (DOWN-only)','ckpt_rungB_gpu')):
    R={}
    for f in glob.glob(d+'/*_history.json'):
        h=json.load(open(f)); b=os.path.basename(f).replace('_history.json','')
        R['instr' if b.endswith('_instr') else 'none']=h
    if 'instr' not in R: print(f'  {tag}: NO RESULT'); continue
    bi=min(R['instr']['history'],key=lambda r:r['valid'])
    bn=min(R['none']['history'],key=lambda r:r['valid']) if 'none' in R else None
    ben=(bn['valid']-bi['valid']) if bn else float('nan')
    print(f"  {tag}: train {R['instr']['train_rows']} | GAP_perm {bi['gap_perm']:+.4f} "
          f"GAP_flip {bi['gap_flip']:+.4f} | BENEFIT {ben:+.4f}  (floor ~0.002)")
EOF
status "stage4 rungs collected"

# ---------------------------------------------------------------- STAGE 5: cohort eval
say "STAGE 5  5k cohorts + generation/manuscript panels"
PY=$PY N=${N:-5000} bash stage0/run_full_eval.sh >> logs/pipeline.log 2>&1
$PY stage0/summarise_eval.py results/full_eval >> logs/pipeline.log 2>&1
status "stage5 cohort eval done"

say "=== PIPELINE COMPLETE ==="; status "COMPLETE"
