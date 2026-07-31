#!/bin/bash
# EXP6 retrospective — launch 6 REST scoring servers on a100-b.
# 3 targets × 2 scorers (film, dabs) -> 6 servers on ports 8088-8093.
#
# Run this ON a100-b. Each server is daemonised and logs to ~/exp6_rest_<port>.log

set -uo pipefail
PROJ="$HOME/edit-small-mol"
SRC_ROOT="$PROJ"
SERVER="$PROJ/experiments/exp6_rest_server.py"
PY="$HOME/miniconda3/envs/quris/bin/python"

declare -a ROWS=(
  "egfr_t790m film 8088"
  "btk        film 8089"
  "kras_g12c  film 8090"
  "egfr_t790m dabs 8091"
  "btk        dabs 8092"
  "kras_g12c  dabs 8093"
)

for row in "${ROWS[@]}"; do
  read -r tgt scr port <<<"$row"
  log="$HOME/exp6_rest_${tgt}_${scr}.log"
  if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "[boot] port $port already up (target=$tgt scorer=$scr) — skipping"
    continue
  fi
  echo "[boot] launching target=$tgt scorer=$scr port=$port"
  setsid nohup "$PY" "$SERVER" \
    --target_dir "$PROJ/data/exp6_retrospective/$tgt" \
    --src_root "$SRC_ROOT" \
    --scorer "$scr" \
    --port "$port" \
    > "$log" 2>&1 < /dev/null &
  disown 2>/dev/null || true
done

sleep 8
echo "--- health checks ---"
for port in 8088 8089 8090 8091 8092 8093; do
  code=$(curl -sf -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health")
  body=$(curl -sf "http://127.0.0.1:${port}/health" | head -c 300)
  echo "port=$port http=$code body=$body"
done
