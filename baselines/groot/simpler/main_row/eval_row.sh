#!/usr/bin/env bash
# SimplerEnv-Bridge evaluation of one model directory: seeded server (GR00T env) + client (SimplerEnv venv), resumable per episode.
#   bash eval_row.sh <row name> <model dir> <gpu> <port> [episodes per task=120] [out name=episodes] [task ...]
# Output: $RUNS/simpler_bridge/eval/<row name>/<out name>.json (+ gpu.txt, server_<out name>.log). Same protocol and seeds as
# ../eval_simpler_bridge.py (7 tasks x 120 episodes, seed 7, 300 env steps, 4 of 8 actions per call).
set -euo pipefail
ROW=$1; MODEL=$2; GPU=$3; PORT=$4; EPISODES=${5:-120}; NAME=${6:-episodes}; shift $(( $# < 6 ? $# : 6 ))
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
export CUDA_VISIBLE_DEVICES=$GPU
OUT=$RUNS/simpler_bridge/eval/$ROW; mkdir -p "$OUT"
cd "$ISAAC_GROOT"
$GROOT_PY -u "$MAIN_ROW_SCRIPTS/simpler_server.py" --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 > "$OUT/server_$NAME.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do grep -q "serving" "$OUT/server_$NAME.log" 2>/dev/null && break; kill -0 $SERVER || { echo "server exited"; exit 1; }; sleep 5; done; sleep 10
nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" > "$OUT/gpu.txt"
$SIM_PY -u "$MAIN_ROW_SCRIPTS/simpler_eval.py" --episodes "$EPISODES" --seed 7 --port "$PORT" --out "$OUT/$NAME.json" ${1:+--tasks "$@"}
echo "bridge eval $ROW $NAME done"
