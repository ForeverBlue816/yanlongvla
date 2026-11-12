#!/usr/bin/env bash
# On-policy states for H_D: 256 SimplerEnv rollouts of the clean-H 2-bit row tcq_w2_rht (simpler_onpolicy_client.py).
#   bash onpolicy.sh <model dir of tcq_w2_rht> <gpu> <port> [out dir] [count=256]
set -euo pipefail
MODEL=$1; GPU=$2; PORT=$3
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
OUT=${4:-$RUNS/simpler_bridge/onpolicy_rinit}; COUNT=${5:-256}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT"; cd "$ISAAC_GROOT"
$GROOT_PY -u "$MAIN_ROW_SCRIPTS/simpler_server.py" --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 > "$OUT/server.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do grep -q "serving" "$OUT/server.log" 2>/dev/null && break; kill -0 $SERVER || { echo "server exited"; exit 1; }; sleep 5; done; sleep 10
$SIM_PY -u "$MAIN_ROW_SCRIPTS/simpler_onpolicy_client.py" --port "$PORT" --out "$OUT" --count "$COUNT"
