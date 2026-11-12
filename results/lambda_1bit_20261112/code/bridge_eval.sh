#!/usr/bin/env bash
# SimplerEnv-Bridge evaluation of one model directory without Slurm (the simpler_eval.sbatch recipe).
#   bash lambda/bridge_eval.sh <row> <model dir> <gpu> <port> [episodes per task=120] [out name=episodes] [task ...]
# Server (envs/groot, simpler_server.py: seeded policy noise) + client (SimplerEnv island, simpler_eval.py); resumable per episode.
set -euo pipefail
ROW=$1; MODEL=$2; GPU=$3; PORT=$4; EPISODES=${5:-120}; NAME=${6:-episodes}; shift $(( $# < 6 ? $# : 6 ))
source /projects/yanlongvla/research/scripts/env.sh
export PYTHONPATH="$VLA_ROOT/src/Isaac-GR00T:$VLA_ROOT/research/scripts" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
SIM_PY=$VLA_ROOT/src/Isaac-GR00T/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python
OUT=$VLA_ROOT/runs/plustrain_20261008/groot/simpler_bridge/eval/$ROW; mkdir -p $OUT
cd $VLA_ROOT/src/Isaac-GR00T
$VLA_ROOT/envs/groot/bin/python -u $VLA_ROOT/research/scripts/simpler_server.py --model-path $MODEL --port $PORT --host 127.0.0.1 > $OUT/server_$NAME.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do grep -q "serving" $OUT/server_$NAME.log 2>/dev/null && break; kill -0 $SERVER || { echo "server exited"; exit 1; }; sleep 5; done; sleep 10
nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU > $OUT/gpu.txt
$SIM_PY -u $VLA_ROOT/research/scripts/simpler_eval.py --episodes $EPISODES --seed 7 --port $PORT --out $OUT/$NAME.json ${1:+--tasks "$@"}
echo "bridge eval $ROW $NAME done"
