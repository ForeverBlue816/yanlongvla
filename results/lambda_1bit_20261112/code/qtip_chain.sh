#!/usr/bin/env bash
# Chamadol's Bridge QTIP baseline (his harness, standard calibration, settings unchanged), built on Lambda and evaluated with our
# SimplerEnv harness (lambda/bridge_eval.sh: seed 7, 7 tasks x 120 episodes) so it pairs with our H100 FP episode by episode.
set -euo pipefail
B=/projects/yanlongvla/runs/bridge_baselines; G=${1:-0}
PY=/projects/yanlongvla/envs/groot/bin/python; H=/projects/yanlongvla/public-reports/baselines/groot
export GROOT_PROFILE=bridge PYTHONPATH=/projects/yanlongvla/src/qtip:/projects/yanlongvla/src/Isaac-GR00T TMPDIR=/projects/yanlongvla/tmp CUDA_VISIBLE_DEVICES=$G
export HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HOME=/projects/yanlongvla/cache/huggingface
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
cd $H/qtip
say "QTIP fits (backbone, head in parallel)"
for key in backbone head_bridge; do
  [ -f $B/q/qtip_w2/$key/accounting.json ] || $PY -u $B/run_with_stubs.py quantize_qtip.py --layers $B/layers/$key --calibration $B/calibration/$key --checkpoint $B/checkpoints/bridge \
      --qtip-repo /projects/yanlongvla/src/qtip --out $B/q/qtip_w2/$key --bits 2 > $B/qtip_$key.log 2>&1 &
done; wait
ls $B/q/qtip_w2/backbone/accounting.json $B/q/qtip_w2/head_bridge/accounting.json
say "import"
cd $H; [ -f $B/ckpt/qtip_w2/bridge/model.safetensors ] || $PY -u import_external.py --checkpoints $B/checkpoints --quantized $B/q/qtip_w2 --out $B/ckpt/qtip_w2 > $B/import_qtip.log 2>&1
say "evaluation (our harness)"
cd /projects/yanlongvla/research
bash lambda/bridge_eval.sh qtip_w2_chamadol $B/ckpt/qtip_w2/bridge $G 5671 120 episodes_a widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube > $B/eval_qtip_a.log 2>&1 &
bash lambda/bridge_eval.sh qtip_w2_chamadol $B/ckpt/qtip_w2/bridge $G 5672 120 episodes_b widowx_put_eggplant_in_basket widowx_put_eggplant_in_sink widowx_open_drawer widowx_close_drawer > $B/eval_qtip_b.log 2>&1 &
wait; say "qtip row done"
