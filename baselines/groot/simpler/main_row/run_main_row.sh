#!/usr/bin/env bash
# GR00T N1.7 SimplerEnv-Bridge, 2-bit main row (tcq_raqft_d1_s) from the official checkpoint to the evaluation. No scheduler needed;
# every step is skipped when its output exists, so the script can be restarted after an interruption.
#   bash run_main_row.sh [gpu A=0] [gpu B=A]
# With one GPU the two fits of a pair run one after the other. Peak GPU memory: about 33 GB (per-source second moments), 25 GB (fine-tune).
set -euo pipefail
A=${1:-0}; B=${2:-$A}
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT=$RUNS/ptq/fit; HES=$RUNS/ptq/hessians; SB=$RUNS/simpler_bridge; mkdir -p "$LOGS"; cd "$HERE"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
base() { CUDA_VISIBLE_DEVICES=$1 $GROOT_PY -u scripts/groot_simpler.py "${@:2}"; }
main() { CUDA_VISIBLE_DEVICES=$1 $GROOT_PY -u scripts/groot_simpler_main.py "${@:2}"; }
pair() {   # two commands: in parallel on two GPUs, in sequence on one
  if [ "$A" != "$B" ]; then eval "$1" & eval "$2" & wait; else eval "$1"; eval "$2"; fi
}

say "1 FP checkpoint, smoke, clean Hessians"
[ -f "$VLA_ROOT/models/groot_n17_simpler_bridge/fp/bridge_manifest.json" ] || base $A --stage prepare > $LOGS/prepare.log 2>&1
[ -f $SB/smoke.json ] || base $A --stage smoke > $LOGS/smoke.log 2>&1
{ [ -f $HES/backbone_simpler_bridge/complete.json ] && [ -f $HES/head_simpler_bridge/complete.json ]; } || base $A --stage calib > $LOGS/calib.log 2>&1

say "2 clean-H 2-bit row tcq_w2_rht (the policy of the on-policy capture; also fixes the per-layer damping)"
pair "[ -f $FIT/head_tcq_rht_simpler_bridge/complete.json ] || base $A --stage fit --part head > $LOGS/fit_base_head.log 2>&1" \
     "[ -f $FIT/backbone_tcq_rht_simpler_bridge/complete.json ] || base $B --stage fit --part backbone > $LOGS/fit_base_backbone.log 2>&1"
[ -f $MODELS_PTQ/tcq_w2_rht/simpler_bridge/accounting.json ] || base $A --stage assemble --row tcq_w2_rht > $LOGS/assemble_base.log 2>&1

say "3 on-policy states (256 SimplerEnv rollouts of tcq_w2_rht). To use our states instead, unpack them into $SB/onpolicy_rinit first (README)"
[ -f $SB/onpolicy_rinit/complete.json ] || bash onpolicy.sh $MODELS_PTQ/tcq_w2_rht/simpler_bridge $B 5631 > $LOGS/onpolicy.log 2>&1

say "4 per-source second moments, H_D"
main $A --stage hsrc >> $LOGS/hsrc.log 2>&1
main $A --stage hd > $LOGS/hd.log 2>&1

say "5 D1 fit on H_D (tcq_w2_rht_d1)"
pair "main $A --stage fit --part head > $LOGS/fit_d1_head.log 2>&1" "main $B --stage fit --part backbone > $LOGS/fit_d1_backbone.log 2>&1"
[ -f $FIT/head_tcq_rht_d1_simpler_bridge/complete.json ] && [ -f $FIT/backbone_tcq_rht_d1_simpler_bridge/complete.json ]
main $A --stage assemble --row tcq_w2_rht_d1 > $LOGS/assemble_d1.log 2>&1

say "6 V0' fine-tune (smoke, then 768 updates; about 16 h on an H100)"
if [ ! -f $FIT/backbone_tcq_raqft_d1_s_simpler_bridge/complete.json ]; then
  [ -f $SB/raqft/backbone_tcq_raqft_d1_s_simpler_bridge_smoke/smoke.json ] || main $A --stage finetune --smoke > $LOGS/finetune_smoke.log 2>&1
  main $A --stage finetune >> $LOGS/finetune.log 2>&1
fi
main $A --stage assemble --row tcq_raqft_d1_s > $LOGS/assemble_main.log 2>&1

say "7 held-out action MSE (informational), SimplerEnv evaluation of the main row (two halves)"
main $A --stage offline > $LOGS/offline.log 2>&1 || say "offline failed (informational)"
pair "bash eval_row.sh tcq_raqft_d1_s $MODELS_PTQ/tcq_raqft_d1_s/simpler_bridge $A 5641 120 episodes_a widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube > $LOGS/eval_main_a.log 2>&1" \
     "bash eval_row.sh tcq_raqft_d1_s $MODELS_PTQ/tcq_raqft_d1_s/simpler_bridge $B 5642 120 episodes_b widowx_put_eggplant_in_basket widowx_put_eggplant_in_sink widowx_open_drawer widowx_close_drawer > $LOGS/eval_main_b.log 2>&1"
say "main row done: $SB/eval/tcq_raqft_d1_s/episodes_{a,b}.json"
