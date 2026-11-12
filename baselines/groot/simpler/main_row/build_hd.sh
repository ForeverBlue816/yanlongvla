#!/usr/bin/env bash
# The Hessians H_D of our main row, on your GPU, from the FP checkpoint and our 256 on-policy states. About 10 minutes, 33 GB GPU memory,
# 70 GB under $VLA_ROOT (ten Hessian sets; the eight that are not *_hd1_* can be deleted afterwards).
#   bash build_hd.sh [gpu=0]
# Steps: FP model directory, smoke, clean second moments (256 calibration observations), per-source second moments (photometric,
# geometric, on-policy), H_D. Needs data/onpolicy_rinit.tar unpacked (the script does it when the states are missing).
set -euo pipefail
G=${1:-0}
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HES=$RUNS/ptq/hessians; SB=$RUNS/simpler_bridge; mkdir -p "$LOGS" "$SB"; cd "$HERE"
export CUDA_VISIBLE_DEVICES=$G
[ -f $SB/onpolicy_rinit/complete.json ] || tar -xf data/onpolicy_rinit.tar -C $SB
[ -f "$VLA_ROOT/models/groot_n17_simpler_bridge/fp/bridge_manifest.json" ] || $GROOT_PY -u scripts/groot_simpler.py --stage prepare > $LOGS/prepare.log 2>&1
[ -f $SB/smoke.json ] || $GROOT_PY -u scripts/groot_simpler.py --stage smoke > $LOGS/smoke.log 2>&1
{ [ -f $HES/backbone_simpler_bridge/complete.json ] && [ -f $HES/head_simpler_bridge/complete.json ]; } || $GROOT_PY -u scripts/groot_simpler.py --stage calib > $LOGS/calib.log 2>&1
$GROOT_PY -u scripts/groot_simpler_main.py --stage hsrc >> $LOGS/hsrc.log 2>&1
$GROOT_PY -u scripts/groot_simpler_main.py --stage hd > $LOGS/hd.log 2>&1
cat $HES/backbone_hd1_simpler_bridge/complete.json $HES/head_hd1_simpler_bridge/complete.json
echo "H_D ready: $HES/{backbone,head}_hd1_simpler_bridge  ->  python hd_for_baselines.py --hd $HES --like <your clean calibration> --out <calibration_hd>"
