#!/usr/bin/env bash
# One-command benchmark: every row x {10, 2} denoising steps on the current GPU -> deploy/results.csv, deploy/table.md (+ per-row JSON).
#   bash deploy/tcq_kernels/run_bench.sh            (inside a GPU job; ~25 min)
#   ROWS="fp w2a8_resident" bash ...                subset of rows
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); R=${VLA_ROOT:-/projects/yanlongvla}
[ -f $R/research/scripts/env.sh ] && source $R/research/scripts/env.sh
export JAX_PLATFORMS=cpu PLUSTRAIN_ANY_GPU=1          # the policy's JAX-side transforms stay on the CPU
PY=${PYTHON:-$R/envs/openpi/bin/python}
export PYTHONPATH=$R/research/scripts:$HERE${PYTHONPATH:+:$PYTHONPATH}
OUT=${OUT:-$HERE/../results}; mkdir -p "$OUT"
cd "$HERE"
[ -f $R/models/pi05_tcq/tcq_raqft_d1/packed.safetensors ] || $PY tcq_format.py            # fit records -> packed 2-bit file (checked layer by layer)
[ -f $R/models/pi05_tcq/tcq_raqft_d1/deploy_size.json ] || $PY export_rest.py             # everything outside the Linear payload + sizes on disk
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
$PY bench.py --all --out "$OUT" ${ROWS:+--rows $ROWS}
