#!/bin/bash
# The LIBERO-Plus client interpreter for run_plus.py --plus-python: the π0.5 harness's simulator environment (openpi venv plus
# the simulator packages), exactly as jobs/evaluate.pbs sets it up. Needs SRC (the baselines source snapshot) in the environment.
set -euo pipefail
: "${SRC:?}"
OPENPI_ROOT=/home/d250009/projects/openpi
HANDOVER=/home/d250009/projects/pi05-handover
SIM_SITE=/scratch/d250009/pi05-handover/runs/envs/sim-site   # robosuite 1.4.0 and the simulator packages, pinned to the versions of the old
# openvla .venv-bf16-stage (pip --no-deps, 7 Nov). That venv was removed from scratch on 6 Nov and the home copy has only part of its files.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1   # as jobs/evaluate.pbs: LIBERO-Plus loads its init states with torch.load
export LIBERO_CONFIG_PATH="$HANDOVER/runs/config/libero-smoke/plus"
export PYTHONPATH="$HANDOVER/runs/envs/libero-plus/site-packages:$SRC:$HANDOVER/runs/toolkits/LIBERO-plus:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/.venv/lib64/python3.11/site-packages:$SIM_SITE"
exec "$OPENPI_ROOT/.venv/bin/python" "$@"
