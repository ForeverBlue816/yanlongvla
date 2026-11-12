#!/usr/bin/env bash
# Site settings for the Bridge main row hand-over. Edit the first block; everything else follows from it.
export VLA_ROOT="${VLA_ROOT:-/path/to/workdir}"                    # writable; runs/ and models/ are created under it (about 60 GB)
export ISAAC_GROOT="${ISAAC_GROOT:-$VLA_ROOT/src/Isaac-GR00T}"      # checkout d2b7e75b937e3ec9aa5dbc798f08b89692c49734, submodules initialised recursively
export GROOT_PY="${GROOT_PY:-$VLA_ROOT/envs/groot/bin/python}"      # the GR00T environment of ../../README.md
export SIM_PY="${SIM_PY:-$ISAAC_GROOT/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python}"   # the SimplerEnv venv of ../README.md
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLA_ROOT/cache/huggingface/hub}"   # must hold the Bridge snapshot as a hub cache (blob symlinks), see README
export OPENPI_CLIENT_MSGPACK="${OPENPI_CLIENT_MSGPACK:-$VLA_ROOT/src/openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
: "${HF_TOKEN:?export HF_TOKEN (read access to nvidia/Cosmos-Reason2-2B)}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAIN_ROW_SCRIPTS="$HERE/scripts"
export PYTHONPATH="$ISAAC_GROOT:$MAIN_ROW_SCRIPTS" PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export TMPDIR="${TMPDIR:-$VLA_ROOT/tmp}"; mkdir -p "$TMPDIR"
unset TCQ_BITS                                                       # 2 bits (the default); any other value writes to a different tree
export RUNS="$VLA_ROOT/runs/plustrain_20261008/groot" MODELS_PTQ="$VLA_ROOT/models/groot_ptq" LOGS="$VLA_ROOT/logs/simpler"
