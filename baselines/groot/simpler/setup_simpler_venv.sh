#!/usr/bin/env bash
# SimplerEnv evaluation island (python 3.10 uv venv) for GR00T N1.7 SimplerEnv-Bridge, mirroring
# src/Isaac-GR00T/gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh without the GPU sanity check (login node has no GPU);
# the check runs separately on a GPU node (simpler_check.sbatch).
set -euxo pipefail
PROJECT_REPO=/projects/yanlongvla/src/Isaac-GR00T
SIMPLER_REPO="$PROJECT_REPO/external_dependencies/SimplerEnv"
UV_ENV="$PROJECT_REPO/gr00t/eval/sim/SimplerEnv/simpler_uv"
export UV_CACHE_DIR=/projects/yanlongvla/cache/uv
SIMPLER_NUMPY="${SIMPLER_NUMPY:-1.26.4}"
rm -rf "$UV_ENV"; mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install "setuptools<71"
BUILD_CONSTRAINTS="$UV_ENV/build-constraints.txt"
printf 'scikit-build-core<0.10\n' > "$BUILD_CONSTRAINTS"
uv pip install gymnasium==0.29.1 "json-numpy>=2.1.1" numpy=="$SIMPLER_NUMPY" opencv-python-headless==4.10.0.84 ray==2.48.0
uv pip install --build-constraint "$BUILD_CONSTRAINTS" -e "$SIMPLER_REPO/ManiSkill2_real2sim"
uv pip install --build-constraint "$BUILD_CONSTRAINTS" -e "$SIMPLER_REPO"
python -c "import sysconfig, pathlib; pathlib.Path(sysconfig.get_path('purelib'), 'gr00t.pth').write_text(pathlib.Path('$PROJECT_REPO').resolve().as_posix() + '\n')"
uv pip install tianshou==0.5.1 pydantic av zmq torchvision==0.22.0 transformers==4.57.3 tyro setuptools==80.9.0 pandas==2.2.3 dm-tree==0.1.9 einops==0.8.1 albumentations==1.4.18 diffusers==0.35.1 scipy==1.15.3
python -c "import simpler_env, mani_skill2_real2sim, gymnasium; print('imports OK')"
echo "SimplerEnv venv ready at: $UV_ENV/.venv/bin/python3"
