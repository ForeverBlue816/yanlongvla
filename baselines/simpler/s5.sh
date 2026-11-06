# Task 5 steps (GR00T N1.7 on SimplerEnv Fractal). Sourced by groot/jobs/groot.pbs, which sets $G, $SRC, $PY_G, $HANDOVER and exports HF_TOKEN.
# Task names: s5__ckpt, s5__env, s5__data, s5__fp_<family>   (families: simpler/plan_task5.py)
ISAAC=$HANDOVER/runs/toolkits/Isaac-GR00T
S5=$G/s5
SIMPLER_PY=$ISAAC/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python

t_s5() {   # t_s5 <step>
  local step=$1
  mkdir -p "$S5"
  case "$step" in
    ckpt)   # official fine-tuned Fractal checkpoint without the DeepSpeed optimizer shards (about 28 GB, not needed for inference; decision G12)
      "$PY_G" - "$S5/checkpoints/fractal" <<'PY' || return 1
import sys, os
from huggingface_hub import snapshot_download
p = snapshot_download('nvidia/GR00T-N1.7-SimplerEnv-Fractal', local_dir=sys.argv[1], ignore_patterns=['global_step*'], token=os.environ['HF_TOKEN'])
print('checkpoint in', p)
PY
      du -sh "$S5/checkpoints/fractal" ;;
    env)    # one-time SimplerEnv environment of the Isaac-GR00T repo (python 3.10 island with ManiSkill2_real2sim), then its own import check
      # the official setup does not install msgpack-numpy, which the rollout client (gr00t/policy/server_client.py) imports (FP runs crashed on 2 Nov)
      cd "$ISAAC" && bash gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh \
        && VIRTUAL_ENV=$ISAAC/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv ~/.local/bin/uv pip install msgpack-numpy imageio-ffmpeg \
        && mkdir -p "$HANDOVER/runs/toolkits/bin" \
        && ln -sf "$("$SIMPLER_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$HANDOVER/runs/toolkits/bin/ffmpeg" \
        && touch "$S5/env_ok" ;;
    data)   # the calibration and held-out episodes of the manifests (data/fractal_*_manifest.json) from the fine-tuning dataset
      "$PY_G" - "$S5/data" "$SRC/data" <<'PY' || return 1
import sys, os, json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
out, man = Path(sys.argv[1]), Path(sys.argv[2])
repo, tok = 'IPEC-COMMUNITY/fractal20220817_data_lerobot', os.environ['HF_TOKEN']
(out / 'revision.txt').write_text(HfApi(token=tok).dataset_info(repo).sha)
n = 0
for name in ('calibration', 'heldout'):
    for e in json.loads((man / f'fractal_{name}_manifest.json').read_text())['episodes']:
        i = e['episode_index']; c = i // 1000
        for f in (f'data/chunk-{c:03d}/episode_{i:06d}.parquet', f'videos/chunk-{c:03d}/observation.images.image/episode_{i:06d}.mp4'):
            hf_hub_download(repo, f, repo_type='dataset', token=tok, local_dir=out); n += 1
print('downloaded', n, 'files')
(out / 'done').write_text('ok')
PY
      ;;
    fp_*)   # FP policy on one Google Robot task: official server + rollout client, 100 episodes, fixed seed 7 (log keeps the success rate)
      local fam=${step#fp_} env_name port log; env_name=simpler_env_google/google_robot_$fam
      port=$((20000 + ${PBS_JOBID%%.*} % 20000)); log=$S5/eval/fp/$fam.log; mkdir -p "$S5/eval/fp"
      [[ -x $SIMPLER_PY ]] || { echo "SimplerEnv environment missing (s5__env)" >&2; return 1; }
      ( cd "$ISAAC" && exec "$PY_G" gr00t/eval/run_gr00t_server.py --model-path "$S5/checkpoints/fractal" --embodiment-tag SIMPLER_ENV_GOOGLE \
          --use-sim-policy-wrapper --port "$port" > "$S5/eval/fp/$fam.server.log" 2>&1 ) &
      local spid=$! t=0
      until grep -q -i "ready\|listening\|serving" "$S5/eval/fp/$fam.server.log" 2>/dev/null; do
        kill -0 $spid 2>/dev/null || { echo "server died" >&2; return 1; }
        sleep 10; t=$((t + 10)); [[ $t -gt 900 ]] && { kill $spid; echo "server start timeout" >&2; return 1; }
      done
      # the rollout always records videos (default /tmp) and calls `ffmpeg`, which the nodes lack (crash of 4 Nov): use the static
      # ffmpeg of the venv's imageio-ffmpeg (symlink in runs/toolkits/bin) and keep the videos on scratch for Task 6
      mkdir -p "$S5/eval/fp/videos/$fam"
      ( cd "$ISAAC" && PATH=$HANDOVER/runs/toolkits/bin:$PATH "$SIMPLER_PY" gr00t/eval/rollout_policy.py --n-episodes 100 --policy-client-host 127.0.0.1 --policy-client-port "$port" \
          --max-episode-steps 300 --env-name "$env_name" --n-action-steps 1 --n-envs 5 --seed 7 --video-dir "$S5/eval/fp/videos/$fam" ) > "$log" 2>&1
      local rc=$?; kill $spid 2>/dev/null; wait $spid 2>/dev/null; return $rc ;;
    fprr_*|fpseed*_*)   # FP again for the open_drawer / close_drawer gap to the model card: fprr_<fam> = same seed 7 (noise of the run), fpseed<N>_<fam> = seed N
      local tag=${step%%_*} fam2 sd=7; fam2=${step#*_}; [[ $tag == fpseed* ]] && sd=${tag#fpseed}
      s5_run "$S5/checkpoints/fractal" "$fam2" "$S5/eval/fp_$tag" "$sd" ;;
    *) echo "unknown s5 step $step" >&2; return 2 ;;
  esac
}

s5_run() {   # s5_run <model dir> <family> <out dir> [seed, default 7]: server + rollout client for one Google Robot family, 100 episodes, seed 7 (as fp_*); the log keeps the success list
  local model=$1 fam=$2 out=$3 seed=${4:-7} env_name port log spid t=0 rc; env_name=simpler_env_google/google_robot_$fam
  port=$((20000 + ${PBS_JOBID%%.*} % 20000)); log=$out/$fam.log; mkdir -p "$out/videos/$fam"
  [[ -x $SIMPLER_PY ]] || { echo "SimplerEnv environment missing (s5__env)" >&2; return 1; }
  ( cd "$ISAAC" && exec "$PY_G" gr00t/eval/run_gr00t_server.py --model-path "$model" --embodiment-tag SIMPLER_ENV_GOOGLE \
      --use-sim-policy-wrapper --port "$port" > "$out/$fam.server.log" 2>&1 ) &
  spid=$!
  until grep -q -i "ready\|listening\|serving" "$out/$fam.server.log" 2>/dev/null; do
    kill -0 $spid 2>/dev/null || { echo "server died" >&2; return 1; }
    sleep 10; t=$((t + 10)); [[ $t -gt 900 ]] && { kill $spid; echo "server start timeout" >&2; return 1; }
  done
  ( cd "$ISAAC" && PATH=$HANDOVER/runs/toolkits/bin:$PATH "$SIMPLER_PY" gr00t/eval/rollout_policy.py --n-episodes 100 --policy-client-host 127.0.0.1 --policy-client-port "$port" \
      --max-episode-steps 300 --env-name "$env_name" --n-action-steps 1 --n-envs 5 --seed "$seed" --video-dir "$out/videos/$fam" ) > "$log" 2>&1
  rc=$?; kill $spid 2>/dev/null; wait $spid 2>/dev/null
  grep -q "^success rate" "$log" || return 1
  return $rc
}
