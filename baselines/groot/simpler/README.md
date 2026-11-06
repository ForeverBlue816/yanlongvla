# GR00T N1.7 on SimplerEnv-Bridge (WidowX): FP baseline evaluation and 2-bit calibration data

Written for Chamadol. Goal: evaluate the official **FP** checkpoint `nvidia/GR00T-N1.7-SimplerEnv-Bridge` on the 7 SimplerEnv WidowX tasks
with a **seeded, per-episode** protocol, so that every 2-bit row (ours and the six PTQ baselines) can be paired with it episode by episode.
Afterwards the same calibration data and the same script serve the baselines. Everything here mirrors `../README.md` (LIBERO-Plus harness).

## 1. Checkpoint (pinned)

| Item | Value |
|---|---|
| Repo / revision | `nvidia/GR00T-N1.7-SimplerEnv-Bridge` @ `940134b3c2948ccfdf8e7393f2d2ca869dc42833` |
| Layout | flat: `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, `model.safetensors.index.json`, `config.json`, `processor_config.json`, `embodiment_id.json`, `statistics.json`, `experiment_cfg/` |
| Do not download | `global_step32000/` (DeepSpeed optimizer states, 28 GB), `rng_state_*.pth`, `scheduler.pt`, `training_args.bin` |
| Embodiment tag | `SIMPLER_ENV_WIDOWX` (projector index 1) |
| Backbone config | `config.json: model_name = nvidia/Cosmos-Reason2-2B` (gated, as for LIBERO: accept the licence, `export HF_TOKEN=...`) |
| Backbone weights | **byte-identical to the LIBERO checkpoints** (we compared every tensor: all 494 `backbone.*` tensors equal; the 537 differing tensors are all `action_head.*`). The quantizable layer set is the same 432 Linear (`groot_common.is_quantizable`): 216 backbone + 216 action head. |
| Inputs | one camera `image_0` 256×256 RGB (no flip), 8-d state `x, y, z, roll, pitch, yaw, pad (= 0), gripper`, instruction |
| Output | 8-step action chunk `x, y, z, roll, pitch, yaw, gripper ∈ [0, 1]`; the env executes **4** steps per policy call |

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('nvidia/GR00T-N1.7-SimplerEnv-Bridge', revision='940134b3c2948ccfdf8e7393f2d2ca869dc42833',
                        allow_patterns=['*.json', '*.safetensors', '*.md', 'experiment_cfg/*']))
PY
```
The snapshot directory is passed directly as `--model-path` (no preparation step; the official server reads sharded weights). For a
quantized row, write a directory with the same files and the fake-quantized weights in one `model.safetensors` (as `export_layers.py` /
`import_external.py` do for LIBERO; the accounting formula is unchanged, retained bytes are those of this checkpoint).

## 2. Environments

1. **GR00T environment** — the one from `../README.md` (Isaac-GR00T `d2b7e75b937e3ec9aa5dbc798f08b89692c49734`, torch 2.9.0+cu128,
   transformers 4.57.3, flash-attn 2.8.3). It serves the policy over ZMQ with the official `gr00t/eval/run_gr00t_server.py`.
2. **SimplerEnv island** — a separate Python 3.10 venv with SAPIEN 2.2.2 / ManiSkill2_real2sim from the fork pinned as the Isaac-GR00T
   submodule `external_dependencies/SimplerEnv` (`8a2d286c926c1371927caa7651a412b4cc331756`). Build it with the upstream script
   `gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh`, or with `setup_simpler_venv.sh` here (identical package list, without the GPU
   sanity check so it can be built on a login node; edit `PROJECT_REPO` at the top), then
   `uv pip install --python <venv>/bin/python msgpack msgpack-numpy tqdm transforms3d`.
   Runtime needs Vulkan on the GPU node: `libvulkan.so.1` and the NVIDIA ICD (`export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`;
   the repo also ships `external_dependencies/SimplerEnv/nvidia_icd.json`). Check: `python -c "import sapien; import simpler_env"`.
   The venv finds `gr00t` through a `.pth` pointing at the Isaac-GR00T checkout, so run the client from that checkout's root.

## 3. Protocol (pre-registered)

| Item | Value |
|---|---|
| Tasks (7) | `widowx_spoon_on_towel`, `widowx_carrot_on_plate`, `widowx_stack_cube`, `widowx_put_eggplant_in_basket`, `widowx_put_eggplant_in_sink`, `widowx_open_drawer`, `widowx_close_drawer` |
| Episodes | 100 per task = 700 per policy |
| Seeds | episode `k` of task index `t` (order above) is reset with `SeedSequence([7, t, k]).generate_state(1)[0] % (2^31 − 1)`; ManiSkill2 derives the object layout and robot initial pose from it, so the initial scene is identical for every policy |
| Horizon | 300 env steps (= `--max-episode-steps 300` in NVIDIA's README), 8-step chunk, **4** executed per call (`--n-action-steps 4`), one environment (no async vector env), no early termination on success |
| Success | the episode counts as a success if any step reports `success` (same rule as `gr00t/eval/rollout_policy.py`) |
| Policy noise | the server's flow-matching noise is not seeded by the client; the FP row is therefore evaluated once (as the LIBERO-Plus rows) |
| Reference numbers (NVIDIA, their protocol: 5 async envs, unseeded) | spoon_on_towel 78, carrot_on_plate 58, put_eggplant_in_basket 53, stack_cube 48, put_eggplant_in_sink 2, close_drawer 97, open_drawer 100 → 62.3 % |

## 4. Commands

Terminal 1 (GR00T environment, GPU):
```bash
cd /path/Isaac-GR00T
python gr00t/eval/run_gr00t_server.py --model-path <snapshot dir or quantized dir> --embodiment-tag SIMPLER_ENV_WIDOWX --use-sim-policy-wrapper --port 5555
```
Terminal 2 (SimplerEnv venv, same node; Vulkan variables set):
```bash
cd /path/Isaac-GR00T
<venv>/bin/python /path/baselines/groot/simpler/eval_simpler_bridge.py --episodes 100 --seed 7 --port 5555 --out results/simpler_bridge/fp/episodes.json
```
`eval_simpler_bridge.py` is resumable (episodes already in `--out` are skipped), prints one JSON line per episode and a per-task summary
at the end. `--tasks` restricts the task list, `--video-dir` records videos. Expect roughly 1–2 s per policy call on an RTX 6000 Ada
(≤ 75 calls per episode).

Output: `episodes.json`, a list of records
`{"task", "task_index", "episode", "seed", "success", "env_steps", "policy_calls", "seconds"}`; pair two policies on `(task, episode)`.
Please send the file and the per-task table (successes / 100 per task, average of the 7 task rates), with the GPU model, the Isaac-GR00T
commit and the SimplerEnv commit.

## 5. Calibration data for the 2-bit baselines (`data/`)

`data/bridge-calibration` and `data/bridge-heldout`: 256 observations each, built from the checkpoint's own fine-tuning data
`IPEC-COMMUNITY/bridge_orig_lerobot` @ `0e9d76d07e9df3ea3eba257b2520d4913833fad2` (LeRobot v2.0; 53,192 episodes; 5 Hz; AV1 video), never
from evaluation rollouts. Selection (`build_bridge_calibration.py`, seed `SeedSequence([20261106, split])`): 64 episodes × 4 frames at
10 / 37 / 63 / 90 % of the episode; 5 episodes for each of the six SimplerEnv task families matched on the instruction (spoon→towel/cloth,
carrot→plate, stack, eggplant, open drawer, close drawer) + 34 uniformly random episodes with a non-empty instruction (length ≥ 8); the two
splits use disjoint episodes. Layout (compact): `observations.parquet` with one row per manifest observation in manifest order, columns
`episode_index, frame_index, timestamp, task_index, image {bytes: PNG 256×256 RGB}, state [8], action [7]`; `meta/tasks.jsonl` maps
`task_index` → instruction; `observation_manifest.json` lists the episodes, frames and the selection rule. The builder reproduces the sets
from the Hub (needs PyAV with dav1d for AV1).

Loader (same contract as the LIBERO one):
```python
import io, json, numpy as np, pyarrow.parquet as pq
from PIL import Image
def load_observations(root):
    m = json.loads(open(f'{root}/observation_manifest.json').read())
    tasks = {r['task_index']: r['task'] for r in map(json.loads, open(f'{root}/meta/tasks.jsonl'))}
    rows = pq.read_table(f'{root}/observations.parquet').to_pylist()
    assert len(rows) == m['observation_count'] == 256
    return [{'observation/image': np.array(Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB')),
             'observation/state': np.asarray(r['state'], np.float32), 'prompt': tasks[r['task_index']]} for r in rows]
```
Feeding `Gr00tPolicy(embodiment_tag='SIMPLER_ENV_WIDOWX', ...)` for the Hessians (as `../calibrate.py` does for LIBERO), one observation
per call, flow-matching noise seed `SeedSequence([7, observation_index])` via `torch.manual_seed` before `policy.reset()`:
```python
obs = {'video': {'image_0': image[None, None]},                                  # (1, 1, 256, 256, 3) uint8
       'state': {k: state[i].reshape(1, 1, 1) for i, k in enumerate(('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'pad', 'gripper'))},
       'language': {'annotation.human.action.task_description': [[prompt]]}}
action, _ = policy.get_action(obs)                                                # keys x, y, z, roll, pitch, yaw, gripper, each (1, 8, 1)
```
Backbone Hessians can be taken from this checkpoint on these observations (we do) — the backbone weights are the LIBERO ones, the
activations are not. The held-out split is for offline checks only (action MSE of a quantized model against FP on the same noise seed).

## 6. Order of work

1. FP baseline with the commands above (700 episodes); send `episodes.json`.
2. The six PTQ baselines (GPTQ, AWQ, VPTQ, AQLM, QuIP#, QTIP; 2-bit, same configurations as the LIBERO rows) calibrated on
   `data/bridge-calibration`, evaluated with the same script and seeds; one `episodes.json` per method plus the accounting file.
