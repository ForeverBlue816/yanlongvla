# SimplerEnv-Bridge (WidowX) line — GR00T N1.7 (2026-11-06)

User request (2026-11-06): build the dataset GR00T needs for SimplerEnv-Bridge, adapt our quantization code to this dataset /
checkpoint, and tell Chamadol how to evaluate the FP baseline `nvidia/GR00T-N1.7-SimplerEnv-Bridge` with the checkpoint we hold here.

## Checkpoint and code
- `nvidia/GR00T-N1.7-SimplerEnv-Bridge` @ `940134b3c2948ccfdf8e7393f2d2ca869dc42833`, flat layout (`model-0000{1,2}-of-00002.safetensors`,
  `config.json`, `processor_config.json`, `embodiment_id.json`: `simpler_env_widowx` → projector 1, `libero_sim` → 2). Downloaded 2026-11-06
  without the DeepSpeed optimizer states (`global_step32000/`, 28 GB): 6.5 GB in `cache/huggingface/hub`.
- **Backbone byte-identical to the LIBERO checkpoints**: 494 tensors equal (all `backbone.*`), 537 differ, all `action_head.*`
  (DiT 456, `vl_self_attention` 64, action/state encoders–decoders 14, `vlln` 2, position embedding 1). Config: `num_inference_timesteps` 4,
  `model_name nvidia/Cosmos-Reason2-2B` (gated; our token has access → official backbone config).
- Isaac-GR00T checkout `src/Isaac-GR00T` (commit recorded in `public-reports/baselines/groot/groot_common.py`), SimplerEnv fork at
  `external_dependencies/SimplerEnv` (ManiSkill2_real2sim 0.5.3, SAPIEN 2.2.2).
- Embodiment `SIMPLER_ENV_WIDOWX`: video `image_0` (256×256, no flip), state 8 = x, y, z, roll, pitch, yaw, pad(=0), gripper, action 7
  (x, y, z, roll, pitch, yaw, gripper∈[0,1]), 8-step action chunk, **4 executed per call** (`examples/SimplerEnv/README.md`),
  `--max-episode-steps 300`, 7 tasks. NVIDIA's own numbers (their protocol, 100 episodes, 5 async envs, unseeded):
  spoon_on_towel 78, carrot_on_plate 58, put_eggplant_in_basket 53, stack_cube 48, put_eggplant_in_sink 2, close_drawer 97, open_drawer 100
  → average 62.3 %.

## Dataset (done 2026-11-06)
- Source = the checkpoint's own fine-tuning data `IPEC-COMMUNITY/bridge_orig_lerobot` @ `0e9d76d07e9df3ea3eba257b2520d4913833fad2`
  (53,192 episodes, 1.89 M frames, 5 Hz, 4 AV1 camera streams; GR00T uses `image_0`). 14,532 episodes have an empty instruction.
- `research/scripts/bridge_calibration.py` (select → download → build → compact): `datasets/bridge-calibration` and `datasets/bridge-heldout`,
  256 observations each = 64 episodes × 4 frames (10/37/63/90 % of the episode), episodes disjoint between splits, seed
  `SeedSequence([20261106, split])`; 5 episodes per SimplerEnv task family matched on the instruction (spoon→towel/cloth 441 candidates,
  carrot→plate 551, stack 11, eggplant 418, open drawer 572, close drawer 493) + 34 random non-empty-instruction episodes (length ≥ 8).
  Frames decoded with PyAV/dav1d (envs/openpi; torchcodec in envs/groot has no FFmpeg) and stored as PNG bytes in parquet, so no video
  decoding downstream; full episodes locally (186 + 175 MB), compact copies (manifest frames only, 23 + 22 MB) in
  `public-reports/baselines/groot/simpler/data/`.
- `bridge_calibration.load_observations(data_dir)` has the `calibration_data.load_observations` contract (+ the dataset action).

## Quantization code (done 2026-11-06, untested on GPU until the calibration job runs)
- `research/scripts/groot_simpler.py`: `prepare` (FP dir `models/groot_n17_simpler_bridge/fp`, done), `smoke`, `calib` (hooks E[xxᵀ] over the
  256 observations, noise `SeedSequence([7, i])`, → `ptq/hessians/{backbone,head}_simpler_bridge`), `fit` (TCQ two-sided
  `groot_tcq.fit_layer(rotation='two')`, → `ptq/fit/{backbone,head}_tcq_rht_simpler_bridge`), `assemble` (→ `models/groot_ptq/<row>/simpler_bridge/`,
  a flat checkpoint the official `run_gr00t_server.py` serves unchanged), `offline` (held-out chunk MSE vs FP).
- Rows: `tcq_w2_rht` (backbone + head calibrated on Bridge), `tcq_w2_rht_bbl` (LIBERO-calibrated backbone fit reused — same weights —
  with the Bridge head; calibration-transfer ablation).
- Evaluation: `research/scripts/simpler_eval.py` (SimplerEnv island, ZMQ `PolicyClient`): one env, episode k of task t reset with seed
  `SeedSequence([7, t, k])` (ManiSkill2 main + episode RNG → object layout / robot init), 300 env steps, 4 of 8 actions per call, success =
  any step, one JSON record per episode (pairing on task × episode). Slurm: `simpler_check` (Vulkan / reset determinism / FP smoke /
  4 episodes), `simpler_calib`, `simpler_fit`, `simpler_eval` (`SIMPLER_ROW`, `SIMPLER_MODEL`, `SIMPLER_EPISODES`).
- SimplerEnv island built by `research/scripts/simpler_setup.sh` (= upstream `setup_SimplerEnv.sh` without the GPU sanity check) at
  `src/Isaac-GR00T/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv` + `msgpack msgpack-numpy tqdm transforms3d`. The login node has no
  `libvulkan.so.1`; whether the GPU nodes do is what `simpler_check` answers first.

## Pre-registration
- FP reference: 7 tasks × 100 seeded episodes (700) with the script above; reported per task and averaged (NVIDIA's table is the sanity
  check, not the reference: different seeding / vectorisation).
- 2-bit rows evaluated with the same seeds and paired per episode; a 2-bit row is read against FP by the paired difference (bootstrap
  CI over the 700 episodes, stratified by task).
- Chamadol: FP baseline first with the same script, then the six PTQ baselines quantized on `data/bridge-calibration` (same observation
  count and loader contract as LIBERO), evaluated with the same script and seeds.

## Status
- 2026-11-06 05:30 UTC: datasets built; FP dir prepared; code written; `simpler_check` (killable, pro6000) and `simpler_calib` (rose, pro6000)
  queued (cluster full: only pro6000 had free GPUs). Arm-B (iso) fit copies paused / cancelled to free killable slots (resumable).
