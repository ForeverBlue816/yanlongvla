# STACK_DIFF (G1): Fractal open_drawer gap, 47-58% vs NVIDIA 65%

## 1. HF repo nvidia/GR00T-N1.7-SimplerEnv-Fractal (public API, no token): 5 commits
| rev | date (UTC) | message | files changed vs previous |
|---|---|---|---|
| 77d1b4b7 | 2026-03-17 16:47 | initial commit | .gitattributes only |
| b86e19bc | 2026-03-17 23:04 | Upload folder using huggingface_hub | all weights, statistics.json, dataset_statistics.json, configs |
| f27b7ea0 | 2026-04-06 03:21 | Fix normalization | experiment_cfg/config.yaml, experiment_cfg/conf.yaml, processor_config.json ONLY (weights and statistics.json byte-identical to b86e19bc) |
| 1aff3ca4 | 2026-04-15 21:01 | Create README.md | README.md |
| fb8357ff | 2026-04-20 23:15 | Upload model-architecture.png | .gitattributes, model-architecture.png |
Result: f27b7ea0, 1aff3ca4, fb8357ff have identical weights, statistics and processor_config. No other revision is a candidate for NVIDIA's table. 77d1b4b7 has no weights. The table checkpoint is b86e19bc (2-4%, so not it) or the fb8357ff weights.
Our checkpoint is fb8357ff weights. The gap is therefore NOT a checkpoint difference.
The "Fix normalization" change is in processor_config.json (use_percentiles=True, use_mean_std=False, use_relative_action=True now). The processor loads from the checkpoint, so we get the fixed version.

## 2. Stack
Isaac-GR00T checkout: d2b7e75b937e (7 Oct 2026, "Sync GR00T N1.7 staging package to Isaac-GR00T").
Server env (python 3.12) vs pyproject/uv.lock; client env = simpler_uv/.venv (python 3.10) vs setup_SimplerEnv.sh.
| package | ours | NVIDIA pin | differs |
|---|---|---|---|
| torch (server) | 2.9.0+cu128 | 2.9.0+cu128 | no |
| torchvision (server) | 0.24.0+cu128 | 0.24.0 | no |
| transformers | 4.57.3 | 4.57.3 | no |
| numpy | 1.26.4 | 1.26.4 | no |
| flash-attn (server) | 2.8.3 in runs/envs/flash-attn (via FA_SITE) | 2.8.3 wheel | no (ruled out; groot env alone has none) |
| pillow (server) | 12.2.0 | 12.2.0 (uv.lock) | no |
| opencv (server) | headless 4.11.0.86 | 4.11.0.86 (lock) | no |
| safetensors / tokenizers | 0.7.0 / 0.22.2 | 0.7.0 / 0.22.2 | no |
| albumentations | 1.4.18 | 1.4.18 | no |
| gymnasium (client) | 1.4.0 | ==0.29.1 (setup_SimplerEnv.sh l.26) | YES |
| opencv (client) | opencv_python_headless 4.10.0.84 AND opencv_python 4.11.0.86 (both installed, share cv2/) | headless 4.10.0.84 only | YES |
| pillow (client) | 12.3.0 | none pinned | n/a |
| torch/torchvision (client) | 2.7.0 / 0.22.0 | torchvision==0.22.0 | no |
| transformers (client) | 4.57.3 | 4.57.3 | no |
| numpy, scipy, ray, tianshou, diffusers, einops, pandas, dm-tree, albumentations | 1.26.4, 1.15.3, 2.48.0, 0.5.1, 0.35.1, 0.8.1, 2.2.3, 0.1.9, 1.4.18 | same | no |
| sapien | 2.2.2 | not pinned (ManiSkill2_real2sim dep) | n/a |
| mani_skill2_real2sim | 0.5.3, submodule commit c2a9e87c (25 Jan 2025) | same submodule | no |
| SimplerEnv | 8a2d286c (squarefk fork, 23 Sep 2025) | same submodule | no |
Not checked: GPU, driver, and sapien/Vulkan renderer (NVIDIA used an unknown GPU). Open_drawer depends on the physics and render step, so this stays a candidate.

## 3. Code path (gr00t/eval/sim/SimplerEnv/simpler_env.py, Isaac-GR00T root)
- Env image: cv2.resize to (H=256, W=320), default INTER_LINEAR (l.110, size set l.243). Same as NVIDIA by construction (same file).
- Policy image: processor eval transform, albumentations SmallestMaxSize(256, INTER_AREA) then FractionalCenterCrop(0.95) then SmallestMaxSize again (gr00t/model/gr00t_n1d7/image_augmentations.py l.485-487), use_albumentations=True, crop 230, target 256 (processor_config.json).
- max episode steps: env._max_episode_steps=10000 (l.28); the cap is the CLI flag --max-episode-steps 300 (README l.129, MultiStepWrapper). Control freq/sim freq come from the SimplerEnv defaults; no override.
- n-action-steps: README uses 1 for Fractal (rollout_policy.py default is 8). Action horizon in the checkpoint: delta_indices 0..7, so 8 predicted steps, execute 1. No action ensembling in this code.
- Denoising: num_inference_timesteps=4 (checkpoint config.json l.57); dtype bfloat16 (config.json l.31, l.52); use_flash_attention true (l.77; flash-attn ruled out).
- NVIDIA exact command already reproduced (47/52/50), so none of these CLI values differs.

## Ranking (likelihood of changing open_drawer)
1. Sampling noise: n=100 per run gives about +-10 pts; 65 is within about 1.5-2 sigma of our 47-58. Run-to-run spread already 47-58. Strongest explanation.
2. Client gymnasium 1.4.0 vs pinned 0.29.1 (step/truncation/seed/space semantics; reset seeding and TimeLimit/obs handling could change initial states or termination).
3. Client cv2: two opencv packages overlaid (4.10 headless + 4.11 full); cv2.resize LINEAR output could differ in rounding. Low.
4. Renderer/GPU/driver (sapien 2.2.2 Vulkan): unknown, not testable by pin.
5. Checkpoint revision, flash-attn, seed, wrapper protocol, torch/transformers/numpy: no difference or ruled out.

## One-factor GPU tests (top 3)
T1 (noise): same fb8357ff, NVIDIA command, 3 more runs at n-episodes 300 (or 1000 total) to give +-3 pts; compare with 65.
T2 (gymnasium): copy venv is not allowed to change; build a NEW venv (python 3.10) by running setup_SimplerEnv.sh into a new UV_ENV dir with gymnasium==0.29.1 left alone (pip check that it stays 0.29.1 after tianshou), or only: new venv = setup script as written. Run the same open_drawer command. Only gymnasium differs.
T3 (cv2): in a NEW venv or via PYTHONPATH overlay, uninstall opencv_python (non-headless) so only headless 4.10.0.84 remains; same command. Only cv2 differs.
