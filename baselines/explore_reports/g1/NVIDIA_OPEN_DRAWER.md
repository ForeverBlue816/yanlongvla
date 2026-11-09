# What NVIDIA's open_drawer = 65% measures (GR00T N1.7 Fractal)
Checkout: /scratch/d250009/pi05-handover/runs/toolkits/Isaac-GR00T (HEAD d2b7e75). SE = external_dependencies/SimplerEnv (fork github.com/squarefk/SimplerEnv per .gitmodules).

## Q1. Which setting / task produced 65
- examples/SimplerEnv/README.md:41: "| `google_robot_open_drawer` | 0/100 (0.0%) | 65/100 (65.0%) |" (N1.6 | N1.7), 100 episodes. Line 12-ish: "These values come from the default SimplerEnv evaluation runs (the per-task success rate are listed below)."
- No mention of "visual matching"/"variant aggregation" for the table (README line 3 only says SimplerEnv "Offers visual matching and variant aggregation"). HF model card nvidia/GR00T-N1.7-SimplerEnv-Fractal: not found any drawer/setting/episode text (only "we automatically measure the success rate").
- NVIDIA's own statement, GitHub issue https://github.com/NVIDIA/Isaac-GR00T/issues/685 (comment by ryhalabi, 12 Jun 2026): "README numbers came from the GR00T rollout_policy.py recipe with episodes raised from 10 to 200, not from SimplerEnv's RT-1 visual-matching script" (paraphrase by fetch tool); README Fractal numbers "generated with seed 2022, the SimplerEnv/ManiSkill default"; "the aggregate task google_robot_close_drawer samples top, middle, and bottom drawers. The official benchmark runs per-drawer environments, so the two protocols are related but not identical"; README uses --n_action_steps 1, --max_episode_steps 300 (official RT-1 script 113).
- So the 65 = the aggregate env google_robot_open_drawer (OpenDrawerCustomInScene-v0), "prepackaged_config" = visual-matching style configs, via GR00T rollout_policy.py, seed 2022 default (no --seed), 100 episodes (this table; issue #685 says an earlier README version used 200 and N1.6 open_drawer was 44% there; table was later rewritten, commit 65cc4a1 per NVIDIA).
- Not the official per-drawer variant-aggregation script: external_dependencies/SimplerEnv/run_evaluations_variant_agg_drawer.sh uses OpenTop/Middle/BottomDrawerCustomInScene-v0 with scenes frl_apartment_stage_simple, modern_bedroom_no_roof, modern_office_no_roof, light_mode brighter/darker, station mk_station2/3, robot_init_x_range 0.65 0.85 3, y -0.2 0.2 3, episode_length 113. The GR00T path does not use it.

## Q2. Our command's definition (confirmed from code)
- gr00t/eval/sim/SimplerEnv/simpler_env.py:236: "google_robot_open_drawer" registered as simpler_env_google/google_robot_open_drawer; line 27 `env = simpler_env.make(env_name)`.
- SE simpler_env/__init__.py:53: `"google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),`; make() sets `kwargs["prepackaged_config"] = True` and gym.make(env_name, obs_mode="rgbd").
- SE ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/open_drawer_in_scene.py:
  - `drawer_ids = ["top", "middle", "bottom"]` (OpenDrawerCustomInSceneEnv, line ~251); reset: `self.drawer_id = self._episode_rng.choice(self.drawer_ids)`.
  - comment "use prepackaged evaluation configs (visual matching)"; scene dummy_drawer, shader_dir rt, station mk_station_recolor, light simple, control_freq 3, max_episode_steps=113 (our wrapper overrides to 10000 / --max-episode-steps 300, simpler_env.py:28).
  - overlay_ids = a0,a1,a2,b0,b1,b2,c0,c1,c2 (9) with paired robot_init_xs/ys/rotzs; `idx_chosen = self._episode_rng.choice(len(overlay_ids))`.
  - urdf versions: choice of ["", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"] (4).
  - success: `qpos >= 0.15`.
  => 3 x 9 x 4 = 108 configurations, each sampled at random per episode (confirmed). Instruction "open {top|middle|bottom} drawer".
- Our s5.sh s5_run (lines ~113-114): rollout_policy.py --n-episodes 100 --max-episode-steps 300 --n-action-steps 1 --n-envs 5, embodiment SIMPLER_ENV_GOOGLE, no --seed when seed=none. Matches README lines 124-130 (README uses --n-episodes 10 as a demo).

## Q3. Issues
- NVIDIA/Isaac-GR00T #685 "Cannot reproduce SimplerEnv Fractal benchmark with N1.6-fractal ..." (closed 22 Jun 2026): open_drawer 25/200 vs README 88/200; NVIDIA reply = seed (see above). Related issues listed by the reporter: #309, #519, #560, #590, #619 (not read). Comment by zifeiyu-qc: A6000 also fails to reproduce.
- simpler-env/SimplerEnv: search "drawer groot" gave no results (not found).

## Strongest cause of 65 vs ~50: the seed-2022 sequence
- SE sapien_env.py:187-188: `self.set_main_rng(2022)`; `self.reset(seed=2022, ...)`; set_episode_rng(None) draws `self._main_rng.randint(2**32)`.
- rollout_policy.py:305-309: no --seed => `env.reset()`; each of the 5 sub-envs has its own main rng seeded 2022 => all 5 envs play the SAME config sequence (issue #685: "all five parallel envs played the same drawer configuration because ManiSkill hardcoded seed 2022"). So NVIDIA's 100 episodes = about 20 distinct configurations x 5 replays (65 = 5 x 13). Their 65 is a single 20-config draw (SE about 10 points per 20 configs), not a 100-sample estimate over 108 configs.
- Our "no seed" runs also default to 2022, so they should replay the same 20 configs; the differences (47-58) then come from sim/GPU/policy nondeterminism, not config choice. UNVERIFIED: whether our logs show identical per-env sequences.
- Secondary: GPU/driver differences (A6000 reports in #685), flash-attn.

## Cheapest GPU test
1. Check configs: run client with --n-envs 1 --n-episodes 20 and no --seed; the 20 configs equal those of NVIDIA's env 0. Run the same 5 times with seed none: if success is near 13/20 +- noise, 65 is the seed-2022 draw.
   Command change in s5_run: `--n-envs 1 --n-episodes 20` (S5_NENVS=1) with seed=none, then compare per-episode success list across repeats (determinism of the 20 configs).
2. Better, cheap: run with `--seed 7 .. 1000+` and --n-episodes 500 (n-envs 5, open_drawer only) to get the true mean over the 108 configs; expected about 50 if the 65 is a lucky 20-config sample.
