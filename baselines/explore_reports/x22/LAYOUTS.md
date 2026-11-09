# X22: layout mix in the official GR00T stack_cube runs

## Answer
Yes. The layout mix explains the 47 vs 39 gap. The mix is not uniform, and it is the same in every run.
The episode draws are NOT independent across envs or runs. They are one fixed pseudo-random chain.

## Code reading (Isaac-GR00T and ManiSkill2_real2sim)
Paths: E = Isaac-GR00T/external_dependencies/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/envs
- rollout_policy.py:300-305: `if seed is not None: reset_seeds=[seed+i ...]; env.reset(seed=reset_seeds)` `else: observations, _ = env.reset()`. The official command has no seed, so reset gets seed=None.
- simpler_env.py:183-184 (and 76-81): `self.env.reset(seed=int(seed) if seed is not None else None)`. None is passed down.
- Autoreset: gymnasium 0.29 AsyncVectorEnv resets a finished env inside its worker with seed=None. No code gives a worker its own seed.
- sapien_env.py:185-188 (construction): `self.set_main_rng(2022)` then `self.reset(seed=2022, ...)`. Every env starts with the same main seed 2022.
- sapien_env.py:490-497: `set_main_rng(None)` returns early when `_main_seed` is set. A random seed is used only when `_main_seed is None`, which never occurs after construction.
- put_on_in_scene.py:184-190: `self.set_episode_rng(seed)`; `episode_id = ... self._episode_rng.randint(len(xy_configs)*len(quat_configs))` = randint(24).
- sapien_env.py:501-502: with seed None, `self._episode_seed = self._main_rng.randint(2**32)`.
- put_on_in_scene.py:205 then calls `super().reset(seed=self._episode_seed)`. sapien_env.py:472 runs `set_main_rng(seed)`, so main_rng = RandomState(episode_seed).
Result: the chain is s0=2022, s_{k+1} = RandomState(s_k).randint(2**32), episode_id_k = RandomState(s_{k+1}).randint(24). No entropy from the OS, time or process id is used.
- 1 env (SyncVectorEnv): the 100 episodes use chain items 1..100 in order. Every run draws the same 100 layouts.
- 5 envs (spawn): each of the 5 workers starts from seed 2022 and draws the SAME chain. All 5 envs do the same layout sequence, shifted only by how fast each episode ends. The first 100 finished episodes come from about the first 20 chain items per env (more for short, successful episodes). Every run draws the same subset.
- Verified: a simulation of the chain gives layout counts for 100 draws that match the 1-env video counts divided by 6 exactly, for all 24 layouts (layout 13 never, layout 1 ten times).
- Run-to-run variance (43..55 and 34..46) comes from policy noise only, not from layouts.

## Empirical: layout of each official episode
Method: first frame of each mp4, downsampled to 128x96, nearest neighbour against 120 frames of the seeded run s5b/eval/fpsl (5 copies of each layout 0..23; video-to-episode map by mtime, which matches the success flags 120/120). Leave-one-out check on the templates: 120/120. A few official frames have a large distance (some layout images may be rare), but the 1-env result matches the simulated chain exactly, so the labels are reliable.

Pooled layout counts (600 episodes per arm), layouts 0..23:
- 5 env: [60,138,30,30,30,60,0,0,0,0,0,30,0,0,0,60,30,30,45,26,0,0,30,1]. 14 layouts appear, 10 never. Layout 1 is 23 percent. Chi-square vs uniform = 953.8 (df 23), p = 9e-187.
- 1 env: [42,60,30,36,18,18,30,18,18,24,24,12,36,0,18,36,42,36,24,18,18,6,24,12]. 23 layouts appear. Chi-square = 160.3, p = 1.5e-22. It is the same counts in all 6 runs (multiples of 6).
- Per-run counts are in layouts.json (per_run).

Success per layout (successes of n), pooled:
- 5 env: L0 43/60, L1 135/138, L2 6/30, L3 23/30, L4 27/30, L5 9/60, L11 4/30, L15 18/60, L16 5/30, L17 0/30, L18 4/45, L19 3/26, L22 5/30, L23 0/1.
- 1 env: L0 31/42, L1 58/60, L2 3/30, L3 26/36, L4 15/18, L5 2/18, L6 27/30, L7 5/18, L8 6/18, L9 10/24, L10 10/24, L11 1/12, L12 5/36, L14 0/18, L15 6/36, L16 6/42, L17 0/36, L18 3/24, L19 6/18, L20 9/18, L21 0/6, L22 3/24, L23 3/12.
- Seeded (fpsl, 5 per layout): rate 30 percent; layouts 0,1,3,4,6 high; 11-18 near 0.

## Reweighting
- Uniform layout weights (mean of per-layout rates): 5 env = 33.4 percent (14 seen layouts); 1 env = 33.4 percent (23 seen layouts); 33.6 percent for 1 env on the 14 layouts common to both. The two arms agree at about 33.4 percent, in line with our seeded 33-37 percent.
- On the 14 common layouts with each arm's own mix: 5 env 47.0 percent, 1 env 40.0 percent.
- Seeded per-layout rates applied to each arm's mix: 5 env predicts 42.3 percent, 1 env 35.4 percent, uniform 30.0 percent (fpsl alone; per-layout n=5 is noisy).
- 5-env per-layout rates on the 1-env mix: 41.4 percent. 1-env rates on the 5-env mix: 44.8 percent.
So the mix explains the gap: the 5-env arm over-samples layouts 1, 0, 3, 4 (success 70 to 98 percent) and never draws layouts 6-10, 12-14, 20, 21. Layout 1 alone is 138 of 600 episodes and gives 135 successes (22.5 points of the 47).
The 5-env mix also has a bias toward short episodes: a successful episode ends sooner, so its env draws the next chain item sooner. This oversamples easy layouts further, such as layout 1.

## Conclusion
The 47 percent is a property of a fixed, repeated layout subset (one RNG chain from seed 2022), not a better policy. Under uniform layouts the official command gives about 33 percent, equal to our seeded protocol. The official 6 runs are not 600 independent samples of layouts: they are 6 noise draws on the same layouts.

Files: classify.py, analyze.py, episodes.json (per-episode layout, success, distances), layouts.json.
