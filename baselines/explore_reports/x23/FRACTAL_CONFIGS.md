# X23: configuration mix in the Fractal open_drawer runs

## Answer
The fixed chain is real, but it does not explain 65 against 50.
It explains why the 5-env official runs sit at 47-56 and not at 54-56.
The chain is slightly UNFAVOURABLE. The uniform mix gives about 54-56 percent, not 65.
So about 9 points remain open and are not a layout effect (for this code version).

## Code (E = Isaac-GR00T/external_dependencies/SimplerEnv)
- Env: simpler_env/__init__.py:53 maps google_robot_open_drawer to OpenDrawerCustomInScene-v0. make() at :79-84 sets prepackaged_config=True.
- Class: ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/open_drawer_in_scene.py. drawer_ids = top, middle, bottom (l.251).
- Draw per reset, all from one `_episode_rng`:
  - l.150 `set_episode_rng(seed)`; l.151 `drawer_id = _episode_rng.choice(drawer_ids)` (3 values).
  - l.210 `idx_chosen = _episode_rng.choice(9)`: overlay a0..c2 (background image) AND robot init x, y, rotz together (l.193-209). 9 values.
  - l.223-230 `new_urdf_version = choice(["", recolor_tabletop_visual_matching_1, ..._2, recolor_cabinet_visual_matching_1])` (4 robot-urdf variants).
  - Lighting is fixed (l.56-57 light_mode simple, l.93-97). Camera mode is None. Not random.
  - 3 x 9 x 4 = 108 configurations. The instruction text "open <drawer> drawer" is burned into every video frame.
- Seeding is the same as in x22: sapien_env.py:185-188 (construct with main seed 2022), :472 (`set_main_rng(seed)`), :492-505 (seed None draws `_episode_seed = _main_rng.randint(2**32)`), :161 of the drawer file passes `seed=self._episode_seed` down.
- GoogleFractalEnv.reset (gr00t/eval/sim/SimplerEnv/simpler_env.py:79-85) passes seed None unless rollout_policy.py:300-305 gives seed+i.
- Chain: s0 = 2022, s_{k+1} = RandomState(s_k).randint(2**32), config_k = (drawer, overlay, urdf) from RandomState(s_k) in the order above. Script: chain.py.
- Seeded run (--seed S, 5 envs): env i first draws cfg(S+i), then the chain of S+i. Also a fixed chain per env, with a different start.
- Deterministic: yes. Every unseeded run, any number of envs, walks the same 2022 chain. 5 workers all start from the same chain.

## Configurations visited
- 1 env: 100 episodes = chain items 1..100: 70 distinct of 108 (26 of 27 drawer x overlay cells). Same 100 in every run (confirmed below).
- 5 envs, official: all 5 envs run the SAME chain, so the 100 episodes are chain items 1..21-24 (about 20 per env). That is 17-19 distinct of 108 and 12-13 of 27 cells. Decoded for every run (decode.py, lock-step model with completion order): max chain index 21-23.
- Seeded 100-episode runs: 63-69 distinct configs, 26-27 of 27 cells. fpbig (500): 27 of 27 cells.

## Method for the data
- Video frame 0 of each episode (mtime order = results order; 2 adjacent swaps fixed by success flag).
- Drawer: read from the burned-in text strip (nearest template, same 3 strings; text distance 6 inside a class, 8100 between classes).
- Overlay (9 values): top 256 rows, nearest template. Templates: the two 1-env runs, whose configs are the simulated chain.
- Check: 200 of 200 one-env episodes classify to the simulated chain (drawer and overlay), leave-one-out with the twin episode excluded. Run 1 and run 2 first frames are identical for every index.
- The robot urdf variant is not separable from the frame. For the 108 count it comes from the chain decode.
- The 5-env order decode (lock-step envs, completion order, max 300 steps; slack of 10 steps) is feasible for all 6 official/pinned runs and for the seeded runs fp, fpfa, seed8, seed9. This confirms the chain model for seeded runs too. fpbig and fprr (101 episodes) were not decoded.

## Per-arm results (successes of n)
| arm | n | succ | cells seen of 27 | top / middle / bottom share | top / middle / bottom rate |
|---|---|---|---|---|---|
| 1env r1 | 100 | 58 | 26 | 36/32/32 | 39/69/69 |
| 1env r2 | 100 | 54 | 26 | 36/32/32 | 36/59/69 |
| 5env official r1 | 100 | 47 | 13 | 31/41/28 | 13/61/64 |
| 5env official r2 | 100 | 52 | 13 | 29/42/29 | 17/74/55 |
| 5env official r3 | 100 | 50 | 13 | 30/42/28 | 3/76/61 |
| 5env pinned venv r1 | 100 | 48 | 12 | 30/42/28 | 17/69/50 |
| 5env pinned venv r2 | 100 | 56 | 13 | 31/42/27 | 13/79/70 |
| 5env pinned venv r3 | 100 | 49 | 13 | 29/43/28 | 0/72/64 |
| seeded 7 (SDPA, fp) | 100 | 48 | 26 | 35/36/29 | 43/42/62 |
| seeded 7 flash (fpfa) | 100 | 50 | 26 | 38/35/27 | 39/63/48 |
| seeded 7 rerun | 101 | 59 | 26 | 34/35/32 | 50/63/62 |
| seeded 8 | 100 | 55 | 26 | 32/35/33 | 41/51/73 |
| seeded 9 | 100 | 51 | 27 | 30/38/32 | 43/53/56 |
| seeded 7, 500 ep (fpbig) | 500 | 271 | 27 | 37/33/29 | 49/44/73 |

Overlay index (robot pose + background), pooled rate:
- 5 env official + pinned (600 ep): n per overlay 0..8 = 60,48,90,60,73,204,30,7,28; rate 97,31,0,92,89,41,67,86,0 percent.
  Overlay 5 is 34 percent of the episodes (7 of the first 22 chain items). Overlays 2 and 8 (never won) are 20 percent.
- 1 env (200 ep): n = 24,28,18,16,22,36,20,18,18; rate 96,75,0,94,64,39,40,89,6 percent.
- Seeded pool (1001 ep): rate 89,61,1,96,85,44,22,81,13 percent. Overlays 2 and 8 are near zero in every arm. Overlay 6 is weak (22 percent). Top drawer is hard when the overlay is 5, 6, 2, 8.
- The per-cell table (27 cells, seeded pool) is in configs.json (key seeded_pool, n and s).

## Reweighting (per-cell rates from the seeded pool, 27 cells, n about 37 per cell)
| arm group | actual | predicted from its own mix | uniform over 27 cells |
|---|---|---|---|
| 1 env, 2 runs (200) | 56.0 | 57.4 | 54.1 (own rates) / 55.9 (pool) |
| 5 env official, 3 runs (300) | 49.7 | 51.3 | 55.8 (own rates, unseen cells filled from pool) |
| 5 env pinned venv, 3 runs (300) | 51.0 | 51.3 | 54.9 |
| seeded 100-ep, 5 runs (501) | 52.5 | 52.7 | 56.2 |
| seeded 500 ep (fpbig) | 54.2 | 54.0 | 56.4 |
- Uniform over 27 cells with seeded per-cell rates: 55.9 plus or minus 1.1 (from cell noise). Direct seeded estimate: 54.2 plus or minus 2.2 (fpbig, 500 ep), 53.3 (pool of 1001).
- So the 5-env fixed chain costs about 4-5 points against uniform. The 1-env chain gains about 1-2 points.
- The mix prediction matches the 5-env data (51.3 against 49.7 and 51.0). Top-drawer failure is large there: 3-17 percent against 40-50 percent in seeded runs.

## What mix would give 65 percent
- Uniform is 55.9. To reach 65 you need to shift weight from the weak cells to the strong ones.
- Remove overlay 2 (the unwinnable pose, 1 percent) from the draw: 62.8. Remove overlays 2 and 8: 68.7. Remove 2, 8, 6: 75.4.
- Smallest tilt of the uniform mix (exponential tilt of cell weights, KL to uniform 0.03) that reaches 65: no cell above 5 percent weight. Small shifts are enough because cell rates are bimodal.
- The official 2022 chain does the opposite: 7 of the first 22 items are overlay 5 (rate 44) and 4 are overlay 2 or 8.

## Can the fixed chain explain 65 against about 50?
No, for this code. The 2022 chain is at the 28th percentile of random chains. It predicts 51, and we measure 50.
A random 20-item chain has a predicted rate with mean 56.0 and sd 8.2 points. 14 percent of random chains predict 65 or more (19 percent of runs reach 65 with run noise).
So a lucky chain could give 65. Our chain is not that chain. NVIDIA's table can come from a favourable chain only if their code or library version draws the chain differently. Their environment ran their stack with the same seed 2022, so this is not shown. No evidence here supports it.
The 5-env official score of about 50 is a property of the fixed chain, as in x22 (Bridge). Here the chain lowers the score, so the layout explains about 5 points of the gap between 55 (uniform) and 50, and none of the gap from 55 to 65.

## Single GPU run to confirm
The layout side is already settled by data in hand: fpbig (500 seeded episodes) gives 54.2 plus or minus 2.2, and the 5-env chain gives 50.
No new run is needed to show that layouts do not carry the 65.
To test the part still open (is the 65 a different chain or a different stack?), run NVIDIA's exact command once on the SimplerEnv and Isaac-GR00T commits named in the model-card README, if they differ from ours. 100 episodes, 5 envs, no seed.
If that run gives about 50, the table value comes from something else (checkpoint revision, table protocol). If it gives 60 or more, the commit changes the chain or the env.
Cheap CPU check first: compare the SimplerEnv commit of the model card with ours (open_drawer_in_scene.py draw order, lines 150, 210, 223).

## Files
chain.py (chain simulation), frames.py, frames2.py, classify.py (frame to drawer and overlay), decode.py (5-env order decode), analyze.py and mc.py (reweighting), episodes.json (per-episode drawer, overlay, success, length), decode.json, configs.json (per-arm and per-group counts), mc.json, first_frames_example.png (four frames of the same overlay; the text shows the drawer).
