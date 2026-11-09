# G3: path A (mentor seeded protocol) vs path B (NVIDIA command) on widowx_stack_cube  (read-only code audit, 2026-11-08)
Paths. A = baselines/groot/simpler/{simpler_server.py,eval_simpler_bridge.py} (below: `ev:` = eval_simpler_bridge.py, `sv:` = simpler_server.py), launched by baselines/simpler/s5.sh t_b5 `ev)` (s5.sh:130-153).
B = Isaac-GR00T gr00t/eval/{run_gr00t_server.py,rollout_policy.py} (`ro:`), env wrapper gr00t/eval/sim/SimplerEnv/simpler_env.py (`se:`), launched by s5.sh t_b5 `of)` (s5.sh:154-162) -> s5_run (s5.sh:99-117).

## 0. What the existing data already says (completed runs under s5b/eval and s5b/official)
- A is deterministic: fp, fprr, fpsl (videos+step log) all give 36/120 with identical per-layout counts; fpnt (no env termination) 37. So video wrapper, per-step log, terminate-after-success do NOT change the result.
- A, other noise: fps8 (seed 8, sdpa) 44/120 = 37%; fpfa (seed 7, flash-attn) 43/120 = 36%; fp (seed 7, sdpa) 36/120 = 30%.
- B (always flash-attn): 5 envs 43,46,46; 1 env 46,40,36.
- Binomial sd for n=100-120 at p=0.4 is 4.5-5 points (difference of two runs: 6.5-7). A sdpa seeds 7 vs 8 differ by 7 points by noise alone. The 15-point gap (30 vs 45) is a 2.3-sigma comparison of one A run against the mean of three B runs; A-with-flash-attn (36%) vs B-1-env mean (41%) is within noise. Per-layout A result is very uneven (layouts 14-18 ~0, 0/1/3/4/6 ~4-5 of 5), so a small fraction of layouts drives the rate.
- Hence the ladder below is mostly "estimate the mean of each arm with enough episodes", not "find a bug". No code-level difference listed in section 1 can plausibly move the MEAN by 15 points except the attention backend and (slightly) the in-flight-episode drop.

## 1. Differences (A line / B line / one-change switch)
| # | Factor | A | B | Switch |
|---|---|---|---|---|
| 1 | Attention backend (server) | sdpa unless FA_SITE set (s5.sh:140 only for row fpfa) | FA when exported (s5.sh:105,160) | A->B: run row fpfa (FA_SITE). B->A: `attn=sdpa` in `b5__of__fp__sdpa__...`. Only difference that moved A (36 -> 43). |
| 2 | Layout choice | enumerated k mod 24, exactly 5 each: ev:143,147,148 (episode_id via patched reset ev:58-63) | random with replacement: se:183-184 drops `options`; PutOnBridgeInSceneEnv.reset draws episode_id = episode_rng.randint(24) (ManiSkill2_real2sim put_on_in_scene.py:188-191) | A->B: ev:147 `options=None` (and drop patch_bridge_env reset patch, ev:197). Same mean, more variance in B. |
| 3 | Env seed | every episode `env.reset(seed=s)`, s=SeedSequence([7,t,k]) (ev:42-43,142,148); also np/random seed ev:145-146 | none; with `--seed` only first reset seed+i (ro:305-309); later episodes autoreset unseeded: main rng (sapien_env.py:490-505) | A->B: ev:148 `seed=None`. B->A: add `--seed 7` (only first reset, weak). Stack cube has no random scene part beyond layout, so low effect. |
| 4 | Flow-matching noise seed | torch.manual_seed(noise_seed) at every episode reset: sv:22-28, client ev:149 | no seed: torch global RNG of server, never reseeded (run_gr00t_server.py:104-169 has no seeding; ro:557 seed_everything only in client process) | A->B: ev:149 `policy.reset()` (no options). Mean unchanged, run-to-run noise appears. |
| 5 | Number of envs / server batch | 1 sync env (ev:135), batch 1 | 5 async spawn envs (ro:444-452, s5.sh:161 `ne=5`), server sees batch of 5 per call: different bf16 kernels/noise shapes (gr00t_n1d7.py:349) | A->B: not possible in ev (SyncVectorEnv, ev:135). B->A: `fa1` (S5_NENVS=1), already run (46,40,36). |
| 6 | Episode count and which are counted | 120, all counted | 100; loop stops when the 100th episode finishes (ro:428-429, `while completed_episodes < n_episodes`), in-flight episodes (up to 4, biased to long = failing, since success ends the episode early) are discarded. Small upward bias, est. +1 to +2 points. n_episodes=max(n,n_envs). | B->A: `S5_NENVS=1` removes it. |
| 7 | policy.reset | per episode, with noise_seed (ev:149) | once at start (ro:310); policy stateless | none needed |
| 8 | Server class | SeededSimPolicy(Gr00tSimPolicyWrapper(Gr00tPolicy(strict=True))) sv:22-29 | Gr00tSimPolicyWrapper(Gr00tPolicy(strict=cfg.strict=True)) run_gr00t_server.py:104-165 | identical except seed (#4) |
| 9 | terminate_on_success | False (ev:134) | True (ro:603) | no effect: ManiSkill done=success already ends episode (ev docstring 16-23; fpnt 37 vs 36 confirms) |
| 10 | Video wrapper | none (video_dir None, ev:133) | on, default /tmp or `--video-dir` (ro:562-566, s5.sh:113) | no effect (fpsl = fp, 36) |
| 11 | Success counting | any success flag over chunks, incl. final_info (ev:156-159) | same rule (ro:320-356) | identical |
| 12 | max steps, n-action-steps | 300, 4 (ev:190-191; s5.sh:147) | 300, 4 (s5.sh:112-113, S5_NAS=4) | identical |

Identical in A and B (same code, same checkpoint $G/s5b/checkpoints/bridge): env id widowx_stack_cube (StackGreenCubeOnYellowCubeBakedTexInScene-v0), robot init fixed (put_on_in_scene.py:208-213), camera via get_image_from_maniskill2_obs_dict, cv2.resize to 256x256 (se:213, 259, default INTER_LINEAR), uint8, proprio xyz + rpy converted with default_rot, pad=0, gripper = proprio[7] (se:207-223), instruction "stack the green block on the yellow block" (put_on_in_scene.py:~345), processor crop 0.95 + resize, statistics.json normalisation, action horizon 40, 4 denoise steps, bf16 (gr00t_policy.py:108-111), gripper map `2.0*(a>0.5)-1` (se:225-227), 4 of the 8/40-step chunk executed via MultiStepWrapper, control frequency set by the env. A patches only WidowXBridgeEnv.reset (ev:58-63) and, optional, step logging / no-env-termination (ev:64-81).

## 2. Ladder (B -> A, one factor at a time; ordered by chance of explaining the gap)
Baselines to compare: B-5env-FA = 43,46,46 (mean 45); B-1env-FA = 46,40,36 (mean 40.7); A-sdpa = 36, 44 (seed 8); A-FA = 43.
Each rung = same server checkpoint, same task; "run" = 100 (B) or 120 (A) episodes. Unit of evidence: pooled successes. To see a 5-point effect at 1 sigma-pair you need ~600 episodes per arm; a 15-point effect needs ~100-200.
1. Sampling noise of A (most likely explanation). Run A-FA with 4 more seeds: row `fpfa` with `seed=8,9,10,11` (s5.sh:133 edit: seed=${fpfa_seed}); 4 x 120 = 480 ep, 4 jobs. If pooled A-FA (5 runs, 600 ep) is ~37-42%, gap is noise + #1/#6 and is closed.
2. Attention backend in A: A-sdpa seeds 9,10,11 (row `fp` with seed edit) = 360 ep, 3 runs; compare pooled with step 1. Seeds 7/8 gave 36/44, so need pooling before believing sdpa->FA = +7.
3. B with 1 env, more repeats: `b5__of__fp__fa1__widowx_stack_cube__r4..r6` (3 jobs x 100 ep). Pool with r1-r3 (6 x 100): tells the real B-1env mean and removes the discard bias (#6).
4. B 5-env repeats: r4..r6 of `fa` (3 x 100) -> B-5env mean with 6 runs; difference to step 3 = batch-5 + discard bias (#5, #6). Expected <= 3 points.
5. Layout sampling: A client with random layouts: ev:147 `options=None` + skip ev:197 patch, seed 7, FA server, 2 runs x 120. Expected mean unchanged; checks that equal weighting of the 24 layouts does not hurt (B draws with replacement, so layout mix per run differs; weights are equal in expectation).
6. Noise seed: A with ev:149 `policy.reset()` and FA, 2 runs x 120 (expected no change; fpfa vs fps8 shows seed noise only).
7. Server batch effect isolated: only if steps 3-4 differ by >5 points; then run B-5env with `--n-envs 5` but `--n-episodes 120` and compute success from the log of the first-completed N vs all completed (needs client edit to print all completed episodes, ro:429/while loop).
8. Skip: terminate_on_success, video, step log, policy.reset, server subclass (data show no effect or stateless).
Decision rule: if pooled A-FA and pooled B-1env-FA agree within 1.5 pooled sd (~4 points for 600 vs 600 episodes), conclude no protocol difference; the earlier A figure (30%) was the sdpa backend plus sampling noise. Report the FP reference as the pooled FA figure.
