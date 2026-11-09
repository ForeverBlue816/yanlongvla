# C0 audit: suspicious results (8 Nov 2026, opus subagent, CPU only, no jobs submitted)

Scope: RESULTS.md, RIGOUR_2026-11-06.md, EXPLANATIONS.md, MENTOR_PENDING_2026-10-31.md, DECISIONS.md, REPORT_2026-11-06.md,
TASK1_METHOD_AUDIT.md, groot/simpler/README.md, GOAL v7 Part C (S1-S8). Data read: s5b/eval/* (Bridge, complete files only;
nz06s5 NOT read, stack_cube has 94 of 120 records), s5bq/paired/*, s5bq/layers (Hessians), s5bq/perturb/aqlm_w2_error.json,
eval/*/plus_*.jsonl (GR00T), readme_baselines/evaluation/*/plus_1000.jsonl (pi0.5), s5/eval/fp*/open_drawer.log (Fractal),
official code in runs/toolkits (Isaac-GR00T d2b7e75, SimplerEnv ManiSkill2_real2sim, quip-sharp).
Side file: noise_ratio_factor.json (A4).

Ranking = (impact on a headline claim) / (cost). Cost "0" = the analysis is already done here; the orchestrator only has to
check one number and edit a sentence.

| rank | id | impact | cost | one line |
|---|---|---|---|---|
| 1 | A1 | high | 0 (CPU done) | Bridge gains rest on 2 tasks; on the other 5 tasks QTIP is significantly BELOW FP |
| 2 | A2 | high | ~2 GPU-h | 2-bit carrot "successes" are abnormally fast (median 28-32 steps vs FP 80); success rule allows a carried object |
| 3 | A3 | high | ~2 GPU-h | Bridge FP is one deterministic numerical draw; H3 tested determinism, not the numerical floor |
| 4 | A4 | med-high | 0 (CPU done) | Bridge pairs read without Holm and without a between-model floor (LIBERO has both) |
| 5 | A5 | high-med | ~3 GPU-h | Same quantizer config re-run gives a different model (VPTQ iter500 vs base: 176 unit flips); no quantizer-draw floor |
| 6 | A6 | medium | 0 (CPU done) | S1 refined: 2-bit rows overshoot NVIDIA's FP, so "restoring an impaired FP" cannot be the whole story; Fractal FP is non-deterministic, S1 Fractal arm halves |
| 7 | A7 | medium | 0 (CPU done) + ~2 GPU-h | H4/H5 noise is under-sized (isotropic factor median 0.55); "max of 6 draws" is a weak design; M is +23 so far |
| 8 | A8 | medium | 0 (CPU seconds) | QTIP vs FP on GR00T: all three replications negative; pooled it is below FP |
| 9 | A9 | medium | ~0.5 h CPU | S3 refined: the official tune sweep is not a monotone optimiser on ill-conditioned H; test on a synthetic H |
| 10 | A10 | low | 0 | Text that contradicts data or other text (E7 numbers, G41 vs E2, README early termination, C2 wording, E3 status) |
| 11 | A11 | low-med | CPU minutes | Fractal: async 5-env runner, completion-order pairing, early-stop sampling bias |
| 12 | A12 | low-med | CPU minutes | Bridge put_eggplant_in_basket is non-deterministic under the same seed (24 flips); cause unknown |
| 13 | A13 | low-med | ~3.5 GPU-h | pi0.5 FP 765 is a reference file from another run (13 Oct), not re-run on our stack |
| 14 | A14 | low | ~2 GPU-h or 0 | Main-row choice by Plus 1000 successes (VPTQ A5, GPTQ G38): winner's curse |
| 15 | A15 | low | 0 | Bridge headline QuIP# row is the appendix configuration (0 sweeps) |
| 16 | A16 | low | closed | Bridge bootstrap: drawer and sink tasks are one cluster each (zero variance); effect checked, negligible |
| 17 | A17 | - | closed | S7 refuted: Plus 1000 is a seeded random draw fixed before GR00T |
| 18 | A18 | - | closed | S8 total match 618 = 618: balance of 64/64 flips has probability ~7%; not suspicious by itself |

---------------------------------------------------------------------------------------------------------------------------

## A1. The Bridge "2-bit above FP" rests on carrot_on_plate and stack_cube only; on the other 5 tasks QTIP is below FP
- File / number: RESULTS.md Bridge table, "+7.4 [+3.8, +11.0]" (QuIP# 0 sweeps), "+7.3" (VPTQ), "+3.3" (QTIP, not separated).
- Recomputed (same bootstrap as simpler/b5_stats.py, per-episode clusters for drawer/sink tasks, 4000 draws):
  without stack_cube and carrot_on_plate (5 tasks): QTIP -4.3 [-7.2, -1.8], QuIP# -0.5 [-3.3, +2.5], VPTQ -1.0 [-4.3, +2.3],
  AQLM -2.3 [-5.8, +1.3]. Without stack_cube only: QTIP -1.1 [-4.9, +2.6], QuIP# +4.0 [+0.7, +7.5], VPTQ +2.2 [-2.2, +6.5].
- Why suspicious: the headline is a mean over 7 tasks of which 2 carry all of it; one of the rows is significantly below FP
  on the other 5 tasks, as on LIBERO. RESULTS says "partly paid back on put_eggplant_in_basket" but not that QTIP is a
  significant loss elsewhere.
- Impact: high on wording ("2-bit above FP on Bridge" -> "2-bit above FP on 2 of 7 tasks; at or below FP on the rest").
- Cheapest test: none needed; check one number with b5_stats.py --tasks (5 tasks) and edit RESULTS.
- Outcomes: confirmed -> report per-task; the Bridge paradox is a 2-task paradox (narrows Track Q to carrot and stack).

## A2. 2-bit carrot successes are abnormally fast; the success rule counts an object that is still held
- Data (s5b/eval, env_steps of successful episodes): carrot_on_plate FP median 80; QTIP 40, QuIP# 32, VPTQ 44, AQLM 32.
  Episodes that only the row solves (row-only): median 32 / 28 / 28 / 32 env steps (7-8 policy calls), 56-73% end at
  <= 40 steps. Random-noise rows (nz06s1-s3) gain carrot episodes at median 64 / 84 / 114 steps (8-34% at <= 40).
  On episodes both solve, the shift is small (QuIP# -28, others -12 to 0). So the new 2-bit successes are a distinct, very
  fast kind of episode, and random noise does not make them. spoon: QuIP# median 28 vs FP 52.
- Code facts (official, unmodified): ManiSkill2_real2sim PutOnInSceneEnv.evaluate: success = src_on_target = xy within the
  target bbox, z offset <= 2 cm, and no contact of the source object with any actor OTHER than the target and the ROBOT
  LINKS. Grasp release is not required (is_src_obj_grasped / consecutive_grasp are computed but not used). sapien_env.get_done
  returns success -> terminated; MultiStepWrapper aggregates done with MAX over the 4 sub-steps; eval_simpler_bridge.py and
  NVIDIA's rollout_policy.py both count "any step". So a carrot held 0-2 cm above the plate for one sub-step ends the
  episode as a success. README and the eval docstring say "no early termination"; that is false (the env terminates).
  NVIDIA uses the same rule, so this does not bias FP-vs-NVIDIA, but it can reward a policy that moves fast and low.
- Impact: high (it can turn the Bridge carrot gain into a measurement artifact; refines S2/Q1).
- Cheapest test (GPU, ~2 GPU-h): rebuild the Bridge QuIP# 0-sweep checkpoint (s5bq/q was deleted; ~1 GPU-h incl. import),
  then replay ONLY the 37 QuIP#-only and the fast FP carrot episodes (seeds and layouts are in the json) with a new
  eval option that (a) logs per sub-step info: success, src_on_target, is_src_obj_grasped, consecutive_grasp, moved_wrong_obj,
  (b) wraps the env so it does not terminate on success, scores success at step 300 and "success held >= k sub-steps",
  (c) writes videos. Also a determinism check: the replayed outcome must equal the stored record.
- Outcomes: most fast row-only successes have is_src_obj_grasped=True at the success step or are not on target at step 300
  -> the carrot gain is a success-rule artifact; re-score all rows with "released and held" and report both. Released and
  stable -> real behaviour change; go to Q2 (action size, gripper timing).

## A3. Bridge FP is one deterministic numerical draw; the right floor was not measured
- File / number: RIGOUR H3 "FP again ... 809 of 840 identical", used to say "the gain comes from the quantized weights,
  not from run noise".
- Why suspicious: H3 shows the run is deterministic, not that FP's 36 / 75 are representative of the FP model. E0 on LIBERO
  shows that any tiny numerical change (kernel) flips 11-15% of units; on Bridge nothing like E0 was run. Random weight
  draws at 6% move stack_cube from 13 to 59 (nz06 s0-s4, nz02: 48, 32, 59, 46, 13, 45), so the per-model numerical spread
  on these two tasks can be very large. NVIDIA's 48 uses batch-5 inference (5 envs): another numerical draw.
- Impact: high (if a no-op numerical change moves FP stack_cube to 50-70, the "2-bit above FP" is a draw of numerics).
- Cheapest test (~2 GPU-h, 6 short tasks): FP only, carrot_on_plate and stack_cube, 120 episodes each, three no-op
  changes: (1) evaluation seed 8 (new noise seeds, same layouts), (2) SDPA math kernel or fp32 inference, (3) server fed
  a batch of 5 (same observation repeated, take element 0) to mimic n-envs 5 numerics.
- Outcomes: all FP variants within ~8 episodes of 36 / 75 -> FP is robustly low on our stack; the 2-bit gains stand
  against FP (still unexplained); B1 then decides the NVIDIA gap. Variants spread to >= 55 on stack or >= 90 on carrot ->
  use the mean of FP variants as the reference, add the spread as the Bridge floor (as E0), redo the B2 readings.

## A4. Bridge pairs are read without multiplicity control and without a between-model floor
- File / number: RIGOUR B2 readings and RESULTS ("AQLM below QuIP# (-4.5 [-8.6, -0.7]) and VPTQ (-4.4 [-8.6, -0.2])",
  "QuIP# above QTIP (+4.0 [+1.3, +6.8])"). LIBERO uses Holm + E0 floor (C2); Bridge uses neither.
- Normal approximation from the stored intervals over the 10 Bridge comparisons (4 vs FP, 6 row pairs): QuIP#-FP p~1e-4,
  VPTQ-FP p~1e-3 survive Holm; QuIP#-QTIP p~0.004 is borderline; AQLM-QuIP# p~0.03 and AQLM-VPTQ p~0.04 fail.
  The paired CI treats both models as fixed: one random draw (nz06s4) is "significant" at -16.4 [-22.2, -10.6] on 3 tasks,
  nz06s2 +5.6; so this CI cannot separate a quantizer effect from a lucky draw of the weights.
- Impact: medium-high (Bridge ranking sentences; "LIBERO ranking not reproduced" sentences).
- Cheapest test: CPU seconds: rerun b5_stats for all pairs, Holm over 10, add a floor from A3 or from the noise draws.
- Outcomes: AQLM orderings and QuIP# > QTIP drop to "not separated" unless the floor is small.

## A5. Re-running the same quantizer config gives a different model; there is no quantizer-draw floor
- File / number: G29 and G37 say the k-means stops on its tolerance after 25-30 iterations, "so 500 iterations change
  nothing". eval/vptq_iter500_w2 vs eval/vptq_w2: 487 vs 495 with 84 up / 92 down (17.6% of units flip). A deterministic
  checkpoint re-evaluated flips 0-1 units (fp_rerun, qtip_w2_rerun). So the two VPTQ checkpoints differ: either the k-means
  does not stop early on all layers, or the quantizer (cuml k-means, init='random', seed 0, GPU) is not deterministic.
  vptq_iter500t0 vs base: 75 / 99; v5k1024 vs base: 101 / 103.
- Why it matters: every row in every table is one quantization draw (RHT signs, k-means init, Adam). E0 covers evaluation
  numerics only. The VPTQ spread 471-495 (S5) and QTIP vs QuIP# 0 sweeps (+19) may be inside a quantizer-draw floor; on
  Bridge, each row's carrot/stack gain is one draw.
- Cheapest test (~3 GPU-h): re-quantize QTIP Bridge with a different RHT sign seed (same Hessians, ~1-1.5 GPU-h) and
  evaluate carrot_on_plate and stack_cube (~0.7 GPU-h); optional: same for LIBERO QTIP on the 200-unit sanity200 set.
  CPU first: if any VPTQ layer log survives (logs/ or scheduler.log), check the k-means iteration counts.
- Outcomes: re-seeded QTIP keeps carrot >= 87 and stack >= 62 -> the Bridge gain is a property of the method, not of one
  draw. It falls toward FP -> the Bridge gains are draw-level; report a quantizer floor next to E0; S5 closes as "inside
  the floor".

## A6. S1 refined: overshoot, and the Fractal arm is weaker than stated
- Bridge: 2-bit carrot 87-107 of 120 (73-89%) vs NVIDIA 58% and our FP 62.5%; stack 62-81 (52-68%) vs NVIDIA 48%.
  "Our FP is impaired and a weight change restores it" predicts values near NVIDIA, not far above it. Our FP is ABOVE
  NVIDIA on spoon (+6) and carrot (+4.5). So S1 can explain at most part of stack_cube.
- Fractal open_drawer: the FP is not deterministic (5 async envs): seed 7 gave 48, the same seed again 59 of 101, seed 8
  55, seed 9 51, flash-attn 50 -> mean 52.5. Rows are paired against the single 48 run. Against the mean: QuIP# 0 sweeps
  +0.5, AQLM and QuIP# 10 sweeps +7.5, QTIP +11.5, VPTQ +16.5, GPTQ best -14.5 (SE per run ~5). Only VPTQ and QTIP stay > 2 SE.
- Also: S1 "the prepared checkpoint equals FP" is confirmed for configs too: config.json, processor_config.json,
  statistics.json, embodiment_id.json have identical md5 in s5b/checkpoints/bridge and s5bq/checkpoints/bridge; config
  dtype bfloat16; Gr00tPolicy casts the whole model to bf16 for FP and rows alike (gr00t_policy.py). So no dtype or
  loader difference between FP and rows was found.
- Impact: medium. Cost 0 (CPU done). B1 (NVIDIA protocol) is still the test for the stack_cube gap itself.
- Outcome for the text: S1 stays a candidate only for stack_cube; carrot/spoon need A2/A3.

## A7. H4/H5 noise size and design
- File / number: RIGOUR H4 "for random noise of relative weight rms s this ratio is about s^2, so s = 0.06 matches".
- Measured on 144 Bridge layers (every 3rd of s5bq/layers; side file noise_ratio_factor.json): the expected
  tr(N H N^T) / tr(W H W^T) for isotropic noise is s^2 x factor, factor median 0.55 (p10 0.22, p90 1.05; visual MLP fc2 and
  attn.proj 0.13, language v_proj up to 2.9). So s = 0.06 gives about half the median 2-bit output error, and much less on
  the vision layers. In weight space the AQLM 2-bit error is 52% of weight rms (s5bq/perturb/aqlm_w2_error.json, median
  0.52), against 6% noise.
- Design: H5's M = max of 6 draws is a weak, threshold-sensitive test. Complete draws so far (s0-s4, s5 not read): stack
  +12, -4, +23, +10, -23 -> M = +23 vs threshold +26 (AQLM): one draw decides. A rank test is more informative: stack,
  the 4 rows (62, 69, 72, 81) are above all 6 draws incl. nz02 (13-59): p = 1/210; carrot, one inversion (draw 91 > AQLM
  87): p ~ 0.01. Noise draws do not shorten episodes (A2), the rows do.
- Impact: medium (cause of the Bridge gains, not the numbers). Cost: CPU done; optional GPU 2 draws at per-layer matched
  s_l = 0.06 / sqrt(factor_l), 3 tasks, ~2 GPU-h.
- Outcomes: matched-per-layer noise still does not raise both tasks -> "structured, not generic" gets stronger; it does
  -> fragility reading.

## A8. QTIP vs FP on GR00T LIBERO-Plus: every replication is negative
- Plus 1000 seed 7: -4.5 (clustered [+1.8, +7.2] for FP-QTIP, excluded by Holm only after adding E0 noise in quadrature);
  seed 8: 605 vs 564 = -4.1; disjoint 500: -6.6 [-12.5, -1.4]. Three of three negative, the independent one excludes 0.
- Impact: medium (the sentence "QTIP not separated from FP" survives only on the Plus 1000 under the strict rule).
- Cheapest test: CPU seconds, pool the disjoint 500 with the Plus 1000 (independent units) with the clustered SE.
- Outcome: pooled interval excludes 0 -> "QTIP is about 5 points below FP on GR00T LIBERO-Plus".

## A9. S3 refined: the sweep is not a monotone optimiser; the code matches official
- quip-sharp lib/algo/quip.py LDLQ_buffered tune: per 8-weight block, continuous optimum hatW_k + inv(H_kk) H_k: delta,
  then cb.quantize = nearest E8P12 codeword in EUCLIDEAN distance, not in the H_kk metric, from a bounded codebook. This
  block-coordinate step is not guaranteed to lower tr(E H E^T); with cond(H) ~7e4 (results.json: 148 of 4096 eigenvalues
  above the floor) inv(H_kk) amplifies, the target leaves the codebook range, and the loss can rise every sweep
  (loss_reg 0.0002 -> 0.0021, weight error 0.33 -> 1.19). On the layer with 2675 eigenvalues above the floor it is flat.
  Our formula matches the official one (checked by reading). So a port bug is unlikely.
- Cheapest test (CPU, ~0.5 h): run the official quip.quantize on one layer with (a) a synthetic full-rank H (X^T X, many
  samples) and (b) our H, sweeps 0..10; record loss_reg.
- Outcomes: loss falls on (a) and rises on (b) -> property of the method on low-rank H, S3 closed, footnote stands. Rises
  on (a) too -> bug in our call (scales, damping) -> fix and rerun QuIP# 10 sweeps (main row on both models).

## A10. Text that does not match data or other text (fix only)
- EXPLANATIONS E7: "the VPTQ variants (v6k3072, v8k12288) ... (535 and 495 against 495)": files give 458 and 426 (535 is
  v6k4096). Wrong numbers.
- DECISIONS G41 and MENTOR A5: "a larger codebook fits the 256 calibration observations more closely" vs EXPLANATIONS E2
  "(a) REFUTED: no overfitting". Contradiction; E2 is the tested one.
- groot/simpler/README.md and eval_simpler_bridge.py docstring: "no early termination" (see A2; the env terminates on
  success; NVIDIA uses terminate_on_success=True explicitly).
- RESULTS "every row except AQLM below FP by more than the floor": c2_groot_main.md lists AQLM-FP as real (-21.7).
- EXPLANATIONS E3 verdict "OPEN until those finish" but the component results are in the same block.
- RESULTS Fractal sentence "FP 48 on open_drawer": the FP is 48 to 59 over 5 runs (A6).
- Spot checks that MATCH the source files (12): Bridge per-task counts of all 6 columns and the 6 means; Bridge paired
  CIs (qtip, quip, vptq, aqlm, fprr, 6 pair files); H4 tables (nz06, nz02, nz06s1 up/down); Fractal family sums of 9 rows;
  GR00T Plus 1000 totals and camera/robot-init splits of 10 rows; R3 seed 8 (605/564/547); R7 (294/261/249); E0 reruns
  (618, 572 with 1 flip); C4 (382 failed, 64 mentor-solved, QTIP 72 gains, 31 on marginal = 48.4%); pi0.5 Plus 1000 of 8
  rows and FP 765 / original_2bit 704; A19 "33 of 65" on layouts 11-23; H2 corrected 30/55 and 6/65.

## A11. Fractal protocol: completion-order pairing and early-stop sampling
- The official runner (5 async envs) stops when >= 100 episodes have finished (fp_fprr: "101it"). Episodes still running
  at the stop are dropped; with early termination on success they are more often failures, so the rate is biased up a
  little, and more for a policy that succeeds faster. Pairing by completion order is not pairing by scene (S4); the FP
  itself is not deterministic (A6).
- Cheapest test: CPU (check whether the logs give per-episode seeds; if not, report unpaired intervals and the 5-run FP
  mean). Longer fix: a Fractal version of eval_simpler_bridge.py (1 env, seeded per episode).
- Impact: low-medium (Fractal "loses 5-9 points" sentences for VPTQ / QuIP# 10 sweeps are near the unpaired edge:
  -5.5 +- 5.2, -5.3 +- 5.2).

## A12. Bridge put_eggplant_in_basket is not deterministic under the same seed
- fprr vs fp: 24 of 120 episodes flip on eggplant_basket, 1 on open_drawer, 6 on close_drawer; 0 on the other 4 tasks.
  The cause is unknown (an unseeded RNG in that scene, or contact physics). The QTIP loss there (-29) is still beyond it.
- Cheapest test: CPU, read put_on_in_scene.py / the basket scene for unseeded RNG use; GPU only if needed.
- Impact: low-medium (Bridge loss sentences on this task; it also shows that "deterministic" holds per task, not per env).

## A13. pi0.5 FP is not run on our stack
- results/plus_episodes/original_fp.jsonl (765) dates from 13 Oct (committed with the harness, mentor's run); every pi0.5
  2-bit row ran later on our stack. E0 (GR00T) shows a stack change flips 13% of units with a net change of a few units.
  Decision 18's identity / FP closed-loop check (exact) on 32 observations reduces the risk.
- Cheapest test: ~3.5 GPU-h, pi0.5 FP Plus 1000 on our GPU (fp_w16 has only the added 1000: 736/1000).
- Impact: low-medium (every pi0.5 "vs FP" number by a few points at most).

## A14. Main rows chosen on the evaluation set
- VPTQ main row = most Plus 1000 successes among 6 in-budget variants (A5 in MENTOR_PENDING); GPTQ best also named by
  successes (G38; A12 says this reason will be dropped). With A5's quantizer-draw noise the max of 6 is biased up by about
  10-15 successes.
- Cheapest test: report the in-budget mean, or evaluate vptq_w2 on the disjoint 500 (~2 GPU-h). Impact low.

## A15. The Bridge headline QuIP# row is the appendix configuration
- Bridge ran QuIP# 0 sweeps; the main QuIP# row (10 sweeps, mentor's choice) was not run on Bridge. The "+7.4" sentence
  must name the configuration. Impact low (wording); ~3 GPU-h if the mentor wants the main row.

## A16 (closed). Bridge bootstrap clusters
- b5_stats.py clusters by layout; the drawer tasks (layout None) and the sink (1 layout) are one cluster each, so they add
  no variance. Recomputed with per-episode clusters: upper bounds move by at most +0.25 pp. Negligible.

## A17 (closed). S7: Plus 1000 selection
- data/plus_units_selection_1000.json: "Preregistered ... 500 Camera Viewpoints + 500 Robot Initial States, seed 7"; shared
  with pi0.5 and fixed before the GR00T rows. R7 shows a slightly larger QTIP gap out of sample, inside noise. No bias found.

## A18 (closed). S8 total match
- Our FP and the mentor's FP: 1000 shared units, 128 flips, 64 each way. Exact balance has probability C(128,64)/2^128 ~
  0.07. Not suspicious alone. The Bridge-vs-LIBERO numerics question is A3.

## Mapping to GOAL v7 seeds
S1 -> A6 (refined: overshoot; Fractal arm weaker; no dtype/loader/config difference found) + A3. S2 -> A2 (refined: rule
allows held objects; fast-success signature found in stored data; terminate-on-success confirmed). S3 -> A9 (refined:
mechanism found, port matches). S4 -> A11. S5 -> A5 (likely inside a quantizer-draw floor). S6 -> not raised further (AQLM
spoon losses are spread over layouts: L0-10 26, L11-23 29 of 55/65 FP 51/50). S7 -> A17 closed. S8 -> A18 closed, A3.
