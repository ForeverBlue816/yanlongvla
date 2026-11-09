# GOAL v8 Part I (L2): items A15, A14, A5/S5, X3 caveat, S8, open C0 items (9 Nov 2026, CPU only)

Scripts and outputs in this folder: part_i.py (part_i_out.txt), a14_a5.py (a14_a5_out.txt), x3_diverge.py (x3_out.txt),
x3_diverge_quip.py, x3_prev.py (x3_prev_out.txt), bridge_table.txt (all Bridge rows, successes per task, flips against FP).
No job was submitted. No result document was edited.

Shared numbers. Evaluation noise of one count: sigma_run = 9.3 successes per 1000 units on GR00T LIBERO
(from 167 to 185 unit flips between eval seeds 7 and 8, sigma = sqrt(flips/2)). Noise of a difference of two runs: 13.2.

---------------------------------------------------------------------------------------------------------------------

## 1. A15: Bridge QuIP# 10 sweeps. Status: test needed (no data on disk)

What exists.
- s5b/eval has no QuIP# 10-sweeps row. Bridge rows with QuIP#: quip_w2 (0 sweeps), quipsl, quipnt (both rebuilds of 0 sweeps), mx*, sh95, orth and others (all built from 0 sweeps).
- s5bq/paired has no 10-sweeps file. s5bq/q is empty. q/quip10_w2 holds only LIBERO layers (backbone, 4 libero heads) and no Bridge head.
- tasks/ has no bq__*quip10_w2 file. So b5_stats.py has nothing to compare. No number can be given.
- Bridge 0 sweeps for reference (of 120): spoon 107, carrot 107, stack 69, basket 41, sink 6, open_drawer 117, close_drawer 118. FP: 101, 75, 36, 56, 0, 120, 115. Mean 67.3% against 59.9%: +7.4 [+3.8, +11.0] (s5bq/paired/quip_w2.json).
- LIBERO prior for the sign: on GR00T Plus 1000, 10 sweeps is 531 and 0 sweeps is 554 (-2.3 pp, inside the 2 pp floor plus noise; not separated). On pi0.5, 10 sweeps is 662 and 0 sweeps 740.

The test.
- Everything needed is on disk: s5bq/layers (weights and Hessians, 432 layers), s5bq/calibration, s5bq/checkpoints/bridge, s5bq/ckpt/quip_w2 (6.5 GB).
- Add row quip10_w2 to BQ_ROWS in groot/jobs/gsched.py (feeder edit, not done here). groot.pbs already gives 10 sweeps to every quip row except quip_w2 (line 107: --tune-iters 0 only for quip_w2).
- New tasks: bq__qk__quip10_w2__backbone (about 3.0 GPU-h), bq__qk__quip10_w2__head_bridge (1.5), bq__imp__quip10_w2 (0.3), then b5__ev__quip10_w2__widowx_<task> for the 7 tasks (0.6 each, spoon is the gate), bq__fin__quip10_w2.
- Cost, all 7 tasks: about 9 GPU-h (wall time about 5 h because backbone and head run in parallel). Carrot and stack only, plus the spoon gate: about 6.6 GPU-h. Sweeps may make the quantizer slower than the X1 estimate; not measured.
- Then run: python simpler/b5_stats.py --fp s5b/eval/fp --row s5b/eval/quip10_w2 --out s5bq/paired/quip10_w2.json, and the same with --fp s5b/eval/quip_w2 for 10 against 0 sweeps. Use /home/d250009/projects/pi05-handover/runs/envs/groot/bin/python.
- Prediction to write down first: on LIBERO the sweeps raise the weight error (0.33 to 1.19 on one layer, X8). A larger error may push the carrot and stack gain up or down; H8 (shrink) does not say which.

Draft RESULTS sentence (valid today, with the number left open).
"The Bridge QuIP# row (+7.4 [+3.8, +11.0] against FP) is QuIP# without tuning sweeps, the appendix configuration of the LIBERO tables; the official 10-sweeps configuration was not run on Bridge (on GR00T LIBERO it is 2.3 points below the 0-sweeps row, not separated), so the Bridge QuIP# result must not be read as the official default."
When the test is done, add: "10 sweeps: <mean> [<CI>] against FP, <diff> [<CI>] against 0 sweeps."

---------------------------------------------------------------------------------------------------------------------

## 2. A14: winner's curse. Status: answered by estimate; one direct test is optional

Which main rows were chosen on the evaluation set (Plus 1000)?
- VPTQ: the most successes among 6 in-budget variants (495, 493, 487, 471, 458, 426; mean 471.7, sd 26.5; the over-budget v6k4096 with 535 is excluded). DECISIONS G41.
- GPTQ best: the row is chosen per layer by calibration error (G38), so the row itself is not chosen on the evaluation set. Only its naming as "main" used the count (max of 4 comparable rows: 476, 454, 439, and mse 300).
- QTIP: one configuration. QuIP#: the official 10 sweeps is the main row (mentor's choice), 0 sweeps in the appendix. AQLM: one configuration. No selection by count.

Estimate (E[max of k] minus mean = c_k x sigma_run, with sigma_run = 9.3).
- VPTQ, k = 6: 1.27 x 9.3 = 11.7 successes = 1.2 pp. This is a lower bound: it counts only evaluation noise, not quantizer-draw noise (item 3).
- Upper bound from the data: chosen row minus mean of the 6 = 23.3 successes = 2.3 pp. This holds if the 6 variants were equal in truth.
- GPTQ best, k = 4: 9.5 successes = 1.0 pp (if the count choice mattered at all).
- So the table values of VPTQ (495) and GPTQ best (476) are inflated by about 1 pp (up to 2 pp). Corrected VPTQ: about 483 (-13.5 pp against FP instead of -12.3).
- No ordering changes. Both rows are below QTIP and QuIP# in the stored C2 readings. The correction makes them lower, so the sentence "VPTQ and GPTQ best are below FP" gets stronger and no "above" sentence depends on them.
- The correction does not apply to the QTIP, QuIP# and AQLM rows.

Bridge: a second selection effect, in the reference and not in the rows.
- The 7 Nov seed-7 FP (the mentor's protocol) is a low draw on stack_cube. 13 FP runs with no-op numerics or other seeds give stack_cube 41.7 +- 3.7 (min 35, max 47; FP seed 7 = 36, 5.7 below the mean). Carrot: 75.5 +- 3.9 over 4 runs (FP 75).
- So the stack gain is overstated by 5.7 episodes (4.8 pp on the task, 0.7 pp on the 7-task mean). QuIP# +7.4 becomes about +6.7. It stays separated from zero (the interval half-width is 3.6).
- The Bridge rows were not picked among variants on Bridge. The 0-sweeps choice was made from LIBERO and pi0.5 results, a different evaluation set, so no selection bias on Bridge.

Draft RESULTS sentence.
"Two main rows were chosen on the evaluation set (VPTQ: the best of 6 in-budget variants; GPTQ best is chosen by calibration error and only labelled by count). With an evaluation noise of 9 successes per run, the expected gain of taking the maximum of 6 is 1.2 points (upper bound 2.3, from the spread of the 6 variants), so the VPTQ row is about 1 to 2 points too high and its gap to FP is -13 to -14 points; QTIP, QuIP# and AQLM were not chosen among variants. On Bridge the seed-7 FP is 5.7 stack_cube episodes below the mean of 13 FP draws, so the Bridge gain over FP is about 0.7 points smaller on the 7-task mean (+6.7 for QuIP#)."

Optional direct test (GPU, about 2 GPU-h): VPTQ on the disjoint 500 units, FP reference 294 exists. Problem: q/vptq_w2 was deleted, so VPTQ must be re-quantized first (backbone about 3 GPU-h, 4 heads; not worth it for a 1 pp effect). Do it together with item 3 (the VPTQ re-quantization there serves both).

---------------------------------------------------------------------------------------------------------------------

## 3. A5 and S5: quantizer-seed floor. Status: S5 answered from existing data; A5 test needed

Is an existing pair a re-quantization with another seed? No.
- All quantizers fix the seed: quantize_qtip.py:51 and quantize_quip.py:59 torch.manual_seed(0) before each layer (this seed gives the RHT signs); quantize_aqlm.py:70-71 random.seed(0) and torch.manual_seed(0); quantize_vptq.py kmeans seed=0, init random.
- Closest pairs, none of them a seed change:
  - X1 rebuild of QuIP# 0 sweeps on Bridge (same seed 0, same Hessians): quipsl equals the stored quip_w2 on 120 of 120 episodes on carrot and stack (EXPLORE X1/X2). So the QuIP# quantizer is deterministic for a fixed seed. This shows determinism, not a floor.
  - vptq_iter500 against vptq_w2 (same seed 0, same structure, 500 against 100 iterations): 487 against 495, 84 up and 92 down. The k-means should stop on tolerance (G37) and give the same model, but the checkpoints differ. Either it does not stop early on every layer, or cuML is not deterministic. Unresolved; no per-layer iteration log survives (checked experiments/vptq_*: no iteration counts).
  - vptq_v5k1024 against vptq_w2: a different structure.
  - rr__qtip_w2 and the seed-8 rows: a rebuild of the same q/ files, not a re-quantization.

S5 from existing data (VPTQ spread against the E0 floor; Plus 1000, flips against vptq_w2 from a14_a5_out.txt).
| row | count | net | flips per 100 | net in units of sigma_diff (13.2) |
|---|---|---|---|---|
| vptq_v5k1024 | 493 | -2 | 20.4 | 0.2 |
| vptq_iter500 | 487 | -8 | 17.6 | 0.6 |
| vptq_iter500t0 | 471 | -24 | 17.4 | 1.8 |
| vptq_v6k3072 | 458 | -37 | 19.5 | 2.8 |
| vptq_v8k12288 | 426 | -69 | 20.7 | 5.2 |
- The three same-structure rows (495, 493, 487; sd 4.2) are inside the evaluation floor (expected sd 9.3 from evaluation noise alone). They cannot be told apart.
- The unit flips (17 to 21 per 100) equal those of a pure evaluation-seed change (16.7 to 18.5 per 100). So a tiny change of the quantizer moves as many units as a new evaluation seed.
- Chi-square of the six counts against evaluation noise alone: 40.8 on 5 degrees of freedom (p < 1e-6), but the variants differ in structure on purpose. The two large-codebook rows (458, 426) and iter500t0 (471, at 1.8 sigma) are below the base. "Lower layer error, fewer successes" is beyond evaluation noise for v8k12288 (5.2 sigma) and v6k3072 (2.8 sigma). Quantizer-draw noise is not measured, so it cannot be excluded for v6k3072.

The A5 test (Bridge, where the gain claims are; GPU).
- Needs a code change first (not done here): add a seed argument to quantize_quip.py line 59 (and quantize_qtip.py line 51), default 0, so old rows stay bit-identical. Add row names quipsd1_w2 and quipsd2_w2 to the feeder (BQ_ROWS style, with the seed passed through groot.pbs).
- Rows: QuIP# 0 sweeps, seeds 1 and 2, plus the existing seed 0. Three draws give a standard deviation; one pair gives only one difference.
- Inputs on disk: s5bq/layers (432 layers with Hessians), s5bq/calibration. Second calibration draw, optional: s5bq/calibration_heldout holds 256 other observations (noise seed 1). Rebuild layers with it using groot/export_layers.py (disk 18 GB, minutes). Cheaper than a seed change in code, and it also tests the calibration draw.
- Per draw: bq__qk backbone 3.0 + head_bridge 1.5 (parallel) + bq__imp 0.3 + carrot and stack 1.2 = 6.0 GPU-h. For 2 new draws: 12 GPU-h (about 5 h wall time). Add the 5 other tasks (3.0 per draw) only if the carrot and stack spread is large.
- Reading to fix in advance: carrot stays at or above 87 and stack at or above 62 for both new draws -> the Bridge gain is a property of the method. A draw falls toward FP (carrot below 80 or stack below 50) -> the gains are draw-level and the quantizer floor goes next to E0.
- VPTQ variant (for S5 proper, cuML determinism): re-quantize vptq_w2 on Bridge with the same settings twice (kmeans seed 0, then 1). 6.0 GPU-h each. First check, free: if the same-seed re-run reproduces the stored vptq_w2 episodes exactly, cuML is deterministic and the iter500 difference is a real configuration effect.
- Optional LIBERO arm: not recommended (backbone 3.0 + four heads of 1.5 each + eval 3.5 = about 12 GPU-h per draw).

---------------------------------------------------------------------------------------------------------------------

## 4. X3 caveat: where fpnt and fpsl diverge. Status: answered from step logs; one cheap GPU test confirms the cause

Method: x3_diverge.py compares per episode the executed action and the source-object position at each step, fpsl (stored seeded run) against fpnt (same seeds, --no-env-termination), over the common prefix.
Result (x3_out.txt):
- carrot_on_plate: 27 of 120 episodes diverge, 93 are bit-identical over the common prefix. stack_cube: 7 of 120 diverge. With QuIP# (quipnt against quipsl): 41 and 15 of 120.
- Episodes 0 and 1 never diverge in any run.
- In 24 of the 34 episodes (carrot and stack, FP rows) the executed action differs first, by 2e-4 to 1e-2, while the object position was still equal. The step is 4 in 15 episodes and 8 or 12 in the others (a multiple of 4: new policy call; the policy executes 4 steps per call). The first policy call is always bit-identical.
- In the other 10 episodes the executed action is equal at the first difference and the object position differs. Eight of them first differ at a step that is not a multiple of 4 (6, 9, 11, 13, 14), so the simulator diverged without any policy change.
- Divergence depends on how the previous episode ended (x3_prev.py). carrot: 23 of 74 episodes diverge when episode k-1 ended early in fpsl (and ran to 300 steps in fpnt), against 4 of 45 when it ran 300 steps in both (Fisher p = 0.006). stack_cube: 6 of 36 against 1 of 83 (p = 0.003). The divergent episodes come in runs (carrot 2 to 5, 13 to 16, 31 and 32, 37 and 38, 74 and 75, 86 and 87, 95 and 96).

Cause, from code (groot/simpler/eval_simpler_bridge.py, run_task, lines 138 to 160).
- One env object and one policy server serve all 120 episodes of a task. Per episode the script re-seeds np.random, random, the ManiSkill reset seed, and (in simpler_server.py:34-37) torch.manual_seed and torch.cuda.manual_seed_all.
- It does not reset anything else: the SAPIEN/PhysX scene objects, the contact and solver warm-start state, the renderer, and the CUDA allocator and cuBLAS workspace state of the server carry over from the previous episode.
- With termination, the previous episode stops at the first success. With --no-env-termination (step wrapper, lines 78-80: done = False) it runs 300 steps and ends in a different physical state. The carried state differs, so episode k starts from a numerically different hidden state.
- Evidence for a simulator part: 10 of 34 episodes diverge first in the object pose with equal actions (8 mid-chunk). Evidence for a history effect: the dependence on the previous episode, the runs of consecutive episodes, and the fact that episodes 0 and 1 never diverge.
- The same history effect explains A12 (basket, open_drawer, close_drawer not reproducible, contact-rich scenes) and shows fpsl is only reproducible against fp because both end each episode at the first success.
- What is not decided: for the 24 episodes that diverge first at a policy call, whether the observation (render, proprio) or the model numerics differ. The step logs hold no observation hash.

Smallest test (GPU, about 1 GPU-h).
1. Add a read-only hash of the image and the 8-d state to the step log (same place as executed_action, lines 69-77 of the script).
2. Run carrot_on_plate episodes 0 to 39 three times with the same seeds and the new logging: fpsl2 (terminate at success), fpnt2 and fpnt3 (--no-env-termination). Use --episodes 40, --step-log and the existing eval_simpler_bridge.py options; feeder names like b5__ev__fpnt2__widowx_carrot_on_plate (same pattern as b5__ev__fpnt__...). Each run is 0.3 to 0.4 GPU-h.
3. Reading: fpnt2 and fpnt3 identical to each other but different from fpsl2/fpsl -> history dependence, deterministic (cause as above). They differ from each other -> plain nondeterminism in sim or render. First differing hash at a policy-call step -> observation side; action differs with equal hashes -> model side (server allocator state).
Fix if confirmed: score X3 only within fpnt against quipnt (already the rule in EXPLORE), and say that "not paired with B1/B2" is a property of the history, not a bug.

---------------------------------------------------------------------------------------------------------------------

## 5. S8: the same no-op numeric change on SimplerEnv and LIBERO. Status: partly answered; LIBERO flash test needed

Existing pairs (unit or episode flips against the same-seed reference; part_i_out.txt).
| setting | change | n | up | down | flips per 100 |
|---|---|---|---|---|---|
| Bridge carrot_on_plate | flash-attn 2.8.3 instead of SDPA, seed 7 (fpfa against fp) | 120 | 13 | 16 | 24.2 |
| Bridge stack_cube | same (fpfa against fp) | 120 | 15 | 8 | 19.2 |
| Bridge, both tasks | same | 240 | 28 | 24 | 21.7 (se 2.7) |
| Bridge, same stack | rerun (fprr, fpsl) | 240 | 0 | 0 | 0 |
| LIBERO FP | math SDPA kernel | 200 | 10 | 13 | 11.5 |
| LIBERO FP | efficient SDPA kernel | 200 | 14 | 14 | 14.0 |
| LIBERO FP | other GPU and software (mentor's FP) | 1000 | 64 | 64 | 12.8 |
| LIBERO FP / QTIP / QuIP# | evaluation seed 8 (not a no-op) | 1000 | 77 to 88 | 88 to 97 | 16.7 / 18.5 / 16.9 |
| Bridge stack_cube | evaluation seed 8 (fps8) and seeds 9 to 11 (not a no-op) | 120 each | | | 26.7, 32.5, 30.0, 25.8 |
| Bridge carrot | evaluation seed 8 (fps8) | 120 | 18 | 19 | 30.8 |
- Net change: Bridge flash-attn -3 (carrot) and +7 (stack) of 120; LIBERO -3 to +0 of 200. Both are near zero on average; the flips are two-sided.
- Reading: a no-op numeric change flips about 22 per 100 episodes on Bridge and about 12 to 14 per 100 on LIBERO. Ratio about 1.7. Pooled LIBERO kernel swaps: 51 of 400 = 12.8% (se 1.7) against 21.7% (se 2.7): difference 9 points, about 2.8 se. Seed changes show the same ratio (Bridge 26 to 33, LIBERO 17 to 19).
- Caveats. Not like for like: the Bridge change is the flash-attn package and the LIBERO changes are torch SDPA kernels. The tasks differ (Bridge FP stack 30%, carrot 62%; LIBERO FP 62%). flash-attn has not been run on LIBERO. Not run: Bridge with the SDPA math or efficient kernel.
- Why it is plausible: Bridge episodes are 75 policy calls with contact-rich grasps, and X3/A12 show the simulator itself is history-dependent. Not tested as a cause.

The test to close it (GPU, about 0.9 GPU-h).
- LIBERO, FP, sanity200 (the same 200 units as the kernel runs): the torch flash SDPA kernel. The feeder task kn__flash does not exist. groot.pbs t_kn already accepts any value of SDPA_BACKEND (serve.py set_sdpa_backend supports math, efficient, flash), so only a line in gsched.py is needed: for kb in ('math','efficient','flash'). Cost 0.9 GPU-h. Compare flips against fp/plus_1000 on the same units.
- Optional like-for-like arm: the flash-attn package on LIBERO needs the groot serve path to take runs/envs/flash-attn; not verified that serve.py can use it. Prefer the first test.
- Reading to fix in advance: LIBERO flash flips at or above 20 per 100 -> the Bridge rate is a property of the numerics change, not of the benchmark. At 11 to 15 -> the benchmark (Bridge episodes) amplify the same change.

---------------------------------------------------------------------------------------------------------------------

## 6. C0 audit items not yet closed (explore/c0/AUDIT.md, 18 items)

Closed (evidence in the documents): A1 (RESULTS: per-task wording, 5-task Holm numbers), A2 (X2, X3: gain is not a first-success artifact), A3 (H7, X19, X20: FP numerics floor measured, 41.7 +- 3.7 on stack), A4 (RESULTS: Holm over 10 pairs), A6 (RESULTS Fractal text, FP mean 52.4), A8 (RESULTS: pooled -5.2 [-7.9, -2.5]), A9 (X8: rise comes from the low-rank Hessian), A10 (E7 numbers, G41 note in DECISIONS line 202, README line 57 and the docstring corrected, C2 wording, E3 annotated), A16, A17, A18 (closed in the audit).

Not closed:
| id | status | what remains |
|---|---|---|
| A5 | test needed | item 3: seed argument, 2 new QuIP# Bridge draws, 12 GPU-h |
| A7 | caveat written (RIGOUR H5 line 289), optional test not run | 2 noise draws at per-layer s_l = 0.06/sqrt(factor_l), 3 tasks, about 2 GPU-h; side file c0/noise_ratio_factor.json holds the factors |
| A11 | partly answered (l1 report) | scene pairing is impossible from stored data (only the first reset is seeded). Stratified unpaired intervals computed, no verdict changes. Early-stop bias stays. Fix = a seeded 1-env Fractal runner, about 1 GPU-h per row family |
| A12 | cause not proven (l1 report) | item 4 now gives a history-dependence reading. Test: replay a fixed action list 5 times on a fresh env and compare the eggplant pose; needs a GPU node for rendering, minutes. Could be merged with the item 4 test |
| A13 | test needed | pi0.5 FP on our stack on the Plus 1000: about 3.5 GPU-h. Note: our own FP run on the added 1000 units is 736/1000 and no same-unit pair exists (FP_REPRO) |
| A14 | answered by estimate (item 2) | RESULTS sentence drafted; optional direct test needs VPTQ re-quantization |
| A15 | test needed (item 1) | about 9 GPU-h, RESULTS sentence drafted |
| X3 caveat | answered, one test for the last doubt (item 4) | about 1 GPU-h |
| S8 | partly answered (item 5) | kn__flash on sanity200, 0.9 GPU-h |
