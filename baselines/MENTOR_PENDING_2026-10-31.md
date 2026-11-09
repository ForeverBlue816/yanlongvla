# For the mentor: answers, assumptions and results (prepared 31 Oct 2026, updated by the loop)

## 1. What you asked for
| item | where it is | status |
|---|---|---|
| (a) per-layer output error of QTIP and both QuIP# rows (GR00T) | `experiments/layer_errors_tables/per_layer_output_error.csv` and `summary.md` in the GR00T run folder (copied here when done) | queued: xp__layererr |
| (b) layers that needed the 10x / 100x damping | only `backbone.model.model.language_model.layers.2.mlp.down_proj`, sigma_reg 0.1 (10x) in BOTH QuIP# rows (0 sweeps and 10 sweeps); no layer needed 100x; no layer of the four action heads needed a retry. Recorded per layer in `damping_retries.json` and, from now on, in `quant.json` (`damping_by_layer`) | done |
| (c) per-unit Plus 1000 files of QTIP and both QuIP# rows | `baselines/groot/data/our_rows/<row>.plus_1000.jsonl` (qtip_w2, quip_w2 = 0 sweeps, quip10_w2 = 10 sweeps; also fp, vptq_w2, vptq_v5k1024_w2, aqlm_w2, gptq_w2, gptq_mse_w2) | done (local commit, not pushed) |
| GPTQ audit, 4 bits, retries (two-day time box to 2 Nov 23:00) | DECISIONS.md G28, G32, G33; RESULTS.md | audit and 4 bits done; rows running |
| AWQ stopped on both models | RESULTS.md (GR00T: sanity gate 0/50; pi0.5: 154/1000 = 15%, "designed for 3 to 4 bits, collapsed at 2 bits") | done |
| VPTQ: 500 iterations, v6 k4096, better row = main | rows vptq_iter500_w2, vptq_iter500t0_w2, vptq_v6k4096_w2 and the fine grid | queued |
| AQLM notes | RESULTS.md and DECISIONS.md: the official layer-wise AQLM procedure (code search plus Adam on codebooks and scales, alternating until the relative error improves by less than 1%, at most 1000 epochs); no block-wise or end-to-end fine-tuning | done |
| harness: damping per layer in quant.json; SDPA note in the README | import_external.py, README.md | done |

## 2. Assumptions made (to confirm or to redo)
| # | assumption | why | what depends on it |
|---|---|---|---|
| A1 | QuIP# main row = the official 10 sweeps on both models, 0 sweeps in the appendix (your point 1 of 31 Oct; point 2 of the same message said 0 sweeps main for GR00T) | point 1 was explicit ("on both") and newer | RESULTS.md ordering only; no run depends on it |
| A2 | AWQ: no further run on either model; the GR00T row stays "stopped by the sanity gate (0/50)" instead of a full Plus 1000 | your point 3: report it as collapsed | nothing to run |
| A3 | "Rerun VPTQ with 500 k-means iterations" is run twice: the plain setting (vptq_iter500_w2) and with the k-means tolerance 0 (vptq_iter500t0_w2) | the layer test shows the k-means stops on its tolerance (1e-5) before 100 iterations, so the plain 500 changes nothing; tolerance 0 is the only way to test more iterations | if you meant only the plain one, ignore the t0 row |
| A4 | GPTQ "more damping": grid 0.03 to 1.0 with and without MSE and act-order; plus a row gptq_best_w2 that picks per layer the variant with the smallest calibration output error (uses only the calibration Hessian) | act-order alone helps most layers but collapsed in the gate because of one outlier layer; a per-layer choice avoids that without touching evaluation data | gptq_best row; payload counts a 16-bit column index only for the layers that chose act-order |
| A5 | VPTQ main row rule: among rows with total linear bpw at most 2.02 (over all 432 layers: backbone and one action head), the most Plus 1000 successes; the rest go to the appendix with the configuration | your point 4 | the VPTQ line of the main table |
| A6 | A centroid count that is not a power of two is counted at log2 k bits per index (mixed-radix packing across vectors) | lets the grid between 2048 and 4096 centroids be tested with honest bits | rows with k not a power of two |
| A7 | v6 k4096 has 12 bits per 6 weights = 2.0 bits plus the codebook; per-layer codebooks (4096 x 6 FP16 = 48 KB) total 0.07 bpw over the GR00T layers, so the linear bpw is 2.072 > 2.02 (the 48 KB is small per layer but the GR00T layers are small: a 1024x1024 layer has 1M weights) | counting rule from your message | reported as over budget; appendix unless it wins by a clear margin and you accept 2.07 |
| A8 | The two days for GPTQ end 2 Nov 23:00 | your message arrived 31 Oct late evening | no new GPTQ rows after that |

## 3. Results of the experiments (filled in as they finish)
(See DECISIONS.md G28 to G34 for the finished ones. Below: the new ones.)

### GPTQ (time box to 2 Nov 23:00)
| row | linear bpw | gate | Plus 1000 | vs FP | vs 486 | vs B2 |
|---|---|---|---|---|---|---|
| plain (percdamp 0.01) | 2.141 | 0/50 | 0 | -61.8 | -48.6 | -55.6 |
| MSE clipping | 2.141 | 22/50 | 300 | -31.8 | -18.6 | see RESULTS.md |
| act-order | 2.148 | 0/50 | stopped | | | |
| act-order + MSE | 2.148 | 23/50 | 454 | -16.4 (-22.0, -10.9) | -3.2 (-7.5, +1.1) | -10.2 (-15.9, -4.3) |
Audit (G28): our GPTQ equals the official IST-DASLab/gptq to four digits at 4 and 2 bits on six layers; error falls about 5x per extra bit. Act-order alone collapsed in the gate although it lowers the mean layer error (one outlier layer, language layer 2 down_proj, gets 4x worse); a per-layer choice (row gptq_best) and the combo with 10% damping are running to test that.

### A3 follow-up (1 Nov 00:00): tolerance 0 is not possible
The cuml k-means used by VPTQ refuses a tolerance of 0 ("invalid parameter (tol<=0)"; both the first `vptq_iter500t0_w2` jobs and the grid variant crashed). The row `vptq_iter500t0_w2` therefore uses a tolerance of 1e-30, which practically never stops the k-means early. The plain row `vptq_iter500_w2` keeps the default tolerance 1e-5 (the layer test showed the k-means stops after 25 to 30 iterations on these layers). Nothing was lost: both rows were requeued with the fix.

### VPTQ fine grid (layer test, 1 Nov; `experiments/vptq_grid_backbone` and `vptq_grid_head` in the GR00T run folder)
Output error on the calibration Hessian, 8 backbone and 5 head layers; total linear bpw over the 432 layers of backbone + one head (codebooks counted, k not a power of two counted at log2 k bits, assumption A6):

| structure | linear bpw | within 2.02? | vision fc2 (base 0.0552) | vision qkv (0.0121) | LM up_proj (0.0074) | head ff (0.0041) | head to_k (0.0262) |
|---|---|---|---|---|---|---|---|
| v4 k256 (the row) | 2.0030 | yes | 0.0552 | 0.0121 | 0.0074 | 0.0041 | 0.0262 |
| v5 k1024 | 2.0174 | yes | 0.0436 | 0.0110 | 0.0070 | 0.0019 | 0.0226 |
| v6 k3072 | 1.9865 | yes | 0.0364 | 0.0100 | 0.0073 | 0.0017 | 0.0228 |
| v6 k3584 | 2.0326 | no | 0.0334 | 0.0092 | 0.0069 | 0.0015 | 0.0217 |
| v6 k4096 (mentor) | 2.0738 | no | 0.0310 | 0.0086 | 0.0066 | 0.0013 | 0.0204 |
| v8 k12288 | 1.9884 | yes | 0.0290 | 0.0085 | 0.0097 | 0.0020 | 0.0299 |
| v7 k8192 | 2.0296 | no | 0.0285 | 0.0080 | 0.0079 | 0.0016 | 0.0243 |
Reading: inside the 2.02 limit the structures that lower the worst layers most are v6 k3072 and v8 k12288 (the second is worse on some layers: up_proj +31%, to_k +14%); v6 k4096 is better everywhere than v4 k256 but costs 2.074 bpw. The success of v5 k1024 (493) was level with the base row (495) despite 10 to 40% lower layer error (G34), so a layer-error gain may not become a success gain. Rows added for the two new in-budget structures: `vptq_v6k3072_w2`, `vptq_v8k12288_w2`; with `vptq_v6k4096_w2` (over budget) they decide the VPTQ main row (A5).

### GPTQ damping grid (`experiments/gptq_grid`, 432 layers, mean output error; base 0.0247)
More damping does not help: d0.03 0.0231, d0.1 0.0231, d0.3 0.0251, d1.0 0.0303 (worse than base on 316 of 432 layers at 0.3). The gains come from MSE clipping (0.0129) and act-order (0.0166), together 0.0090; adding damping 0.03 / 0.1 / 0.3 / 1.0 to act-order + MSE gives 0.0093 / 0.0102 / 0.0119 / 0.0154. The per-layer best over all 20 variants is 0.0087 (a 3% gain over act-order + MSE), so the row gptq_best_w2 is not expected to beat the act-order + MSE row (454) by much; it removes the one layer (language layer 2 down_proj) that act-order + MSE makes 40x worse. Answer to "more damping": it does not help on these Hessians.

### (a) and (b) are done: per-layer output errors of QTIP and both QuIP# rows (1 Nov 02:00)
Files: `groot/data/layer_errors/summary.md` and `damping_retries.json` (small; the full per-layer CSV, 432 layers x three rows, is `per_layer_output_error.csv` in the GR00T run folder `experiments/layer_errors_tables/`, 150 KB).
- Mean relative output error over the layers of the backbone / of an action head: QTIP 0.0046 / 0.0035 to 0.0039; QuIP# 0 sweeps 0.0062 / 0.0047 to 0.0053; QuIP# 10 sweeps 0.0074 / 0.0054 to 0.0059. The order of the rows' errors (QTIP < QuIP# 0 sweeps < QuIP# 10 sweeps) is the order of their Plus 1000 successes (573 > 554 > 531).
- The worst layers are not the outlier layer: in the backbone the SigLIP-style visual blocks (visual.blocks.11.mlp.linear_fc1 and the qkv of blocks 17 to 20, 0.013 to 0.020), and in every action head the attn1.to_v of the DiT blocks 0, 4, 8, 12, 16, 20 (0.017 to 0.027). With 10 sweeps two language-model layers join the worst five: layers.13.self_attn.o_proj 0.026 and layers.15.self_attn.v_proj 0.023.
- The layer you suspected, language_model.layers.2.mlp.down_proj (the only layer that needed a retry, sigma_reg 0.1 in both QuIP# rows): output error QTIP 0.00035, QuIP# 0 sweeps 0.00024, QuIP# 10 sweeps 0.00013; weight error 0.27 / 0.31 / 0.33. So after the 10x damping its output error is among the smallest in the model, not the largest.

### A5 result (1 Nov 10:20): the VPTQ main row stays the original v4 k256
All VPTQ rows with a total linear bpw of at most 2.02 (original v4 k256: 495; 5-weight 1024 centroids: 493; 500 iterations: 487; 500 iterations with tolerance 1e-30: 471; 6-weight 3072 centroids: 458; 8-weight 12288 centroids: 426) are level with or below the original, so the original is the main row. The only VPTQ row clearly above it is 6-weight 4096 centroids (535) at 2.074 bpw (over your limit by 0.054). The offline layer error did not predict the closed-loop success (the larger-codebook structures have the lowest calibration layer error and the lowest success), which we read as a fit to the 256 calibration observations. Question: do you accept 2.07 bpw for VPTQ (it would then be the VPTQ main row), or do you want the bits kept under 2.02?

## A9 (2026-11-01, goal v3) Bisection on 200 units, not 50
- Assumed: component tests (only vision / only language / only head quantized, the rest FP) use 200 units (every 5th unit of the Plus 1000, 100 camera and 100 robot-init), paired against our own FP on the same units, because the 50-unit gate has a standard error of about 7 points.
- Why: the goal allows 50 or 200; 200 is the cheapest size that separates a 10-point effect from the noise floor E0.
- Depends on it: EXPLANATIONS.md blocks on collapses (plain GPTQ, AQLM, AWQ), component sensitivity of QTIP.
- Redo if the mentor wants the full 1000: change `sanity200.json` to the whole selection (about 3.5 h per component and method).

## A10 (2026-11-01, goal v3) pi0.5 GPTQ best is run on the Plus 1000 only
- Assumed: `gptq_best_w2` on pi0.5 gets the sanity sample and the four Plus 1000 parts, not Plus 2000 / LIBERO 2000.
- Why: the goal asks for Plus 1000 with gate, bpw and paired stats; the other benchmarks are ~25 GPU hours more.
- Redo: `python jobs/scheduler.py enable` style items for the added 1000 and LIBERO if the mentor wants the full row set.

## A11 (2026-11-02) Which attention kernel is "default"? (question for the mentor)
All our rows ran with the PyTorch default for `scaled_dot_product_attention` (SDPA_BACKEND unset), and we never logged which kernel that resolves to. Forcing `math` or `efficient` changes 11.5% / 14% of units (E0), so the kernel matters when pairing against his rows.
Done: `groot/serve.py` now prints `{"sdpa_kernels": [...], "q_shape": ..., "dtype": ...}` for the first 3 attention calls of every server start (applies to jobs started from now on; the old rows stay unlogged).
Ask: (1) which GPU, PyTorch version and attention kernel did his FP 618 / 2-bit 486 runs use? (2) Should we pin one kernel (flash / efficient / math) for the final rows? No re-quantization is needed to find out: quantization does not run attention at evaluation time, one evaluation start-up prints the answer.

## A12 (2026-11-02) Official settings only, or tuned settings for every method? (question for the mentor)
We already tuned some methods and not others, so the budgets are unequal:
- GPTQ: 9 official-option variants (act-order, MSE clipping, 1% / 10% damping) and a per-layer pick by calibration error (G38); the plain row (0/1000) is the as-published one.
- VPTQ: 8 structure / iteration variants (G34 to G41).
- AQLM, QTIP, QuIP#: one setting each (QuIP# 0 and 10 sweeps). AQLM departs from the official code in one place (1% Hessian damping; the official code adds none, decision 24). AQLM damping 0/0.1/1/10% and its stopping rule (goal item 1.4) are not run yet.
Question: (a) report every method at its official settings only; or (b) official-settings row plus one tuned row per method, each tuned with a similar small budget and chosen by calibration error (never by Plus 1000)? We propose (b). Also: the GPTQ main row is currently named by the rule "smallest calibration error per layer", but G38 also mentions that it has the most successes; we will drop the second reason.

## A13 (2026-11-02) VPTQ at true 2 bits: which setting is the main row?
The official VPTQ README gives v8 k65536 with no residual (index 16 bits / 8 = 2 bits) as its "~2 bit" model, plus `--enable_perm --enable_norm`. That figure ignores the codebook (65536 x 8 x FP16 per layer); on our layers the codebook adds about 1.28 bits per weight (3.50 in total), so it is not 2 bits.
Closest to official that still fits 2 bits per weight: v8 k12288 (8-weight vectors as official; 1.988 bits; iterations 100, tolerance 1e-5, damping 1% as official; norm, permutation and residual off), Plus 1000 = 426.
Mentor's choice (decision 12, point 2 of 23 Oct): v4 k256 (2.003 bits, same other settings), Plus 1000 = 495, 69 more successes (above the noise line of ~20).
Question: keep v4 k256 as the VPTQ main row (our proposal; v8 k12288 listed as the closest-to-official row), or make v8 k12288 the main row? A variant with `enable_perm` / `enable_norm` on has not been run (it costs extra bits, not yet counted).

## A14 (2026-11-06, rigour C3) QuIP# lattice tables: counted once per model or once per checkpoint group?
The accounting of the QuIP# rows adds the shared E8P12 lattice tables (2,420,744 bytes) to the backbone group and again to each head group, so the table shows 2.031 linear bpw (whole 5.601). Counted once per model (the payload rule of the quantizer says "shared lattice tables once"), it is 2.023 (whole about 0.002 lower). Evidence: `RIGOUR_2026-11-06.md` C3, `experiments/rigour_v5/c3_bits.md`. Our reading: once per model. Either way QuIP# stays above the 2.02 rule by 0.003 to 0.011. Question: correct the QuIP# bpw in RESULTS.md to 2.023, and is the 2.02 limit a hard limit for QuIP# (then 0 sweeps and 10 sweeps would need a smaller padding or a different tables storage)?

## A15 (2026-11-06, rigour C2) Reading of "better": Holm over the main rows with cluster-robust intervals and the E0 perturbation noise
We call a pair of rows "better / worse" only if it passes Holm at 0.05 over the main-row family (GR00T 21, pi0.5 28 comparisons) with a standard error clustered by base task plus the E0 perturbation noise (1.14 pp), and the difference exceeds 2 pp. This is stricter than the paired intervals in the tables, which stay as they are. Under it QTIP is not separated from FP or from QuIP# on GR00T. Question: is this the reading you want for the baseline claims, or do you prefer the per-pair intervals?

## A16 (2026-11-06, scratch quota) Free space for the rigour jobs
At 17:21 the scratch quota stopped R3 and R7 (RIGOUR_2026-11-06.md). About 50 GB are free now; I cut the checkpoint slots from 2 to 1 so the jobs run one at a time. The quota size cannot be read (`quota` fails). Question: may old pi0.5 data under `runs/quantization` (324 GB) or `readme_baselines_offload` (219 GB) be moved or deleted, or can the quota be raised? Recommendation: raise the quota; until answered, nothing old is touched.

## A17 (2026-11-06, SimplerEnv-Bridge) 840 or 700 episodes for the FP baseline
README section 3 gives 120 episodes x 7 tasks = 840 per policy; section 6 says "700 episodes". I run 840 (section 3 is the protocol table, and 120 is a multiple of 24 that covers every layout of the object tasks equally). Recommendation: 840; please correct section 6 if you agree.

## A19 (2026-11-07, SimplerEnv-Bridge) stack_cube FP is 30% against NVIDIA's 48%
(8 Nov: H6 shows NVIDIA's command on our stack gives 45%, so the 30% is the seeded protocol; G3 (explore/g3/LADDER.md) finds no code difference that moves the mean except the attention kernel and dropped in-flight episodes; more repeats planned. 9 Nov, X19 + X22: answered. NVIDIA's command does NOT sample the 24 layouts evenly. ManiSkill2_real2sim seeds every env with 2022 at construction (sapien_env.py:185-188) and reset(seed=None) never draws a new seed (sapien_env.py:492-497), so every run walks the same fixed layout chain. With 1 env, all 6 runs see identical layout counts (23 layouts); with 5 envs, only 14 layouts appear and layout 1 (an easy one) is 23% of episodes; layouts 6-10, 12-14, 20, 21 never appear. Weighted uniformly over the layouts, both official arms give 33.4%, the same as your seeded protocol (33 to 37%). So NVIDIA's 48% (and our 47.0% / 39.2% with their command) is a fixed easy layout subset, not a better policy; the seeded protocol is the right one. H9 (pre-registered) confirmed it on 9 Nov: NVIDIA's command with a seed per run gives 33.8% (6 x 100) against 47.0% with the fixed chain. Closed.) Under your README protocol (24 layouts x 5 episodes, seed 7, one env, `--n-action-steps 4`), the Bridge FP row reproduces NVIDIA on 6 of 7 tasks (mean 59.9 against 62.3), but stack_cube gives 36 of 120 (30.0%, SE 4.2) against 48. Failures are concentrated: layouts 11 to 23 give 6 of 65 (corrected 7 Nov 13:30 from 10). The QTIP 2-bit row gives 72 of 120 on the same task and layouts (33 of 65 on layouts 11 to 23), so the low FP rate is not a property of the layouts alone. Both protocols sample the 24 layouts about evenly, so the layout set does not explain it. Question: do you know how NVIDIA got 48 (5 async envs, a different checkpoint revision than 940134b, or another seed)? Our proposal: keep the seeded protocol for all Bridge rows (the paired differences stay valid) and report stack_cube with this note; optionally one 20-minute run with NVIDIA's command (100 unseeded episodes, 5 envs) if you want the gap explained.

## A20 (2026-11-07, SimplerEnv-Fractal) open_drawer FP is about 50% against the model card's 65%
Our Fractal command equals NVIDIA's README (`--n-action-steps 1 --n-envs 5`, 300 steps, checkpoint `nvidia/GR00T-N1.7-SimplerEnv-Fractal`). Five SDPA runs gave 48 to 53 of 100 on open_drawer; the other families are near the card (all 68.2 against 72.5). Ruled out: render noise (H1: the sapien rasterizer is deterministic for a seed) and the attention kernel (R5: flash-attn 2.8.3 built for sm_120 gives 50 of 100, seed 7). Question: do you know which setting the card used for open_drawer (another checkpoint revision, renderer or ray-tracing setting, GPU, or seed range)? Our proposal: report open_drawer with this note; every 2-bit Fractal row is paired against our own FP, so the paired differences stay valid. Update 8 Nov: NVIDIA's exact README command (flash-attn, no seed) gives 47, 52 and 50 of 100 in three repeats on our stack, so the seed and our protocol are ruled out too (on Bridge the same check reproduced NVIDIA on all 7 tasks, so the stack itself works). Left: checkpoint revision, SimplerEnv / ManiSkill2 version of the drawer scenes, or the card's setup. (The R5 pre-registration named A14 for this question; A14 is the QuIP# table question, so the question is A20.)

(9 Nov, A20 update, X23: open_drawer configurations follow the same fixed 2022 chain as Bridge. With NVIDIA's 5-env command only about 22 chain items are used, and they are harder than average: the chain predicts 51.3, we get 49.7 to 51.0; a uniform mix gives 55.9. So the layout chain does not explain 65 on our code. Question for you: did NVIDIA's 65 come from another SimplerEnv commit or another table protocol? A random chain gives 65 or more only 14% of the time.)

## A21 (2026-11-08, SimplerEnv-Bridge) The Bridge "2-bit above FP" comes from a 5% weight shrink, not from quantization
(9 Nov update: a plain 0.95 W gives the same gain, so the gain is not evidence for quantization. But X25 shows the 2-bit error shrinks the layer OUTPUTS by only 0.2 to 0.9% on real activations, and the QuIP# error without its weight-space shrink still gives carrot 109. So the 2-bit rows reach the gain by a route that is not a plain output shrink; the title claim is too strong and is corrected in RESULTS.md.)
Finding (H8, pre-registered, RESULTS.md): the 2-bit error of QuIP# and QTIP shrinks every one of the 432 quantized weights by 4 to 7% along itself. FP x 0.95 on those 432 weights (no quantizer) gives the same carrot / stack_cube gain (+25.0 [+15.4, +35.4]); FP x 1.05 loses (-16.7). On 7 tasks 0.95 W is 69.9% against FP 59.9% and the 2-bit rows 62.7 to 67.3%. Questions: (1) Should the paper report the Bridge 2-bit rows with this explanation and the 0.95 W control row next to them? (2) Do you know of a reason this GR00T Bridge checkpoint would have weights slightly too large (e.g. its fine-tuning recipe, no weight decay, EMA)? We are now testing why a shrink helps (dose 0.85 to 0.99, layer groups, smaller actions, sampler noise, attention temperature).
