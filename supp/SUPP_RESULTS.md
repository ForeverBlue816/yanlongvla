# Supplementary results for LoopQ

Status of 2026-11-19 08:30 UTC. Experiment log entry: `EXPERIMENTS.md` E224. Every number below comes from a finished run; the
log or result file and the commit that recorded it follow the number. Arms that have not finished are listed with their state and
have no row in any table.

Conventions used in every LIBERO-Plus table:

- Instances: the 1000 paired instances of the main table (500 camera, 500 initial state; 245 Spatial, 251 Object, 256 Goal, 248 Long).
- One run per instance, flow-noise seed `SeedSequence([7, suite, task, 0])`, the seed of the reference rows in this repository.
  Further seeds are not run (decision of 2026-11-19), so the tables have no standard-deviation values.
- GPU: NVIDIA RTX 6000 Ada Generation for every new row and for the 2-bit reference rows; the 1-bit rows were run on an H100.
- Reference rows are quoted from their existing per-instance files and were not re-run.

## Relation to statements in the main text (与正文表述的关系)

No arm of A–D has finished yet, so none of the six statements can be read against new numbers today. The table records what each
statement will be checked with and the state of that run.

| Statement in the main text | Checked by | State on 2026-11-19 08:30 UTC | Supports / does not support |
|---|---|---|---|
| "We perform one fixed-point iteration" | C (second iteration, and the per-module distance between first and second on-policy second moments) | not started (second batch) | no result yet |
| "Trace normalization gives every source the same total energy" | B1–B3 | not started (third batch) | no result yet; one recorded fact is given under B below |
| "the two steps address different causes, and their effects add" | D1, D2 | fine-tunes running (started 08:16 UTC) | no result yet |
| "constraining the projected attention outputs W_O·o … focuses the limited trainable capacity" | D3 | not started (second batch) | no result yet |
| "student-side queries … essential for GR00T" | D4 | not started (second batch) | no result yet |
| "standard codebook learning rates destabilize entire layers" | D5 | not started (second batch) | no result yet |

## Open points for the author

1. **Noise seeds — decided 2026-11-19.** One seed per instance (the reference rows' seed); no further seeds for now.
2. **Reference row of D — decided 2026-11-19.** The D arms are compared with the 60.4 row (604 / 1000,
   `runs/plustrain_20261008/groot/groot_n17_tcq_raqft_rht_s/plus/complete.json`, E203). The other recorded runs of the default
   recipe are listed with their files in the reference table below.
3. **GR00T AWQ** has no LIBERO-Plus row in the baseline records: it was stopped at the 0 / 50 sanity gate
   (`baselines/RESULTS.md:38`, commit be5728c).
4. **H2 (Jetson AGX Thor)** is not run, by instruction of 2026-11-19.

## Reference rows (existing files, not re-run)

GR00T N1.7, 2 bit, LIBERO-Plus, successes (percent).

| Row | All | Camera | Init. state | Spatial | Object | Goal | Long | File |
|---|---|---|---|---|---|---|---|---|
| FP | 618 (61.8) | 326 (65.2) | 292 (58.4) | 183 (74.7) | 126 (50.2) | 178 (69.5) | 131 (52.8) | `groot_n17_fp/plus/episodes.jsonl` |
| Trellis format, demonstration-only calibration (the 55.9 row) | 559 (55.9) | 272 (54.4) | 287 (57.4) | 175 (71.4) | 122 (48.6) | 154 (60.2) | 108 (43.5) | `groot_n17_tcq_w2_rht/plus/episodes.jsonl` |
| Closed-loop calibration only (the 57.5 row) | 575 (57.5) | 289 (57.8) | 286 (57.2) | 174 (71.0) | 127 (50.6) | 167 (65.2) | 107 (43.1) | `groot_n17_tcq_w2_rht_d1/plus/episodes.jsonl` |
| Default fine-tune started from the 55.9 row (the 60.4 row) | 604 (60.4) | 311 (62.2) | 293 (58.6) | 184 (75.1) | 123 (49.0) | 175 (68.4) | 122 (49.2) | `groot_n17_tcq_raqft_rht_s/plus/episodes.jsonl` |
| Default fine-tune started from the 57.5 row, RTX 6000 Ada | 581 (58.1) | 300 (60.0) | 281 (56.2) | 175 (71.4) | 126 (50.2) | 169 (66.0) | 111 (44.8) | `groot_n17_tcq_raqft_d1_s/plus_6000ada/episodes.jsonl` |
| Default fine-tune started from the 57.5 row, A40 | 594 (59.4) | 304 (60.8) | 290 (58.0) | 174 (71.0) | 133 (53.0) | 169 (66.0) | 118 (47.6) | `groot_n17_tcq_raqft_d1_s/plus/episodes.jsonl` |

Files are under `runs/plustrain_20261008/groot/`. The baseline record of QTIP run with its toolkit on GR00T is 573 / 1000
(`baselines/RESULTS.md:22`, commit be5728c).

Held-out action error and readout error of the same rows (256 held-out observations, RTX 6000 Ada;
`runs/plustrain_20261008/groot/shift/tcq_raqft_d1_s_offline.json` and the TCQ document, E212):

| Row | Action MSE vs FP | Readout error L_readout |
|---|---|---|
| Demonstration-only calibration | 5.70e-3 | not in that file |
| Closed-loop calibration only | 5.146e-3 | 7997.8 |
| Default fine-tune from the 57.5 row | 4.586e-3 | 7232.9 |

## A — sources of the closed-loop calibration matrix

H = H_demo + Σ_{s∈S} w_s H_s with w_s = tr(H_demo) / (|S| tr(H_s)), |S| = the number of sources used. The per-source second moments
are the stored ones of the 57.5 row; the matrix is composed per layer at fit time. Composing {lighting, crop, on-policy} this way
reproduces the stored matrix of the 57.5 row exactly (maximum absolute difference 0.0, checked on one backbone and one head layer).

| Arm | Sources | State |
|---|---|---|
| A1 | demo + lighting | queued (third batch) |
| A2 | demo + crop | queued (third batch) |
| A3 | demo + on-policy | fits running since 08:16 UTC (2 of 5 fit parts started) |
| A4 | demo + lighting + crop | queued behind A3 |
| A5 | demo + lighting + on-policy | queued (third batch) |
| A6 | demo + crop + on-policy | queued (third batch) |

Reference rows: demonstration only 559 (55.9), all three sources 575 (57.5).

## B — weighting of the sources

| Arm | Weighting | State |
|---|---|---|
| B1 | second moments added without trace normalisation | queued (third batch) |
| B2 | normalised, total weight × 0.5 | queued (third batch) |
| B3 | normalised, total weight × 2 | queued (third batch) |

Recorded fact that bears on B1: the per-layer weights of the 57.5 row have mean 0.333 to 0.343 per source in every part of the model
(`runs/plustrain_20261008/groot/shift/hd1_summary.json`). The trace of each source's second moment is therefore within a few percent
of the trace of H_demo, and B1 is close to "total weight × 3" with unit weights (on the two layers checked, the added trace is 2.98
and 3.00 times tr(H_demo)).

## C — second fixed-point iteration

Not started. Plan: 256 rollouts of the 57.5 row from the same initial-state distribution, on-policy second moments from them, the
calibration matrix rebuilt with the first-iteration lighting and crop sources, one more quantization pass, LIBERO-Plus. The
per-module (vision / language / action head) relative Frobenius distance between the trace-normalised first and second on-policy
second moments is reported with it.

## D — fine-tune objective

Each arm starts from the 57.5 row and changes one thing in the default recipe (768 updates, 216 / 40 split, Adam, the default
learning rates, EMA 0.99). Reference: the 60.4 row (604 / 1000). The exported checkpoint of every arm is chosen by the default rule: lowest readout error on the 40
held-out observations, including the arms that do not train the readout term (D2, D3).

| Arm | Change | State |
|---|---|---|
| D1 | readout term only (β = 0) | running since 08:16 UTC (A40) |
| D2 | velocity term only | running since 08:16 UTC (L40S) |
| D3 | raw prefix K, V aligned instead of W_O·o | queued (second batch) |
| D4 | teacher-side queries | queued (second batch) |
| D5 | no per-step RMS renormalisation of the tables | queued (second batch) |
| D6 | no Frobenius anchor (μ = 0) | queued (second batch) |

## E1 — LoopQ hyper-parameters

Exported from the run records: `runs/plustrain_20261008/groot/ptq/fit/backbone_tcq_rht_d1/complete.json`,
`groot/ptq/raqft/backbone_tcq_raqft_d1_s/manifest.json`, `pi05_tcq/fit/d1/complete.json`, `pi05_tcq/raqft/raqft_d1/manifest.json`
(code commit 4a6c069 of the private repository; values unchanged since the runs).

| Item | GR00T N1.7 | π0.5 |
|---|---|---|
| Quantized layers | 432 Linear (216 backbone + 216 action head per suite checkpoint) | 414 Linear (288 backbone + 126 action expert) |
| Trellis code | bitshift, L = 16, K = 2 (1 bit: K = 1), V = 2 | same |
| Lookup table | 2^9 entries × 2, FP16, `quantlut_sym` | same |
| Tile | 16 × 16 | same |
| Incoherence | two-sided randomized Hadamard | same |
| Scale | FP16 per output row; `scale_override` 0.9 | same |
| Rounding | LDLQ with Viterbi, 1 pass | same |
| Damping | 0.01 of the mean diagonal on every layer (ladder 0.01 / 0.1 / 1.0, first value used on 432 / 432 and 414 / 414 layers) | same |
| Bits per weight, linear layers | 2.018 (2 bit), 1.0177 (1 bit) | 2.0162 (2 bit), 1.0162 (1 bit) |
| Token embedding | unchanged | INT8 per row |
| Demonstration calibration set | 256 observations | 256 observations |
| Lighting source | 2 copies per observation (512): brightness and contrast U(0.8, 1.2), hue ±5°, Gaussian blur σ U(0.5, 1.0) | same |
| Crop source | 1 copy per observation (256): square crop of 90–95 % of the area, resized back | same |
| On-policy source | 256 rollout states of the 2-bit policy, initial state drawn from the task's 50 official LIBERO initial states, one state from the first 40 % of the policy calls | same (policy: the earlier accepted 2-bit row) |
| Source weights | w_s = tr(H_demo) / (3 tr(H_s)) per layer | same |
| Fixed-point iterations | 1 | 1 |
| Fine-tune: trainable | lookup tables and row scales (as log s) of all 432 layers, RMSNorm / LayerNorm weights (117 backbone + 9 per suite head) | lookup tables and row scales of the 288 backbone layers, 147 norm tensors; action expert frozen |
| Fine-tune: loss | readout term (W_O metric, student-side queries) + β · velocity term + anchor | readout term (W_O metric, student-side queries, 10 steps × 18 layers) + β · velocity-direction term (steps 0–5) + anchor |
| β | set once by gradient-norm matching at update 0: 2.907e7 | 0.1 |
| Anchor | μ · Σ_l ‖Ŵ_l − W_l‖²_F / ‖W_l‖²_F, μ set so that the anchor is 10 % of the initial total loss: μ = 71.44 | same rule: μ = 1.176 |
| Updates, latents per update | 768, 8 | 768, 8 |
| Optimizer | Adam; tables 2e-5, norm weights 2e-5, log row scales 1e-4; 5 % linear warm-up, cosine to 0; gradient clip 1.0 | same |
| EMA | 0.99, exported | same |
| Table renormalisation | every table rescaled to its initial RMS after each step, factor folded into the row scale | same |
| Split | 216 train / 40 held out (12 / 10 / 10 / 8 by suite) | same |
| Checkpoint selection | lowest held-out readout error, evaluated every 64 updates (4 latents) | same |
| Non-finite guard | a non-finite gradient skips the update; stop above 38 events | same |
| 1 bit | same settings with K = 1 (`LAMBDA_1BIT_2026-11-12.md`, commit 970fa1f) | same |

The GR00T manifest also carries four inherited fields that the recipe does not use (`lr` 3e-4, `lr_scale` 2e-5, `beta` 0.1,
`epochs` 3); the effective values are the ones in the table.

## E2 — baselines

From the baseline records (`baselines/`, commits be5728c / 0c6b1b7 / 82e1284). Calibration for every baseline: the same 256
demonstration observations as LoopQ (π0.5: `baselines/data/calibration_manifest.json`, noise seed 0, all 10 flow steps; GR00T:
`baselines/groot/README.md:24`, noise seed `SeedSequence([7, i])`), second moments accumulated in FP64 over every forward call.

| Baseline | Toolkit and commit | Key settings | bpw, linear (π0.5 / GR00T) | LIBERO-Plus, π0.5 / GR00T |
|---|---|---|---|---|
| AQLM | Vahe1994/AQLM `e79a896` | 2 codebooks × 8 bits, groups of 8, 16-bit codebook values, init 100 iterations, lr 1e-4, up to 1000 epochs × 100 steps, tolerance 0.01, beam 1, damping 0.01; no block or global fine-tuning | 2.0172 / 2.020 | 639 / 401 |
| VPTQ | microsoft/VPTQ `5a8dfb9` (algorithm branch) | vector length 4, 256 centroids, no residual, Hessian-weighted k-means (100 iterations, seed 0), damping 0.01, block 128 | 2.0025 / 2.003 | 685 / 495 |
| QuIP# | Cornell-RelaxML/quip-sharp `1d8f873` | E8P12, `scale_override` 0.9, σ_reg 1e-2, Hadamard incoherence, 10 tuning sweeps, no fine-tuning | 2.0283 / 2.031 | 662 / 531 |
| QTIP | Cornell-RelaxML/qtip `e90c668` | bitshift trellis L 16, K 2, V 2, 16 × 16 tiles, `quantlut_sym`, 9-bit table, `scale_override` 0.9, σ_reg 1e-2; no fine-tuning | 2.0237 / 2.018 | 726 / 573 |
| GPTQ | re-implementation after IST-DASLab/gptq (`baselines/quantize.py:135-178`) | 2 bits, groups of 128, block 128, damping 0.01; main row = best variant per layer by calibration error | 2.1465 / 2.148 | 657 / 476 |
| AWQ | re-implementation after llm-awq (`baselines/quantize.py:65-113`) | 2 bits, groups of 128, 20-point scale grid, 20-point clip grid (max shrink 0.5) | 2.1466 / not recorded | 154 / none (stopped at the 0 / 50 gate) |

Counts are from `baselines/RESULTS.md` lines 8–15 (π0.5) and 22–39 (GR00T). Other recorded variants: QuIP# with 0 tuning sweeps
740 / 554; GPTQ plain 393 / 0. On GR00T one layer (language model layer 2, `mlp.down_proj`) needed σ_reg 0.1 for QuIP#
(`baselines/groot/data/layer_errors/damping_retries.json`).

## E3 — cost

Wall-clock seconds as recorded by each stage; one GPU per stage. "not logged" = no record in a log or result file.

| Stage | GR00T N1.7 | π0.5 |
|---|---|---|
| Rollout collection (256 on-policy states) | not logged | not logged |
| Second moments, demonstrations | backbone 69 s; head 102 s (Spatial), 51 s (Object); RTX 6000 Ada | 194 s, RTX 6000 Ada |
| Second moments, lighting / crop / on-policy | backbone 185 / 91 / 92 s; head (Spatial) 201 / 117 / 119 s; RTX A6000 | 340 / 192 / 187 s, RTX 6000 Ada |
| Composing the calibration matrix (CPU) | backbone 392 s, head 379 s per suite | 1822 s |
| One quantization pass, demonstration-only matrix | backbone 7870 s, head (Spatial) 3449 s; GPU not logged | not run for π0.5 |
| One quantization pass, closed-loop matrix | backbone 7891 s; heads 6864 / 10089 / 6870 / 10080 s (Spatial / Object / Goal / Long); RTX 6000 Ada and RTX A6000 (from the scheduler ledger) | 14020 s for all 414 layers; GPUs not logged per layer |
| Fine-tuning, 768 updates | 18.7 h (67,231 s recorded by the run, held-out evaluations included; 17.48 h summed over the updates), RTX 6000 Ada | 3.27 h summed over the updates, RTX 6000 Ada |
| Fine-tuning at 1 bit | 16.9 h, H100 | 3.0 h, H100 |

GR00T files: `runs/plustrain_20261008/groot/ptq/hessians/*/complete.json`, `groot/ptq/fit/*/complete.json`,
`groot/ptq/raqft/backbone_tcq_raqft_d1_s/history.json`, `groot/drivers/ledger.log`. π0.5 files: `pi05_tcq/hessians/*/complete.json`,
`pi05_tcq/fit/d1/complete.json`, `pi05_tcq/raqft/raqft_d1/history.json`. 1 bit: `LAMBDA_1BIT_2026-11-12.md`.

Baselines: quantization time is recorded for two rows only. QTIP on GR00T: 5302 s summed over the layers of the backbone and the
four heads (quantizer loop only; `baselines/groot/qtip/accounting_*.json`), GPU not logged. AQLM on π0.5: two records that disagree,
"about 45 GPU-hours for all 414 layers" (`baselines/DECISIONS.md:29` at commit 9addac9^) and "~13 GPU-hour quantization"
(`baselines/jobs/evaluate.pbs:19`). Every other baseline row: not logged. The GPU model used for quantizing the baselines: not logged.

## F — SimplerEnv Bridge

Model and protocol of the baseline instructions (`baselines/groot/simpler/README.md`): `nvidia/GR00T-N1.7-SimplerEnv-Bridge` at
revision `940134b`, 7 WidowX tasks × 120 seeded episodes, 4 of 8 actions executed per call. All four rows were built and evaluated
on one machine (H100), so they are paired per episode (`LAMBDA_1BIT_2026-11-12.md`, per-episode files
`results/lambda_1bit_20261112/bridge_*.episodes_{a,b}.json`, commits 970fa1f / 9d0b5eb).

| Task (of 120) | FP | QTIP | Closed-loop calibration only | LoopQ |
|---|---|---|---|---|
| Spoon on towel | 101 | 110 | 100 | 87 |
| Carrot on plate | 78 | 93 | 82 | 68 |
| Stack cube | 42 | 72 | 72 | 60 |
| Eggplant in basket | 62 | 86 | 68 | 77 |
| Eggplant in sink | 2 | 1 | 0 | 2 |
| Open drawer | 120 | 119 | 120 | 120 |
| Close drawer | 117 | 118 | 118 | 115 |
| Total (of 840) | 522 | 599 | 560 | 529 |
| Mean (%) | 62.1 | 71.3 | 66.7 | 63.0 |

- QTIP here is the QTIP toolkit row rebuilt on that machine (`qtip_w2_chamadol`). Our trellis row with demonstration-only
  calibration on the same machine: 597 (71.1 %).
- Paired differences (clustered by task and layout, 95 %): LoopQ − closed-loop calibration only −3.7 [−6.0, −1.3] pp;
  LoopQ − QTIP −8.3 [−12.4, −4.2] pp; LoopQ − FP +0.8 [−2.9, +4.4] pp.
- Second machine (RTX 6000 Ada, same protocol, `baselines/groot/simpler/results/ntu_rerun_20261116/`, commit 6e881a3): FP 507
  (60.4 %), demonstration-only trellis row 536 (63.8 %), closed-loop calibration only 512 (61.0 %), fine-tune 514 (61.2 %). No QTIP
  toolkit row exists on that machine.

On Bridge the fine-tuned row is not above the calibration-only row on either machine, and both are below the QTIP row on the H100.

## G — GR00T N1.7 at 1 bit

LIBERO-Plus, same instances and seed, H100 (`results/lambda_1bit_20261112/groot_1bit_*.plus_1000.jsonl`, commit 9d0b5eb).
Linear layers 1.0177 bits per weight.

| Row | All | Camera | Init. state | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|---|---|
| Closed-loop calibration only | 81 (8.1) | 35 (7.0) | 46 (9.2) | 44 (18.0) | 24 (9.6) | 13 (5.1) | 0 (0.0) |
| LoopQ | 292 (29.2) | 148 (29.6) | 144 (28.8) | 99 (40.4) | 83 (33.1) | 80 (31.3) | 30 (12.1) |

LoopQ − calibration only, paired: +21.1 [+15.2, +27.2] pp.

A 1-bit QTIP row for GR00T does not exist in the records. It is scheduled in the fourth batch; it will run on this cluster's RTX
6000 Ada, a different GPU model from the two rows above.

## H1 — operating points on the RTX 6000 Ada

π0.5, trellis checkpoint, from `DEPLOY_TCQ_2026-11-14.md` (commit 3a09a2a; raw files `deploy/results/`). Median of 200 timed
CUDA-graph replays after 20 warm-up replays, batch 1. GB = 2^30 bytes. Disk = weight files. Memory = peak GPU memory allocated by
the process after load with the full-chunk graph captured (allocated / reserved); load-time peaks are not included.

| Operating point | Disk (GB) | Memory, allocated / reserved (GB) | Prefill (ms) | Chunk, 10 steps (ms) | Chunk, 2 steps (ms) |
|---|---|---|---|---|---|
| FP (BF16) | 6.47 | 6.65 / 6.78 | 51.8 | 81.6 | 54.6 |
| W2, decoded at load | 1.58 | 6.16 / 6.29 | 52.4 | 82.1 | 55.0 |
| W2, decoded on the fly | 1.58 | 1.99 / 2.84 | 86.5 | 133.1 | 94.5 |
| W2A8 integer kernels, decoded at load | 1.58 | 3.72 / 4.93 | 38.2 | 74.6 | 42.8 |
| W2A8 integer kernels, decoded on the fly | 1.58 | 1.92 / 2.77 | 40.6 | 79.8 | 47.2 |

Disk 1.58 GB = 0.64 GB of packed 2-bit payload + 0.94 GB for everything else (INT8 token embedding 0.49, FP32 tensors 0.45).

## H2 — Jetson AGX Thor

Not run (instruction of 2026-11-19).

## H3 — bit-exactness of the decode modes

From `DEPLOY_TCQ_2026-11-14.md`, Stage 1 (commit 3a09a2a). Reference = the W2A8 layer written in PyTorch operations
(`tcq_format.reference_linear`).

- Packed file: the windowed read reproduces the stored trellis state indices in 414 / 414 layers.
- Per layer, on the inputs it receives inside the model: decoded INT8 weights, INT8 activations, scales, INT32 accumulators and BF16
  outputs are bit-identical to the reference in 414 / 414 layers, for decoding at load and for decoding on the fly.
- Final actions on 40 held-out observations: decoded at load, on the fly, and hybrid are bit-identical to the reference on 40 / 40;
  CUDA graph equals eager in every row.
- Checked on two GPU models (RTX 6000 Ada, RTX A6000) with the same outcome.
- Scope: this holds for the W2A8 path. W2 with FP activations on the kernel path is a different arithmetic from the evaluated dense
  W2 model (0.18 % median relative difference in the actions), not a bit-exact path.

## π0.5 fine-tune hyper-parameters (requested 2026-11-19, outside the supplementary list)

Goal: bring the trellis-format fine-tuned row to the level of the 74.9 row, and report the row before and after fine-tuning on
the expanded 2000 instances.

| Row | Selection 1000 | Added 1000 | Expanded 2000 | State |
|---|---|---|---|---|
| Closed-loop calibration only (`tcq_d1`) | 734 | running | — | added 1000 started 08:16 UTC |
| Fine-tuned, default settings (`tcq_raqft_d1`) | 733 | 700 | 1433 | existing |
| Earlier-format fine-tuned row (the 74.9 row) | 749 | 699 | 1448 | existing |
| FP | 765 | 718 | 1483 | existing |
| Variants: rates × 3, rates × 10, 2304 updates, velocity term weighted to the readout term's size | — | — | — | queued |

Existing numbers: `PI05_TCQ_2026-11-10.md` (E215, E216, and the added-1000 section).
