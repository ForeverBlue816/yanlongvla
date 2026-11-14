# Inference kernels for the trellis-coded format: speed and memory of π₀.₅ W2A8 (2026-11-14)

User task (2026-11-14): build inference kernels for the current trellis-coded format and measure speed and memory of π₀.₅ W2A8; the
2-bit policy should be at least as fast as full precision end to end, and faster where possible. Offline only: one GPU, no evaluation
queue. Stage 0 profile, Stage 1 correct kernels (40 held-out observations: integer paths bit-exact, final actions within 1e-3 relative),
Stage 2 three operating points: (i) decode everything once at load, (ii) decode on the fly everywhere, (iii) backbone on the fly,
action expert decoded per chunk. Batch 1, 10 and 2 denoising steps, 20 warm-up and 200 timed calls, median and 95th percentile.

Model: the π₀.₅ LoopQ main row **`tcq_raqft_d1`** (bitshift trellis L = 16, 2 bits / weight, V = 2, 16 × 16 tiles, 512 × 2 table per layer,
FP16 row scales, two-sided randomized block-Walsh rotation; 414 Linear layers) with 8-bit activations taken per token in the rotated
space. Code: `deploy/tcq_kernels/` (README there), one command `bash deploy/tcq_kernels/run_bench.sh`. Results: `deploy/results.csv`,
`deploy/table.md`, per-row JSON in `deploy/results/`. The kernel code is published with this report (`deploy/tcq_kernels/`, user 2026-11-14); the policy loader, checkpoints
and observation sets it plugs into are not.

GPU: NVIDIA RTX 6000 Ada Generation (48 GB), driver 595.71.05, PyTorch 2.7.1+cu126 (CUDA 12.6, cuDNN 9.5.1), NVRTC 12.6. Every row of
the result table was measured in one job on one card (job 1738590, gpu-6000ada-2, 2026-11-14); the Stage 0 profile and the tuning
runs used other cards (gpu-6000ada-1, gpu-a6000-1).

## What existed before

The TCQ rows had no W2A8 runtime. LIBERO-Plus ran `tcq_raqft_d1` as dense BF16 weights in the original basis with FP activations
(row `w2_dense` below): full-precision speed and full-precision GPU memory, 2-bit only on disk. The INT8 runtime of 2026-10-15/20
(INT8_FUSION_2026-10-20.md) belongs to the older codebook format and ended at 0.87–0.95× of BF16.

**LIBERO-Plus with 8-bit activations for exactly this model: none exists.** The Plus numbers of `tcq_raqft_d1` (733 selection, 700
added) are W2 with FP activations. The W2A8 rows that were evaluated (`a8_studentq` 741, `a8_raqft`) are the older format. Nothing was
queued. Offline, W2A8 (INT8 tables, 8-bit activations) differs from the evaluated W2 model by 0.20 % of the action norm (median over the
40 held-out observations; 39 of 40 below 1 %, one observation at 11 % on the 6000 Ada and 43 % on the A6000), against 2.8 % (median)
between either of them and FP.

## Stage 0 — where the time goes

CUDA-graph replays, one held-out observation, 968 prefix tokens, 3 × 256 image tokens, 10 action tokens; CUPTI kernel classes.

| Stage | FP (BF16), ms | of which matrix products | W2A8 in PyTorch ops, ms | of which INT8 products |
|---|---:|---:|---:|---:|
| Vision encoder (3 images) | 8.35 | 5.53 | 131.3 | 3.70 |
| LLM prefill | 42.05 | 33.68 | 367.1 | 11.21 |
| Action expert, one step | 3.53 | 2.18 (0.95 of it attention) | 17.5 | 0.81 |

- FP: the prefill is matrix products (39.2 of 50.4 ms); the expert step is launch-bound (1,131 launches, 1.2 ms of quantizable products).
- W2A8 written with PyTorch operations (the definition the kernels are checked against): the Hadamard transforms take 82–84 % of the
  prefill and 70 % of an expert step (eager timing per operation: Hadamard 389 ms, activation quantizer 27, product 20, scale 19,
  everything else 20 in the LLM prefill). Decode is not in this path (weights held as INT8).
- The INT8 products themselves need 11.2 ms where FP's BF16 products need 33.7 ms: that 22 ms is the budget the rotation, quantizer
  and scale kernels must stay under.

## Stage 1 — correctness

Reference W2A8 model = `tcq_format.reference_linear` in PyTorch operations (FP32 butterflies in a fixed order, FP32 division in the
quantizer, round-half-even, INT32 accumulation, BF16 outputs).

- Packed file: the windowed read (state of pair t = bits [4t, 4t + 16) of the tile's stream) reproduces the stored state indices in
  414 / 414 layers.
- Layers, on the inputs each one receives inside the model (prefill, and denoising step 0): decoded INT8 weights, INT8 activations,
  scales, INT32 accumulators (decoded-at-load and on-the-fly) and BF16 outputs are bit-identical to the reference in **414 / 414
  layers**. No layer mismatches.
- Final actions, 40 held-out observations: operating points (i), (ii), (iii) are **bit-identical** to the reference on 40 / 40
  (relative error 0). CUDA graph = eager (max |Δ| 0) in every row.
- The fused kernels (shared input rotation, three output rotations in one launch, the MLP middle with GELU evaluated in the kernel)
  are bit-identical to the unfused chain; the in-kernel GELU equals PyTorch's on all 65,536 BF16 inputs.
- Vision call structure: the kernel rows run the per-camera vision-encoder calls in lockstep (the six Linear layers of each block see
  the 768 tokens of all cameras in one launch; PyTorch's own operations stay per image). For W2A8 this is bit-identical to the
  per-image calls: 40 / 40 observations for (i) and (ii) on the A6000, and max |Δ| 0 in every W2A8 row of the benchmark on the 6000 Ada.
  Plain batching of the images is not exact (PyTorch's BF16 attention arithmetic depends on the batch size: actions move by 2e-3
  relative for FP, 3e-3 to 1e-2 for W2A8), so it is not used.
- The checks were run on two GPU models (RTX 6000 Ada, RTX A6000) with the same outcome.
- The split kernels of the action expert (gate and up, and q / k / v, as separate (token, layer) blocks) are bit-identical to the
  fused ones.
- W2 without activation quantization on the kernel path (`w2_fly`, `w2_hybrid`: BF16 activations in the rotated space, INT8 tables):
  0.18 % (median) from the evaluated dense W2 model, the same distance as W2A8; it is a different arithmetic, not a bit-exact path.

## Stage 2 — results

| Row | Weights on disk (GB) | Peak GPU memory, allocated / reserved (GB) | Prefill, ms | Full chunk, 10 steps, ms | Full chunk, 2 steps, ms | vs FP, 10 / 2 steps |
|---|---:|---:|---:|---:|---:|---:|
| Full precision (BF16), official path | 6.47 | 6.65 / 6.78 | 51.8 (52.3) | 81.6 (83.3) | 54.6 (56.3) | 1.00× / 1.00× |
| Full precision (BF16), lockstep vision calls | 6.47 | 6.64 / 6.78 | 51.0 (51.6) | 81.1 (81.7) | 53.9 (55.3) | 1.01× / 1.01× |
| 2-bit weights, (i) decoded at load | 1.58 | 6.16 / 6.29 | 52.4 (52.9) | 82.1 (85.3) | 55.0 (56.3) | 0.99× / 0.99× |
| 2-bit weights, (ii) on the fly | 1.58 | 1.99 / 2.84 | 86.5 (86.8) | 133.1 (133.6) | 94.5 (94.9) | 0.61× / 0.58× |
| 2-bit weights, (iii) hybrid | 1.58 | 2.21 / 2.97 | 86.5 (86.9) | 129.8 (130.3) | 93.7 (94.2) | 0.63× / 0.58× |
| W2A8, (i) decoded at load | 1.58 | 3.72 / 4.93 | 38.2 (38.6) | 74.6 (75.3) | 42.8 (44.3) | 1.09× / 1.28× |
| W2A8, (ii) on the fly | 1.58 | 1.92 / 2.77 | 40.6 (41.7) | 79.8 (80.1) | 47.2 (48.0) | 1.02× / 1.16× |
| W2A8, (iii) hybrid | 1.58 | 2.15 / 2.89 | 40.7 (41.9) | 78.7 (79.5) | 47.5 (48.2) | 1.04× / 1.15× |

| Row | Vision encoder, ms | LLM prefill, ms | Expert step (10-step schedule), ms | All 10 steps, ms | Expert step (2-step schedule), ms | Graph = eager | Same as per-image calls |
|---|---:|---:|---:|---:|---:|---|---|
| Full precision (BF16), official path | 8.55 | 43.44 | 3.43 (3.58) | 34.3 | 2.72 | yes | yes |
| Full precision (BF16), lockstep vision calls | 7.50 | 43.99 | 3.44 (3.48) | 34.3 | 2.72 | yes | max abs diff 2.3e-03 |
| 2-bit weights, (i) decoded at load | 8.58 | 43.99 | 3.43 (3.44) | 34.3 | 2.72 | yes | yes |
| 2-bit weights, (ii) on the fly | 18.18 | 68.77 | 4.69 (4.70) | 47.2 | 3.99 | yes | max abs diff 2.6e-03 |
| 2-bit weights, (iii) hybrid | 18.18 | 68.87 | 4.35 (4.36) | 44.1 | 3.65 | yes | max abs diff 2.7e-03 |
| W2A8, (i) decoded at load | 10.14 | 28.69 | 3.66 (3.67) | 36.6 | 2.94 | yes | yes |
| W2A8, (ii) on the fly | 10.57 | 31.42 | 3.78 (3.79) | 37.8 | 3.03 | yes | yes |
| W2A8, (iii) hybrid | 10.52 | 31.13 | 3.66 (3.67) | 37.2 | 2.94 | yes | yes |

Median (95th percentile) of 200 timed CUDA-graph replays after 20 warm-up replays, batch 1; GB = 2^30 bytes. NVIDIA RTX 6000 Ada Generation, PyTorch 2.7.1+cu126, CUDA 12.6, cuDNN 90501, NVRTC 12.6.

Operating points: (i) every layer decoded once at load and held as INT8; (ii) decoded from the 2-bit stream on every call; (iii) backbone
on the fly, action expert decoded once per chunk. "2-bit weights" rows have FP activations: (i) is the evaluated model (dense BF16
weights in the original basis), (ii) / (iii) run the kernels with BF16 activations in the rotated space. Weights on disk: 0.64 GB of
packed 2-bit payload + 0.94 GB for everything else (INT8 token embedding 0.49, FP32 tensors 0.45); FP checkpoint 6.47 GB. Peak GPU
memory is the process after load with the full-chunk graph captured; the loader builds the FP skeleton first, so load-time peaks (7.1 GB)
are not deployment numbers.

- **Fastest: W2A8 (i), 74.6 ms per 10-step chunk and 42.8 ms per 2-step chunk: 1.09× and 1.28× FP**, at 3.72 GB (FP 6.65 GB).
- **Smallest: W2A8 (ii), 1.92 GB allocated (0.29× FP), 79.8 ms and 47.2 ms: 1.02× and 1.16× FP.**
- (iii) sits between them in memory (2.15 GB) and is level with (ii) in time (78.7 / 47.5 ms): decoding inside the expert's product
  kernel costs 0.12 ms per step, about what one per-chunk decode of the expert costs, so (iii) buys nothing on this GPU.
- Prefill: 38.2 ms against 51.8 ms (1.36×). LLM prefill 28.7 vs 43.4 ms; vision encoder 10.1 vs 8.5 ms (slower: at 256–768 tokens and
  width 1152 the INT8 products save 2 ms and the rotations cost 4); action expert 3.66 vs 3.43 ms per step (slower by 7 %).
- 2-bit weights without 8-bit activations: decoded at load it is FP in speed and memory; decoded on the fly it is 0.61× FP (BF16
  products on decoded tables plus the rotations). The INT8 product is what pays for the on-the-fly decode.
- FP with the same lockstep vision calls: 81.1 / 53.9 ms (1.01×), so the call structure accounts for 0.5 ms of the gain. For FP and for
  the BF16-activation rows lockstep is not bit-neutral on this card (max |Δ| 2e-3 to 3e-3 in the actions: cuBLAS BF16 products depend
  on the token count); for the W2A8 rows it is exact.

Second GPU model, RTX A6000 (gpu-a6000-1; same code, thread plan not re-tuned; 10-step rows from job 1738557, 2-step rows from job
1738769; `deploy/results/a6000/`):

| Row | Prefill, ms | Full chunk, 10 steps, ms | Full chunk, 2 steps, ms | vs FP, 10 / 2 steps |
|---|---:|---:|---:|---:|
| Full precision (BF16), official path | 70.9 | 114.8 | 77.6 | 1.00× / 1.00× |
| 2-bit weights, (i) decoded at load | 70.7 | 115.3 | 77.6 | 1.00× / 1.00× |
| 2-bit weights, (ii) on the fly | 132.3 | 197.8 | 142.4 | 0.58× / 0.55× |
| W2A8, (i) decoded at load | 72.3 | 119.3 | 80.0 | 0.96× / 0.97× |
| W2A8, (ii) on the fly | 77.7 | 129.3 | 86.2 | 0.89× / 0.90× |
| W2A8, (iii) hybrid | 77.7 | 125.6 | 86.2 | 0.91× / 0.90× |

On that card the 2-bit policy is not faster than FP (memory is the same as on the Ada: 3.72 / 1.92 / 2.15 GB against 6.65). The
speed-up above is specific to the RTX 6000 Ada.

## What makes it fast, and what does not

1. INT8 products on cuBLASLt for the prefill (11–12 ms instead of 34 ms), with the tile decode amortised over all tokens.
2. One rotation + quantization per input tensor: q / k / v and gate / up share their input signs (162 of 414 input rotations removed,
   exactly).
3. Radix-16 Walsh passes in shared memory with a conflict-free first pass; the wide (16384) kernels run near the memory-traffic floor
   of the INT32 accumulators.
4. Fused MLP middle (scale + rotate of gate and up, GELU, product, rotate + quantize of the down input) in one launch.

Not removable without a refit: the 414 output rotations and 252 input rotations (README, "Rotations that can be removed exactly").

Where it is still slower than FP: the action expert. At 10 tokens a product is as fast as cuBLAS (4–9 µs per layer either way), but
the two-sided rotation adds launches (per block: 3 rotate + quantize, 3 scale + rotate, 1 fused middle against FP's GELU and product),
about 13 µs per block, 0.23 ms per step. The remaining lever is to fuse the residual add and the norm between two layers into the
rotation kernels (not bit-exact against PyTorch's norm; not done).
