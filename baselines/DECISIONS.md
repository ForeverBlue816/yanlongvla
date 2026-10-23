# Baseline implementation decisions

Decisions for the baseline rows run with this harness. Priority of sources: (1) `README.md` in this folder (the
mentor's protocol), (2) the mentor's instructions, (3) the official repository of each method. Runs, logs and results:
`~/projects/pi05-handover/runs/readme_baselines/` (`STATUS.md` has the live table).

## Decisions made

| # | Topic | Decision | Source / reason | Date |
|---|---|---|---|---|
| 1 | Rows to report | 2-bit only: RTN, GPTQ, VPTQ, AQLM, AWQ, QuIP#, QTIP. 3-bit and 4-bit rows stopped; their partial results are kept but not extended. | Mentor: "You only need to evaluate 2bit performance", especially VQ methods | 2026-10-24 |
| 2 | Row order | RTN, GPTQ, VPTQ, AQLM, AWQ, QuIP#, QTIP. QuIP# and QTIP are optional extras, done last. | User order; mentor: QuIP#/QTIP last | 2026-10-23 |
| 3 | Model | Official `pi05_libero`, converted with openpi to PyTorch (`model.safetensors` sha256 `1869ec96…`) | README | 2026-10-23 |
| 4 | Quantized layers | The 414 `nn.Linear` layers: SigLIP 162, Gemma LM 126, action expert 126, all at the nominal width. The expert's AdaRMS `*layernorm.dense` Linears are norms and stay unquantized. | README (norms never quantized) | 2026-10-23 |
| 5 | Fake-quant checkpoints | Dequantized BF16 weights in the unmodified openpi model. The writer copies the token embedding (stored only as the tied `lm_head.weight`) to `embed_tokens.weight` before dropping the heads; the loader refuses a checkpoint without it. | Bug fix: every earlier baseline checkpoint loaded a random embedding | 2026-10-23 |
| 6 | Calibration | README manifest: 256 observations from 60 LIBERO training episodes, noise seed 0, all 10 flow steps. Recomputed with this repository's code; identical to the earlier Hessians (max relative difference 0.0). | README | 2026-10-23 |
| 7 | Training | Pure PTQ for every row: no layer-wise or end-to-end fine-tuning (AQLM global fine-tuning and QuIP#/QTIP fine-tuning are not used). Every `quant.json` says so. | Mentor point 5 | 2026-10-23 |
| 8 | Bits accounting | README formula. Payload counts every stored tensor: codes, codebooks, scales, zero points, AWQ scales, padding, RHT signs. `quant.json` reports linear and whole-model bpw. | README; mentor point 2 | 2026-10-23 |
| 9 | RTN | Asymmetric min/max, groups of 128 along the input dim, FP16 scale and b-bit zero per group. SigLIP fc2 (4304 inputs) gets a shorter last group of 80; its scale and zero are counted. | README step 2a | 2026-10-23 |
| 10 | GPTQ | README reference implementation: groups of 128, 1% damping, block 128, no act-order. | README | 2026-10-23 |
| 11 | AWQ | Per-input-channel scale s = rms(x)^α with rms(x) from diag of the shared Hessian, α on a 20-point grid, chosen by the Hessian-weighted output error tr(E H Eᵀ). Then the official `auto_clip` on the scaled weight: per output row and group of 128, clip to ±max\|w\|·(1 − i/20) for i = 0…9 and keep the range with the smallest group output error eᵀH_gg e (exact with the shared second moment; the official code samples tokens). q/k projections are not clipped, as in the official code. RTN g128 asymmetric on the clipped weight. One FP16 scale vector per layer, counted (scales are per layer, so they cannot all fold into a shared norm); clipping stores nothing extra. | README silent; official llm-awq `auto_scale` + `auto_clip` with the shared Hessian; mentor point 4 | 2026-10-24 |
| 12 | VPTQ | W2 = v4-k256-0: one 8-bit index per 4 weights (2.000 bpw) plus one FP16 256×4 codebook per layer, counted. Official `algorithm` branch at `5a8dfb9`, Hessian-weighted k-means (100 iterations, seed 0), 1% damping, no outliers, no norm or permutation. Not the official v8-k65536 (its 1 MB codebooks add ~1.28 bpw that the official accounting ignores). | Mentor point 2 | 2026-10-23 |
| 13 | AQLM | W2 = 2x8g8: two 8-bit codebooks over groups of 8 weights (2.000 bpw) plus codebooks and per-output-channel scales, counted. Official `quantize_layer` with the shared Hessian, no global fine-tuning. | Mentor point 1; README step 2b | 2026-10-23 |
| 14 | QuIP#/QTIP dimensions | Randomized Hadamard transform. 1152 = 36 × 32 needs a size-36 Hadamard in the toolkit's table (verify when porting). 4304 = 16 × 269 has none: pad fc1 outputs / fc2 inputs to 4480 = 140 × 32 with zeros, quantize the padded matrix, and count the padded codes and all RHT sign vectors in the payload. | Mentor point 3; README step 2b | 2026-10-24 |
| 15 | Evaluation protocol | README: LIBERO 4 × 10 × 50 (and the 1900 column), LIBERO-Plus preregistered 1000 and the 2000 expansion, seed 7, paired bootstrap against the published rows. Plus 2000 = the Plus 1000 episodes + the 1000 added units (same units and seeds); the added 1000 are also reported separately, as the expansion file asks. | README | 2026-10-23 |
| 16 | Splitting evaluation | Plus in 250-unit parts, LIBERO one part per suite, packed into 12-hour PBS jobs (~11 h estimated work each). Episode outcomes do not depend on the split: seeds are set per unit and per episode. | Fit the 12 h walltime; resumable | 2026-10-23 |
| 17 | Sanity gate | Before a row's full evaluation: checkpoint structure check (embedding, exactly 414 layers changed, finite) and a 100-episode sample (50 LIBERO spatial + 50 Plus units, kept out of the results). A row with 0 successes, or ≤5% while the reference succeeds on ≥50% of the same episodes, is stopped and gets a debug job. The same rule runs on the full evaluation. | Fast feedback; user instruction | 2026-10-23 |
| 18 | RTN W2 | Stopped as a genuine collapse: 0/292 episodes (reference 248). Identity checkpoint and FP closed loop exact; RTN W3/W4 near FP; 2-bit RTN action error 1.38 (language model alone 0.76, SigLIP 0.19, expert 0.012). | Debug job `debug/rtn_w2_20261023_1849` | 2026-10-23 |
| 19 | Queue | At most 5 of the user's jobs waiting, no PBS dependency holds; jobs build their own prerequisites under file locks. A cron watchdog restarts the scheduler. | User instruction | 2026-10-23 |

## Open questions, decided

| # | Question | Decision | Reason | Date |
|---|---|---|---|---|
| D1 | Finish RTN W2's Plus 1000 for a complete number? | No, unless GPUs sit idle near the end. Report it as a collapse with the 292 paired episodes. | 0/292 already settles it; ~5 GPU-hours better spent on VQ rows | 2026-10-24 |
| D2 | AWQ W2 scores 28/100 on its sample. Is our AWQ complete? | No: official AWQ also searches a weight clipping range per group, which matters most at 2 bits. Add the clipping search (with the shared Hessian as the output-error metric), re-run AWQ W2's sample, and keep whichever version the official algorithm defines (with clipping) for the table. Pause AWQ W2's full evaluation until then. **Done 2026-10-24 00:38:** clipping added (decision 11; on a synthetic test it cuts the 2-bit Hessian-weighted error by 38%), the scale-only AWQ W2 checkpoint and sample deleted (`scheduler.py reset`), AWQ W2 rebuilt and re-sampled in bundle b028. | README priority 2: follow the official method | 2026-10-24 |
| D3 | Run the full 10,030-instance LIBERO-Plus? | No. | README marks it optional; deadline | 2026-10-24 |
| D4 | 3-bit / 4-bit partial results | Keep on disk and mention them in an appendix if useful; no further GPU time. | Decision 1 | 2026-10-24 |
| D5 | Commit the code? | Commit to a local branch `baselines-readme` at each milestone (row added, bug fixed); do not push until the user says so. | Reproducibility: snapshots then record a real commit | 2026-10-24 |
