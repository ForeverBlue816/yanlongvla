# π₀.₅ — plane-allocation track (2026-10-26)

Status: scheduled 2026-10-26 13:20 UTC (user: "根据他的调整下我们的实验, raqft_on_vptq 这个可以先别做" — raqft_on_vptq stays cancelled, the
rest of §4 applies). Results are appended in §Results.

## Specification (user, 2026-10-26, verbatim)

π₀.₅ plane-allocation track (2026-10-26). Equal-storage reallocation of RVQ planes across layers, built on the frozen a8_studentq pipeline. Fake-quant evaluation as for every other row; kernel support for a per-layer plane count is a separate deployment task.

## 0. Format constraint (fixed)
Every quantized Linear keeps the frozen format: RHT, 8-weight groups, 256×8 INT8 codebooks, row scales. The only free variable is the plane count p_l ∈ {1, 2, 3} per layer (1.0 / 2.0 / 3.0 bits per weight for its codes). No intra-layer mixing, no other code widths. Storage constraint: total code bytes (plus the extra codebooks) must not exceed the all-2-plane base; report linear and whole-model bpw with the same accounting; target linear bpw ≤ 2.02.

## 1. Allocation rule (pre-registered)
Fit all 414 layers at 1, 2 and 3 planes with the frozen fitter (same calibration data, Hessians, RHT seeds; sequential residual fit, p=1 is the first plane, p=3 adds a third residual plane). Measure, per layer and per plane count, the held-out readout error with all other layers at FP: the same readout metric as RAQ-FT (W_O-weighted cross-attention outputs of the expert reading the prefix, 256 held-out observations), and separately the expert-side error for expert layers. Compute marginal utility per byte: u_l(2→3) = Δerr / Δbytes and u_l(2→1) likewise. Then greedy equal-storage reallocation: repeatedly move a layer 2→1 where the error increase per byte is smallest, and spend the freed bytes 3-planes-first on the layers with the largest error reduction per byte, until no swap improves the sum. Freeze the resulting allocation as alloc_v1 before any rollout and record it (per-layer p_l, bytes, predicted Δerr).
Two allocations to fit and screen, nothing else:
  alloc_v1 — the greedy result above.
  alloc_prior — a hand-set control: expert at 1 plane everywhere; backbone k_proj and v_proj at 3 planes (the tensors the expert reads directly); remaining freed bytes to the backbone layers with the highest D1 propagated-error score, 3 planes each, until the budget is spent.

## 2. Screens (PTQ level, before fine-tune)
Held-out action MSE and readout error for base (2-plane, 704), alloc_v1, alloc_prior. Roll out Plus selection 1000 for alloc_v1 (and alloc_prior only if its held-out numbers are at least as good as alloc_v1's), paired vs original_2bit 704 and FP 765. This is the PTQ-level result; report it regardless.

## 3. Fine-tune and final row
Apply the frozen raqft_studentq recipe unchanged to alloc_v1 (prefix codebooks / scales / norms trainable, 768 updates, same data and seeds), then A8 → a8_studentq_alloc. Plus selection 1000 paired vs a8_studentq 741, then added 1000 paired vs 698, then LIBERO 2000 paired vs 1941.
Gate: a8_studentq_alloc replaces a8_studentq as the main row only if the paired gain on Plus selection has 95% CI lower bound > 0 AND added-1000 agrees in sign. Otherwise it is reported as the equal-storage allocation ablation. No retuning of the allocation after rollouts; a new allocation is a new named row.

## 4. Queue and budget
Cancel AM-VQ fitting (keep its diagnostics). B-AM finishes its Plus run. Row B continues on the A40 with its Plus run tomorrow.
On the 6000 Ada freed by B-AM: alloc per-layer fits and readout screen (estimate ≤ 6 GPU-h; the 414×3 fits are reuse of the frozen fitter) → alloc_v1 Plus selection → ~~raqft_on_vptq fine-tune~~ (reassigned to Chamadol 2026-10-28, see baselines/raqft_handover) → alloc_v1 RAQ-FT → their Plus runs. Report GPU-hours before starting the fits.
Deadlines: PTQ-level alloc result by 2026-10-29; a8_studentq_alloc Plus selection by 2026-10-31; added 1000 and LIBERO by 2026-11-02.

## 5. Reporting
Table: row, planes per module (vision / LLM attn / LLM MLP / expert), linear and whole-model bpw, held-out MSE, readout error, Plus selection /1000 with Δ vs 704 / 741 / FP and 95% CI, added 1000, LIBERO 2000. Plus one figure: per-layer marginal utility per byte for 2→3 and 2→1, grouped by module, so the allocation is explainable.

## Implementation (agent) — `scripts/plustrain_alloc.py`, controller keys `alloc_*`, rows `alloc_v1`, `alloc_prior`, `raqft_studentq_alloc`, `a8_studentq_alloc`

**Fits.** The frozen fitter's outputs already exist for most plane counts and were checked for provenance: backbone p=2 = the base's
own per-layer fits (`models/hd_srvq_backbone_layers/M2`, byte-identical SHA to the base checkpoint), backbone p=3 =
`hd_srvq_backbone_layers/M3` (same calibration moments / Fisher SHAs and the same action-Fisher weighting as the M2 fits), expert
p=1/2/3 = `hd_srvq_a_layers/M1..M3` (same recipe SHA and moments as the base's expert). Only the 288 backbone layers at p=1 are new:
`backbone_vq_fit.fit_backbone(planes=1)` on the base's calibration artifacts (`runs/backbone/calibration`, moments / Fisher SHAs
asserted equal to the base fits'), RHT signs from `layer_signs` (asserted equal to the base's), row scale asserted equal to the
base's. The fitter's `planes in (2,3)` assertion was widened to (1,2,3); the 2- and 3-plane paths are unchanged. Byte accounting per
layer and plane: codes out×in/8 bytes + one FP16 256×8 codebook (4 096 B); row scale, signs and mask are plane-independent.
All-2-plane code+codebook bytes: 679 468 032.

**Screen (one error per layer and plane count, every other layer at FP, 256 held-out observations).** FP engine (`HybridEngine('fp')`,
no CUDA graphs); per observation the FP prefix KV, the FP expert trajectory with its readout records (`plustrain_raqft.Readout`,
queries='fp', W_O metric — exactly the RAQ-FT metric) and the FP teacher path are computed once. Then for every (layer, p) the FP
`nn.Linear` is swapped for the deployed quantized module built from the stored fit tensors (`BackboneVQLinear` / `VQLinear`, the
same arithmetic as the quantized policy) and:
- backbone layers: the prefix is recomputed and `Readout.loss` gives the readout error of the perturbed KV against the FP readouts
  (mean over 10 steps × 18 expert layers of ‖W_O Δo‖², RAQ-FT's reduction); also relative to the FP readout energy;
- expert layers: the expert-side error = Σ_t ‖v_q(x_t) − v_fp(x_t)‖² along the FP trajectory with FP prefix KV; also relative to
  Σ_t ‖v_fp‖².
Per-observation results are saved (resumable; two GPU workers).

**Pre-registered interpretation choices (fixed before any screen result):**
1. The utilities use *relative* errors (readout error / FP readout energy for backbone layers; velocity error / FP velocity energy
   for expert layers) so that the two error kinds share a scale; `err[l, p]` is the mean over the 256 observations.
2. Greedy rule as specified: downgrade (2→1) the layer with the smallest relative-error increase per byte freed; spend the freed
   bytes (plus any unspent carry) on upgrades (2→3) in order of the largest reduction per byte; commit the swap set only if the sum of
   errors decreases (a downgrade that costs nothing is also taken); stop when no swap improves the sum. Total bytes never exceed the
   base; unspent bytes are reported.
3. `alloc_prior`: expert at 1 plane; LLM k_proj / v_proj at 3 planes; the remaining bytes go, 3 planes each, to backbone layers ranked by
   a D1 propagated-error score defined as: per LLM block i, s_i = mean over {K, V} × {image, language_state} of the reader-weighted
   propagated fraction of the KV error (D1 of `raqft_studentq`); a layer in block j (not k/v_proj) scores Σ_{i>j} s_i, vision / projector
   layers Σ_i s_i (upstream of every block); ties by smaller plane bytes. D1 has no per-Linear score, so this is the agent's reading
   and is recorded as such.
4. alloc_prior rolls out only if both its held-out action MSE and its full-model readout error are ≤ alloc_v1's.

**Assembly.** Base checkpoint tensors with the chosen fits' `codes` / `codebooks` (and the mask byte set to 2^p − 1); signs and row
scale asserted equal to the base's; `assemble_evening.write_checkpoint` → `models/hd_srvq_alloc_v1` / `alloc_prior` with the base's
accounting (`whole_model_bpw` = payload × 8 / active original elements; linear bpw recomputed from the per-layer payloads); total
payload ≤ base asserted.

**Screens, rollouts, fine-tune.** `--stage offline`: held-out action MSE vs the FP reference actions (`offline_metrics.relative_action_mse`,
same as every A8 row) and the full-model readout error for base / alloc_v1 / alloc_prior. Plus selection 1000 via the frozen
`plustrain_run.py` (same units and noise seeds as 704 / 741 / 765; the π₀.₅ Plus rows run on the rollout GPU type of the earlier rows,
L40S / 6000 Ada first free). `raqft_studentq_alloc` = the frozen `raqft_studentq` recipe with base `alloc_v1` (the trainable layer
installer sums over the stored plane count per layer); `a8_studentq_alloc` = its A8 composition; selection → added 1000 → LIBERO 2000
paired as specified.

**GPU-hours (reported before the fits start; measured durations of the same stages in this pipeline where available).**

| Stage | GPU-h (estimate) | basis |
|---|---:|---|
| backbone p=1 fits (288 layers) | ~1.7 | M2 fits took 1.68 h, M3 2.34 h |
| per-layer screen (414 × 3, 256 obs) | ~6 (2 workers × ~3 h) | ~60 ms prefix forward per backbone eval, 10 expert steps per expert eval; refined after the first observation |
| allocate + assemble + held-out screens | ~0.7 | |
| alloc_v1 Plus selection 1000 | ~3.2 | raqft_studentq Plus took 3.2 h |
| alloc_prior Plus (conditional) | ~3.2 | |
| raqft_studentq_alloc fine-tune | ~1.3 | raqft_studentq took 1.3 h |
| a8_studentq_alloc offline + Plus selection | ~8.7 | a8_studentq Plus took 8.4 h |
| added 1000 | ~10 | a8_studentq confirmation took 10 h |
| LIBERO 2000 | ~12 | a8_studentq full took 12 h |
| **PTQ-level result (through alloc_v1 Plus)** | **~12 (15 with alloc_prior)** | above the ≤ 6 GPU-h estimate for fits + screen alone (~8) |
| **full track** | **~47–50** | |

## Addendum to §1 (user, 2026-10-26, verbatim) — K-FAC predictor

Addendum to §1 (K-FAC predictor, run alongside the measured search; it does NOT decide the allocation):
For every layer l and plane count p ∈ {1,2,3}, also compute the Kronecker-factored first-order estimate of the readout distortion, D̂_l(p) = Σ_i G_l,ii · (ΔW_l(p))_i H_l^in (ΔW_l(p))_iᵀ, with H_l^in the existing layer Hessian and G_l = diag E[δ_l²] from Hutchinson probes of the readout objective (8 probes × 256 calibration observations, FP model, < 1 GPU-hour for all 414 layers). From it derive the predicted marginal utility per byte for 2→3 and 2→1, exactly as for the measured values.
Report, per module (vision / LLM attention / LLM MLP / expert) and overall: Spearman correlation between predicted and measured utility, and the rank agreement of the top-20 layers to upgrade and the top-20 to downgrade. Also run the greedy allocation once on the predicted utilities only (alloc_kfac, no rollout) and report how many layers' plane counts differ from alloc_v1 and the measured readout error of alloc_kfac vs alloc_v1.
The frozen allocation remains alloc_v1 (measured). If predicted and measured agree (overall Spearman ≥ 0.8 and alloc_kfac within 2% readout error of alloc_v1), the paper presents K-FAC as the analytic allocation rule with the search as validation; otherwise K-FAC goes to the appendix as a predictor comparison.

### Implementation of the addendum (agent) — `plustrain_alloc.py --stage kfac`, controller key `alloc_kfac`

- **Probe.** FP engine; for each of the 256 calibration observations (bank a) one grad-enabled pass: prefix forward (KV depends on the
  backbone layer outputs), then the FP expert's 10 flow steps with the attention instrumented to compute, with gradient, the FP32
  cross-attention readout over the FP prefix KV at every expert layer and project it with W_O (the RAQ-FT readout objective, 180
  tensors); the trajectory is held fixed between steps (x_t detached), as the measured screen holds the FP queries fixed. Every
  quantized Linear's output is captured (the graph is rooted at a layer's input where nothing upstream requires grad). For 8 probes
  ε ~ N(0, I) over the readouts, δ_l = ∂⟨R, ε⟩/∂(output of l) by one `autograd.grad` each; G_l,ii = mean over observations × probes of
  Σ_tokens δ². Deterministic seeds (20261026, observation index).
- **D̂.** In the RHT-transformed basis (the fitter's; the quadratic form is invariant): ΔW = W_t − Ŵ_t(p) from the stored fit tensors,
  H^in = the base's calibration Hessian of the layer (backbone: `runs/backbone/calibration` moments; expert: the step-averaged
  expert input moments of the base fit), D̂_l(p) = Σ_i G_l,ii (ΔW_i H ΔW_iᵀ) in float64.
- **Comparison (in `allocate`, written to `alloc/kfac/comparison.json`).** Predicted utilities per byte with the same byte accounting;
  Spearman (scipy) between predicted and measured u(2→3) and u(2→1) per module and overall; overlap of the top-20 upgrade and top-20
  downgrade sets; `alloc_kfac` = the same greedy procedure on D̂ (one objective, raw scale), the number of layers whose plane count
  differs from alloc_v1, and `figures/plane_alloc_kfac_vs_measured.png`. `alloc_kfac` is assembled and gets the held-out screens
  (full-model readout error and action MSE) but no rollout. Decision recorded in `alloc/offline.json` (`kfac_decision`): analytic rule
  iff overall Spearman ≥ 0.8 for both utilities and alloc_kfac's full-model readout error is within 2% of alloc_v1's. The frozen
  allocation remains alloc_v1.
- Cost: probe ≈ 256 × (1 forward + 8 backward) ≈ 15–20 min; D̂ ≈ 10 min (1 242 quadratic forms). Runs as its own rose job alongside the
  screen once the p=1 fits exist (D̂ needs them); nothing in the measured search depends on it.

## Results

**Fits and screen (2026-10-26/27, RTX 6000 Ada).** Backbone p=1 fits: 288 layers in 1.1 GPU-h (three worker jobs, dynamic claiming).
Per-layer screen: 256 observations in 10.6 GPU-h (one worker did 250 observations, ~150 s per observation; the ≤ 6 GPU-h estimate
was too low by ~1.8×). Mean relative error per module (other layers FP; backbone = readout, expert = velocity):

| module | n | p=1 | p=2 | p=3 | p1/p2 | p3/p2 |
|---|---:|---:|---:|---:|---:|---:|
| vision | 162 | 3.69e-4 | 8.39e-5 | 5.45e-5 | 4.4 | 0.65 |
| LLM attention | 72 | 1.88e-3 | 2.70e-4 | 7.75e-5 | 7.0 | 0.29 |
| LLM MLP | 54 | 3.15e-3 | 6.48e-4 | 1.75e-4 | 4.9 | 0.27 |
| expert | 126 | 3.38e-4 | 3.64e-5 | 7.33e-6 | 9.3 | 0.20 |

Preview of the greedy rule on these numbers (the frozen alloc_v1.json is written by the `post` job, which also assembles and screens):
79 layers → 1 plane, 203 → 3 planes, 132 stay at 2; by module (1 / 2 / 3 planes): vision 8 / 58 / 96, LLM attention 3 / 10 / 59, LLM MLP
8 / 41 / 5, expert 60 / 23 / 43; predicted change of the summed relative error −31% (−0.0228 of 0.0726). The expert's low relative
velocity errors make it the main donor of bytes (60 of 126 expert layers at 1 plane); the relative scales of the two error kinds are
a pre-registered choice (see above) and the held-out screens / Plus decide.

**Updated GPU-hour estimate for the remaining stages (user 2026-10-27 §2; measured stages of this pipeline, the screen-type stages scaled
by the 1.8× overrun).** post (K-FAC probe + D̂, allocate, assemble, held-out screens incl. additivity) ~1.3 GPU-h; alloc_v1 Plus
selection ~3.5 h; alloc_prior Plus (conditional) ~3.5 h; raqft_studentq_alloc fine-tune ~1.5 h (starts after the alloc_v1 Plus result,
regardless of it); a8_studentq_alloc offline + Plus selection ~9 h; added 1000 ~10 h; LIBERO 2000 ~12 h. Remaining ≈ 37–41 GPU-h;
spent so far 11.7 GPU-h (fits 1.1, screen 10.6). PTQ-level result expected 2026-10-28 (deadline 10-29); a8_studentq_alloc Plus
selection 2026-10-29/30 (deadline 10-31); added 1000 + LIBERO by 2026-11-01/02 if the L40S queue cooperates.

**Additivity check.** `offline.json` also reports, for base / alloc_v1 / alloc_prior / alloc_kfac, the sum over backbone layers of the
per-layer relative readout errors at the chosen plane counts against the measured full-model relative readout error (ratio
measured / predicted): how additive the single-layer screen is.

**K-FAC.** First probe run (228366) ended on an assertion: the last LLM block's q/o/MLP layers receive no gradient from the readout
objective (only that block's K/V reach the expert), so G = 0 and D̂ = 0 for those layers is now recorded as structurally unused (the
fitter treats their action-Fisher the same way). Re-run inside the `post` job.

**Status 2026-10-27 05:00 UTC (user follow-up §4).** fit1 and screen complete (E160); post job 228912 (rose, L40S) pending since 01:30 UTC with a Slurm
start estimate of 2026-10-31; a duplicate copy 229297 was submitted on an RTX 6000 Ada (allowed fitting GPU), first to start wins, the other is cancelled.
Frozen allocations, held-out screen, additivity check and the alloc_v1 Plus run all wait on this job.

**Status 2026-10-27 15:40 UTC.** Allocations frozen (E164): alloc_v1 / alloc_prior / alloc_kfac. K-FAC Spearman overall 0.74 (2→3) / 0.63 (2→1) < 0.8
(expert 0.18 / 0.31; LLM attention 0.98 / 0.97). Post job 229297 failed after writing alloc_v1/alloc_prior (matplotlib missing in the openpi env); the
GPU part (assembly + held-out screen + additivity) is re-queued as race copies 230282 (6000 Ada) / 230283 (L40S), rose.

## Pre-registration of alloc_v2 / alloc_v2_kfac (user decisions 2026-10-28 morning; recorded 2026-10-27 16:05 UTC, before any held-out screen or rollout of an assembled allocation)

- **alloc_v2**: expert — all 126 layers at 1 plane (justified by the lossless whole-expert-1-plane row, 193/200). Backbone (vision, LLM attention,
  LLM MLP; 288 layers) — the current backbone budget plus the bytes freed by the expert, allocated by the same greedy as alloc_v1 on the measured
  readout utilities only, under the same storage ceiling (code + codebook bytes ≤ the uniform 2-plane base). Greedy as frozen in
  `plustrain_alloc.greedy` (the rule alloc_v1 was built with): the freed bytes are first spent on 2→3 upgrades in order of the largest
  readout-error reduction per byte; then, repeatedly, downgrade (2→1) the backbone layer with the smallest error increase per byte freed and
  spend the freed bytes (plus carry) on further upgrades, committing a swap set only if the summed error decreases (a downgrade with no error
  increase is taken as a free improvement). *Note on the user's wording "free-downgrade rule (< 1% increase)": the frozen alloc_v1 code has no
  1%-threshold; its free-downgrade rule is "no increase". alloc_v2 uses the identical code so the two candidates differ only in the expert
  rule and the budget; the wording is recorded here for the user to overrule.*
- **alloc_v2_kfac**: the same expert rule; the backbone allocated by the same greedy on the K-FAC predicted utilities (D̂) instead of the measured
  ones. Reported: the number of backbone layers whose plane count differs from alloc_v2, and, after the screen, the readout-error difference.
- **Frozen now**: alloc_v1 (primary, frozen 2026-10-27 15:30 UTC), alloc_v2 (second pre-registered candidate), alloc_v2_kfac, alloc_prior. Any re-run of
  the allocate stage asserts the frozen plane counts are reproduced.
- **Screen** (one GPU job, or CPU if no GPU has started it by 2026-10-28 20:00 Beijing): held-out readout error and action MSE for base, alloc_v1,
  alloc_v2, alloc_v2_kfac, alloc_prior; additivity check for v1 and v2 (and every assembled row).
- **Rollouts**: Plus selection 1000 for alloc_v1 and alloc_v2 only (paired vs base 704 and FP 765); alloc_v2_kfac and alloc_prior get none.
  RAQ-FT + A8 proceeds on whichever of v1 / v2 has the better Plus selection count (tie → v2, fewer expert bytes); the fine-tuned row is gated vs
  a8_studentq 741 as before.
- **K-FAC rule (amended before the screen)**: decided on the backbone only. If alloc_v2_kfac is within 2% readout error of alloc_v2 and differs
  from it in fewer than 10% of the 288 backbone layers (< 28.8, i.e. ≤ 28), the paper presents K-FAC as the allocation rule for the prefix with the
  search as validation; otherwise K-FAC is an appendix predictor comparison. The module-level Spearman table (LLM attention 0.98, LLM MLP 0.90,
  vision 0.85, expert 0.18 for 2→3) is reported either way.

**Frozen allocations (2026-10-27 16:20 UTC; `alloc/alloc_v2.json`, `alloc/alloc_v2_kfac.json`; alloc_v1 / alloc_prior re-verified identical).**

| allocation | bytes (ceiling 679 468 032) | predicted Σ rel. error change (screen metric) | vision 1/2/3 | LLM attn 1/2/3 | LLM MLP 1/2/3 | expert 1/2/3 |
|---|---|---|---|---|---|---|
| alloc_v1 (primary) | 679 401 216 | −0.0228 | 8 / 58 / 96 | 3 / 10 / 59 | 8 / 41 / 5 | 60 / 23 / 43 |
| alloc_v2 | 679 414 528 | +0.0136 | 10 / 88 / 64 | 3 / 13 / 56 | 7 / 35 / 12 | 126 / 0 / 0 |
| alloc_v2_kfac | 679 408 384 | (predicted on D̂) | 12 / 56 / 94 | 3 / 13 / 56 | 7 / 37 / 10 | 126 / 0 / 0 |
| alloc_prior | 678 980 864 | +0.0244 | 0 / 25 / 137 | 0 / 36 / 36 | 0 / 54 / 0 | 126 / 0 / 0 |

- alloc_v2: the expert rule frees 39 444 480 bytes; 53 504 bytes unspent. On the per-layer screen metric the forced expert downgrade costs more than
  the backbone upgrades recover (+0.0136 vs −0.0228 for v1) — exactly the expert-scale question the screen cannot settle (the 1-plane expert row was
  lossless in rollouts, 193/200), hence both v1 and v2 get the Plus selection.
- The pre-spend of the freed bytes packs fewer, larger vision layers to 3 planes (64 vs 96 in v1): the per-byte ranking with one large budget fills
  with the highest-utility-per-byte layers until the budget is exhausted, whereas v1's swap loop upgrades small layers one downgrade at a time.
- alloc_v2_kfac differs from alloc_v2 in **52 of 288 backbone layers (18%)** — already more than the 10% of the amended K-FAC rule, so K-FAC is the
  appendix predictor comparison unless the screen shows otherwise on the readout-error half (the rule needs both halves).

**Held-out screen (job 231187, RTX 6000 Ada, 2026-10-28 08:34–09:12 UTC; `alloc/offline.json`; 256 held-out observations, FP reference).**

| row | held-out action MSE (relative) | readout error (relative, vs FP energy) | additivity: measured / Σ per-layer prediction | whole-model bpw |
|---|---|---|---|---|
| base (uniform 2 planes) | 0.0372 | 0.1179 | 1.73 | 2.9226 |
| alloc_v1 | **0.0256** | 0.0697 | 1.45 | 2.9224 |
| alloc_v2 | 0.0295 | **0.0626** | 1.43 | 2.9225 |
| alloc_v2_kfac | 0.0301 | 0.0663 | 1.45 | 2.9224 |
| alloc_prior | 0.0319 | 0.0796 | — | 2.9214 |

- Both pre-registered candidates beat the base on both screens at equal storage: alloc_v1 −31% action MSE / −41% readout error; alloc_v2 −21% / −47%.
  They split: v1 has the lower action MSE, v2 the lower readout error (its whole backbone budget is larger). The Plus selection decides (both roll out).
- Additivity: the per-layer screen under-predicts the full-model readout error by 1.43–1.73×, consistently across allocations (the base worst), so the
  per-layer utilities rank allocations correctly but are not additive in absolute value.
- **K-FAC rule (amended, backbone only): fails on both halves** — alloc_v2_kfac is 7.1% worse than alloc_v2 in readout error (limit 2%) and differs in
  52/288 = 18% of backbone layers (limit 10%). K-FAC is the appendix predictor comparison; the module-level Spearman table is reported either way.
- alloc_prior: no rollout (user 2026-10-28); for the record it is worse than both candidates on both screens.

## 2026-10-28 14:30 UTC — user decision (afternoon): any GPU type for the Plus rows; fine-tune both candidates now

Slurm's start estimate for the two rose L40S Plus jobs (231799 / 231800, submitted 10:20 UTC) was 2026-11-02; the user decided the time is too
tight. Changes, all recorded here:

1. **Plus selection for alloc_v1 and alloc_v2 may run on any GPU type.** Both rose jobs were cancelled and replaced by one killable 2-GPU pack
   on A40 (job **232232**, `slurm/alloc_plus_pack.sbatch`, one model per GPU; requeue on preemption, resume from `episodes.jsonl`). The Plus guard
   (`plustrain_guard.py`) accepts a non-L40S device only with `PLUSTRAIN_ANY_GPU=1` and now records the device in its proof. **Caveat:** the
   reference counts 704 (base) and 765 (FP) were measured on L40S; the kernel arithmetic differs across GPU models (E162: FP actions differ
   cross-device, MSE 1.2e-4 on GR00T), so a non-L40S alloc count vs 704 is a cross-device comparison. The paired bootstrap is still on identical
   units and noise seeds; the device of every row is stated next to the number.
2. **RAQ-FT on both candidates now, not after the Plus.** Rows `raqft_studentq_alloc_v1` (base alloc_v1) and `raqft_studentq_alloc_v2` (base
   alloc_v2), the frozen `raqft_studentq` recipe unchanged (768 updates, student queries, bank (a)), one killable 2-GPU pair job on A40
   (**232230**, `slurm/alloc_raqft_pair.sbatch`; resume from `training_state.pt` on preemption). The **selection rule is unchanged**: the winner is
   decided by the PTQ-level Plus counts (tie → v2, `alloc/winner.json`), and only the winner's fine-tune continues as `a8_studentq_alloc` (W2A8,
   gated vs a8_studentq 741). The loser's fine-tune is kept on disk, not rolled out. Cost: one extra ~9 h fine-tune.
3. The comparison targets stay as stated in the morning: alloc_v1 / alloc_v2 (PTQ bases, no fine-tune) vs the uniform base **704**; the fine-tuned
   winner vs **749** (raqft_studentq) and its A8 row vs **741**.
4. Killable submit budget (5 per user) now: B2 229308 (one copy, GROOT_COPIES 2 → 1; copy 232155 cancelled), row B medium 230618, figure v3
   capture 232076, the pair 232230 and the pack 232232.

## alloc_v3 — pre-registration (user spec 2026-10-29; written 2026-10-28 16:40 UTC, before the first probe job)

One-shot end-to-end test of the improved analytic predictor. Does not delay the alloc_v1 / alloc_v2 Plus pack (232232) or their fine-tune pair
(232448): the probe job is submitted on the rose QoS (no killable slot taken); those two stay first in the killable queue.

### §1 Predictor (replaces the diagonal-G K-FAC)
For every quantized layer l (414) and p ∈ {1, 2, 3}, with ΔW_l(p) = Ŵ_l(p) − W_l from the existing fits (RHT basis, exactly as `stage_kfac_dhat`
reconstructs Ŵ: Σ_{q<p} codebook_q[codes_q] × row scale; the layer input is transformed by the same RHT signs, as the deployed module does):

  D̂_l(p) = mean over observations n and probes s of ⟨g_{l,n,s}, ΔW_l(p)⟩²,  ⟨g, ΔW⟩ = Σ_τ δ_τᵀ (ΔW x_τ),

computed in hooks (layer input x_τ captured in the forward pass, output gradient δ_τ from the backward pass of the probed objective; g is
never materialised; per layer and plane one matmul x ΔWᵀ, then the probe-wise inner product with δ). Probes: Rademacher v_s, S = 4, seed
SeedSequence([20261029, n, s]); N = 256 calibration observations (bank a, `datasets/libero-calibration`), latent per observation
`make_latent(20261005)(100000 + n, 0)` (the screen's and the K-FAC probe's rule); all quantities at the FP model (`HybridEngine('fp')`, no CUDA graph).
- Backbone layers (288): r_n = the RAQ-FT readout objective = the 180 W_O-projected FP32 cross-attention readouts of the FP expert over the FP prefix
  KV (10 steps × 18 expert layers, FP queries along the FP trajectory, trajectory held fixed between steps as in the K-FAC probe); objective
  v_sᵀ r_n with v_s Rademacher over all 180 × (8 × 10 × 1024) entries; backprop to every backbone layer (the prefix pass is grad-enabled).
- Expert layers (126): r_n = the final action chunk of the 10-step sampler (prefix KV fixed, computed without grad), unnormalised with the policy's
  own output transform (an affine map per action dim, so v_sᵀ a_n = (v_s ⊙ scale)ᵀ x_10 + const; the scale is read from the policy and verified
  against `engine.infer` on the first observation), restricted to the executed window and valid dims a[:5, :7] (the held-out action MSE mask
  `offline_metrics.relative_action_mse`); v_s Rademacher over the 35 entries; backprop through the unrolled 10 steps (x_{t+1} = x_t + dt·v_t with
  grad through x_t). Expert weights are shared across steps: the inner product is accumulated over all 10 steps per probe before squaring,
  ⟨g, ΔW⟩ = Σ_t Σ_τ δ_{t,τ}ᵀ ΔW x_{t,τ}.
- Also stored: the Hutchinson estimate of the exact Gauss–Newton value for the uniform-2 model, D̂_total(2) = mean_{n,s} (Σ_l ⟨g_{l,n,s}, ΔW_l(2)⟩)²,
  separately for the backbone objective (sum over the 288 backbone layers) and the expert objective (sum over the 126 expert layers), next to
  Σ_l D̂_l(2) → the predictor's own additivity ratio.
- Checks before use (recorded in `alloc/v3/checks.json`): on three layers (one LLM attention, one LLM MLP, one expert layer; the first of each
  module in model order) compare the hook inner product ⟨g_{n,s}, ΔW_l(2)⟩ and D̂_l(2) against central finite differences of the probed objective
  along ΔW_l(2) (f(W + εΔW) − f(W − εΔW)) / 2ε with only that layer perturbed, FP64, 2 observations (n = 0, 1), S = 4; relative error < 5 %
  required. FP64 on CPU is specified; the engine hard-codes CUDA, so the FP64 check runs on the GPU in FP64 (same arithmetic precision; stated
  in the record). On the expert layer, additionally verify that the step-accumulated D̂ differs from the per-step-sum Σ_t (Σ_τ δᵀΔW x)² (recorded
  value and ratio; must differ).

### §2 Calibration (Design 4; existing measurements only)
Per module m ∈ {vision, LLM attention, LLM MLP, expert}: predicted utilities U_pred(2→3) = D̂(2) − D̂(3), U_pred(2→1) = D̂(1) − D̂(2); measured
utilities from the held-out screen `alloc/utilities.json` (relative readout error for backbone layers, relative expert velocity error for expert
layers; other layers FP). Calibration layers: 10 per module, stratified by predicted utility — for 2→3 the 5 layers at ranks ⌊k·(n_m − 1)/4⌋,
k = 0..4, of U_pred(2→3) sorted ascending; for 2→1 the same for U_pred(2→1) (if a layer is picked twice it counts once and the set is not refilled).
Fit log U_meas = log s_m + β_m log U_pred on those points by ordinary least squares (points with a non-positive utility on either side are
dropped from the fit and listed); calibrated utility U_cal = s_m · U_pred^{β_m} for every layer of the module. Report s_m, β_m, then Spearman
between calibrated predictions and ALL measured utilities per module and overall, for 2→3 and 2→1, in the K-FAC table's format
(K-FAC: vision 0.85 / 0.79, LLM attn 0.98 / 0.97, LLM MLP 0.90 / 0.77, expert 0.18 / 0.31, overall 0.74 / 0.63 for 2→3 / 2→1).
Units: backbone calibrated utilities are relative readout error; they are converted to relative action-MSE units with one exchange rate λ fitted
through the origin by least squares on the allocation-level pairs (Δ readout error, Δ held-out action MSE vs the base) of alloc_v1, alloc_v2,
alloc_v2_kfac and alloc_prior from the screen (job 231187). Expert calibrated utilities are relative expert velocity error along the FP
trajectory and are used as action units directly (λ_expert = 1), as the spec states.

### §3 Allocation
Exact multiple-choice knapsack by dynamic programming over the byte budget (bytes scaled by the gcd of the per-plane byte counts) over 414 layers
× {1, 2, 3}, byte ceiling **679 468 032** (= 2 planes everywhere, as for v1 / v2), objective = Σ_l U_cal,l(p_l) in action units (maximise the
predicted MSE reduction). The expert is NOT forced to one plane. Reported: planes per module (1/2/3 counts), what the expert gets, layers differing
from alloc_v1 and from alloc_v2, predicted total relative action MSE = 0.0372 (base, screen) − predicted reduction, per layer whether the utility
curve is convex (U(1→2)/byte ≥ U(2→3)/byte) and the number of violations. Frozen as `alloc/alloc_v3.json` before any screen or rollout.

### §4 Offline screen (GPU if available, CPU allowed)
alloc_v3 assembled as `models/hd_srvq_alloc_v3` (payload ≤ base asserted); held-out readout error, relative action MSE (total and per DoF),
additivity ratio, next to base / v1 / v2 / v2_kfac (same script, one job, device recorded). No PTQ-level rollout for v3.

### §5 End-to-end
`raqft_studentq_alloc_v3` = the frozen raqft_studentq recipe on alloc_v3; `a8_studentq_alloc_v3` = + A8. Plus selection 1000 on the same GPU
model as a8_studentq (741; L40S), paired vs a8_studentq and vs the fine-tuned measured-allocation row a8_studentq_alloc (winner of v1 / v2), same
units and seeds. Pre-registered reading: (i) predictor = the paper's allocation rule if a8_studentq_alloc_v3 is non-inferior to a8_studentq_alloc
(paired Δ 95 % CI lower bound ≥ −3 pp) and beats 741 (CI lower bound > 0); (ii) if v3 beats the measured row with CI lower bound > 0, v3 is the
main row and the measured search the validation experiment; (iii) otherwise the predictor goes to the appendix with the Spearman table and the
measured allocation stays. No retuning of S, the calibration layer set, λ or the solver after any rollout.

### §6 Budget, order, two flags
Probe + predictor ≈ 2–3 GPU-h (backbone: 256 × 4 backward passes + 288 × 3 × 256 matmuls; expert: 256 × 4 unrolled 10-step backward passes).
**Flag 1 — fine-tune time:** the spec budgets 1.5 h; the frozen raqft_studentq recipe took **9.06 h** on a 6000 Ada (job 197014) and is used
unchanged, so the chain is ≈ 9 h + A8 offline 0.3 h + Plus ≈ 3 h. **Flag 2 — the 2026-11-02 deadline:** rose L40S start estimates are currently
2026-11-02 for any new job; the deadline is only reachable if the Plus rows of a8_studentq_alloc and a8_studentq_alloc_v3 may use another GPU
type (then both run on the same type, paired). Order: the probe job queues now on rose without touching the killable budget of the v1 / v2 Plus
and fine-tunes; the fine-tune of alloc_v3 starts only after `raqft_studentq_alloc_v1/_v2` are complete (controller rule). The spec's mention of
an "alloc_v2_kfac Plus" is noted: v2_kfac has no rollout under the 2026-10-28 morning pre-registration and none is added.

**2026-10-28 18:00 UTC (user):** no deadline on any of this; use every available GPU. The 2026-11-02 remark in the alloc_v3 pre-registration §6 is
therefore not a constraint (the spec's date is kept as written, the flag is withdrawn). Race copies added so each line is queued on every allowed
GPU type (loser cancelled when one starts): alloc Plus pack 232232 (A40, killable) + rose copies on L40S / 6000 Ada; RAQ-FT pair 232448 (6000 Ada,
killable) + rose copy on L40S; alloc_v3 232494 / 232495 (rose, 6000 Ada / L40S).
