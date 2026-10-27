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
On the 6000 Ada freed by B-AM: alloc per-layer fits and readout screen (estimate ≤ 6 GPU-h; the 414×3 fits are reuse of the frozen fitter) → alloc_v1 Plus selection → raqft_on_vptq fine-tune → alloc_v1 RAQ-FT → their Plus runs. Report GPU-hours before starting the fits.
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

**K-FAC.** First probe run (228366) ended on an assertion: the last LLM block's q/o/MLP layers receive no gradient from the readout
objective (only that block's K/V reach the expert), so G = 0 and D̂ = 0 for those layers is now recorded as structurally unused (the
fitter treats their action-Fisher the same way). Re-run inside the `post` job.
