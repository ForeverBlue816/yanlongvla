# GR00T N1.7 — AM-RAQ track: action-metric readout consistency (2026-10-23)

Status: scheduled 2026-10-23. Fine-tune jobs `groot_amraq_am` / `groot_amraq_am0` (221204 / 221205) are submitted by the controller (killable
QoS, resumable) behind row A (`groot_raqft_readout` 221202) and its distillation control (221203). GPU rule (user, 2026-10-23):
GR00T jobs take any free eligible GPU model (RTX 6000 Ada, L40S, A6000, A40, RTX Pro 6000; the controller moves a pending job to
whichever model has a free GPU); the GPU model is recorded in every row's complete.json and in the fine-tune manifests. Results are appended to §7 as they arrive; nothing in
§0–§6 changes after the first rollout.

## Specification (user, 2026-10-23, verbatim)

GR00T N1.7 — AM-RAQ track: action-metric readout consistency (2026-10-23).
Read PLUSTRAIN_2026-10-08.md §P2 and the GR00T RAQ-FT track spec (rows A/B/C) first. This track changes ONLY the readout metric inside the fine-tune loss. Trainable set, data, budget, optimizer, base checkpoint and evaluation protocol are those of row A unless stated.

## 0. Definition (fixed before any result)
Let o = {o_{l,t,h}} be the reader readouts (cross-attention outputs before W_O, 16 DiT reader blocks, 4 flow steps, all heads, all action tokens), o^FP from the FP model on its own trajectory, o^q from the quantized model on its own trajectory (student queries, same observation, same noise seeds). Δo = o^q − o^FP.
Row A (existing) uses the local metric: L_A = Σ_{l,t,h} ‖W_{O,h} Δo_{l,t,h}‖².
AM-RAQ uses the end-to-end action metric: the first-order change of the FP expert's velocity output when its readouts are perturbed by Δo, weighted in action space:
  L_AM = Σ_t ‖ Ω(t)^{1/2} · JVP( v_θ^FP ; o^FP_t → Δo_t ) ‖²
computed by one forward-mode (JVP) pass through the FP expert at flow step t on the FP trajectory, with the tangent injected at every reader block simultaneously (hook the cross-attention outputs; the tangent elsewhere is zero). This is ‖Ω^{1/2} J_{v←o} Δo‖² exactly, with all cross-block interactions, and no backprop through the sampling loop.
Ω(t): γ^h = 1 on the executed window, 0.5 on the rest, 0 on padded dims; λ_DoF = 1 / E‖J_d‖² per group (translation / rotation / gripper) measured once on the calibration set from the FP model; ω(t) uniform over the 4 steps. Record the measured λ values.
Total loss: L = L_AM / s + 0.1 · L_H5b, where s is the calibration-set mean of L_AM at step 0 of training (fixed scalar, so the gradient scale matches row A; record s). Stop-gradient on the FP side; gradients flow from the tangent into o^q and from there into the prefix codebooks, row scales and norms (row A trainable set).

## 1. Rows
A        raqft_groot_prefix  — local W_O metric (already specified).
A-AM     raqft_groot_prefix_am — L_AM + 0.1 H5b, same everything else.
A-AM0    raqft_groot_prefix_am_nob — L_AM only (β = 0), to see whether the action metric subsumes the velocity-direction term.
Same base checkpoint (groot_n17_ptq_w2), same 256 observations (bank a), 8 latents, 768 updates, final iterate, same joint-backbone / per-suite-head rule.

## 2. Implementation checks before training
(a) JVP correctness: on 2 observations, compare the JVP against finite differences (ε = 1e-3 along a random Δo direction) — relative error < 1e-3 in FP32.
(b) Cost: wall-clock per update for A vs A-AM on the same GPU; if A-AM exceeds 2× A, fall back to the static diagonal metric (Hutchinson estimate of diag(JᵀΩJ) per (l,t,h) from the FP model, 8 probes × 256 observations, stored once) and name the row raqft_groot_prefix_amdiag instead. Record which variant actually trained.
(c) Equivalence sanity: with Ω = I and the tangent restricted to a single block, L_AM should correlate with L_A (Spearman > 0.8 over 64 samples); report the number.

## 3. Evaluation (paired, same instances and noise seeds)
Held-out action MSE (256 obs, per DoF group) for base, A, A-AM, A-AM0 before any rollout. Roll out A-AM0 only if its held-out MSE is within 5% of A-AM; otherwise drop it.
Plus selection 1000 for A-AM (and A-AM0 if kept), paired vs A, vs base, vs FP.
Gate (pre-registered): A-AM is the main GR00T fine-tune row if its paired gain over A has 95% CI lower bound > 0; otherwise A stays main and A-AM is an appendix ablation ("action-metric vs local metric"). Confirm the chosen row on LIBERO medium 1000. No retuning of Ω, β or s after rollouts; any change is a new named row.
If A-AM passes, run the same metric on π₀.₅ as raqft_studentq_am (prefix trainable set, from the Fisher base), one Plus selection 1000 paired vs raqft_studentq (741), so the component is reported on both models.

## 4. Diagnostics for the paper (cheap, run regardless)
(a) Per reader block: ‖Ω^{1/2} J_{v←o_l}‖_F² from the FP model (the action sensitivity of each block's readout), text vs image tokens, per flow step. This is the figure that says which readouts matter for control.
(b) Readout error before/after fine-tune for A and A-AM, measured under both metrics (local and action), so the table shows what each loss optimized.

## 5. Priority and budget
A-AM runs immediately after row A in the queue, sharing its slot order; it must not delay the GR00T Plus FP/PTQ rows or the INT8-fusion jobs. Budget ≤ 1 GPU-day per fine-tune row, ≤ 4 h per Plus 1000. Hard stop: if A-AM has not started by 2026-10-28, keep A as main and report A-AM as an appendix row.

## 6. Reporting
One table: row, metric, held-out MSE (total, per DoF), Plus selection /1000 with Δ vs A / base / FP (pp, 95% CI), LIBERO medium if run, wall-clock per update, JVP/finite-difference check, s and λ values.

## Implementation notes (agent)

- Code: `scripts/groot_amraq.py` (fine-tune, modes `am` = A-AM, `am0` = A-AM0), `scripts/groot_offline.py` (held-out action MSE),
  rows `raqft_am_bb` / `raqft_am0_bb` in `groot_ptq.ROWS` (fit keys `backbone_am` / `backbone_am0` + the per-suite 2-bit heads),
  `plustrain_groot.PTQ_ROWS`, controller gates (`plustrain_controller.py`), `slurm/groot_amraq.sbatch` (envs/groot).
- Readout taps: forward pre-hooks on `attn1.to_out[0]` of the 16 cross-attention (reader) blocks of the AlternateVLDiT; the
  captured tensor is the pre-W_O cross-attention output over all heads and action tokens. The same hook injects a tangent Δo
  (added to the captured value) for the JVP / VJP / finite-difference evaluations.
- J Δo is computed with `torch.autograd.functional.jvp` over a function that injects the tangents at all 16 reader blocks and
  runs the FP head at step t on the FP trajectory (step features, AdaLN embedding, noisy action and flow time taken from the
  recorded FP trajectory); one forward-mode evaluation per step, exact (no finite differences, no truncation of cross-block
  interactions). The gradient to the student is g = 2 Jᵀ Ω u from one VJP, applied through a surrogate
  Σ (o^q · stopgrad(g) / s) whose backward equals dL/do^q; the FP side carries no gradient.
- Ω(t): γ^h = 1 for executed steps 0–7, 0.5 for steps 8–15, 0 for steps 16–39 and dims 7–31 (112 non-zero entries, checked);
  λ_DoF = 1 / E‖J_d‖² per group, Hutchinson over 4 random unit tangents × 4 steps × 32 calibration observations on the FP
  model; ω(t) uniform; s = mean of L_AM over the first 64 calibration observations at step 0. λ, s, the JVP/FD check and the
  Spearman check are recorded in `runs/plustrain_20261008/groot/raqft/backbone_am*/manifest.json` before training.
- Cost rule: two updates with the row-A loss vs two with L_AM at step 0 (wall-clock, including the 8 latents). If L_AM costs
  more than 2× row A, the run switches to the static diagonal metric D = diag(JᵀΩJ) (Hutchinson, 8 Rademacher probes ×
  256 observations × 4 steps on the FP model), re-calibrates s for that loss and records `variant = amdiag`
  (row reported as `raqft_groot_prefix_amdiag`); otherwise `variant = am_jvp`. The variant is also written to the fit's
  complete.json.
- Trainable set, data, optimizer and budget are row A's (`groot_raqft.py`, 2026-10-23 revision): backbone codebooks (FP32
  masters, FP16 round trip) + FP16 row scales; codes, RHT signs and the 2-bit heads frozen; 256 calibration observations
  (bank a), 8 latents per observation, 3 epochs = 768 updates, Adam 3e-4 / 2e-5, clip 1.0, final iterate; one backbone
  fine-tune with the Spatial head, evaluated with all four 2-bit heads. Deviations shared with row A: LayerNorm/RMSNorm
  parameters are not trainable (the 2-bit heads are frozen per the plan; the backbone norms stay at FP values), and the
  reader uses the student's own per-step hidden states as queries.
- Held-out action MSE (`groot_offline.py`): 256 held-out observations, `torch.manual_seed(index)` before each call so every row
  sees the same flow noise, actions in the dataset convention (16 steps × 7 dims, before the LIBERO env mapping), MSE vs the
  FP model total and per DoF group; A-AM0 is rolled out only if its total is within 5% of A-AM's.
- Rollout order and gates (controller): offline MSE → Plus selection 1000 for A-AM (and A-AM0 if the gate passes) → paired
  bootstrap of A-AM vs A (`story_plus_summary.paired`, 2000 draws, seed 20260930) → LIBERO medium for A-AM only if the 95% CI
  lower bound of the gain over A is > 0 (`groot/offline/am_gate.json`); otherwise A-AM is an appendix row.
- Diagnostics (per-block, per-step ‖Ω^{1/2} J_{v←o_l}‖_F² text vs image reader blocks; readout errors under both metrics for
  base / A / A-AM) run as a separate stage after the fits; π₀.₅ `raqft_studentq_am` only if A-AM passes the gate.

## Correction (2026-10-25): Δo is row A's readout difference

The first A-AM run (job 225092) took §0 literally: o^q from the quantized model's own reader on its own trajectory and o^FP from
the FP model on its own trajectory. Under that reading Δo also contains the 2-bit head's own attention error and the divergence of
the two trajectories, which the backbone-only trainable set cannot remove, and it is not row A's Δo. The spec states that this
track changes only the readout metric relative to row A, and check (c) (Spearman > 0.8 between the Ω = I single-block action
metric and L_A) presupposes a shared Δo; the literal run measured Spearman 0.157 and was stopped after 32 updates. Implemented
now (`readout_deltas` in groot_amraq.py): Δo_{l,t} = pre-W_O output of the FP head's cross-attention in block l with the
student's own step-t hidden state as query, over the student's tokens minus over the FP tokens (row A's reader, before W_O);
J_{v←o} is still evaluated by one JVP through the FP expert at FP step t with the tangent at all 16 reader blocks. Check (a) in
that run used a unit-norm tangent over ~1e6 readout elements, so ε = 1e-3 moved each element by ~1e-6 and the central difference
was FP32-noise dominated (rel. err. 1.18 / 0.41); it now uses an unnormalised N(0,1) tangent, and an FP32 CPU re-check
(groot_amraq_jvpcheck.py, ε ∈ {1e-2, 1e-3, 1e-4}, plus the VJP adjoint identity) is recorded in
runs/.../raqft/backbone_am/jvp_check_cpu.json. λ is now estimated with N(0, I) tangents, so it equals 1/E‖J_d‖_F² literally
(a common rescaling of all groups, cancelled by s). Checks (a) and (c) are now hard gates: a failing run stops before training.
The cost check of the literal run passed (78.0 s vs 48.6 s per update on an A40, ratio 1.60, exact-JVP variant).

JVP verification (CPU, 2026-10-25): in FP64 the central difference converges to the JVP at ~2e-8 relative error once the step no
longer crosses kinks of the piecewise-smooth expert (unit tangent, ε = 1e-3: 2.1e-8 / 2.6e-8), and the VJP satisfies the adjoint
identity to ~5e-8; in FP32 the ε = 1e-3 check stays at 2.6e-3 to 8.2e-3 (N(0,1) tangent, kink crossings) or 0.23 to 0.68 (unit
tangent, rounding) whatever the implementation. Deviation from §2(a): the gate runs the expert in FP64 (unit tangent, ε = 1e-3,
threshold 1e-3); the FP32 value is recorded next to it.

## Corrections (user, 2026-10-25, verbatim)

A-AM corrections (2026-10-25). Three changes to the spec; do not start the A-AM run until all four checks below pass as hard gates.

1. Δo definition — fix it explicitly and pin it.
   Δo is whatever row A computes: FP head cross-attention evaluated with the student's queries, student-token readouts minus FP-token readouts, taken before W_O. Write this definition in one sentence into the A-AM doc and into row A's doc, identically. Add hard check (d): on one fixed batch of 8 observations, the Δo tensors produced by the row-A code path and the A-AM code path must be bit-identical (max abs diff == 0, same shapes, same block/step/head ordering). A-AM differs from A in the metric applied to Δo and in nothing else.

2. Check (a) — accepted as revised.
   JVP vs finite differences in FP64 on CPU (relative error ≈ 2e-8) plus the adjoint identity ⟨JVP(u), w⟩ = ⟨u, VJP(w)⟩. Keep the FP32 numbers in the log for the record; they are not a gate. Record the deviation from the original spec as you have done.

3. Check (c) — replace it. The Spearman test is withdrawn; it compares two quantities that are not expected to correlate across samples, and 0.8 was never a meaningful threshold. The correct equivalence test is the truncated-pullback identity:
   With Ω = I and the JVP tangent truncated immediately after W_O of the same block (i.e. the pullback stopped at the first linear map), L_AM_trunc must equal L_A per sample: max over 64 samples of |L_AM_trunc − L_A| / L_A < 1e-5 in FP32 (< 1e-10 in FP64). This is an exact identity, not a statistical one; it is the "W_O is the first-order special case" claim in the paper, so it must hold to the bit up to rounding. Report the number. The 64-observation Spearman job (225276) can finish for information only; do not gate on it.

4. Gates and the 32 wasted steps.
   All four checks (a), (b), (c)-new, (d) run in the training script before the first optimizer step and abort on failure with the failing value printed. A failed check never produces a checkpoint or a training log entry. Verify this by injecting a deliberate failure once (e.g. scale the tangent by 2) and confirming the script stops.

5. Check (b) cost ratio is still outstanding: report wall-clock per update for A vs A-AM on the same GPU once row A finishes (~14:50). The 2× fallback rule to raqft_groot_prefix_amdiag stands.

6. s (the L_AM normalization scalar) and the λ_DoF values must be recomputed with the corrected Δo; the old values are void. Record the new ones.

Queue unchanged: A-AM (225277) after row A; A-AM0 and the distillation control stay killable. The 2026-10-28 start deadline for A-AM stands; if the four checks are not all green by 2026-10-27, report that instead of starting.

### Implementation of the corrections (agent, 2026-10-25)

Δo definition (pinned 2026-10-25, identical in GROOT_2026-10-15.md, GROOT_AMRAQ_2026-10-23.md, groot_raqft.py and groot_amraq.py): Δo_{l,t} is the pre-W_O output of the FP head's cross-attention in reader block l, evaluated with the student's own step-t hidden state as query over the student's backbone tokens, minus the same over the FP backbone tokens; the difference is taken in FP32 from the BF16 readouts.

- Check (d): `groot_raqft.row_a_deltas` (row A's call sequence, additive function; the row A training loop is untouched) vs
  `groot_amraq.readout_deltas` on observations 0–7 with the training seed of latent 0: same record count, same (step, block) order,
  same shapes, max |Δo_A − Δo_AM| == 0 in FP32, and the per-record ‖post-W_O difference‖² equal; in addition the FP trajectory used
  for the Jacobian (the real `get_action` after the same `torch.manual_seed`) must draw the same noise as the student trajectory,
  and replaying its step-1 readouts through the JVP's forward function must reproduce the captured readouts.
- Check (a): FP64 gate (unit-norm N(0,1) tangent over all readouts, ε = 1e-3, relative error < 1e-3) plus the adjoint identity
  ⟨JVP(r), w⟩ = ⟨r, VJP(w)⟩ (relative difference < 1e-6); the FP32 numbers are recorded, not gated.
- Check (c): truncated-pullback identity on 64 observations with the FP head in FP32 (max relative difference < 1e-5) and in FP64
  (< 1e-10); the JVP of the map o ↦ W_O o at o^FP with tangent Δo, summed over the 64 records of a sample, against
  Σ‖post-W_O(student tokens) − post-W_O(FP tokens)‖² from the same readouts. The Spearman test is withdrawn; its 64-observation
  CPU job (225276) finishes for information only.
- Check (b): wall-clock of 2 updates with row A's loop (line for line) vs 2 with L_AM on the same GPU, 8 latents each, reported
  in the manifest; > 2× selects the `amdiag` variant (never aborts).
- Order and mechanism: (d) → (a) → (c) → λ_DoF (N(0,1) tangents, 4 probes × 4 steps × 32 observations) → s (mean L_AM at step
  0 over 64 observations) → (b) → first optimizer step. Every gate writes checks.json; a failure writes checks_failed.json, prints the
  failing value and exits with status 3 before manifest.json, progress.json, history.json or any fit file exists.
  `--checks-only` runs (d), (a), (c) and exits; `--inject-fail jvp` scales the JVP tangent by 2 so that (a) must fail.
- λ_DoF and s of the literal run (λ ≈ 5.1e6 / 6.7e6 / 4.3e6, s ≈ 6.1e7) are void; the files of that run are kept under
  `runs/.../raqft/backbone_am/void_literal_20261025/`. The new values are recorded in the manifest of the run that trains.

### Gate results before the A-AM start (CPU, BF16 student, 2026-10-25; jobs 225379 / 225404)

| Gate | Value | Threshold | Result |
|---|---|---|---|
| (d) Δo bit-identity, 8 observations | max abs diff 0.0; shapes, (step, block) order equal; FP/student noise identical; JVP-step replay diff 0.0 | == 0 | pass |
| (a) JVP vs finite difference, FP64, unit tangent, ε = 1e-3 | 1.24e-8 | < 1e-3 | pass |
| (a) adjoint identity ⟨Jr, w⟩ = ⟨r, Jᵀw⟩, FP64 | 2.4e-12 | < 1e-6 | pass |
| (c) truncated pullback, 64 observations, FP32 | max rel. diff 2.66e-8 | < 1e-5 | pass |
| (c) truncated pullback, 64 observations, FP64 | max rel. diff 3.2e-16 | < 1e-10 | pass |
| injected failure (tangent × 2, job 225379) | (a) rel. err. 1.000, adjoint 0.500 → exit 3; only checks.json / checks_failed.json written | must stop | stopped |

(b) and the new λ / s are measured by the GPU run itself (same GPU for A and A-AM) and reported in §7 with the training result.
Withdrawn Spearman check, for information only (job 225276, 64 observations): 0.25 (single block, Ω = I) and 0.39 (full metric,
unit λ) against L_A.

## Row A follow-up and A-AM pre-start (user, 2026-10-25, second message) — implementation

1. Non-finite steps in row A. 29 of 768 updates (3.8%) had an infinite gradient norm (loss finite each time); 21 distinct
   calibration observations, the set is **not** identical across epochs (epoch 0: 10, epoch 1: 8, epoch 2: 11 observations;
   7 recur: 203, 205, 224, 225, 228, 240, 253; 225 in all three). 15 of the 21 are LIBERO-Spatial frames (tasks 30–38, the suite
   of the 2-bit head used in the fine-tune); the others are Goal/Object/10 frames (1, 25, 56, 57, 72, 129). Full list with
   episode / frame / task in `runs/.../raqft/backbone_raqft/nonfinite.json`. Diagnostic (`groot_raqft_nonfinite.py`, 4 recurring
   observations × latents 0–3 × student at base and at final masters, row A's step line for line): per-block/step readout loss,
   L_H5b, max |Δo|, max |o^FP|, non-finite forward outputs, and separate backward passes of the readout term and of the H5b term
   with the first modules carrying non-finite gradients and the parameters affected. Then (i) λ_DoF and s with and without the
   21 observations, (ii) the FP32 suite on exactly these observations (L_AM, JVP, gradients; 4-step dry run on the recurring
   four) — `groot_amraq.py --nonfinite-suite` → `backbone_am/nonfinite_suite.json`, which the training run reads as gate (e):
   A-AM does not train unless that file says passed.
   **Mechanism found (2026-10-25, GROOT_2026-10-15.md 'Non-finite updates'):** the FP16 round trip inside `TrainableVQLinear.weight()`
   casts the codebook / row-scale gradients to FP16 in backward; sums above 65 504 become inf (readout term only; H5b never).
   A-AM shares this layer, so the same overflow will occur in A-AM, and in the FP32 suite as well (the round trip is dtype-independent).
   **Pending user decision:** (A) straight-through FP16 round trip (forward bit-identical, FP32 gradients) in A-AM only — A-AM would
   then differ from row A in gradient numerics as well as in the metric; (B) the same fix in both rows and a re-run of row A
   (~15 h on an A40 plus queue); (C) no change, both rows keep the 5% drop cap and FP16-rounded gradients. Until decided, A-AM stays
   queued behind gate (e), which fails if the FP32 suite shows non-finite gradients.
2. Row A evaluation order: held-out action MSE (256 observations, total and per DoF group) base vs row A runs on CPU
   (`groot_offline.py`, GROOT_OFFLINE_DEVICE=cpu) before any rollout; Plus selection 1000 (paired vs FP and base) is queued as the
   primary (job 225656, rose); LIBERO medium (225555, rose) is kept as confirmation since the GR00T Plus FP / PTQ rows are complete.
3. Queue: A-AM keeps its rose copies; A-AM0 and the distillation control stay on killable QoS with one copy each and yield.
   **Rows B and C of the GR00T RAQ-FT track (trainable-set decision of 2026-10-20) are deferred past the GR00T deadline: they are
   untested, not abandoned.**
4. Check (b) stays a gate at 2× (ratio > 2 → `amdiag` variant); the ratio with the corrected Δo is measured by the GPU run.

### A-AM GPU gate run (A40, job 225570, 2026-10-25 07:48–08:00 UTC)

A rose copy reached an A40 before the FP32 non-finite suite had run. Gates on the GPU: (d) max |Δo_A − Δo_AM| = 0.0 (8 observations,
same shapes/order/noise, replay 0.0); (a) FP64 JVP vs finite difference 5.0e-9 / 1.3e-9, adjoint identity 2e-15 / 3e-15 (FP32
record only: 0.59 / 0.23); (c) truncated pullback max relative difference 1.6e-8 (FP32) and 2.5e-16 (FP64) over 64 observations.
Gate (e) found `nonfinite_suite.json` missing and stopped the run with exit 3 (checks.json / checks_failed.json only; no manifest,
no training). A-AM is held in the controller until the user's decision on the FP16-overflow fix (above) and the FP32 suite
(job 225654) are in; then it is resubmitted and repeats all gates.

## Decisions (user, 2026-10-26, verbatim)

Decisions (2026-10-26):
1. Overflow fix: option A — apply the FP16 row-scale gradient fix to A-AM only; do not rerun row A. Record in both row docs that row A dropped 29/768 steps (row-scale overflow only, codebook gradients never overflowed) and A-AM does not. Unpause A-AM's gate suite.
2. Branch on row A's Plus selection 1000 (paired vs base): if Δ ≥ +3 pp with 95% CI lower bound > 0, A-AM runs next as queued. Otherwise swap A-AM's slot for row B (prefix + DiT codebooks/scales trainable, same loss, same data and budget), and A-AM is deferred; it can later be run as B-AM.
3. Queue the backbone-only (FP head, 2-bit backbone) Plus selection 1000 now, ahead of A-AM0 and the distillation control. We need the Plus ceiling for prefix-side fine-tuning to read row A's number.
4. State which GPU model evaluated row A's LIBERO medium; if it was the A40, flag the mismatch in the table and re-evaluate on a 6000 Ada when a slot is free.

### Implementation (agent, 2026-10-26)

- Gradient numerics: row A dropped 29 of 768 updates (3.8%) because row-scale gradients overflowed FP16 in the trainable layer's round trip (row-scale overflow only; codebook gradients never overflowed on real data); row A is not re-run. A-AM uses a straight-through FP16 round trip for the row scale (forward bit-identical, FP32 row-scale gradients) and does not drop updates for this reason; its codebook path is row A's (option A, user decision 2026-10-26). Code: `groot_amraq.row_scale_ste` (asserts per layer that the forward weight is unchanged); recorded in checks.json
  (`row_scale_ste_layers`). Unit test on fitted layers: row-scale gradients finite under a 1e6 output-gradient stress (max 2.1e9);
  codebook gradients still overflow under that synthetic stress, as decided, but never did on real data.
- Gate suite unpaused: the FP32 non-finite suite now runs under rose QoS on CPU (226172) with the fix in place; gate (e) reads it.
- Branch (§2): `plustrain_controller.groot_rowA_branch()` pairs row A's Plus selection with base when it completes and writes
  `groot/offline/rowA_branch.json`; A-AM is submitted only if `next == 'A-AM'` and the suite passed; otherwise row B is prepared
  for A-AM's slot.
- §3: the backbone-only Plus selection 1000 already exists (run 2026-10-21 on an RTX 6000 Ada): **569/1000** (FP 618, base 486,
  head-only 2-bit 519). It is the Plus ceiling for prefix-side fine-tuning against which row A is read; no new run needed.
- §4: row A's LIBERO medium (888) ran on an **RTX A40**; FP / base / backbone-only ran on RTX 6000 Ada. A GPU-matched re-run on an
  RTX 6000 Ada is queued (226174, pinned, output `medium_6000ada`). Row A's Plus selection runs on an RTX 6000 Ada (matched).

### Branch outcome (2026-10-25 13:10 UTC)

Row A Plus +2.4 pp vs base [−1.9, +6.6] → row B (GROOT_2026-10-15.md, 'Row B') takes A-AM's slot; A-AM is deferred and can be run later as B-AM on top of row B. The FP32 non-finite suite job for A-AM was cancelled; B-AM will get its own suite on row B's masters. All A-AM code and gates stay in place.

### B-AM (user plan + amendment 2026-10-25): run in parallel with row B

`groot_amraq.py --mode b_am` (row `raqft_b_am`, fit keys `backbone_bam` / `head_bam_<suite>`): row B's trainable set, routing and Δo
(`groot_rowb.build_models`), L = L_AM / s + 0.1 L_H5b with Ω and λ_DoF as in the A-AM spec; λ (pooled over suites, each observation
through its own suite's FP head) and s recomputed for row B's set; row-scale straight-through round trip. Gates before the first
optimizer step: (d) Δo bit-identity with row B's (= row A's) code path per routed suite, (a) FP64 JVP / adjoint, (c) truncated
pullback (FP32 and FP64, routed heads cast per observation), (e) FP32 non-finite suite on row A's 21 observations (routed), (b) cost
rule vs row B's loop on the same GPU (> 2× → diagonal metric per suite head, row `raqft_groot_b_amdiag`). Prerequisites (e) and a
checks-only (d)/(a)/(c) pass run as one GPU job on a 96 GB RTX Pro 6000 (CPU nodes booked until ~10-28); the B-AM fine-tune is
submitted when both pass. The FP32 suite of the deferred A-AM was cancelled.

### B-AM gates and prerequisites (2026-10-25)

GPU gate run on an RTX Pro 6000 (226390): (d) Δo bit-identity with row B's code path 0.0 (shapes/order/noise identical, replay 0.0);
(a) FP64 JVP vs finite difference 2.4e-8 / 2.4e-8, adjoint 3.7e-15 / 1.0e-15; (c) truncated pullback 1.9e-8 (FP32) / 6.1e-16 (FP64);
(e) FP32 suite on row A's 21 non-finite observations (routed to their suites): 21/21 finite (L_AM, JVP, gradients), 4-step dry run
4/4 finite. λ_DoF (pooled over suites, N(0, I) tangents) = translation 19.27, rotation 21.34, gripper 63.77; removing the 21
observations changes λ by 0.6% / 1.0% / 3.1% and the step-0 L_AM mean by 5.7% (s here = 109.8 with unit metric scaling of part (i);
the training run recomputes s with its own λ and records it). B-AM fine-tune submitted (226502, rose, race copies); (b) is measured
by that run before its first optimizer step.

### B-AM result (2026-10-26 12:50 UTC)

Plus selection 1000 (RTX 6000 Ada, paired): **496/1000** (camera 246/500, robot init 250/500). Δ vs base 486: **+1.0 pp [−3.5, +5.1]**;
vs row A 510: −1.4 pp [−7.3, +4.1]; vs backbone-only ceiling 569: −7.3 pp [−12.6, −1.7]; vs FP 618: −12.2 pp [−17.5, −7.2]. By
category vs base: camera −2.2 [−9.1, +4.1], robot init +4.2 [−1.2, +9.7]. Held-out action MSE vs FP (RTX Pro 6000, same device as
base): 1.16e-2 vs base 7.53e-3 (+54%; translation +78%, rotation +44%, gripper +40%) although the training loss fell from 0.19 to 0.12
with no dropped updates. Pre-registered reading: B-AM's gain over base has a CI that includes 0 → B-AM is not the GR00T main row; the
metric comparison B-AM vs row B (paired Δ) is reported when row B's Plus completes (2026-10-27).

## 7. Results

Held-out action MSE vs FP (256 observations, CPU, same noise per observation; total / translation / rotation / gripper):

| Row | metric | total | translation | rotation | gripper | Plus /1000 | LIBERO medium | s/update | checks |
|---|---|---:|---:|---:|---:|---|---|---|---|
| base `groot_n17_ptq_w2` | — | 6.30e-3 | 5.00e-3 | 1.85e-4 | 2.86e-2 | 486 | 872 | — | — |
| ceiling: backbone-only 2-bit (FP head) | — | | | | | 569 | 927 | — | — |
| A `raqft_bb` | local W_O (L2 after W_O) | 6.41e-3 | 4.72e-3 | 1.31e-4 | 3.03e-2 | 510 (+2.4 pp vs base [−1.9, +6.6]; −10.8 vs FP; −5.9 vs ceiling) | 888 (+1.6 pp vs base [−1.7, +5.1]) **A40 — GPU mismatch, 6000 Ada re-run queued** | 71 (A40) | 29 non-finite dropped (E140) |
| A-AM `raqft_am_bb` | action metric (JVP) | pending | | | | | | | (d)/(a)/(c) green; (e) pending |
| A-AM0 `raqft_am0_bb` | action metric, β = 0 | pending | | | | | | | |

(A-AM / A-AM0 not started. Branch 2026-10-26: row A Plus Δ vs base +2.4 pp [−1.9, +6.6] does not meet Δ ≥ +3 pp with CI lower bound > 0 → **A-AM deferred**, to be run later as B-AM on top of row B.)
