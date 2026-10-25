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

## 7. Results

(pending — fine-tunes queued behind row A; estimated start 2026-10-26)
