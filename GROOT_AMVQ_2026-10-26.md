# GR00T — AM-VQ track: action-metric codebook fitting for the 2-bit base (2026-10-26)

Status: scheduled 2026-10-26 05:30 UTC; one killable GPU job (`groot_amvq`, race copies on 6000 Ada / L40S / A40 / RTX Pro 6000)
runs Fisher → fit → gate 1 → diagnostics and yields to the rose rollouts of row B / B-AM. Results are appended in §Results.

## Specification (user, 2026-10-26, verbatim)

GR00T AM-VQ track (2026-10-26): action-metric codebook fitting for the 2-bit base. Runs now, in parallel with the Plus evaluations of row B and B-AM; it must not delay them.

Definition: as in the AM-VQ spec of 2026-10-23 — Kronecker metric G_out ⊗ H_in per quantized Linear layer, with G_out = diag E[δ²] from Hutchinson probes of ⟨v_θ, Ω^{1/2} ε⟩ through the full FP GR00T graph, and Ω identical to the one used in B-AM (same executed-window weights and the same λ_DoF values; reuse B-AM's recorded λ, do not recompute). H_in, RHT seeds, calibration observations, codebook format, bpw and the GPTQ-style column feedback are those of groot_n17_ptq_w2. Only the metric in the K-means assignment, centroid update, second RVQ plane and row-scale WLS changes. Uniform ω only (no "late" variant).

Row: groot_amvq_w2 (2-bit backbone + 2-bit head, same layer set as the base). Verify bpw byte-identical to the base.

Gate 1 (offline, before any rollout): held-out action relative MSE on the 256 held-out observations, total and per DoF group, vs groot_n17_ptq_w2. Continue only if total MSE improves by ≥ 5% relative. Otherwise stop, report, and keep it as an appendix row.
Gate 2: Plus selection 1000 on RTX 6000 Ada, paired vs base (486) and FP (618). AM-VQ counts as a component only if Δ vs base has 95% CI lower bound > 0.

If gate 2 passes and the time allows: one fine-tune from the AM-VQ base, using whichever of row B / B-AM has the better Plus result, same trainable set, loss, data and budget. It must start by 2026-10-28 12:00 Beijing time; otherwise skip it and report AM-VQ as a PTQ-level result only.

Diagnostics (regardless of gates): per-layer G_out energy concentration by module (vision / LLM / vl_self_attention / DiT), and per-row DoF shares; same figures as the B-AM readout-sensitivity plot, so the paper can show the metric on both the codebook and the readout side.

Priority: row B Plus → B-AM Plus → AM-VQ Fisher + fit + gate 1 → AM-VQ Plus → (conditional) fine-tune from AM-VQ base → row B / B-AM medium.

## Implementation (agent, 2026-10-26) — `scripts/groot_amvq.py`

- **G_out (stage `fisher`).** For each of the 432 quantized Linear layers (216 backbone, shared; 216 per suite head) and each
  calibration observation (bank a, 256; each observation routed to its own suite's FP head as in B-AM): the FP trajectory is taken
  from the deployed `get_action`; then, with the graph made differentiable at the vision patch embedding, the text embedding and
  the head's state / action encoders (all parameters frozen), the backbone tokens and the four flow-step velocities v_t are
  recomputed and the outputs of every quantized layer captured. For 2 probes per observation, ε_t ~ N(0, I) over the 40 × 32 action
  window (independent per step), the scalar Σ_t ⟨v_t, Ω_g^{1/2} ε_t⟩ is differentiated once per DoF group g (Ω_g = Ω restricted to
  that group's dims) with `torch.autograd.grad` w.r.t. all captured outputs (one backward through the full FP graph per group and
  probe); G_out^g[o] accumulates Σ_tokens δ_o² over the layer's calls (DiT layers: 4 steps). Because the groups are independent,
  G_out = Σ_g G_out^g (unbiased for diag E[δ²] under Ω) and G_out^g / G_out are the per-row DoF shares. Ω: executed steps 1, steps
  8–15 0.5, padded 0, with B-AM's recorded λ_DoF (backbone_bam/manifest.json: translation 19.27, rotation 21.34, gripper 63.77; not
  recomputed). Normalisation: per key, by observations × probes (backbone over all 256; each head over its suite's 74 / 66 / 65 / 51).
  Checkpointed every 16 observations (requeue-safe).
- **Fit (stage `fit`).** `backbone_vq_fit.fit_backbone` — the π₀.₅ action-Fisher fit (group-wise √Fisher residual weighting in
  K-means assignment and centroid update, both RVQ planes, row-scale / codebook WLS, GPTQ-style column feedback with the full
  layer H) — with fisher[o, g] = G_out[o] for every input group g, i.e. the Kronecker metric G_out ⊗ H_in. H_in = the base's
  calibration Hessian of the same layer, RHT signs = `layer_signs(name, in_features)` (the base's seeds), 2 planes × 256 × 8, FP16 row
  scale, the base's 10-byte mask tensor. Per layer the payload is asserted equal to the base's, tensor shapes / dtypes and signs
  identical; per fit key the summed payload is asserted equal to the base's. Also recorded per layer: the action-metric error
  (G_out-weighted H-error) of the base fit and of the AM-VQ fit, plain relative weight MSE and H-error for both.
- **Row** `groot_amvq_w2` = fit keys `backbone_amvq` + `head_amvq_<suite>` (both 2-bit, the base's layer set), assembled per suite by
  `groot_ptq.stage_assemble`; `accounting.json` compared with the base's (linear payload bytes, layer count) in gate 1.
- **Gate 1 (stage `gate1`).** Held-out action MSE (256 held-out observations, same noise per observation, dataset action convention,
  total and per DoF group) for FP, base and AM-VQ on this job's GPU (`groot_offline.py`, per-device directory); pass iff the total
  MSE vs FP improves by ≥ 5% relative to the base → `amvq/gate1.json`. Only then does the controller queue the Plus selection
  (RTX 6000 Ada, pinned, rose, after row B's and B-AM's Plus in plan order).
- **Gate 2.** Plus selection 1000 paired vs base (486) and FP (618); component iff the 95% CI lower bound of Δ vs base > 0 →
  `amvq/gate2.json`. If passed and both row B and B-AM Plus are complete, the better of the two recipes is fine-tuned from the AM-VQ
  base (`GROOT_FT_BASE=amvq`: same trainable set, loss, data, budget; fit keys `backbone_rowb_amvq` / `backbone_bam_amvq`, rows
  `raqft_rowb_amvq` / `raqft_b_am_amvq`; for B-AM the gate suite and checks run first on the same GPU). The decision and its time are
  written to `amvq/finetune.json`; if no decision could be made by 2026-10-28 04:00 UTC (12:00 Beijing) the fine-tune is skipped
  and AM-VQ is reported as a PTQ-level result only.
- **Diagnostics (stage `diag`, regardless of gates).** Per layer: share of Σ G_out in the top 1% / 10% output rows, Gini, per-row DoF
  shares and the fraction of rows dominated by each DoF group, grouped by module (vision / LLM / vl_self_attention / DiT) →
  `figures/groot_amvq_gout.png`. Readout side for B-AM: ‖Ω^{1/2} J_{v←o_l}‖_F² per reader block and flow step (Hutchinson, N(0, I)
  tangents at one block, 32 observations routed to their suites, FP model, B-AM's Ω) → `figures/groot_bam_readout_sensitivity.png`;
  numbers in `amvq/diagnostics.json`.
- **Scheduling.** The AM-VQ job runs on killable QoS so the rose Plus runs of row B / B-AM preempt it; row B / B-AM LIBERO medium
  runs wait until the AM-VQ Plus and the fine-tune decision are settled (or gate 1 fails, or the deadline passes).

## Results

(pending)
