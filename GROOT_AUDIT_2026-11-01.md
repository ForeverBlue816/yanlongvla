# GR00T base audit (user, 2026-11-01): is a broken layer behind the 486?

## Specification (user, verbatim)

> Urgent GR00T base audit (2026-11-01). Chamadol's pure-PTQ QTIP (573) and QuIP# (554/531) beat our GR00T base (486) and QTIP beats B2 (556); on π₀.₅ our base is above every baseline. Check our GR00T plain-VQ fit for a broken layer: (1) per-layer relative output error for all 432 layers, flag any layer > 3× the module median; (2) the LLM layer-2 mlp.down_proj specifically — QuIP# needed 100× damping there because of a ~6000× Hessian outlier channel; report our damping, the layer's error, and the max/mean diagonal ratio of its Hessian; (3) refit any flagged layers with 10× / 100× damping, re-assemble, held-out MSE, and if the flagged set changes the model, Plus selection 1000 as groot_n17_ptq_w2_v2. Also publish B2, base and FP per-unit Plus files to the GR00T harness results/. Report GPU-hours first; this goes ahead of the AM-VQ chain's remaining steps except jobs already running.

## Budget (estimate, reported first)

Refit 33 layers × 2 dampings ≈ 0.5–1 GPU-h; re-assembly CPU; held-out MSE ≈ 0.2 GPU-h; Plus selection 1000 ≈ 2.2 h on one RTX 6000 Ada
(B2's Plus: 2 h 12 min, job 235123). Total ≈ 3–4 GPU-h. The AM-VQ chain's running jobs (raqft_studentq_amvq Plus 236798, a8_studentq_amvq
Plus 236808) were not touched; its remaining step (added 1000 for the A8 row) waits behind this audit.

## (1) Per-layer relative output error — `runs/plustrain_20261008/groot/ptq/audit_layer_errors.json` (CPU, from the fit metadata)

Metric: tr(ΔW H ΔWᵀ) / tr(W H Wᵀ) on the 256 calibration observations (`hessian_relative_error` of every fit record), 432 layers per suite
checkpoint (216 backbone shared + 216 head × 4 suites = 1080 fits). Module median = median over the layers with the same module name
(DiT self-attention (odd) and cross-attention (even) blocks separated; 39 module classes). Medians: backbone 0.0053, head 0.0021.

**Flagged (> 3× module median): 32 of 1080; none catastrophic (largest flagged error 0.020).**

| Where | Layers | Error range | Ratio to median |
|---|---|---|---|
| DiT self-attention blocks 19–31 `attn1.to_k` (all four suites) | 20 | 0.0032–0.0048 | 3.1–4.8 |
| DiT self-attention blocks 24–30 `attn1.to_q` (libero_10 only) | 5 | 0.0046–0.0057 | 3.0–3.5 |
| DiT `ff.net.2` blocks 16, 22, 23, 24 (libero_10 ×4, libero_goal ×1) | 5 | 0.0014–0.0023 | 3.1–5.1 |
| Vision tower `blocks.11.mlp.linear_fc1`, `blocks.14.mlp.linear_fc2` | 2 | 0.0204, 0.0162 | 3.2, 3.1 |

The largest errors in absolute terms are the cross-attention `attn1.to_v` of DiT blocks 0–16 and `vl_self_attention.blocks.3.to_out`
(0.028–0.029, relative weight error 0.15–0.17) — at their module median, not outliers.

## (2) LLM layer-2 `mlp.down_proj` — `audit_hessian_layer2.json`

| | layer 0 | layer 1 | **layer 2** | layer 3 |
|---|---|---|---|---|
| Hessian diag max | 5.98 | 18.9 | **2.44 × 10⁶** (channel 1821; next largest 3.76) | 1.75 |
| diag mean | 0.100 | 0.030 | 397 | 0.017 |
| **max / mean** | 60 | 620 | **6144** | 101 |
| max / median | 119 | 2234 | 2.5 × 10⁸ | 158 |
| our damping ÷ median diag | 0.02 | 0.04 | **406** | 0.016 |

Our damping: GPTQ feedback with 1 % of the mean diagonal (`vq_core.gptq_encode_uniform(damping=.01)`, after the fitter normalises H by
its mean diagonal; `fit_uniform` had no damping argument until this audit). On layer 2 that is 3.97 absolute = 406× the median diagonal:
the feedback is effectively switched off for every channel except the outlier. Recorded error of the layer: 0.0000 (the metric is
dominated by channel 1821), relative weight error 0.131 (normal); **error with the outlier channel excluded: 0.0553** (`audit_layer2_base_excl.json`) — 7× the down_proj
median (0.0078) and the largest of any layer in the model; diagonal-only error without the channel 0.146 ≈ the plain weight error 0.131, i.e.
the feedback gave the other 6143 channels nothing, while the outlier column itself is reproduced to 1.5e-6. On the outlier-excluded metric
layer 2 is flagged; it is in the refit set.

## (3) Refit plan

`groot_ptq.py --stage refit` (new; `vq_fit.fit_uniform(damping=…)` passthrough): the 32 flagged layers + layer-2 down_proj at relative
damping 0.1 and 1.0 (10× / 100× the base 0.01), outputs `groot/ptq/fit/refit_d<damping>/<key>/`; layer 2 additionally at 0.001, 1e-4, 1e-5 and 1e-6
(agent addition, second job groot_audit_refit_l2: on that layer the base damping is already 406× the median diagonal, so lower is the direction that can change anything).
Each refit record carries the full-H error and the error with the max-diagonal channel excluded. `--stage v2` composes
`fit/backbone_v2` + `fit/head_v2_<suite>` (base fits, flagged layers replaced where a damped refit has the lower error) → row `ptq_w2_v2`
(`models/groot_ptq/ptq_w2_v2/<suite>`), `groot_offline.py` row `ptq_w2_v2` (held-out MSE vs FP, same device as the base row), then
`plustrain_groot.py --stage plus --row groot_n17_ptq_w2_v2` if any layer was replaced. Refit race 236878 (6000 Ada rose) / 236879 (L40S) /
236880 (A40), first to start wins.

## Published

`public-reports/baselines/groot/results/`: `groot_fp` (618), `groot_2bit_ours_ptq` (486, + accounting), `groot_2bit_ours_b2` (556, + accounting)
per-unit Plus files; README section; commit 65cda2a.

## Results

