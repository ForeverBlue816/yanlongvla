# Figure `fig_readout_sensitivity` — versions, rules and printed numbers

Files: `figures/fig_readout_sensitivity_v3.{pdf,png,json}` (current), `figures/fig_readout_sensitivity_v3_supp_language_share_per_layer.png`,
script `figures/fig_readout_sensitivity_v3.py` (reads only `amvq/diagnostics.json` and the two capture outputs below). Font TeX Gyre Termes
(Times-compatible; Times New Roman is not installed on the cluster), mathtext `stix`. Layout 7.2 × 2.5 in, width ratios 1.3 : 0.85 : 0.95, wspace 0.3.
Previous versions: v2 (`…_v2.*`, E161/E169, 2026-10-27/28), v1 (E159).

## v3 (user 2026-10-28) — what changed and why

**Panel (c) replaced.** v2's (c) overlaid the expert's reader-block-0 cross-attention (last flow step) over the third-person image for FP / 2-bit PTQ /
ours on one frame (libero_spatial task 260, noise seed 1194133538, the policy call before the first gripper close). Measured on that frame (E169): at
expert layer 0, last step, only **7 %** of the attention row reaches the third-person image and **72 %** goes to the language tokens; within the image the
map is near-uniform (target-box share 0.023 vs 0.016 uniform; the maximum cell is a border cell in every model), so the three overlays are
indistinguishable. The informative quantity is the **token-group split**, which shifts under 2-bit quantization. Rule change recorded here as v3:

- **(c1)** token-group attention of the expert's first reader layer (layer 0), last flow step (9 of 10), π₀.₅-LIBERO: three horizontal stacked bars
  (FP, 2-bit PTQ = `bb_M2_ae_M2_fp32`, ours = `raqft_studentq`); segments language (#2F2F2F α 0.85, white label), third-person image (#D8ECFA), wrist
  image (#EBD1E3), action tokens (#FFF4A8). Values = mean over the **256 held-out observations** (`datasets/libero-heldout`; never used for fitting),
  **same latent per observation across models** (`make_latent(20261005)(100000 + i, 0)`, the screen's rule); the language share is printed inside the
  bar; to the right of the 2-bit and ours bars the **paired difference of the language share vs FP** with a 95 % bootstrap interval (2000 draws over
  observations, seed 20260930, percentile). The same numbers for all 18 expert layers are in the JSON (`c1.per_layer_mean_share_last_step`,
  `c1.per_layer_delta_vs_fp_language_share`) and in the supplementary line plot (language share per layer, three models).
- **(c2)** (optional, only with (c1) done): an image-reading layer, selected **on FP only over the 256 calibration observations** = the expert layer with the
  largest mean third-person-image attention mass (last step). Same frame as v2; three sub-images: the FP map, |A_2bit − A_FP|, |A_ours − A_FP|, one
  shared colour scale, L1 distance to FP printed under each. **Dropped if the selected layer's image mass is below 20 %** (the JSON records the
  per-layer FP image mass and the decision either way).
- **(a)** fixes: step 0 / step 1 end labels in a darker tone of their line colour (#7FA8CF, #5B93C9); y-axis upper limit 5e-5; right margin trimmed
  (xlim +1.7 → +1.2 blocks; the labels must stay clear of panel (b)'s y labels, which is why it is not trimmed further).
- **(b)** unchanged. Top of (c) aligned with (a) and (b).

Data: `scripts/paper_fig_c1_capture.py` (openpi env, any GPU type; no CUDA graph; the expert's eager attention hooked at all 18 layers at flow steps 0 and
9, mean over 8 heads × 10 action tokens, summed per token group: third-person image [0, 256), wrist image [256, 512), padding image [512, 768) (masked,
asserted 0), language [768, L−10), action tokens [L−10, L)) → `runs/plustrain_20261008/diagnostics/fig_c1/{groups.npz, capture.json}`; the v2 frame
capture `diagnostics/fig_c/{attention.npz, capture.json, frame.png}` is reused for (c2).


## v3 printed numbers (capture job 232076, A40, 2026-10-29 ~05:00 UTC; `diagnostics/fig_c1/{groups.npz, capture.json}`)

Panel (c1), reader block 0, last flow step, mean over the 256 held-out observations (same latent per observation across models;
attention row sums 0.9996–1.0004, padding-image mass 0):

| model | language | third-person image | wrist image | action tokens | Δ language vs FP (95 % paired bootstrap) |
|---|---|---|---|---|---|
| FP | 0.753 | 0.078 | 0.126 | 0.043 | — |
| 2-bit PTQ | 0.688 | 0.101 | 0.153 | 0.058 | -0.065 [-0.066, -0.065] |
| ours | 0.719 | 0.094 | 0.140 | 0.047 | -0.035 [-0.035, -0.034] |

Under 2-bit quantisation the language share of the first reader layer drops by 6.5 pp (PTQ) and the fine-tune recovers about half of it
(ours −3.5 pp); the shift goes to the two image segments and the action tokens. The difference is negative on every one of the 256
observations for both models (fraction positive 0.000), hence the narrow intervals.

**(c2) dropped by the pre-registered rule.** FP image-reading layer on the 256 calibration observations: the largest mean third-person-image
mass is layer 4 with 0.126 < 0.20, so no image-map panel is shown; the per-layer FP image mass is in the JSON
(`c1.c2_selection.fp_calibration_image_mass_by_layer`) and in the last column below.

Per expert layer (last step; language share; Δ vs FP with 95 % CI; FP third-person-image mass on the calibration set):

| layer | FP | 2-bit PTQ | ours | FP image mass (cal.) |
|---|---|---|---|---|
| 0 | 0.753 | 0.688 (-0.065 [-0.066, -0.065]) | 0.719 (-0.035 [-0.035, -0.034]) | 0.080 |
| 1 | 0.875 | 0.873 (-0.002 [-0.002, -0.002]) | 0.852 (-0.023 [-0.024, -0.023]) | 0.007 |
| 2 | 0.413 | 0.408 (-0.005 [-0.006, -0.005]) | 0.379 (-0.034 [-0.035, -0.033]) | 0.054 |
| 3 | 0.419 | 0.434 (+0.015 [+0.014, +0.016]) | 0.400 (-0.019 [-0.020, -0.018]) | 0.113 |
| 4 | 0.234 | 0.250 (+0.015 [+0.014, +0.016]) | 0.233 (-0.002 [-0.003, -0.001]) | 0.126 |
| 5 | 0.660 | 0.671 (+0.012 [+0.011, +0.013]) | 0.645 (-0.015 [-0.016, -0.014]) | 0.060 |
| 6 | 0.356 | 0.380 (+0.024 [+0.023, +0.025]) | 0.352 (-0.004 [-0.005, -0.003]) | 0.120 |
| 7 | 0.230 | 0.239 (+0.008 [+0.007, +0.009]) | 0.227 (-0.003 [-0.004, -0.002]) | 0.118 |
| 8 | 0.317 | 0.328 (+0.010 [+0.009, +0.012]) | 0.313 (-0.004 [-0.005, -0.003]) | 0.121 |
| 9 | 0.208 | 0.219 (+0.011 [+0.009, +0.013]) | 0.206 (-0.002 [-0.004, -0.001]) | 0.097 |
| 10 | 0.480 | 0.488 (+0.008 [+0.005, +0.011]) | 0.467 (-0.013 [-0.015, -0.010]) | 0.117 |
| 11 | 0.676 | 0.690 (+0.014 [+0.012, +0.016]) | 0.667 (-0.009 [-0.011, -0.007]) | 0.016 |
| 12 | 0.338 | 0.367 (+0.029 [+0.025, +0.032]) | 0.338 (-0.000 [-0.002, +0.002]) | 0.102 |
| 13 | 0.323 | 0.344 (+0.021 [+0.017, +0.024]) | 0.326 (+0.003 [+0.001, +0.006]) | 0.053 |
| 14 | 0.181 | 0.200 (+0.019 [+0.017, +0.022]) | 0.184 (+0.003 [+0.001, +0.005]) | 0.080 |
| 15 | 0.191 | 0.184 (-0.006 [-0.010, -0.003]) | 0.186 (-0.005 [-0.007, -0.002]) | 0.044 |
| 16 | 0.261 | 0.207 (-0.055 [-0.059, -0.050]) | 0.238 (-0.023 [-0.026, -0.020]) | 0.024 |
| 17 | 0.085 | 0.098 (+0.013 [+0.011, +0.014]) | 0.093 (+0.007 [+0.006, +0.009]) | 0.019 |

Supplementary `figures/fig_readout_sensitivity_v3_supp_language_share_per_layer.png`: the language share per layer for the three models.

## v2 (user 2026-10-27, "Nature level") — summary

(a)/(b) re-rendered from `amvq/diagnostics.json` (step 3 / step 0 = 6.1×, top block 0; DoF shares vision 13/19/68, LLM 15/21/64, VL self-attn 23/32/44,
DiT 29/34/37 %; top-10 %-row concentration 0.38 / 0.41 / 0.23 / 0.52). (c) attention overlays on one frame with fixed target / gripper boxes
((40, 126, 72, 150) and (17, 98, 94, 126) in the 224 × 224 input, fixed before any overlay was rendered); printed within-image box shares FP 0.023 / 0.069,
PTQ 0.022 / 0.070, ours 0.021 / 0.066 (uniform 0.016 / 0.047). Full record: E161, E169.
