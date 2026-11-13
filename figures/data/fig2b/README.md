# Figure 2(b) data — where the quantization error sits: demonstration H (A) vs closed-loop H (D), GR00T N1.7

A = `tcq_w2_rht` (H from the 256 demonstration observations), D = `tcq_w2_rht_d1` (H_D = H_demo + trace-balanced photometric /
geometric / on-policy second moments). Same format, rotation and bit budget; FP weights as reference; original weight space.
H_policy = the on-policy source matrix of arm D. CPU only (`scripts/paper_fig2b.py`, `scripts/paper_fig2b_render.py`).

## Result in one paragraph — read first

**The maps of A and D look alike at this scale; by the stated rule the figure stops here** (nothing was rescaled and no layer was
picked by eye). In the chosen layer the on-policy error is 56 % lower for D than for A, but that gain does not appear in a
per-direction error picture: summed over directions with their on-policy energy, Σ_k e[k]·p_k, D is not lower than A (ratio 1.016).
The whole reduction of tr(ΔW H_policy ΔWᵀ) (ratio 0.436) comes from the cross terms between directions — D's errors along different
demonstration eigen-directions cancel under the on-policy inputs — which neither the error maps nor the scatter can show.

## 1. Layer choice (rule: largest relative reduction A → D of tr(ΔW H_policy ΔWᵀ) / tr(W H_policy Wᵀ), all 1080 fitted layers)

D is lower than A under H_policy on 1065 of 1080 layers (median reduction 11.2 %, mean 13.8 %); under H_demo D is lower on 94 of 1080.
Backbone only: 213 of 216, median 2.8 %, maximum 13.9 %. The large reductions are all in the action head.

| rank | checkpoint | layer | shape | on-policy error A | D | reduction |
|---|---|---|---|---|---|---|
| 1 (used) | head, libero_object | transformer_blocks.16.attn1.to_out.0 | 1536 × 1536 | 0.02040 | 0.00889 | 56.4 % |
| 2 (backup) | head, libero_object | transformer_blocks.20.attn1.to_out.0 | 1536 × 1536 | 0.02474 | 0.01090 | 56.0 % |
| 3 (backup) | head, libero_object | transformer_blocks.26.ff.net.2 | 1536 × 6144 | 0.00109 | 0.00049 | 54.6 % |
| 4 | head, libero_goal | transformer_blocks.6.attn1.to_out.0 | 1536 × 1536 | 0.01766 | 0.00802 | 54.6 % |
| 5–9 | head, libero_object | transformer_blocks.{27,24,25,28,23}.ff.net.2 | 1536 × 6144 | | | 53.2 – 51.5 % |
| 10 | head, libero_spatial | transformer_blocks.20.attn1.to_out.0 | 1536 × 1536 | 0.01779 | 0.00874 | 50.9 % |

Full list: `ranking.json` (top 10), `ranking_all.json` (all layers).

## 2. Numbers for the first layer (`numbers.json`; backups in `numbers_backup1.json`, `numbers_backup2.json` say the same)

| quantity | A (demonstration H) | D (closed-loop H) | D / A |
|---|---|---|---|
| relative weight error ‖ΔW‖² / ‖W‖² | 0.1430 | 0.1359 | 0.951 |
| error under H_demo | 0.00503 | 0.00569 | 1.131 |
| error under H_policy | 0.02040 | 0.00889 | **0.436** |
| share of the model's error in the 20 % of directions with the highest p_k / λ_k | 42.5 % | 39.2 % | |
| error energy in those directions | 47.58 | 41.70 | 0.877 |
| share of the model's error in the lower-λ half of the directions | 82.3 % | 79.1 % | |
| Σ_k e[k]·p_k (on-policy error without cross terms) | 7.726 | 7.851 | 1.016 |

The lower-λ half of the directions holds 0.84 % of the demonstration energy. The 20 % high-ratio directions hold 0.12 % of the
demonstration energy and 3.4 % of the on-policy energy; they all lie in the three lowest-λ deciles.
By λ-decile (largest first), D / A of the error energy: 1.19, 1.19, 1.17, 1.14, 1.08, 1.01, 0.96, 0.91, 0.88, 0.87 (`by_lambda_decile`).

## 3. The two statements

- **(i) Is A's error concentrated in low-λ directions? Yes.** 82 % of A's error energy is in the half of the directions that carries
  0.84 % of the demonstration energy; error energy falls monotonically with λ (rank correlation −0.998). The same holds for D (79 %,
  −0.998): this is what any Hessian-weighted quantizer does, not something specific to demonstration calibration.
- **(ii) Does D have visibly less error than A in the directions with high p_k / λ_k? No, not visibly.** D has 12 % less error energy
  there (ratio 0.877; per-direction median 0.875) and 9 % more elsewhere (median 1.095). On a shared colour scale the two maps are
  indistinguishable: pooled-map correlation 0.985 (eigenbasis) and 0.845 (channel basis), mean map value D / A = 0.992 and 0.975.

## 4. Files

`directions.csv` (k, λ_k, p_k, e_A, e_D, flag for the top-20 % ratio directions); `map_demo.npy`, `map_closedloop.npy` (|ΔW U| pooled
to 64 × 128, rows = output channels, columns = directions sorted by λ, largest first) and `map_*_channels.npy` (plain input-channel
basis); `numbers.json`; the renders `map_demo`, `map_closedloop` (+ `_channels`), `scatter` as PDF / SVG / PNG at 300 dpi, and
`contact_sheet.png`. `*_backup1`, `*_backup2`: the same for the second and third ranked layers.
The renders are included so the "look alike" statement can be checked; they are not proposed as a figure.

Supplementary (`numbers.json: onpolicy_error_by_H_policy_direction`): split along the eigenvectors of H_policy itself, the reduction is
spread out (D / A = 0.47 on the top direction, 0.20 on the top five, 0.46 on the rest), so there is no single on-policy direction that
explains it either.

Colours: `figures/loopq_style.py` is a placeholder palette (no paper style file existed on the cluster).


---

# Second attempt (2026-11-13): output error per layer

`scripts/paper_fig2b_layers.py`, reusing the ranking's per-layer errors. One row per quantized Linear of the deployed model
(216 backbone + 216 action head; action-head errors are the mean over the four suite checkpoints, ratios are ratios of those means,
per-suite values kept in `layers.csv`). ratio = D / A of the relative output error tr(ΔW H ΔWᵀ) / tr(W H Wᵀ).

| module | layers | ratio_policy, median [p10, p90] | ratio_demo, median [p10, p90] | layers with ratio_policy < 1 | parameter-weighted ratio_policy | parameter-weighted ratio_demo | worse on both / better on both |
|---|---|---|---|---|---|---|---|
| Vision encoder | 104 | 0.974 [0.919, 0.987] | 1.008 [1.004, 1.019] | 101 | 0.958 | 1.012 | 3 / 3 |
| LLM | 112 | 0.972 [0.943, 0.980] | 1.010 [1.005, 1.020] | 112 | 0.963 | 1.011 | 0 / 2 |
| Action head | 216 | 0.855 [0.664, 0.979] | 1.023 [1.001, 1.061] | 214 | 0.840 | 1.026 | 2 / 20 |
| Backbone (vision + LLM) | 216 | 0.972 [0.936, 0.985] | 1.009 [1.004, 1.019] | 213 | 0.962 | 1.012 | 3 / 5 |
| All | 432 | 0.954 [0.742, 0.984] | 1.011 [1.004, 1.037] | 427 | 0.903 | 1.018 | 5 / 25 |

By layer type (median ratio_policy): DiT cross-attention out 0.54, DiT ff-down 0.82, DiT self-attention out 0.83,
DiT q / k / v 0.86–0.89, DiT ff-up 0.91; VL self-attention ff-down 0.76, out 0.82; LLM 0.94–0.98; vision blocks 0.95–0.99, patch mergers 0.93.

**What the two pictures show.** The heat-map: closed-loop calibration lowers the error on the policy's own states in 427 of 432 layers,
but almost all of the gain is in the action head (median −14 %, up to −50 % in the cross-attention output projections and the
feed-forward output layers of the later DiT blocks), while the backbone moves by a median of −2.8 %.
The scatter: the gain is bought with error on the demonstrations, along a nearly straight line in the action head (about 4.6 %
less on-policy error per 1 % more demonstration error, r = -0.95; backbone slope 2.2, r = -0.62).

**What does not fit.** Five layers are worse on both measures, all by less than 0.3 % on-policy: vision blocks 1 and 2 (`attn.qkv`,
`attn.proj`, `attn.qkv`), DiT block 5 `to_k` and DiT block 3 `ff.net.0.proj` (light-coral cells). Twenty-five layers are better on
both (20 in the action head). No layer trades the other way (better on demonstrations, worse on-policy). The LLM, the largest
module by parameters, changes least (median −2.8 %).

Files: `layers.csv` (432 rows), `numbers_layers.json`, `heatmap.{svg,pdf,png}` + `heatmap_values.csv`, `tradeoff.{svg,pdf,png}`,
`tradeoff_marginals.{svg,pdf,png}`, `contact_sheet_layers.png`.
- `heatmap.svg` is written by hand: every cell is its own `<rect>` (432 cells, each with a `<title>` naming the layer and its value),
  the colour bar is 60 vector `<rect>` slices, all labels are `<text>` in Times, no embedded raster. The PDF / PNG use the same layout.
  Colour = 100·(1 − ratio_policy) on one scale from 0 to the 99th percentile (49.0 %); negative cells light coral.
  `heatmap_values.csv` has one row per cell (value, fill colour, position in points).
- The scatter SVGs are matplotlib vector output with text kept as text (font TeX Gyre Termes, the Times clone on the cluster) and no raster.
- Columns: vision blocks 0–23 then the four patch mergers (MLP rows only); LLM layers 0–15; action head: the four VL self-attention
  blocks, then DiT blocks 0–31 (even blocks cross-attend to the backbone tokens, odd blocks self-attend).
- `figures/loopq_style.py`: `CMAP_SEQ` is now white → navy as the request describes; the earlier dark-to-hot map used by the attention
  overlays and the first-attempt maps is kept as `CMAP_HEAT`. Both are placeholders for the paper's palette.
