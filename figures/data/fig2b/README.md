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
