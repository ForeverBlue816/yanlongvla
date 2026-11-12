# Attention heat-maps — π₀.₅, libero_goal_303 inputs, calls 0 / 10 / 18

The action expert's attention over the prefix tokens under FP, the accepted 2-bit base (`bb_M2_ae_M2_fp32`) and LoopQ (`tcq_raqft_d1`),
with the same teacher-forced FP queries as `figures/readout_viz_pi05` (only the prefix keys differ between models). Softmax attention of
the 10 action tokens at the first flow step, averaged over the 8 heads, 18 expert layers and 10 action tokens.
Prefix layout (read from the model): third-person camera 256 tokens (16 × 16 patches), wrist camera 256, an unused third image slot 256
(masked, attention 0), language 200 (28 valid), then the 10 action tokens themselves.

## What the maps show — read before using them

- **The three maps look alike at this scale.** Correlation with the FP map over the 16 × 16 grid: base 0.94–0.96, LoopQ 0.98–0.995
  (all 19 calls: base 0.965, LoopQ 0.989 on average). The colour scale of each row is shared and nothing is rescaled per model.
- **The visible difference is one hotspot per call**, not a change of the overall pattern: at call 0 the base has an extra hotspot near
  the robot's shoulder that FP and LoopQ do not have (largest single-patch difference 81 % of the FP maximum); at call 10 the base misses
  the top-left hotspot that FP and LoopQ share (41 %). LoopQ's largest single-patch differences are 23 %, 6 % and 22 % of the FP maximum.
- **The hotspots are not on the task objects.** In all three models the attention on the third-person image is spread thinly, with a
  few isolated high patches in the background (image corners, floor, robot base) — the pattern usually attributed to register-like
  tokens. The drawer, its handle and the gripper do not stand out. As a picture of "what the model looks at" these maps are weak.
- **Only 4–9 % of the attention goes to the third-person image at all**: about half is on the language tokens, a quarter on the wrist camera.

| call (env step) | model | third-person image | wrist image | language | action tokens | corr. with FP map | max patch diff / FP max | L1 diff / FP mass |
|---|---|---|---|---|---|---|---|---|
| 0 (10) | FP | 8.5 % | 23.5 % | 50.5 % | 17.4 % | — | — | — |
| | 2-bit | 9.0 % | 20.9 % | 51.5 % | 18.6 % | 0.949 | 0.81 | 0.165 |
| | LoopQ | 8.6 % | 23.5 % | 50.1 % | 17.8 % | 0.987 | 0.23 | 0.109 |
| 10 (90) | FP | 7.9 % | 23.7 % | 51.7 % | 16.8 % | — | — | — |
| | 2-bit | 6.5 % | 23.3 % | 52.7 % | 17.5 % | 0.960 | 0.41 | 0.259 |
| | LoopQ | 7.8 % | 24.3 % | 51.0 % | 16.9 % | 0.995 | 0.06 | 0.116 |
| 18 (154) | FP | 4.2 % | 28.8 % | 49.7 % | 17.4 % | — | — | — |
| | 2-bit | 4.0 % | 28.0 % | 50.0 % | 18.0 % | 0.942 | 0.38 | 0.199 |
| | LoopQ | 4.3 % | 28.8 % | 49.4 % | 17.5 % | 0.980 | 0.22 | 0.133 |

LoopQ is closer to FP than the base on every number in the table, and its split across image / wrist / language matches FP's to within
0.7 points, but the differences are small in absolute terms: this is supporting evidence, not a striking picture.

## Files

- `attn_maps.npz`: `weights` (model × 19 calls × 978 tokens), `maps` (model × call × image block × 16 × 16, policy image orientation), layout fields.
- `attn_overlay_call{00,10,18}`: FP / 2-bit / LoopQ on the 768-px upright agentview frame, bilinear up-sampling, shared scale per row, 55 % opacity, CMAP_SEQ.
- `attn_diff_call*`: 2-bit − FP and LoopQ − FP with CMAP_DIV on a shared scale set by the larger of the two differences (this is the
  only place where the scale is the difference's own). `attn_diff_samescale_call*`: the same differences on the row's attention scale,
  i.e. as large as they are in the overlays.
- `frames/`: the 768-px renders of the three calls (FP episode states replayed; replay error < 1 mm). `contact_sheet.png`, `numbers.json`.
- PDF + SVG + PNG at 600 dpi, transparent.

Notes: the overlays are on the upright display frame; the maps are mirrored left-right with it (the policy sees the frame rotated by
180°). `CMAP_SEQ` did not exist in a paper style file on the cluster; a placeholder was added to `figures/loopq_style.py`.
The inputs are the GR00T FP rollout's observations (π₀.₅ has no saved rollout).
