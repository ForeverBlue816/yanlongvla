# Readout visualisation — π₀.₅ (same 19 inputs as the GR00T libero_goal_303 episode)

What the action expert reads from the prefix under FP, the accepted 2-bit PTQ base (`bb_M2_ae_M2_fp32`, Plus 704) and LoopQ
(`tcq_raqft_d1`: TCQ on H_D + stabilised readout fine-tune; its Plus run was in progress when this was made).

- Inputs: the 19 policy-call observations of the libero_goal_303 episode used for the GR00T figures (the GR00T FP rollout — π₀.₅ has no
  saved rollout of its own), and the 256 held-out observations for the PCA basis.
- Queries are teacher-forced: the FP expert runs its own 10-step trajectory on the FP prefix (same noise seed per call), and every model's
  readout uses those same queries, suffix keys / values and masks over **its own prefix KV cache**. The only difference between models
  is the quantized backbone. (The fine-tune itself used the quantized model's own queries.)
- r[call, layer, step] = Σ_h W_O,h (FP expert) × attention readout, mean over the 10 action tokens; 18 expert layers, 10 flow steps,
  d = 1024. Feature per call = first flow step over all 18 layers (18,432 dims), standardised with the FP held-out mean / std.

## Numbers

| quantity | 2-bit base | LoopQ | LoopQ / base |
|---|---|---|---|
| distance to FP, full space (standardised, first step) | 219.2 | 110.7 | **0.505** (closer on 19 / 19 calls) |
| in PC1–PC3 | 10.87 | 3.14 | 0.289 (closer on 19 / 19) |
| in PC1–PC10 | 16.31 | 6.19 | 0.379 |
| all flow steps, unstandardised | 244.5 | 160.0 | 0.654 |

Variance explained by the PCA of the FP held-out features: 24.1 % (3 components), 44.7 % (10).
Share of the squared readout error inside PC1–PC3: 0.26 % (base), 0.09 % (LoopQ).
The gate (LoopQ / base < 0.9 in the full space, closer on ≥ 75 % of calls, PC1–PC3 agrees) passes.

All rows measured (standardised full-space distance to FP, ratio to the base; unstandardised all-step ratio in brackets):
`tcq_d1` (TCQ on H_D, no fine-tune, Plus 734) 137.0, 0.625 [0.709]; `raqft_studentq` (old format + fine-tune, Plus 749) 89.6, 0.409 [0.818];
`tcq_raqft_d1` (LoopQ) 110.7, 0.505 [0.669]. Standardisation changes the size of the ratio a lot (0.41 vs 0.82 for `raqft_studentq`): it
up-weights directions that vary little across held-out observations. Both conventions agree on the direction for every row.

## Figures (PDF + SVG + PNG at 600 dpi, transparent, 2.3 in × 1.6 in)

`readout_path_3d_view1..4` / `_sheet` (A), `readout_path_2d` (B), `readout_strip` (C, call 2), `readout_distance` (D), `readout_drift_2d` (E),
`contact_sheet.png`, `contact_sheet_drift.png`. Same styling and rules as the GR00T set (`../readout_viz/README.md`).
Camera views chosen from the data: view 1: elev 45, azim 170 (on-screen LoopQ / base 0.29), view 2: elev 28, azim 160 (on-screen LoopQ / base 0.28), view 3: elev 45, azim -150 (on-screen LoopQ / base 0.29), view 4: elev 12, azim 160 (on-screen LoopQ / base 0.25).

**Which picture is faithful.** E and D. E (deviation from FP; axes = the two leading directions of the pooled errors of both models)
shows 96 % / 92 % of the squared error on screen and an on-screen ratio of 0.494 against 0.505 in the full space.
A and B are projections that hold 0.1–0.3 % of the error and show a larger gap (0.29) than the full space (0.505); unlike for GR00T they
agree with it in direction on all 19 calls, so they are usable with that caveat in the caption.

Caveats for a caption: the base and LoopQ here differ in more than the fine-tune (weight format RVQ → TCQ, calibration clean → H_D);
`tcq_d1` isolates the format + calibration part (0.625). The inputs are another model's FP rollout. Colours are the placeholder
`figures/loopq_style.py`.

Files: `readouts_all.npz` (five rows: fp, base, raqft_studentq, tcq_d1, tcq_raqft_d1), `readouts_goal303.npz` (the plotted triple),
`pca_basis.npz`, `numbers.json`, `numbers_drift.json`. `heldout_fp.npz` (175 MB) is not in the repositories; `scripts/paper_readout_pi05.py` regenerates it.
