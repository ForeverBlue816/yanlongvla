# Readout visualisation (GR00T N1.7, libero_goal_303)

What the action head reads under FP, the 2-bit base (`tcq_w2_rht`) and LoopQ (`tcq_raqft_d1_s`) on identical inputs: the 19 policy calls
of the FP rollout of libero_goal_303, teacher-forced into all three models (libero_goal checkpoints), same flow noise per call.
Readout r[call, block, step] = FP W_O applied to the mean over the 40 action tokens of the pre-W_O cross-attention output (the tensor in
the fine-tune loss); 16 cross-attention blocks, 4 flow steps, d = 1536. Feature per call = first flow step, concatenated over blocks,
standardised with the FP held-out mean / std (256 held-out observations). PCA fitted on the FP held-out features only.

## Read this before using a figure

**In the full feature space LoopQ is only 6.5 % closer to FP than the base** (ratio 0.935, closer on 17 of 19 calls). The pre-set gate
("clearly closer", ratio < 0.9) did not pass for the all-block feature. The figures were made anyway at the user's request
(2026-11-12), in two sets, and every caption needs the numbers below:

| feature | base → FP | LoopQ → FP | LoopQ / base, full space | LoopQ / base in PC1–PC3 | in PC1–PC10 | variance explained (3 / 10 PCs) | share of squared error inside PC1–PC3 (base / LoopQ) |
|---|---|---|---|---|---|---|---|
| all 16 cross-attention blocks (files without suffix) | 76.73 | 71.77 | **0.935** (17 / 19 calls) | 0.658 (13 / 19) | 0.738 | 20.9 % / 44.3 % | 2.3 % / 1.2 % |
| blocks 0 and 4 only (files `*_blocks0_4`) | 29.96 | 25.53 | **0.852** (19 / 19 calls) | 0.611 (16 / 19) | 0.666 | 47.7 % / 79.8 % | 16.5 % / 9.7 % |

- **The path figures (A, B) overstate the gap.** PC1–PC3 hold about 2 % of the readout error for the all-block feature, and the gap
  there (0.66) is five times the full-space gap (0.935). For blocks 0 and 4 the projection shows 0.61 against 0.85 in the full space.
- **Blocks 0 and 4 were picked after looking at the results**: they are the two cross-attention blocks with the largest gain
  (per-block LoopQ / base at the first flow step: block 0 0.72, block 4 0.64; the other 14 blocks 0.90–0.98).
- **Camera views were chosen from the data** (the cameras in which the path is spread out and the base-to-FP gap is most visible):
  all blocks — view 1: elev 20, azim 150 (on-screen LoopQ / base 0.55), view 2: elev 36, azim 160 (on-screen LoopQ / base 0.54), view 3: elev 20, azim -170 (on-screen LoopQ / base 0.52), view 4: elev 36, azim 120 (on-screen LoopQ / base 0.59); blocks 0 and 4 — view 1: elev 28, azim -80 (on-screen LoopQ / base 0.52), view 2: elev 45, azim -70 (on-screen LoopQ / base 0.50), view 3: elev 12, azim -80 (on-screen LoopQ / base 0.53), view 4: elev 12, azim 100 (on-screen LoopQ / base 0.52).
- No value is scaled or altered in any figure; what differs between figures is the feature subset, the projection and the camera.
- Action outputs on the same calls: distance to FP 0.151 (base) vs 0.145 (LoopQ), ratio 0.959, LoopQ closer on 12 of 19 calls.

## Figures (PDF + SVG + PNG at 600 dpi, transparent, 2.3 in × 1.6 in)

| file | content |
|---|---|
| `readout_path_3d_view1..4`, `readout_path_3d_sheet` | A: the three per-call sequences in PC1–PC3 (FP white tube, base coral, LoopQ navy, time by fading, floor shadows, error sticks base → FP every 2nd call) |
| `readout_path_2d` | B: the same in PC1–PC2 |
| `readout_strip` | C: PC1–PC10 coefficients at the call with the largest base-to-FP distance (call 17 for all blocks, call 4 for blocks 0 and 4). With the 1-unit threshold almost every cell is outlined for both models (9 / 10 of 10 for all blocks), so the outline does not separate them |
| `readout_distance` | D: full-space distance to FP per call, base and LoopQ |
| `readout_drift_2d` | E (added): the deviation from FP itself. FP is the origin; axes are the two leading directions of the pooled errors of both models (so the axes favour neither); dashed rings = mean on-screen distance of each model |
| `contact_sheet`, `contact_sheet_blocks0_4`, `contact_sheet_drift` | review sheets on white |

E numbers: all blocks — on-screen LoopQ / base 0.889 (full space 0.935), share of squared error shown 46 % / 42 %;
blocks 0 and 4 — 0.780 (full space 0.852), 73 % / 62 %.

**Which picture is the more faithful:** neither A nor B — both show a projection that contains ~2 % of the error and exaggerate the gap.
D (full-space distance) and E (deviation from FP, 42–73 % of the error on screen, on-screen ratio close to the full-space ratio) are the
faithful ones. E also shows what the numbers say: both 2-bit models sit at a nearly constant offset from FP, in different directions,
and LoopQ's offset is smaller.

Not drawn: axes fitted on the base's error alone. In those axes LoopQ / base is 0.37, but in axes fitted on LoopQ's error alone it is
2.19 the other way — each model's error looks small in the other's leading directions, so that picture would show a gap that is not there.

Colours and fonts: `figures/loopq_style.py` did not exist on the cluster; the file here defines INK / NAVY / CORAL_DARK / CMAP_DIV with
placeholder values and TeX Gyre Termes (Times-compatible). Replace it with the paper's palette and re-run `scripts/paper_readout_fig.py`.

Files: `readouts_goal303.npz` (r: model × call × block × step × d, float32; actions), `pca_basis.npz` / `pca_basis_blocks0_4.npz`,
`numbers*.json` (all values above, per-call values, views). `heldout_fp.npz` (94 MB, FP readouts on the 256 held-out observations) is
not in the repositories; regenerate with `scripts/paper_readout.py`.
