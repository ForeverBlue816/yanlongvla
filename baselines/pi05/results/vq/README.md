# pi0.5 vector-quantization rows: 1 bit, plus the mentor's VPTQ structure at 2 and 3 bits

Per-row folders: `quant.json` (structure, bits per weight with codebooks and scales counted), `libero/<suite>.jsonl`
(500 episodes per suite), `plus_1000.jsonl` (LIBERO-Plus 1st 1000 = selection units), `plus_added_1000.jsonl`
(2nd 1000 = added units) and `paired_stats/` (paired bootstrap 95% CIs; `libero_vs_original_fp`, `plus_1000_vs_original_fp`,
`plus_2000_vs_original_2bit_expanded`). Tables: `python3 baselines/jobs/experiments/report_tables.py <run folder>`.
All rows below are finished (every LIBERO suite and both LIBERO-Plus halves).

## Settings

Official toolkits: QTIP = Cornell-RelaxML/qtip, QuIP# = Cornell-RelaxML/quip-sharp, VPTQ = microsoft/VPTQ, AQLM = Vahe1994/AQLM.
The official quantization code runs unchanged; this list gives only the settings that differ. Every row: PTQ stage only
(no layer or end-to-end fine-tuning), 256 calibration observations.

| row | status | setting | ours | official |
|---|---|---|---|---|
| `qtip_w1` | NON-OFFICIAL (no 1-bit recipe) | trellis bits K | 1 | smallest official K = 2 |
| | | scale_override | 0.9 (2-bit recipe) | – |
| | | L / V / decode | 16 / 2 / quantlut_sym, tlut 9 bits (2-bit recipe) | same |
| `quip_w1` | NON-OFFICIAL (no 1-bit codebook) | codebook | E81B: the official 256-point E8 grid alone (8 bits / 8 weights), `quip_e81b.py` | smallest official codebook E8P12 = 2 bits |
| | | opt_scale | 0.606, from the official Gaussian-MSE procedure | – |
| | | LDLQ refinement sweeps | 10 | 10 |
| `aqlm_w1` | official config 1x8 g8 | Hessian damping | 1% of the mean diagonal | none |
| `vptq_v4k16_w1` | NON-OFFICIAL (no 1-bit recipe) | vector length / centroids | 4 / 16 (1.000 bpw) | 2-bit recipe 8 / 65536 |
| `vptq_w2` | mentor's fixed 2-bit structure (23 Oct) | vector length / centroids | 4 / 256 (2.003 bpw, codebook counted) | 8 / 65536 (3.3 bpw with the codebook counted) |
| `vptq_w3` | the same structure, extended to 3 bits | vector length / centroids | 4 / 256 + 16 residual (3.003 bpw) | 8 / 65536 + residual |
| all VPTQ rows | | layer norm / permutation | off / off | on / on |
| | | k-means weights | diagonal of the undamped Hessian | diagonal of H + 0.01 I |

The official VPTQ 2-bit configuration (vector length 8, 65536 centroids) is not actually 2 bits per weight once its
codebook is counted — on our layers it comes to about 3.3 bpw. The mentor fixed vector length 4 with 256 centroids
instead, which is a true 2 bits per weight (`vptq_w2`); we extend the same structure to 1 bit (`vptq_v4k16_w1`) and 3
bits (`vptq_w3`). We also tried the official vector length (8) with fewer centroids chosen for a true 2/3 bits per
weight; on the one GR00T test we ran it scored 69 of 1000 units below the mentor's v4 structure (MENTOR_PENDING A13,
2 Nov), so only the mentor's v4 structure is reported here, at every bit width.

### pi0.5 on LIBERO (500 episodes per suite; difference vs FP over 2000 episodes, paired 95% CI)

| row | linear bpw | whole bpw | spatial | object | goal | LIBERO-10 | total /2000 | vs FP (pp) |
|---|---|---|---|---|---|---|---|---|
| QTIP 1-bit, K=1 (NON-OFFICIAL) | 1.0196 | 2.1158 | 460 | 461 | 250 | 141 | 1312 | -30.8 [-38.1, -23.0] |
| AQLM 1-bit, 1x8g8 | 1.0122 | 2.1097 | 350 | 148 | 103 | 5 | 606 | -66.1 [-74.5, -57.1] |
| QuIP# 1-bit, E81B (256-point E8) (NON-OFFICIAL) | 1.0172 | 2.1138 | 278 | 126 | 92 | 0 | 496 | -71.6 [-79.5, -63.0] |
| VPTQ 1-bit, v4 k16 (NON-OFFICIAL) | 1.0002 | 2.1000 | 0 | 0 | 0 | 0 | 0 | -96.4 [-98.3, -93.8] |
| QTIP 2-bit, default recipe | 2.0237 | 2.9255 | 493 | 498 | 489 | 471 | 1951 | +1.1 [-0.5, +3.2] |
| VPTQ 2-bit, v4 k256 | 2.0025 | 2.9084 | 492 | 493 | 490 | 465 | 1940 | +0.6 [-1.4, +3.0] |
| QuIP# 2-bit, 10 sweeps | 2.0283 | 2.9292 | 490 | 490 | 487 | 458 | 1925 | -0.1 [-2.0, +2.3] |
| AQLM 2-bit, 2x8g8 | 2.0172 | 2.9202 | 491 | 495 | 477 | 452 | 1915 | -0.7 [-4.1, +2.8] |
| QTIP 3-bit, default recipe | 3.0277 | 3.7352 | 490 | 494 | 486 | 457 | 1927 | -0.1 [-1.2, +1.2] |
| VPTQ 3-bit, v4 k256 + r16 | 3.0027 | 3.7149 | 492 | 495 | 490 | 476 | 1953 | +1.2 [-0.3, +3.0] |
| QuIP# 3-bit, 10 sweeps | 3.0324 | 3.7389 | 495 | 494 | 489 | 470 | 1948 | +1.0 [-0.3, +2.5] |
| AQLM 3-bit, default recipe | 3.0222 | 3.7307 | 494 | 498 | 486 | 470 | 1948 | +1.0 [-0.6, +2.8] |

### pi0.5 on LIBERO-Plus (1st 1000 = selection units, 2nd 1000 = added units; paired 95% CI)

| row | 1st 1000 | vs FP 765 (pp) | 2nd 1000 | 2000 | 2000 vs original_2bit (pp) |
|---|---|---|---|---|---|
| QTIP 1-bit, K=1 (NON-OFFICIAL) | 399 | -36.6 [-44.2, -28.6] | 355 | 754 | -30.3 [-36.1, -24.8] |
| AQLM 1-bit, 1x8g8 | 181 | -58.4 [-65.7, -50.6] | 154 | 335 | -51.3 [-58.5, -44.1] |
| QuIP# 1-bit, E81B (256-point E8) (NON-OFFICIAL) | 138 | -62.7 [-69.3, -55.8] | 109 | 247 | -55.7 [-62.5, -48.5] |
| VPTQ 1-bit, v4 k16 (NON-OFFICIAL) | 0 | -76.5 [-81.9, -71.0] | 0 | 0 | -68.0 [-75.6, -60.9] |
| QTIP 2-bit, default recipe | 726 | -3.9 [-7.4, -0.7] | 685 | 1411 | +2.5 [-1.1, +6.4] |
| VPTQ 2-bit, v4 k256 | 685 | -8.0 [-13.4, -2.8] | 632 | 1317 | -2.2 [-5.3, +1.0] |
| QuIP# 2-bit, 10 sweeps | 662 | -10.3 [-14.9, -6.1] | 611 | 1273 | -4.4 [-8.2, -0.7] |
| AQLM 2-bit, 2x8g8 | 639 | -12.6 [-17.9, -7.4] | 589 | 1228 | -6.7 [-9.9, -3.2] |
| QTIP 3-bit, default recipe | 755 | -1.0 [-4.2, +2.1] | 704 | 1459 | +4.9 [+0.7, +9.0] |
| VPTQ 3-bit, v4 k256 + r16 | 753 | -1.2 [-4.7, +2.0] | 710 | 1463 | +5.1 [+1.7, +8.5] |
| QuIP# 3-bit, 10 sweeps | 746 | -1.9 [-4.7, +0.9] | 710 | 1456 | +4.8 [+1.3, +8.4] |
| AQLM 3-bit, default recipe | 746 | -1.9 [-5.1, +1.1] | 704 | 1450 | +4.5 [+1.3, +7.7] |
