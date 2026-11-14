| Row | Weights on disk (GB) | Peak GPU memory, allocated / reserved (GB) | Prefill, ms | Full chunk, 10 steps, ms | Full chunk, 2 steps, ms | vs FP, 10 / 2 steps |
|---|---:|---:|---:|---:|---:|---:|
| Full precision (BF16), official path | 6.47 | 6.65 / 6.78 | 51.8 (52.3) | 81.6 (83.3) | 54.6 (56.3) | 1.00× / 1.00× |
| Full precision (BF16), lockstep vision calls | 6.47 | 6.64 / 6.78 | 51.0 (51.6) | 81.1 (81.7) | 53.9 (55.3) | 1.01× / 1.01× |
| 2-bit weights, (i) decoded at load | 1.58 | 6.16 / 6.29 | 52.4 (52.9) | 82.1 (85.3) | 55.0 (56.3) | 0.99× / 0.99× |
| 2-bit weights, (ii) on the fly | 1.58 | 1.99 / 2.84 | 86.5 (86.8) | 133.1 (133.6) | 94.5 (94.9) | 0.61× / 0.58× |
| 2-bit weights, (iii) hybrid | 1.58 | 2.21 / 2.97 | 86.5 (86.9) | 129.8 (130.3) | 93.7 (94.2) | 0.63× / 0.58× |
| W2A8, (i) decoded at load | 1.58 | 3.72 / 4.93 | 38.2 (38.6) | 74.6 (75.3) | 42.8 (44.3) | 1.09× / 1.28× |
| W2A8, (ii) on the fly | 1.58 | 1.92 / 2.77 | 40.6 (41.7) | 79.8 (80.1) | 47.2 (48.0) | 1.02× / 1.16× |
| W2A8, (iii) hybrid | 1.58 | 2.15 / 2.89 | 40.7 (41.9) | 78.7 (79.5) | 47.5 (48.2) | 1.04× / 1.15× |

| Row | Vision encoder, ms | LLM prefill, ms | Expert step (10-step schedule), ms | All 10 steps, ms | Expert step (2-step schedule), ms | Graph = eager | Same as per-image calls |
|---|---:|---:|---:|---:|---:|---|---|
| Full precision (BF16), official path | 8.55 | 43.44 | 3.43 (3.58) | 34.3 | 2.72 | yes | yes |
| Full precision (BF16), lockstep vision calls | 7.50 | 43.99 | 3.44 (3.48) | 34.3 | 2.72 | yes | max abs diff 2.3e-03 |
| 2-bit weights, (i) decoded at load | 8.58 | 43.99 | 3.43 (3.44) | 34.3 | 2.72 | yes | yes |
| 2-bit weights, (ii) on the fly | 18.18 | 68.77 | 4.69 (4.70) | 47.2 | 3.99 | yes | max abs diff 2.6e-03 |
| 2-bit weights, (iii) hybrid | 18.18 | 68.87 | 4.35 (4.36) | 44.1 | 3.65 | yes | max abs diff 2.7e-03 |
| W2A8, (i) decoded at load | 10.14 | 28.69 | 3.66 (3.67) | 36.6 | 2.94 | yes | yes |
| W2A8, (ii) on the fly | 10.57 | 31.42 | 3.78 (3.79) | 37.8 | 3.03 | yes | yes |
| W2A8, (iii) hybrid | 10.52 | 31.13 | 3.66 (3.67) | 37.2 | 2.94 | yes | yes |

Median (95th percentile) of 200 timed CUDA-graph replays after 20 warm-up replays, batch 1; GB = 2^30 bytes. NVIDIA RTX 6000 Ada Generation, PyTorch 2.7.1+cu126, CUDA 12.6, cuDNN 90501, NVRTC 12.6.
