# GR00T N1.7 SimplerEnv-Bridge: VPTQ v8, 12288 centroids (true 2 bits)

VPTQ with the official vector length 8 and 12288 centroids: 1.988 bits per weight with the codebook counted (`quant.json`),
the same structure as the GR00T LIBERO-Plus row `vptq_v8k12288_w2` (`../../../results/groot_2bit_ours_vptq_v8k12288_w2.plus_1000.jsonl`, 426/1000).
Calibration: the 256 Bridge observations. Evaluation: the seeded protocol and layout table of `../../README.md` (`eval_simpler_bridge.py`, unchanged; as FP and the
other 2-bit Bridge rows), 120 episodes per task, seed 7.
`widowx_<task>.json`: one record per episode. `paired_vs_fp.json`: paired bootstrap over (task, layout) clusters (`simpler/b5_stats.py`).

| task | FP | VPTQ v8 k12288 |
|---|---|---|
| spoon_on_towel | 101 (84.2%) | 80 (66.7%) |
| carrot_on_plate | 75 (62.5%) | 92 (76.7%) |
| stack_cube | 36 (30.0%) | 62 (51.7%) |
| put_eggplant_in_basket | 56 (46.7%) | 89 (74.2%) |
| put_eggplant_in_sink | 0 (0.0%) | 7 (5.8%) |
| open_drawer | 120 (100.0%) | 119 (99.2%) |
| close_drawer | 115 (95.8%) | 108 (90.0%) |
| total /840 | 503 (59.9%) | 557 (66.3%) |

Difference to FP: +6.4 pp, paired 95% CI [+1.4, +11.4].
