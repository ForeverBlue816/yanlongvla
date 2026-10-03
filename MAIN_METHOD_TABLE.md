# FlowVQ main-method validation — 2026-10-03

Frozen b* prefix schedule + conditional centroids + per-subset affine; quantized backbone, FP32 AdaRMS tables and int8 embeddings. No additional model fitting or evaluation seeds.

| Model | Linear bpw | Whole-model bpw | Bytes read / inference | Held-out relative MSE | Medium | Full |
|---|---:|---:|---:|---:|---:|---:|
| FP | 16.000000 | 16.571214 | 17,434,050,720 | 0.00000000 | 970/1000 (97.00%) | 1928/2000 (96.40%) |
| Backbone M3 + M2 | 2.908467 | 3.638965 | 2,048,853,828 | 0.01304602 | 974/1000 (97.40%) | 1948/2000 (97.40%) |
| Backbone M3 + FlowVQ | 2.915447 | 3.644593 | 1,703,069,508 | 0.00730857 | 968/1000 (96.80%) | 1936/2000 (96.80%) |
| Backbone M2 + M2 (medium only) | 2.020137 | 2.922590 | 1,644,469,572 | 0.03721696 | 982/1000 (98.20%) | — |
| Backbone M2 + FlowVQ | 2.027117 | 2.928218 | 1,298,685,252 | 0.04291682 | 964/1000 (96.40%) | 1939/2000 (96.95%) |

Linear bpw includes quantized eligible Linear weight payload plus codebooks, indices, scales, masks, signs and affine metadata; native biases are reserved. Whole-model bpw uses the same 3,353,433,872 active original parameter denominator, with retained native FP32 tensors counted. FP is the accepted native BF16 policy with FP32 exceptions and its original embedding; removal of unused heads was numerically exact.

Bytes read are independent offset-derived logical compressed tensor reads per 10-step inference, including three vision calls and 15 distinct prompt embedding rows in the audited observation. They exclude activations/KV and are not measured GPU DRAM or latency. The actual quality runtime uses BF16 reconstruction caches: both FlowVQ deployments have 12,702,343,332 logical fallback bytes per inference.

Held-out MSE is global squared action error / FP action energy on the executed 5×7 window, using the same 256 observations from 40 trajectories and fixed noise 0. Both new deployments passed strict reload, retained tensor equality and four action-repeat guards. Absolute MSE and paired trajectory intervals are in the machine-readable evidence.

Medium uses episodes 0–24; full uses 0–49, all four suites, ten tasks each, evaluation seed 7. Reuse is checkpoint- and source-verified; the BF16-backbone c episodes are excluded. Paired 95% intervals below use the existing suite-stratified hierarchical task/episode bootstrap with 2000 resamples and are conditional on this single evaluation seed.

Backbone M3 + FlowVQ versus FP: +0.40 pp, paired 95% CI [-0.80, +1.60] pp.

Backbone M2 + FlowVQ versus FP: +0.55 pp, paired 95% CI [-1.45, +2.90] pp.

Backbone M3 + FlowVQ versus backbone M3 + uniform M2: -0.60 pp, paired 95% CI [-2.25, +0.85] pp. No matching M2-backbone uniform full result exists.

Sources: [complete machine-readable evidence](results/flowvq_main_complete.json), [independent accounting](results/flowvq_main_accounting.json), [paired held-out MSE](results/flowvq_main_offline.json).
