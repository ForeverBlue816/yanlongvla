# D3 completed screening evidence

Relative weight-noise amplitude: **1**. All 11 conditions have exact paired coverage: two suites, 10 tasks per suite, 10 episodes per task, evaluation seed 7. Offline metrics use 256 frozen calibration observations and noise seed 0, executed first 5 x 7 physical action dimensions.

The clean reference uses the validated head-removed, int8-embedding model. Clean offline errors are zero by self-comparison; clean is not one of the ten perturbed conditions used for proxy correlation.

| Perturbed flow step | Spatial | LIBERO-10 | Success % | Relative action MSE | Paired marginal MMD² |
|---|---:|---:|---:|---:|---:|
| clean | 100/100 | 91/100 | 95.5 | 0 | 0 |
| 0 | 100/100 | 93/100 | 96.5 | 2.24094453e-06 | 8.22422865e-08 |
| 1 | 100/100 | 92/100 | 96.0 | 2.76384751e-06 | 1.10234277e-07 |
| 2 | 99/100 | 96/100 | 97.5 | 4.68062199e-06 | 1.61186139e-07 |
| 3 | 98/100 | 94/100 | 96.0 | 0.00207001112 | -2.2138041e-09 |
| 4 | 99/100 | 94/100 | 96.5 | 0.0021049366 | 1.15142886e-07 |
| 5 | 99/100 | 94/100 | 96.5 | 0.00210944689 | 5.59252605e-07 |
| 6 | 99/100 | 95/100 | 97.0 | 0.00197881919 | 1.99935731e-06 |
| 7 | 100/100 | 97/100 | 98.5 | 0.000119974033 | 9.2375975e-06 |
| 8 | 99/100 | 95/100 | 97.0 | 0.000580807938 | 5.16585603e-05 |
| 9 | 98/100 | 89/100 | 93.5 | 0.0142145203 | 0.00285095384 |

Decision status: `selected_from_screening`.
Selected proxy by the preregistered rule: `relative_mse`. Step-3 evidence gate: `ready_for_review`.

| Proxy | Spearman with failure | Conditional bootstrap 95% interval |
|---|---:|---|
| relative_mse | 0.376607792 | [-0.147757, 0.776669] |
| mmd2 | -0.277825420 | [-0.487814, 0.507241] |

Paired 95% interval for the MSE-minus-MMD correlation difference: [-0.194667, 0.849331].

Intervals use 2000 paired, suite-stratified task/episode resamples conditional on evaluation seed 7 and the ten fixed perturbation conditions. They do not establish across-seed robustness or remove uncertainty from selecting a proxy on this diagnostic grid.

Finite-sample negative MMD estimates are retained. Tolerance brackets are measured amplitude intervals, not confidence intervals. Final allocation requires a selected proxy with all ten tolerances measured and bracketed. No HD-SR-VQ quantization gain follows from this diagnostic table.
