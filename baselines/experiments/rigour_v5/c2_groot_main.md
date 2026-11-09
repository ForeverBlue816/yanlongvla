### groot: 7 rows, 21 comparisons, Holm at 0.05; E0 floor = 2.0 pp; p-values use the paired se widened by the E0 perturbation noise (1.14 pp) in quadrature; the interval column is the paired interval with a standard error clustered by base task

| row A | row B | A | B | A-B (pp) | 95% interval | p | Holm | beyond floor | real |
|---|---|---|---|---|---|---|---|---|---|
| aqlm_w2 | fp | 401 | 618 | -21.7 | [-27.7, -15.7] | 3.28e-11 | yes | yes | **yes** |
| aqlm_w2 | gptq_best_w2 | 401 | 476 | -7.5 | [-13.0, -2.0] | 0.0133 | no | yes | no |
| aqlm_w2 | qtip_w2 | 401 | 573 | -17.2 | [-22.8, -11.6] | 2.52e-08 | yes | yes | **yes** |
| aqlm_w2 | quip10_w2 | 401 | 531 | -13.0 | [-17.9, -8.1] | 2.04e-06 | yes | yes | **yes** |
| aqlm_w2 | quip_w2 | 401 | 554 | -15.3 | [-21.0, -9.6] | 8.78e-07 | yes | yes | **yes** |
| aqlm_w2 | vptq_w2 | 401 | 495 | -9.4 | [-14.3, -4.5] | 0.000647 | yes | yes | **yes** |
| fp | gptq_best_w2 | 618 | 476 | +14.2 | [+8.7, +19.7] | 2.67e-06 | yes | yes | **yes** |
| fp | qtip_w2 | 618 | 573 | +4.5 | [+1.8, +7.2] | 0.0115 | no | yes | no |
| fp | quip10_w2 | 618 | 531 | +8.7 | [+5.3, +12.1] | 3.02e-05 | yes | yes | **yes** |
| fp | quip_w2 | 618 | 554 | +6.4 | [+3.4, +9.4] | 0.000846 | yes | yes | **yes** |
| fp | vptq_w2 | 618 | 495 | +12.3 | [+7.4, +17.2] | 7.98e-06 | yes | yes | **yes** |
| gptq_best_w2 | qtip_w2 | 476 | 573 | -9.7 | [-14.8, -4.6] | 0.000612 | yes | yes | **yes** |
| gptq_best_w2 | quip10_w2 | 476 | 531 | -5.5 | [-9.7, -1.3] | 0.024 | no | yes | no |
| gptq_best_w2 | quip_w2 | 476 | 554 | -7.8 | [-13.0, -2.6] | 0.00723 | no | yes | no |
| gptq_best_w2 | vptq_w2 | 476 | 495 | -1.9 | [-6.6, +2.8] | 0.477 | no | no | no |
| qtip_w2 | quip10_w2 | 573 | 531 | +4.2 | [+1.2, +7.2] | 0.0294 | no | yes | no |
| qtip_w2 | quip_w2 | 573 | 554 | +1.9 | [-1.2, +5.0] | 0.329 | no | no | no |
| qtip_w2 | vptq_w2 | 573 | 495 | +7.8 | [+3.4, +12.2] | 0.0021 | yes | yes | **yes** |
| quip10_w2 | quip_w2 | 531 | 554 | -2.3 | [-5.2, +0.6] | 0.216 | no | yes | no |
| quip10_w2 | vptq_w2 | 531 | 495 | +3.6 | [+0.5, +6.7] | 0.0658 | no | yes | no |
| quip_w2 | vptq_w2 | 554 | 495 | +5.9 | [+1.8, +10.0] | 0.0132 | no | yes | no |
