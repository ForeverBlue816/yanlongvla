### pi05: 8 rows, 28 comparisons, Holm at 0.05; E0 floor = 2.0 pp; p-values use the paired se widened by the E0 perturbation noise (1.14 pp) in quadrature; the interval column is the paired interval with a standard error clustered by base task

| row A | row B | A | B | A-B (pp) | 95% interval | p | Holm | beyond floor | real |
|---|---|---|---|---|---|---|---|---|---|
| aqlm_w2 | gptq_best_w2 | 639 | 657 | -1.8 | [-4.2, +0.6] | 0.278 | no | no | no |
| aqlm_w2 | qtip_w2 | 639 | 726 | -8.7 | [-12.7, -4.7] | 0.000224 | yes | yes | **yes** |
| aqlm_w2 | quip_w2 | 639 | 662 | -2.3 | [-6.1, +1.5] | 0.308 | no | yes | no |
| aqlm_w2 | quipt0_w2 | 639 | 740 | -10.1 | [-14.1, -6.1] | 1.45e-05 | yes | yes | **yes** |
| aqlm_w2 | vptq_w2 | 639 | 685 | -4.6 | [-7.5, -1.7] | 0.0145 | no | yes | no |
| aqlm_w2 | fp_w16 | 639 | 765 | -12.6 | [-17.3, -7.9] | 1.89e-06 | yes | yes | **yes** |
| aqlm_w2 | original_2bit | 639 | 704 | -6.5 | [-9.5, -3.5] | 0.000654 | yes | yes | **yes** |
| gptq_best_w2 | qtip_w2 | 657 | 726 | -6.9 | [-11.1, -2.7] | 0.00413 | no | yes | no |
| gptq_best_w2 | quip_w2 | 657 | 662 | -0.5 | [-4.3, +3.3] | 0.825 | no | no | no |
| gptq_best_w2 | quipt0_w2 | 657 | 740 | -8.3 | [-12.0, -4.6] | 0.000158 | yes | yes | **yes** |
| gptq_best_w2 | vptq_w2 | 657 | 685 | -2.8 | [-5.4, -0.2] | 0.113 | no | yes | no |
| gptq_best_w2 | fp_w16 | 657 | 765 | -10.8 | [-15.3, -6.3] | 2.89e-05 | yes | yes | **yes** |
| gptq_best_w2 | original_2bit | 657 | 704 | -4.7 | [-7.1, -2.3] | 0.00515 | no | yes | no |
| qtip_w2 | quip_w2 | 726 | 662 | +6.4 | [+3.5, +9.3] | 0.000683 | yes | yes | **yes** |
| qtip_w2 | quipt0_w2 | 726 | 740 | -1.4 | [-4.0, +1.2] | 0.428 | no | no | no |
| qtip_w2 | vptq_w2 | 726 | 685 | +4.1 | [-0.2, +8.4] | 0.0998 | no | yes | no |
| qtip_w2 | fp_w16 | 726 | 765 | -3.9 | [-6.6, -1.2] | 0.0278 | no | yes | no |
| qtip_w2 | original_2bit | 726 | 704 | +2.2 | [-1.4, +5.8] | 0.312 | no | yes | no |
| quip_w2 | quipt0_w2 | 662 | 740 | -7.8 | [-10.9, -4.7] | 7.45e-05 | yes | yes | **yes** |
| quip_w2 | vptq_w2 | 662 | 685 | -2.3 | [-6.1, +1.5] | 0.302 | no | yes | no |
| quip_w2 | fp_w16 | 662 | 765 | -10.3 | [-14.2, -6.4] | 6.13e-06 | yes | yes | **yes** |
| quip_w2 | original_2bit | 662 | 704 | -4.2 | [-7.3, -1.1] | 0.0324 | no | yes | no |
| quipt0_w2 | vptq_w2 | 740 | 685 | +5.5 | [+1.1, +9.9] | 0.0291 | no | yes | no |
| quipt0_w2 | fp_w16 | 740 | 765 | -2.5 | [-5.2, +0.2] | 0.159 | no | yes | no |
| quipt0_w2 | original_2bit | 740 | 704 | +3.6 | [+0.4, +6.8] | 0.0707 | no | yes | no |
| vptq_w2 | fp_w16 | 685 | 765 | -8.0 | [-12.8, -3.2] | 0.00321 | no | yes | no |
| vptq_w2 | original_2bit | 685 | 704 | -1.9 | [-4.8, +1.0] | 0.308 | no | no | no |
| fp_w16 | original_2bit | 765 | 704 | +6.1 | [+2.1, +10.1] | 0.00935 | no | yes | no |
