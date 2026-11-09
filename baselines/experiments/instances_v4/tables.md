# GR00T N1.7 on LIBERO-Plus 1000: success by group and a look at the instances

_GR00T N1.7 has no plain LIBERO table: the mentor dropped it, so every GR00T number here is LIBERO-Plus 1000 (paired against FP on the same units). Differences below about 20 of 1000 successes are inside the noise floor (EXPLANATIONS.md E0). Generated 4 Nov 2026 with every finished row._

Rows with 1000 finished episodes: FP, QTIP, QuIP# 0 sweeps, QuIP# 10 sweeps, VPTQ, AQLM, GPTQ, GPTQ MSE, GPTQ act-order+MSE, GPTQ combo, VPTQ v5 k1024, GPTQ best, VPTQ 500 iter, VPTQ 500 iter t0, VPTQ v6 k4096, VPTQ v6 k3072, VPTQ v8 k12288.


#### By suite

| | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---|---|---|---|
| FP | 178/245 (73%) | 128/251 (51%) | 173/256 (68%) | 139/248 (56%) |
| QTIP | 170/245 (69%) | 119/251 (47%) | 174/256 (68%) | 110/248 (44%) |
| QuIP# 0 sweeps | 168/245 (69%) | 118/251 (47%) | 156/256 (61%) | 112/248 (45%) |
| QuIP# 10 sweeps | 165/245 (67%) | 113/251 (45%) | 151/256 (59%) | 102/248 (41%) |
| VPTQ | 161/245 (66%) | 111/251 (44%) | 143/256 (56%) | 80/248 (32%) |
| AQLM | 143/245 (58%) | 98/251 (39%) | 104/256 (41%) | 56/248 (23%) |
| GPTQ | 0/245 (0%) | 0/251 (0%) | 0/256 (0%) | 0/248 (0%) |
| GPTQ MSE | 110/245 (45%) | 81/251 (32%) | 73/256 (29%) | 36/248 (15%) |
| GPTQ act-order+MSE | 133/245 (54%) | 115/251 (46%) | 124/256 (48%) | 82/248 (33%) |
| GPTQ combo | 147/245 (60%) | 100/251 (40%) | 117/256 (46%) | 75/248 (30%) |
| VPTQ v5 k1024 | 147/245 (60%) | 115/251 (46%) | 143/256 (56%) | 88/248 (35%) |
| GPTQ best | 138/245 (56%) | 122/251 (49%) | 128/256 (50%) | 88/248 (35%) |
| VPTQ 500 iter | 153/245 (62%) | 101/251 (40%) | 142/256 (55%) | 91/248 (37%) |
| VPTQ 500 iter t0 | 153/245 (62%) | 99/251 (39%) | 136/256 (53%) | 83/248 (33%) |
| VPTQ v6 k4096 | 175/245 (71%) | 117/251 (47%) | 136/256 (53%) | 107/248 (43%) |
| VPTQ v6 k3072 | 143/245 (58%) | 105/251 (42%) | 125/256 (49%) | 85/248 (34%) |
| VPTQ v8 k12288 | 141/245 (58%) | 98/251 (39%) | 118/256 (46%) | 69/248 (28%) |


#### By perturbation

| | Camera Viewpoints | Robot Initial States |
|---|---|---|
| FP | 327/500 (65%) | 291/500 (58%) |
| QTIP | 294/500 (59%) | 279/500 (56%) |
| QuIP# 0 sweeps | 285/500 (57%) | 269/500 (54%) |
| QuIP# 10 sweeps | 272/500 (54%) | 259/500 (52%) |
| VPTQ | 258/500 (52%) | 237/500 (47%) |
| AQLM | 201/500 (40%) | 200/500 (40%) |
| GPTQ | 0/500 (0%) | 0/500 (0%) |
| GPTQ MSE | 156/500 (31%) | 144/500 (29%) |
| GPTQ act-order+MSE | 225/500 (45%) | 229/500 (46%) |
| GPTQ combo | 228/500 (46%) | 211/500 (42%) |
| VPTQ v5 k1024 | 259/500 (52%) | 234/500 (47%) |
| GPTQ best | 238/500 (48%) | 238/500 (48%) |
| VPTQ 500 iter | 250/500 (50%) | 237/500 (47%) |
| VPTQ 500 iter t0 | 251/500 (50%) | 220/500 (44%) |
| VPTQ v6 k4096 | 280/500 (56%) | 255/500 (51%) |
| VPTQ v6 k3072 | 220/500 (44%) | 238/500 (48%) |
| VPTQ v8 k12288 | 212/500 (42%) | 214/500 (43%) |


#### By perturbation and suite

| | ('Camera Viewpoints', 'libero_spatial') | ('Camera Viewpoints', 'libero_object') | ('Camera Viewpoints', 'libero_goal') | ('Camera Viewpoints', 'libero_10') | ('Robot Initial States', 'libero_spatial') | ('Robot Initial States', 'libero_object') | ('Robot Initial States', 'libero_goal') | ('Robot Initial States', 'libero_10') |
|---|---|---|---|---|---|---|---|---|
| FP | 90/120 (75%) | 62/126 (49%) | 100/129 (78%) | 75/125 (60%) | 88/125 (70%) | 66/125 (53%) | 73/127 (57%) | 64/123 (52%) |
| QTIP | 86/120 (72%) | 55/126 (44%) | 104/129 (81%) | 49/125 (39%) | 84/125 (67%) | 64/125 (51%) | 70/127 (55%) | 61/123 (50%) |
| QuIP# 0 sweeps | 86/120 (72%) | 50/126 (40%) | 98/129 (76%) | 51/125 (41%) | 82/125 (66%) | 68/125 (54%) | 58/127 (46%) | 61/123 (50%) |
| QuIP# 10 sweeps | 81/120 (68%) | 57/126 (45%) | 93/129 (72%) | 41/125 (33%) | 84/125 (67%) | 56/125 (45%) | 58/127 (46%) | 61/123 (50%) |
| VPTQ | 82/120 (68%) | 55/126 (44%) | 90/129 (70%) | 31/125 (25%) | 79/125 (63%) | 56/125 (45%) | 53/127 (42%) | 49/123 (40%) |
| AQLM | 67/120 (56%) | 51/126 (40%) | 63/129 (49%) | 20/125 (16%) | 76/125 (61%) | 47/125 (38%) | 41/127 (32%) | 36/123 (29%) |
| GPTQ | 0/120 (0%) | 0/126 (0%) | 0/129 (0%) | 0/125 (0%) | 0/125 (0%) | 0/125 (0%) | 0/127 (0%) | 0/123 (0%) |
| GPTQ MSE | 52/120 (43%) | 49/126 (39%) | 43/129 (33%) | 12/125 (10%) | 58/125 (46%) | 32/125 (26%) | 30/127 (24%) | 24/123 (20%) |
| GPTQ act-order+MSE | 56/120 (47%) | 57/126 (45%) | 78/129 (60%) | 34/125 (27%) | 77/125 (62%) | 58/125 (46%) | 46/127 (36%) | 48/123 (39%) |
| GPTQ combo | 79/120 (66%) | 45/126 (36%) | 73/129 (57%) | 31/125 (25%) | 68/125 (54%) | 55/125 (44%) | 44/127 (35%) | 44/123 (36%) |
| VPTQ v5 k1024 | 71/120 (59%) | 57/126 (45%) | 88/129 (68%) | 43/125 (34%) | 76/125 (61%) | 58/125 (46%) | 55/127 (43%) | 45/123 (37%) |
| GPTQ best | 62/120 (52%) | 65/126 (52%) | 76/129 (59%) | 35/125 (28%) | 76/125 (61%) | 57/125 (46%) | 52/127 (41%) | 53/123 (43%) |
| VPTQ 500 iter | 77/120 (64%) | 46/126 (37%) | 88/129 (68%) | 39/125 (31%) | 76/125 (61%) | 55/125 (44%) | 54/127 (43%) | 52/123 (42%) |
| VPTQ 500 iter t0 | 79/120 (66%) | 51/126 (40%) | 85/129 (66%) | 36/125 (29%) | 74/125 (59%) | 48/125 (38%) | 51/127 (40%) | 47/123 (38%) |
| VPTQ v6 k4096 | 92/120 (77%) | 56/126 (44%) | 83/129 (64%) | 49/125 (39%) | 83/125 (66%) | 61/125 (49%) | 53/127 (42%) | 58/123 (47%) |
| VPTQ v6 k3072 | 71/120 (59%) | 46/126 (37%) | 73/129 (57%) | 30/125 (24%) | 72/125 (58%) | 59/125 (47%) | 52/127 (41%) | 55/123 (45%) |
| VPTQ v8 k12288 | 69/120 (58%) | 41/126 (33%) | 77/129 (60%) | 25/125 (20%) | 72/125 (58%) | 57/125 (46%) | 41/127 (32%) | 44/123 (36%) |


#### By difficulty level (the unit field difficulty_level)

| | ('Camera Viewpoints', 1) | ('Camera Viewpoints', 2) | ('Camera Viewpoints', 3) | ('Camera Viewpoints', 4) | ('Camera Viewpoints', 5) | ('Robot Initial States', 1) | ('Robot Initial States', 2) | ('Robot Initial States', 3) | ('Robot Initial States', 4) | ('Robot Initial States', 5) |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 67/77 (87%) | 96/118 (81%) | 76/103 (74%) | 38/70 (54%) | 50/132 (38%) | 79/90 (88%) | 71/92 (77%) | 86/126 (68%) | 33/87 (38%) | 22/105 (21%) |
| QTIP | 69/77 (90%) | 87/118 (74%) | 72/103 (70%) | 32/70 (46%) | 34/132 (26%) | 74/90 (82%) | 69/92 (75%) | 81/126 (64%) | 39/87 (45%) | 16/105 (15%) |
| QuIP# 0 sweeps | 64/77 (83%) | 89/118 (75%) | 62/103 (60%) | 35/70 (50%) | 35/132 (27%) | 77/90 (86%) | 70/92 (76%) | 72/126 (57%) | 29/87 (33%) | 21/105 (20%) |
| QuIP# 10 sweeps | 65/77 (84%) | 85/118 (72%) | 60/103 (58%) | 31/70 (44%) | 31/132 (23%) | 75/90 (83%) | 69/92 (75%) | 75/126 (60%) | 25/87 (29%) | 15/105 (14%) |
| VPTQ | 60/77 (78%) | 83/118 (70%) | 61/103 (59%) | 28/70 (40%) | 26/132 (20%) | 75/90 (83%) | 64/92 (70%) | 67/126 (53%) | 20/87 (23%) | 11/105 (10%) |
| AQLM | 53/77 (69%) | 67/118 (57%) | 44/103 (43%) | 21/70 (30%) | 16/132 (12%) | 65/90 (72%) | 56/92 (61%) | 51/126 (40%) | 18/87 (21%) | 10/105 (10%) |
| GPTQ | 0/77 (0%) | 0/118 (0%) | 0/103 (0%) | 0/70 (0%) | 0/132 (0%) | 0/90 (0%) | 0/92 (0%) | 0/126 (0%) | 0/87 (0%) | 0/105 (0%) |
| GPTQ MSE | 40/77 (52%) | 56/118 (47%) | 37/103 (36%) | 16/70 (23%) | 7/132 (5%) | 46/90 (51%) | 43/92 (47%) | 39/126 (31%) | 9/87 (10%) | 7/105 (7%) |
| GPTQ act-order+MSE | 57/77 (74%) | 70/118 (59%) | 53/103 (51%) | 24/70 (34%) | 21/132 (16%) | 73/90 (81%) | 60/92 (65%) | 60/126 (48%) | 26/87 (30%) | 10/105 (10%) |
| GPTQ combo | 52/77 (68%) | 74/118 (63%) | 50/103 (49%) | 23/70 (33%) | 29/132 (22%) | 68/90 (76%) | 53/92 (58%) | 59/126 (47%) | 19/87 (22%) | 12/105 (11%) |
| VPTQ v5 k1024 | 66/77 (86%) | 74/118 (63%) | 61/103 (59%) | 25/70 (36%) | 33/132 (25%) | 69/90 (77%) | 64/92 (70%) | 65/126 (52%) | 25/87 (29%) | 11/105 (10%) |
| GPTQ best | 60/77 (78%) | 77/118 (65%) | 54/103 (52%) | 23/70 (33%) | 24/132 (18%) | 73/90 (81%) | 65/92 (71%) | 58/126 (46%) | 27/87 (31%) | 15/105 (14%) |
| VPTQ 500 iter | 58/77 (75%) | 80/118 (68%) | 59/103 (57%) | 27/70 (39%) | 26/132 (20%) | 68/90 (76%) | 64/92 (70%) | 68/126 (54%) | 23/87 (26%) | 14/105 (13%) |
| VPTQ 500 iter t0 | 57/77 (74%) | 80/118 (68%) | 57/103 (55%) | 29/70 (41%) | 28/132 (21%) | 69/90 (77%) | 59/92 (64%) | 61/126 (48%) | 22/87 (25%) | 9/105 (9%) |
| VPTQ v6 k4096 | 65/77 (84%) | 90/118 (76%) | 67/103 (65%) | 28/70 (40%) | 30/132 (23%) | 71/90 (79%) | 65/92 (71%) | 68/126 (54%) | 33/87 (38%) | 18/105 (17%) |
| VPTQ v6 k3072 | 63/77 (82%) | 68/118 (58%) | 47/103 (46%) | 24/70 (34%) | 18/132 (14%) | 76/90 (84%) | 61/92 (66%) | 63/126 (50%) | 25/87 (29%) | 13/105 (12%) |
| VPTQ v8 k12288 | 52/77 (68%) | 70/118 (59%) | 58/103 (56%) | 20/70 (29%) | 12/132 (9%) | 69/90 (77%) | 58/92 (63%) | 53/126 (42%) | 18/87 (21%) | 16/105 (15%) |


## Instances (main rows: QTIP, QuIP# 10 sweeps, VPTQ, AQLM)


#### Units by FP result and by the number of main 2-bit rows that solve them

| | 0 of 4 solve | 1 of 4 solve | 2 of 4 solve | 3 of 4 solve | 4 of 4 solve |
|---|---|---|---|---|---|
| FP solves | 51 | 65 | 98 | 147 | 257 |
| FP fails | 241 | 65 | 34 | 31 | 11 |

FP solves and no main row solves: 51 units. FP fails and every main row solves: 11 units (the noise level of the comparison).


#### Base tasks that lose most (FP successes minus the mean main-row successes)

| | units | FP solves | main rows solve (mean) | loss |
|---|---|---|---|---|
| libero_10 KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close | 25 | 16 | 4.5 | 11.5 |
| libero_10 KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 26 | 25 | 17.2 | 7.8 |
| libero_10 STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_ | 24 | 17 | 9.5 | 7.5 |
| libero_goal put_the_wine_bottle_on_the_rack | 25 | 20 | 13.0 | 7.0 |
| libero_10 KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_ | 24 | 20 | 13.5 | 6.5 |
| libero_10 LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yel | 25 | 9 | 3.5 | 5.5 |
| libero_10 LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the | 24 | 16 | 10.8 | 5.2 |
| libero_spatial pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 25 | 20 | 15.0 | 5.0 |
| libero_object pick_up_the_salad_dressing_and_place_it_in_the_basket | 26 | 23 | 18.0 | 5.0 |
| libero_goal put_the_cream_cheese_in_the_bowl | 24 | 17 | 12.0 | 5.0 |
| libero_goal push_the_plate_to_the_front_of_the_stove | 26 | 7 | 2.2 | 4.8 |
| libero_spatial pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plat | 24 | 22 | 17.5 | 4.5 |


#### Base tasks with no loss or a gain (the 5 best)

| | units | FP solves | main rows solve (mean) | loss |
|---|---|---|---|---|
| libero_spatial pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 24 | 17 | 17.5 | -0.5 |
| libero_spatial pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_ | 19 | 16 | 16.8 | -0.8 |
| libero_spatial pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 26 | 17 | 18.2 | -1.2 |
| libero_goal put_the_bowl_on_the_stove | 26 | 19 | 21.0 | -2.0 |
| libero_object pick_up_the_cream_cheese_and_place_it_in_the_basket | 26 | 16 | 18.2 | -2.2 |


### The units FP solves and no main row solves (first 25)

| suite | perturbation | level | unit |
|---|---|---|---|
| libero_spatial | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_301_15_100_0_0_initstate_0 |
| libero_spatial | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_300_15_100_0_0_initstate_0 |
| libero_spatial | Robot Initial States | 5 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_372 |
| libero_spatial | Robot Initial States | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_init |
| libero_spatial | Robot Initial States | 5 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_366 |
| libero_spatial | Robot Initial States | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_276 |
| libero_object | Camera Viewpoints | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_68_15_100_0_0_initstate_0 |
| libero_object | Camera Viewpoints | 5 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_190_0_0_initstate_0 |
| libero_object | Camera Viewpoints | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_290_15_100_0_0_initstate_0 |
| libero_object | Camera Viewpoints | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_69_0_100_0_0_initstate_0 |
| libero_object | Camera Viewpoints | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_292_15_100_0_0_initstate_0 |
| libero_object | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_322_15_100_0_0_initstate_0 |
| libero_object | Robot Initial States | 5 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_454 |
| libero_object | Robot Initial States | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_200 |
| libero_object | Robot Initial States | 3 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_286 |
| libero_object | Robot Initial States | 4 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_273 |
| libero_object | Robot Initial States | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_254 |
| libero_goal | Camera Viewpoints | 4 | push_the_plate_to_the_front_of_the_stove_view_287_15_100_0_0_initstate_0 |
| libero_goal | Robot Initial States | 5 | put_the_bowl_on_the_plate_view_0_0_100_0_0_initstate_219 |
| libero_goal | Robot Initial States | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_294 |
| libero_goal | Robot Initial States | 3 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_136 |
| libero_goal | Robot Initial States | 2 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_196 |
| libero_goal | Robot Initial States | 4 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_491 |
| libero_goal | Robot Initial States | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_314 |
| libero_goal | Robot Initial States | 3 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_247 |


### Agreement with pi0.5 on the shared units (majority of the main rows solves the unit)

| both models solve | only GR00T | only pi0.5 | neither |
|---|---|---|---|
| 495 | 83 | 211 | 211 |
