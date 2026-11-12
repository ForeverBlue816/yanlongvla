# Paper frames — GR00T N1.7

Rendered at 768 x 768 from replayed simulator states (groups A, B, D); group C is the stored 256 x 256 calibration data.
Every frame has `<stem>_{agentview,wrist}.png` (policy orientation: rendered image rotated 180 degrees, as every LIBERO client feeds it), `<stem>_{agentview,wrist}_upright.png` (display copy: upright, not mirrored) and `<stem>.json`. Contact sheets use the upright copies.

A selection rule (FP succeeds, tcq_w2_rht fails, tcq_raqft_d1_s succeeds in the recorded Plus files; ours = the A40 file, 594): qualifying units per suite {'Camera Viewpoints': {'libero_spatial': 8, 'libero_object': 14, 'libero_goal': 14, 'libero_10': 15}, 'Robot Initial States': {'libero_spatial': 5, 'libero_object': 5, 'libero_goal': 10, 'libero_10': 7}}; per category the first by unit id in each suite is used.
D units (first camera unit and first robot-init unit of A with a > 3 cm deviation): ['libero_spatial_694', 'libero_spatial_316']

## Notes (missing frames, re-run outcomes that differ from the recorded Plus file)

- A libero_object_655 tcq_raqft_d1_s: re-run success = False, recorded Plus file = True

## A

| file stem | suite | unit id | model row | policy call | env step | episode success | noise seed | frame |
|---|---|---|---|---|---|---|---|---|
| A/libero_10_363/fp_call000pct | libero_10 | 363 | fp | 0 | 10 | True | 498375094 | call000pct |
| A/libero_10_363/fp_call025pct | libero_10 | 363 | fp | 10 | 90 | True | 498375094 | call025pct |
| A/libero_10_363/fp_call050pct | libero_10 | 363 | fp | 20 | 170 | True | 498375094 | call050pct |
| A/libero_10_363/fp_call075pct | libero_10 | 363 | fp | 29 | 242 | True | 498375094 | call075pct |
| A/libero_10_363/fp_final | libero_10 | 363 | fp | 38 | 316 | True | 498375094 | final |
| A/libero_10_363/tcq_raqft_d1_s_call000pct | libero_10 | 363 | tcq_raqft_d1_s | 0 | 10 | True | 498375094 | call000pct |
| A/libero_10_363/tcq_raqft_d1_s_call025pct | libero_10 | 363 | tcq_raqft_d1_s | 10 | 90 | True | 498375094 | call025pct |
| A/libero_10_363/tcq_raqft_d1_s_call050pct | libero_10 | 363 | tcq_raqft_d1_s | 20 | 170 | True | 498375094 | call050pct |
| A/libero_10_363/tcq_raqft_d1_s_call075pct | libero_10 | 363 | tcq_raqft_d1_s | 29 | 242 | True | 498375094 | call075pct |
| A/libero_10_363/tcq_raqft_d1_s_final | libero_10 | 363 | tcq_raqft_d1_s | 30 | 256 | True | 498375094 | final |
| A/libero_10_363/tcq_w2_rht_call000pct | libero_10 | 363 | tcq_w2_rht | 0 | 10 | False | 498375094 | call000pct |
| A/libero_10_363/tcq_w2_rht_call025pct | libero_10 | 363 | tcq_w2_rht | 10 | 90 | False | 498375094 | call025pct |
| A/libero_10_363/tcq_w2_rht_call050pct | libero_10 | 363 | tcq_w2_rht | 20 | 170 | False | 498375094 | call050pct |
| A/libero_10_363/tcq_w2_rht_call075pct | libero_10 | 363 | tcq_w2_rht | 29 | 242 | False | 498375094 | call075pct |
| A/libero_10_363/tcq_w2_rht_final | libero_10 | 363 | tcq_w2_rht | 64 | 530 | False | 498375094 | final |
| A/libero_10_688/fp_call000pct | libero_10 | 688 | fp | 0 | 10 | True | 1657564835 | call000pct |
| A/libero_10_688/fp_call025pct | libero_10 | 688 | fp | 8 | 74 | True | 1657564835 | call025pct |
| A/libero_10_688/fp_call050pct | libero_10 | 688 | fp | 16 | 138 | True | 1657564835 | call050pct |
| A/libero_10_688/fp_call075pct | libero_10 | 688 | fp | 24 | 202 | True | 1657564835 | call075pct |
| A/libero_10_688/fp_final | libero_10 | 688 | fp | 31 | 259 | True | 1657564835 | final |
| A/libero_10_688/tcq_raqft_d1_s_call000pct | libero_10 | 688 | tcq_raqft_d1_s | 0 | 10 | True | 1657564835 | call000pct |
| A/libero_10_688/tcq_raqft_d1_s_call025pct | libero_10 | 688 | tcq_raqft_d1_s | 8 | 74 | True | 1657564835 | call025pct |
| A/libero_10_688/tcq_raqft_d1_s_call050pct | libero_10 | 688 | tcq_raqft_d1_s | 16 | 138 | True | 1657564835 | call050pct |
| A/libero_10_688/tcq_raqft_d1_s_call075pct | libero_10 | 688 | tcq_raqft_d1_s | 24 | 202 | True | 1657564835 | call075pct |
| A/libero_10_688/tcq_raqft_d1_s_final | libero_10 | 688 | tcq_raqft_d1_s | 31 | 261 | True | 1657564835 | final |
| A/libero_10_688/tcq_w2_rht_call000pct | libero_10 | 688 | tcq_w2_rht | 0 | 10 | False | 1657564835 | call000pct |
| A/libero_10_688/tcq_w2_rht_call025pct | libero_10 | 688 | tcq_w2_rht | 8 | 74 | False | 1657564835 | call025pct |
| A/libero_10_688/tcq_w2_rht_call050pct | libero_10 | 688 | tcq_w2_rht | 16 | 138 | False | 1657564835 | call050pct |
| A/libero_10_688/tcq_w2_rht_call075pct | libero_10 | 688 | tcq_w2_rht | 24 | 202 | False | 1657564835 | call075pct |
| A/libero_10_688/tcq_w2_rht_final | libero_10 | 688 | tcq_w2_rht | 64 | 530 | False | 1657564835 | final |
| A/libero_goal_303/fp_call000pct | libero_goal | 303 | fp | 0 | 10 | True | 3438583167 | call000pct |
| A/libero_goal_303/fp_call025pct | libero_goal | 303 | fp | 5 | 50 | True | 3438583167 | call025pct |
| A/libero_goal_303/fp_call050pct | libero_goal | 303 | fp | 10 | 90 | True | 3438583167 | call050pct |
| A/libero_goal_303/fp_call075pct | libero_goal | 303 | fp | 14 | 122 | True | 3438583167 | call075pct |
| A/libero_goal_303/fp_final | libero_goal | 303 | fp | 18 | 161 | True | 3438583167 | final |
| A/libero_goal_303/tcq_raqft_d1_s_call000pct | libero_goal | 303 | tcq_raqft_d1_s | 0 | 10 | True | 3438583167 | call000pct |
| A/libero_goal_303/tcq_raqft_d1_s_call025pct | libero_goal | 303 | tcq_raqft_d1_s | 5 | 50 | True | 3438583167 | call025pct |
| A/libero_goal_303/tcq_raqft_d1_s_call050pct | libero_goal | 303 | tcq_raqft_d1_s | 10 | 90 | True | 3438583167 | call050pct |
| A/libero_goal_303/tcq_raqft_d1_s_call075pct | libero_goal | 303 | tcq_raqft_d1_s | 14 | 122 | True | 3438583167 | call075pct |
| A/libero_goal_303/tcq_raqft_d1_s_final | libero_goal | 303 | tcq_raqft_d1_s | 21 | 184 | True | 3438583167 | final |
| A/libero_goal_303/tcq_w2_rht_call000pct | libero_goal | 303 | tcq_w2_rht | 0 | 10 | False | 3438583167 | call000pct |
| A/libero_goal_303/tcq_w2_rht_call025pct | libero_goal | 303 | tcq_w2_rht | 5 | 50 | False | 3438583167 | call025pct |
| A/libero_goal_303/tcq_w2_rht_call050pct | libero_goal | 303 | tcq_w2_rht | 10 | 90 | False | 3438583167 | call050pct |
| A/libero_goal_303/tcq_w2_rht_call075pct | libero_goal | 303 | tcq_w2_rht | 14 | 122 | False | 3438583167 | call075pct |
| A/libero_goal_303/tcq_w2_rht_final | libero_goal | 303 | tcq_w2_rht | 37 | 310 | False | 3438583167 | final |
| A/libero_goal_739/fp_call000pct | libero_goal | 739 | fp | 0 | 10 | True | 3090413492 | call000pct |
| A/libero_goal_739/fp_call025pct | libero_goal | 739 | fp | 3 | 34 | True | 3090413492 | call025pct |
| A/libero_goal_739/fp_call050pct | libero_goal | 739 | fp | 6 | 58 | True | 3090413492 | call050pct |
| A/libero_goal_739/fp_call075pct | libero_goal | 739 | fp | 9 | 82 | True | 3090413492 | call075pct |
| A/libero_goal_739/fp_final | libero_goal | 739 | fp | 11 | 99 | True | 3090413492 | final |
| A/libero_goal_739/tcq_raqft_d1_s_call000pct | libero_goal | 739 | tcq_raqft_d1_s | 0 | 10 | True | 3090413492 | call000pct |
| A/libero_goal_739/tcq_raqft_d1_s_call025pct | libero_goal | 739 | tcq_raqft_d1_s | 3 | 34 | True | 3090413492 | call025pct |
| A/libero_goal_739/tcq_raqft_d1_s_call050pct | libero_goal | 739 | tcq_raqft_d1_s | 6 | 58 | True | 3090413492 | call050pct |
| A/libero_goal_739/tcq_raqft_d1_s_call075pct | libero_goal | 739 | tcq_raqft_d1_s | 9 | 82 | True | 3090413492 | call075pct |
| A/libero_goal_739/tcq_raqft_d1_s_final | libero_goal | 739 | tcq_raqft_d1_s | 10 | 97 | True | 3090413492 | final |
| A/libero_goal_739/tcq_w2_rht_call000pct | libero_goal | 739 | tcq_w2_rht | 0 | 10 | False | 3090413492 | call000pct |
| A/libero_goal_739/tcq_w2_rht_call025pct | libero_goal | 739 | tcq_w2_rht | 3 | 34 | False | 3090413492 | call025pct |
| A/libero_goal_739/tcq_w2_rht_call050pct | libero_goal | 739 | tcq_w2_rht | 6 | 58 | False | 3090413492 | call050pct |
| A/libero_goal_739/tcq_w2_rht_call075pct | libero_goal | 739 | tcq_w2_rht | 9 | 82 | False | 3090413492 | call075pct |
| A/libero_goal_739/tcq_w2_rht_final | libero_goal | 739 | tcq_w2_rht | 37 | 310 | False | 3090413492 | final |
| A/libero_object_292/fp_call000pct | libero_object | 292 | fp | 0 | 10 | True | 110576128 | call000pct |
| A/libero_object_292/fp_call025pct | libero_object | 292 | fp | 5 | 50 | True | 110576128 | call025pct |
| A/libero_object_292/fp_call050pct | libero_object | 292 | fp | 10 | 90 | True | 110576128 | call050pct |
| A/libero_object_292/fp_call075pct | libero_object | 292 | fp | 15 | 130 | True | 110576128 | call075pct |
| A/libero_object_292/fp_final | libero_object | 292 | fp | 19 | 163 | True | 110576128 | final |
| A/libero_object_292/tcq_raqft_d1_s_call000pct | libero_object | 292 | tcq_raqft_d1_s | 0 | 10 | True | 110576128 | call000pct |
| A/libero_object_292/tcq_raqft_d1_s_call025pct | libero_object | 292 | tcq_raqft_d1_s | 5 | 50 | True | 110576128 | call025pct |
| A/libero_object_292/tcq_raqft_d1_s_call050pct | libero_object | 292 | tcq_raqft_d1_s | 10 | 90 | True | 110576128 | call050pct |
| A/libero_object_292/tcq_raqft_d1_s_call075pct | libero_object | 292 | tcq_raqft_d1_s | 15 | 130 | True | 110576128 | call075pct |
| A/libero_object_292/tcq_raqft_d1_s_final | libero_object | 292 | tcq_raqft_d1_s | 15 | 131 | True | 110576128 | final |
| A/libero_object_292/tcq_w2_rht_call000pct | libero_object | 292 | tcq_w2_rht | 0 | 10 | False | 110576128 | call000pct |
| A/libero_object_292/tcq_w2_rht_call025pct | libero_object | 292 | tcq_w2_rht | 5 | 50 | False | 110576128 | call025pct |
| A/libero_object_292/tcq_w2_rht_call050pct | libero_object | 292 | tcq_w2_rht | 10 | 90 | False | 110576128 | call050pct |
| A/libero_object_292/tcq_w2_rht_call075pct | libero_object | 292 | tcq_w2_rht | 15 | 130 | False | 110576128 | call075pct |
| A/libero_object_292/tcq_w2_rht_final | libero_object | 292 | tcq_w2_rht | 34 | 290 | False | 110576128 | final |
| A/libero_object_655/fp_call000pct | libero_object | 655 | fp | 0 | 10 | True | 4070272344 | call000pct |
| A/libero_object_655/fp_call025pct | libero_object | 655 | fp | 5 | 50 | True | 4070272344 | call025pct |
| A/libero_object_655/fp_call050pct | libero_object | 655 | fp | 10 | 90 | True | 4070272344 | call050pct |
| A/libero_object_655/fp_call075pct | libero_object | 655 | fp | 15 | 130 | True | 4070272344 | call075pct |
| A/libero_object_655/fp_final | libero_object | 655 | fp | 19 | 168 | True | 4070272344 | final |
| A/libero_object_655/tcq_raqft_d1_s_call000pct | libero_object | 655 | tcq_raqft_d1_s | 0 | 10 | False | 4070272344 | call000pct |
| A/libero_object_655/tcq_raqft_d1_s_call025pct | libero_object | 655 | tcq_raqft_d1_s | 5 | 50 | False | 4070272344 | call025pct |
| A/libero_object_655/tcq_raqft_d1_s_call050pct | libero_object | 655 | tcq_raqft_d1_s | 10 | 90 | False | 4070272344 | call050pct |
| A/libero_object_655/tcq_raqft_d1_s_call075pct | libero_object | 655 | tcq_raqft_d1_s | 15 | 130 | False | 4070272344 | call075pct |
| A/libero_object_655/tcq_raqft_d1_s_final | libero_object | 655 | tcq_raqft_d1_s | 34 | 290 | False | 4070272344 | final |
| A/libero_object_655/tcq_w2_rht_call000pct | libero_object | 655 | tcq_w2_rht | 0 | 10 | False | 4070272344 | call000pct |
| A/libero_object_655/tcq_w2_rht_call025pct | libero_object | 655 | tcq_w2_rht | 5 | 50 | False | 4070272344 | call025pct |
| A/libero_object_655/tcq_w2_rht_call050pct | libero_object | 655 | tcq_w2_rht | 10 | 90 | False | 4070272344 | call050pct |
| A/libero_object_655/tcq_w2_rht_call075pct | libero_object | 655 | tcq_w2_rht | 15 | 130 | False | 4070272344 | call075pct |
| A/libero_object_655/tcq_w2_rht_final | libero_object | 655 | tcq_w2_rht | 34 | 290 | False | 4070272344 | final |
| A/libero_spatial_316/fp_call000pct | libero_spatial | 316 | fp | 0 | 10 | True | 222116177 | call000pct |
| A/libero_spatial_316/fp_call025pct | libero_spatial | 316 | fp | 4 | 42 | True | 222116177 | call025pct |
| A/libero_spatial_316/fp_call050pct | libero_spatial | 316 | fp | 8 | 74 | True | 222116177 | call050pct |
| A/libero_spatial_316/fp_call075pct | libero_spatial | 316 | fp | 13 | 114 | True | 222116177 | call075pct |
| A/libero_spatial_316/fp_final | libero_spatial | 316 | fp | 16 | 139 | True | 222116177 | final |
| A/libero_spatial_316/tcq_raqft_d1_s_call000pct | libero_spatial | 316 | tcq_raqft_d1_s | 0 | 10 | True | 222116177 | call000pct |
| A/libero_spatial_316/tcq_raqft_d1_s_call025pct | libero_spatial | 316 | tcq_raqft_d1_s | 4 | 42 | True | 222116177 | call025pct |
| A/libero_spatial_316/tcq_raqft_d1_s_call050pct | libero_spatial | 316 | tcq_raqft_d1_s | 8 | 74 | True | 222116177 | call050pct |
| A/libero_spatial_316/tcq_raqft_d1_s_call075pct | libero_spatial | 316 | tcq_raqft_d1_s | 13 | 114 | True | 222116177 | call075pct |
| A/libero_spatial_316/tcq_raqft_d1_s_final | libero_spatial | 316 | tcq_raqft_d1_s | 22 | 191 | True | 222116177 | final |
| A/libero_spatial_316/tcq_w2_rht_call000pct | libero_spatial | 316 | tcq_w2_rht | 0 | 10 | False | 222116177 | call000pct |
| A/libero_spatial_316/tcq_w2_rht_call025pct | libero_spatial | 316 | tcq_w2_rht | 4 | 42 | False | 222116177 | call025pct |
| A/libero_spatial_316/tcq_w2_rht_call050pct | libero_spatial | 316 | tcq_w2_rht | 8 | 74 | False | 222116177 | call050pct |
| A/libero_spatial_316/tcq_w2_rht_call075pct | libero_spatial | 316 | tcq_w2_rht | 13 | 114 | False | 222116177 | call075pct |
| A/libero_spatial_316/tcq_w2_rht_final | libero_spatial | 316 | tcq_w2_rht | 27 | 230 | False | 222116177 | final |
| A/libero_spatial_694/fp_call000pct | libero_spatial | 694 | fp | 0 | 10 | True | 291047450 | call000pct |
| A/libero_spatial_694/fp_call025pct | libero_spatial | 694 | fp | 4 | 42 | True | 291047450 | call025pct |
| A/libero_spatial_694/fp_call050pct | libero_spatial | 694 | fp | 8 | 74 | True | 291047450 | call050pct |
| A/libero_spatial_694/fp_call075pct | libero_spatial | 694 | fp | 11 | 98 | True | 291047450 | call075pct |
| A/libero_spatial_694/fp_final | libero_spatial | 694 | fp | 14 | 129 | True | 291047450 | final |
| A/libero_spatial_694/tcq_raqft_d1_s_call000pct | libero_spatial | 694 | tcq_raqft_d1_s | 0 | 10 | True | 291047450 | call000pct |
| A/libero_spatial_694/tcq_raqft_d1_s_call025pct | libero_spatial | 694 | tcq_raqft_d1_s | 4 | 42 | True | 291047450 | call025pct |
| A/libero_spatial_694/tcq_raqft_d1_s_call050pct | libero_spatial | 694 | tcq_raqft_d1_s | 8 | 74 | True | 291047450 | call050pct |
| A/libero_spatial_694/tcq_raqft_d1_s_call075pct | libero_spatial | 694 | tcq_raqft_d1_s | 11 | 98 | True | 291047450 | call075pct |
| A/libero_spatial_694/tcq_raqft_d1_s_final | libero_spatial | 694 | tcq_raqft_d1_s | 11 | 102 | True | 291047450 | final |
| A/libero_spatial_694/tcq_w2_rht_call000pct | libero_spatial | 694 | tcq_w2_rht | 0 | 10 | False | 291047450 | call000pct |
| A/libero_spatial_694/tcq_w2_rht_call025pct | libero_spatial | 694 | tcq_w2_rht | 4 | 42 | False | 291047450 | call025pct |
| A/libero_spatial_694/tcq_w2_rht_call050pct | libero_spatial | 694 | tcq_w2_rht | 8 | 74 | False | 291047450 | call050pct |
| A/libero_spatial_694/tcq_w2_rht_call075pct | libero_spatial | 694 | tcq_w2_rht | 11 | 98 | False | 291047450 | call075pct |
| A/libero_spatial_694/tcq_w2_rht_final | libero_spatial | 694 | tcq_w2_rht | 27 | 230 | False | 291047450 | final |
| A/selection | None | None | None | None | None | None | None | None |

Per unit also: `eef_xyz.csv` (end-effector xyz per env step, three models) and `episodes.json` (re-run episode records, env step of every policy call).

## B

| file stem | suite | unit id | model row | policy call | env step | episode success | noise seed | frame |
|---|---|---|---|---|---|---|---|---|
| B/libero_10/camera_unit682 | libero_10 | 682 | None | 0 | 10 | None | None | Camera Viewpoints |
| B/libero_10/robot_init_unit292 | libero_10 | 292 | None | 0 | 10 | None | None | Robot Initial States |
| B/libero_10/unperturbed | libero_10 | None | None | 0 | 10 | None | None | unperturbed LIBERO |
| B/libero_goal/camera_unit715 | libero_goal | 715 | None | 0 | 10 | None | None | Camera Viewpoints |
| B/libero_goal/robot_init_unit324 | libero_goal | 324 | None | 0 | 10 | None | None | Robot Initial States |
| B/libero_goal/unperturbed | libero_goal | None | None | 0 | 10 | None | None | unperturbed LIBERO |
| B/libero_object/camera_unit667 | libero_object | 667 | None | 0 | 10 | None | None | Camera Viewpoints |
| B/libero_object/robot_init_unit292 | libero_object | 292 | None | 0 | 10 | None | None | Robot Initial States |
| B/libero_object/unperturbed | libero_object | None | None | 0 | 10 | None | None | unperturbed LIBERO |
| B/libero_spatial/camera_unit608 | libero_spatial | 608 | None | 0 | 10 | None | None | Camera Viewpoints |
| B/libero_spatial/robot_init_unit260 | libero_spatial | 260 | None | 0 | 10 | None | None | Robot Initial States |
| B/libero_spatial/unperturbed | libero_spatial | None | None | 0 | 10 | None | None | unperturbed LIBERO |

## C

| file stem | suite | unit id | model row | policy call | env step | episode success | noise seed | frame |
|---|---|---|---|---|---|---|---|---|
| C/clean | libero_spatial | 182 | None | None | None | None | None | calibration observation (LIBERO training frame) |
| C/geometric | libero_spatial | 182 | None | None | None | None | None | geometric copy, shift_perturb stream 0, j = 3 (arm D) |
| C/onpolicy | libero_spatial | 182 | tcq_w2_rht | None | None | None | None | on-policy state of the 2-bit policy tcq_w2_rht, D1 capture set (official init state by rollout seed) |
| C/photometric | libero_spatial | 182 | None | None | None | None | None | photometric copy, shift_perturb stream 0, j = 1 (arm D) |

## D

| file stem | suite | unit id | model row | policy call | env step | episode success | noise seed | frame |
|---|---|---|---|---|---|---|---|---|
| D/libero_spatial_316/fp_step034 | libero_spatial | 316 | fp | 3 | 34 | True | 222116177 | first env step with |eef_base - eef_FP| > 3 cm |
| D/libero_spatial_316/tcq_raqft_d1_s_step034 | libero_spatial | 316 | tcq_raqft_d1_s | 3 | 34 | True | 222116177 | first env step with |eef_base - eef_FP| > 3 cm |
| D/libero_spatial_316/tcq_w2_rht_step034 | libero_spatial | 316 | tcq_w2_rht | 3 | 34 | False | 222116177 | first env step with |eef_base - eef_FP| > 3 cm |
| D/libero_spatial_694/fp_step091 | libero_spatial | 694 | fp | 10 | 91 | True | 291047450 | first env step with |eef_base - eef_FP| > 3 cm |
| D/libero_spatial_694/tcq_raqft_d1_s_step091 | libero_spatial | 694 | tcq_raqft_d1_s | 10 | 91 | True | 291047450 | first env step with |eef_base - eef_FP| > 3 cm |
| D/libero_spatial_694/tcq_w2_rht_step091 | libero_spatial | 694 | tcq_w2_rht | 10 | 91 | False | 291047450 | first env step with |eef_base - eef_FP| > 3 cm |

