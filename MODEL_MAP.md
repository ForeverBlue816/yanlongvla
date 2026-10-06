# October7 addendum — full policy received and D3 replaced

The [complete addendum](ADDENDUM_2026-10-07.md) retains C1–C5 and D1, and incorporates the all-task D3 replacement. D3 has1600offline observations/40tasks/32latents plus three pairedPlus1000FP-prefix interventions. The three A8 feasibility rows are separate from method selection. D1/D3/A8 implementations remain pending; these instructions are not results.

The>=3pp paired-significant promotion gate versus704/1000 is unchanged. Final deployment now requires the A8 feasibility decision and a validated winning-backbone + uniformM2-expert + A8 composition before LIBEROfull2000/1900 and expandedPlus2000. Legacy direct-to-final dispatch is disabled; current baseline fitting/evaluation continues. The expandedPlus set still reuses the originalsubset and reports added1000 separately.

# Active comparison policy — October6 evening C1–C5

[PRECISION_2026-10-06.md](PRECISION_2026-10-06.md) governs the new round. C1 (`hd_srvq_precision_c1`) protects all36 Gemma K/V projections in nativeBF16:3.000463596439 whole-model bpw. C2 (`hd_srvq_precision_c2`) uses3bit linears in Gemma14–17:3.054192145406bpw. Both exports strictly reload; Plus quality evaluation is pending. C3 starts from original2bit with2304updates/16latents/steps0–9; C4 uses originalH5b768updates/8latents/steps0–5 onC2. The uniformM2 expert remains frozen.

Both mixed-calibration coverage variants remain conditional on accepted data generation. A positive paired Plus gain with95%CI lower>0 triggers originalH5b fitting on that variant and the mixed256; this user-approved trigger does not change the>=3pp promotion gate. All selection uses paired seed7 Plus500+500. Candidate diagnostics include per-layer/type K/V errors, prefix attention entropy and S1 energy decomposition. The winner retains LIBERO2000/1900 and nested expandedPlus2000; coverage retains medium1000/950. Older sections below describe historical rounds.

# Active comparison policy — October6

The new authority is [PLUS_ROUND_2026-10-06.md](PLUS_ROUND_2026-10-06.md). H5a/b/c4/c16 and S2 cached-KV gain correction are evaluated on fixed Plus500+500, alongside the two mixed-calibration coverage variants. The four H5 fit artifacts are complete; new evaluation remains pending validation/results. S2 gains are fit from original training256 only; an S2 deployment requires the gain artifact and cache transform in addition to the original2-bit checkpoint. No standalone task8/9 or H4/H4b work remains.

Coverage and the eventual final deployment retain full40-task LIBERO1000/2000, with950/1900 columns alongside. Promotion uses only Plus; expanded Plus1000+1000 includes the original500+500, with added-instance results reported separately. Older sections below are historical and their previous gates do not govern the new queue.

# October5 causal-round model map (supersedes historical pending statuses below)

H3 uses the unchanged FP checkpoint with controlled prefix K/V noise;13 primary group-local conditions plus2 distinct equal-total-energy controls. There are no state tokens in this config. Native KV anchors: M2 image0.06225/language0.21506; M3 image0.01417/language0.06230.

H5 output variants are planned as `hd_srvq_causal_h5a`, `hd_srvq_causal_h5b`, `hd_srvq_causal_h5c4`, and `hd_srvq_causal_h5c16`. GPU pilots/fitting are pending; these names do not yet imply existing validated checkpoints. H5a uses early velocity-direction backbone importance; H5b changes backbone codebooks only; H5c4/16 changes valid-language activation weighting in layer-Hessian calibration. All keep the exact uniformM2 expert and baseline tensor layout/bpw. H5 evaluation waits for full H4 failure labels. TV failed its detector gate and is diagnostic only. “Action-aware” remains an ablation label, not a method name.

Current evidence: [causal round](results/causal_round.json), [H2 distributions](results/causal_h2_followup.json), [paired Plus dimensions](results/causal_plus_comparison.json). Four historical Plus comparators and plus2 all have the same accepted500 camera+500 initial-state records. No fullFP10030 restart.

# X3 precision clarification — offline only

The theory check now widens the identical accepted expert weights and sampler arithmetic toFP32 with TF32 disabled; the fixed observation-only backbone KV is widened for attention. Deployment and accepted rollouts remain BF16. JVP gains are primary, paired finite-amplitude slopes secondary, with explicit zero-run floor and factor-three stability checks. The native-BF16 pilot is preserved separately. First complete reduced-step row: FP1-step474/500 seed7, paired -1.4pp versus matchingFP10 (95%CI[-4.8,+1.2]); other configurations remain pending.

# Evening story round — 2026-10-03 in progress

X0 confirms both deployed FlowVQ experts used BF16-backbone activation moments with int8 embedding, not quantized-backbone activations. M3's MSE advantage remains unexplained. X1 fits plain layer-Hessian M2/M3 backbones with the identical uniform M2 expert; policy validation is pending. X2 reduced-step models use freshly captured FP32 AdaRMS tables and preserved backbone/embedding tensors. FP1-step exact-fold and heldout MSE0.009263133559734162 are verified; logical reads7,082,226,468 bytes. The one-observation X3 pilot uses native BF16 arithmetic; it exposed a small-perturbation rounding floor and is not a500-observation result. Current evidence: [story progress](results/story_progress_20261003.json). Main-method validation below is complete; older pending statements are historical snapshots.

# Current result — FlowVQ main-method end-to-end validation complete

Both quantized-backbone FlowVQ policies now have strict reload, paired held-out MSE, independent safetensors accounting, and full four-suite seed7 validation. The authoritative five-row result is [MAIN_METHOD_TABLE.md](MAIN_METHOD_TABLE.md); all paired intervals and source hashes are in [complete evidence](results/flowvq_main_complete.json). The requested table is complete; no deferred experiment is authorized or dispatched by this controller.

# 2026-10-03 main-method compositions

New standalone deployable checkpoints: `models/hd_srvq_bb_M3_flowvq_fp32` and `models/hd_srvq_bb_M2_flowvq_fp32`. Both use the exact accepted `evening_c_M2` expert: b* prefix masks [1,1,1,1,1,1,1,1,1,3], conditional-centroid codebooks and per-subset affine, with FP32 AdaRMS tables and int8 embedding. Independent whole-model bpw 3.6445933006309184 / 2.928218271423245. Strict reload and held-out relative MSE 0.007308567466560175 / 0.0429168180781574 verified. Full evaluation pending; no BF16-backbone c episode reuse. See [current phase](MAIN_METHOD_2026-10-03.md).

# E059 — Evening pipeline complete; final d result

All authorized evening branches completed successfully by2026-10-02 22:44:40Singapore; no active GPU/CPU jobs. M3-backbone/M2-expert/FP32-table/int8 full remains1948/2000=97.4% at3.638965wholebpw. No M2-backbone full evaluation was launched, in accordance with the frozen selection rule.

Final d expert isolation medium is966/1000=96.6%, versusM1941/M2968/b*963/c962 andFP970. d stores81,913,068expertbytes/2.104199bpw and reads449,742,060logicalbytes per inference (1.121323×M1), hence fails the1.1×read ceiling despite+2.5pp point-estimate recovery. d relativeactionMSE0.004484082144878243 is61.55%higher than c; its+0.4pp medium increment overc does not meet the1.5pp alternative component gate. b* alone meets the combined target by point estimate, with recoveryCI crossing0. Full statistics and229source hashes: [complete ablations](results/evening_ablations_medium.json). No model/runtime tensors changed during this audit.

# E058 — Quantized-backbone full validation complete

M3 backbone + expert M2 + FP32 AdaRMS tables + int8 completes full seed7:1948/2000=97.4%, matchingFP1928/2000=96.4%; paired difference+1.0pp,95%CI[-0.35,+2.65]pp. Actual whole-model storage remains3.638965bpw/1,525,378,532tensorbytes. Meets the operational≤1.5pp loss gate, without a superiority/equivalence claim. M2-backbone98.2% remains medium-only. [Full result](results/evening_deployment_full.json).

In BF16-backbone medium isolation, b*=963/1000 versusM1=941/M2=968; +2.2pp recovery at1.098345×M1 logical reads meets the target by point estimate only (pairedrecoveryCI[-0.90,+5.30]pp). b* stores79,553,772bytes, exactly uniformM2 storage. c=962/1000 but449,742,060readbytes/1.121323×M1 fails the1.1×budget; stored81,913,068bytes includes2,359,296extra bytes. No measuredDRAM/speed claim. d pending. [Comparison frames](results/evening_bc_comparison_frames.json).

P4 allthree offline validations complete, no rollouts: actualexpertbpw0.543593/0.517906/1.070108 and relativeactionMSE0.07653418/0.11357073/0.00741770 forgroup16-K256-M1/group8-K16-M1/group16-K256-M2 respectively. [P4 offline](results/evening_p4_offline.json). Older pending sections below are historical.

# E057 — Scheduling update; checkpoint quality results unchanged

User-requested independent execution now retains ablation job180151 and adds four single-L40S workers180284–180287 under controller180278 (up to6GPUs, ceiling8). P4 exports are complete; two offline workers are allocated and the third is waiting for a GPU. P4 remains rollout-free, with no completed action-MSE result yet. Strict selection chose M3-backbone/M2-expert/FP32-table/int8 by the existing minimum-held-out-MSE rule; the2000-episode full protocol reuses1000validated medium episodes. No new full/P4 quality result is claimed. [Promotion](results/evening_promotion.json); [schedule snapshot](results/evening_parallel_schedule_20261002.json).

# E056 — Both M2-expert backbone compositions complete medium

Both compositions retain P3 FP32 tables and int8 embeddings. Medium is four suites × ten tasks × 25 episodes, seed 7; matched FP is 970/1000 (97.0%). M3 backbone: 974/1000 (97.4%), +0.4pp, paired 95% CI [-1.50,+2.30]pp. M2 backbone: 982/1000 (98.2%), +1.2pp, paired 95% CI [-1.00,+3.80]pp. Both pass the operational FP-minus-1.5pp point-estimate gate; neither has completed full validation.

| Backbone + expert M2 | Tensor payload bytes | Whole-model bpw | Held-out relative action MSE | Logical compressed reads per inference (bytes) |
|---|---:|---:|---:|---:|
| M3 + FP32 tables + int8 | 1,525,378,532 | 3.638965 | 0.01304602 | 2,048,853,828 |
| M2 + FP32 tables + int8 | 1,225,088,996 | 2.922590 | 0.03721696 | 1,644,469,572 |

Expert actual storage remains 2.043593 bpw; expert logical reads are 795,526,380 bytes/inference in both. Logical reads are representation accounting, not measured DRAM traffic or latency. Source: [complete medium comparison](results/evening_deployments_medium.json) and [checkpoint accounting](results/evening_checkpoint_accounting.json). No tensor layout or runtime source changed in this report update. Older pending statements below are historical snapshots.

# E055 — M3-backbone/M2-expert medium complete

The int8/FP32-table composition at3.6389649302wholebpw completes seed7 medium974/1000 vsmatchingFP970/1000 (+0.4pp; paired95%CI[-1.50,+2.30]pp). This is a medium pass, not full validation. The2.9225899010bpw M2-backbone candidate is still running; no final comparison/promotion yet. See `results/evening_deployment_M3_medium.json`.

# E054 — M2 expert default confirmed

The complete single-seed 2000-episode protocol gives M2 1923 successes versus FP1928, a0.25pp loss (paired95%CI[-1.40,+0.75]pp), passing the≤1.5pp gate. Current backboneM3/M2 +expertM2 +FP32table +int8 combinations have validated held-out actionMSE0.01304601522565916/0.037216964264135996; their medium/full accuracy is not yet established. Group64int4 embedding passes its standalone calibrationMSE gate (8.246323803397018e-5) and completes200-episode screening; combined deployments remainint8. See `results/p1_full.json`, `results/embedding_group64.json`, `results/evening_progress_20261002.json`.

# Evening deployment update — E053

## Evening composition storage audit — 2026-10-02

Independent safetensors-header validation confirms backbone M3/M2 + expert M2 + FP32 AdaRMS tables + int8 embedding payloads of1,525,378,532/1,225,088,996bytes. Actual all-Linear bpw2.908467/2.020137; whole-model3.638965/2.922590. Both have542,205,320reserved bytes, including543,456native backbone bias bytes; biases remain excluded from quantizable-Linear numerators. ExpertM2 actual2.043593bpw. These are assembled storage measurements; combined-policy accuracy is still pending. [Accounting](results/evening_checkpoint_accounting.json).


Deployment now composes backbone M3/M2 with expert M2, int8 embedding and validated FP32 AdaRMS tables. The denominator remains the original active weights; all tables, scales and masks count. Expert-M2 default is conditional on full validation. The group64 packed-int4 candidate costs 4.25 embedding bpw including FP16 scales; it is not the failed per-row int4 candidate. Current c/d affine read overhead exceeds the requested 1.1× M1 expert budget; see `results/evening_preflight.json`. New assembled payloads are independently header-validated: M3-backbone/M2-expert 1,525,378,532 bytes (3.6389649302 whole-model bpw); M2-backbone/M2-expert 1,225,088,996 bytes (2.9225899010 bpw). Quality is still pending. See `results/evening_deployment_exports.json`.

# π₀.₅ LIBERO model map

## Current deployment accounting — 2026-10-01

Measured hook/action evidence: E011. All new experiments use one seed, overriding the earlier protocol. Both language-output heads are removed; the language embedding remains stored once. Expert head calls = 0, VLM head calls = 0, and all 256 paired single-seed action chunks remain identical at the recorded precision after removal.

| Item | Current value |
|---|---:|
| Active original unique weight elements | 3,353,433,872 |
| Eligible Linear weight elements (all components) | 2,704,306,176 |
| Eligible expert attention/MLP weight elements | 311,427,072 |
| Native-dtype payload after removing heads | 6,946,308,960 bytes |
| Payload after adding row-int8 language embedding | 6,420,690,272 bytes |
| Int8 embedding codes + FP32 row scales | 527,675,904 bytes |
| Whole-model bpw with int8 embedding (before VQ) | 15.317291 |
| Logical tensor reads / action chunk, first frozen observation | 17,434,020,060 bytes |

Storage includes every persistent tensor; container/header bytes are separate. Logical reads count three vision calls, one language pass and ten expert calls, with distinct lookup rows per embedding invocation. They are not DRAM counters and exclude KV/activation/cache effects. Original native FP32 normalization/conditioning and action/time tensors are preserved.

Budget implication, analytically calculated from the measured tensor inventory: retained tensors excluding eligible Linear weights use 1,012,077,920 bytes, a 2.414428 whole-model-bpw floor at the current native dtypes. If every eligible Linear retains at least one 8-bit index per 8 weights, even an ideal one-plane representation needs at least 3.220857 whole-model bpw before codebook/scales/mask overhead. Thus the later 3.0/2.5 whole-model targets need a separately validated retained-tensor or allocation change; they do not follow automatically from int8 embeddings and head removal. This is a budget bound under those explicit assumptions, not a measured quantized-model result. The 2026-10-01 phase now starts backbone quantization concurrently with full expert validation and authorizes fixed-timestep AdaRMS folding plus validated int4 embeddings; independent P3 deployment floors and action validation are now measured below; these changes are not yet composed into P2 artifacts.

Expert M1 full-protocol quality update:1869/2000=93.45% versusoriginalFP1928/2000=96.40%, paired difference-2.95pp (95%CI[-6.10,+.30]pp). Expertactual1.030335bpw does not establish lossless performance; nativebackbone/int8 whole-model13.927086bpw. [Full quality result](results/p1_M1_full.json).

## Exported backbone RVQ artifacts — 2026-10-01 (M3 screening passed; full protocol pending)

All288backboneLinear layers fitted at M3 and M2, with the validated uniformM1 expert fixed. The two self-contained local artifacts are `models/hd_srvq_bb_M3_ae_M1` and `models/hd_srvq_bb_M2_ae_M1`. Independent safetensors shape/dtype/offset/file-size accounting agrees with export metadata. Strictpolicy reload and256held-out action checks nowcomplete; originalFP relativeMSE M3=.0170782926020,M2=.0430040726108. M3 screening now190/200=95.0%, exactlyFP-1.5pp; paired95%CI[-8,+5]pp. M2screen and backbonefull pending. [Complete M3 screening](results/backbone_M3_screen.json).

| Backbone depth | Backbone Linear bpw | All Linear bpw | Whole-model bpw | All stored tensor bytes |
|---|---:|---:|---:|---:|
| M3 | 3.021028 | 2.791781 | 4.665800 | 1,955,806,652 |
| M2 | 2.017084 | 1.903451 | 3.949425 | 1,655,517,116 |

Both retain1,012,077,920bytes outside eligibleLinear representations, including native conditioning and int8embedding. P3 is independently validated below; these P2 artifacts still use the original retained layout. Expertpayload40,109,292bytes (1.030335bpw) unchanged. Denominator3,353,433,872originalactiveelements unchanged. Dense quality-runtime caches excluded fromcheckpoint bytes and tobe reported separately; compresseddecoder logicalreads M3=6,394,054,428 andM2=5,989,670,172bytes/inference; currentdensefallback reads17,437,380,732bytes plus5,408,612,352bytes densecache. No measuredDRAM/kernelclaim. [Full storage aggregate](results/backbone_checkpoint_accounting.json).

## P3 measured reserved-tensor deployment — 2026-10-01

On all256calibrationobservations, fixed-ten-step FP32 tables preserve native full10x7 actions bit-identically (maxabs0). FP16 tables fail the1e-6 gate (maxabs.0813788513). Selected FP32 tables+schedule occupy4,546,600bytes, replacing474,419,200bytes of39matrices+theirbiases. Original active-element denominator remains3,353,433,872.

| Embedding with exact FP32 tables | Reserved bytes | Reserved-only whole-model bpw floor | Offline relative action MSE | Decision |
|---|---:|---:|---:|---|
| int8 row absmax | 542,205,320 | 1.293493 | 1.99510e-5 | Keep int8 |
| int4, FP16 row scale | 278,367,368 | 0.664077 | 0.03763057 | Reject: offline gate fails |

Both artifacts strictly reload and repeat native/canonical actions exactly. Int4 paired200episode screening is complete:190/200=95.0%, -1.5pp versusFP, paired95%CI[-7.0,+3.5]pp. The screen gate passes, but does not reverse the failed offline prerequisite. [Complete paired validation](results/reserved_validation.json). These are native-Linear P3 artifacts, not composed backbone RVQ results. Quantized-Linear bytes must be added to the listed floors; .664077 is a rejected candidate floor. [Complete measured aggregate](results/reserved_offline.json).

## Historical constructor audit (E002; not current deployment denominator)

The following original inventory includes the now-removed expert head. Its old arithmetic tables describe the constructor, and are superseded by the current accounting above for all new results.


Pinned openpi: `215abfb217dbac7d5f1273282331b9b1866c0479`. Evidence: E002 in EXPERIMENTS.md.

## Source configuration

- Active LIBERO action horizon: 10; model action dimension: 32. Generic Pi0Config defaults to horizon 50; the LIBERO override is authoritative.
- Flow: 10 Euler steps, dt = −0.1, t starts at 1 and decreases toward 0.
- LIBERO outputs slice the first 7 dimensions; client replan K = 5. State input has 8 dimensions.
- Prefix image/language KV is built once before the denoising loop; expert uses the cache with use_cache=False.
- Two real views plus one masked padding view pass through the image path. Source-default discrete_state_input=False for this LIBERO config.
- Sampling is decorated with torch.no_grad(): D1 needs a separately validated differentiable unroll.
- Native inference preserves selected FP32 parameters; FP16 is a method storage choice, not the native baseline dtype.

## Unique instantiated parameters (not checkpoint verification yet)

| Component | Parameters | Logical bytes from constructor dtype (not measured allocation) |
|---|---:|---:|
| siglip | 412,442,352 | 826,831,584 |
| vision_projector | 2,361,344 | 4,722,688 |
| language_embeddings | 526,647,296 | 1,053,294,592 |
| language_transformer | 1,981,884,416 | 3,963,920,384 |
| expert_attention_mlp | 311,427,072 | 622,854,144 |
| expert_adarms | 116,505,600 | 466,022,400 |
| unused_expert_lm_head | 263,323,648 | 526,647,296 |
| action_io | 66,592 | 266,368 |
| time_mlp | 2,099,200 | 8,396,800 |
| TOTAL | 3,616,757,520 | 7,472,956,256 |

The expert `lm_head` is constructed but never called by action sampling. Count it as stored overhead until conversion/runtime handling is verified; exclude it from active bytes-read. Tied language embeddings are counted once.

## Analytic bytes-read model (NOT a timing or measured memory-traffic result)

Quantize eligible Linear weights only; retained tensors use an idealized 16-bit representation. Each backbone weight is charged once, each active expert weight ten times. Excludes codebooks, codes metadata, scales, KV/activations, multi-view rereads and hardware cache effects; native BF16/FP32 runtime differs. Full storage accounting must include these overheads before any bpw claim.

| Eligible bits | Backbone once (bytes) | Expert ×10 (bytes) | Expert share |
|---|---:|---:|---:|
| 16 | 5,846,670,816 | 8,601,969,280 | 59.5348% |
| 3 | 1,958,242,272 | 3,541,279,360 | 64.3925% |
| 2 | 1,659,132,384 | 3,151,995,520 | 65.5147% |

## Ideal storage including all instantiated parameters

These are arithmetic lower bounds, excluding codebook/scale metadata. All retained tensors, including the unused expert head, are charged at 16 bits. Their minimum alone can make a whole-model target infeasible. Remove unused tensors and/or change embedding storage only after validating numerical equivalence.

| Eligible bits | Whole-model bpw | Retained share | 16-bit retained floor (bpw) |
|---|---:|---:|---:|
| 16 | 16.000000 | 25.2284% | 4.036550 |
| 3 | 6.279697 | 25.2284% | 4.036550 |
| 2 | 5.531981 | 25.2284% | 4.036550 |

## Every expert Linear

| Name | in → out | Bias | Immediate input / conditioning |
|---|---|---|---|
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.0.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.1.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.1.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.1.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.2.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.2.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.2.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.3.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.3.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.3.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.4.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.4.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.4.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.5.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.5.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.5.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.6.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.6.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.6.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.7.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.7.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.7.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.8.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.8.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.8.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.9.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.9.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.9.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.10.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.10.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.10.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.11.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.11.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.11.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.12.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.12.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.12.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.13.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.13.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.13.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.14.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.14.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.14.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.15.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.15.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.15.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.16.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.16.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.16.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.17.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.17.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.17.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.norm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.lm_head` | 1024 → 257152 | False | unused in action inference |

## Pinned evidence paths

- `src/openpi/training/config.py`: pi05_libero override.
- `src/openpi/models_pytorch/pi0_pytorch.py`: action I/O, timestep MLP, sampling loop.
- `src/openpi/models_pytorch/gemma_pytorch.py`: model construction and prefix/expert routing.
- `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`: AdaRMS and gated residual.
- `src/openpi/policies/libero_policy.py`, `examples/libero/main.py`: action slicing and replan window.


## Reserved tensor audit after D3 — E027

Paper primary rate is quantizable-Linear bpw; whole-model rate is secondary. The complete41-tensor >1Melement inventory is [large_reserved_tensors.csv](results/reserved_floor/large_reserved_tensors.csv), including all37protected FP32 AdaRMS matrices andboth protected FP32 timeMLP weights. [Analytical floors](results/reserved_floor/summary.json) count packed embedding/scales and preserve norms,biases,timeMLP,AdaRMS. Int4embedding and reservedint8 are accounting scenarios, not validated deployments.

E032 uniform VQ deployment artifacts:126expertLinears /311427072originalweights, all other tensors retained from validatedint8embeddingcheckpoint. ActualexpertbpwM3/M2/M1=3.0568505489/2.0435929732/1.0303353974; allLinear14.5094708684/14.3927844360/14.2760980035; whole14.1152850620/14.0211857441/13.9270864262. Includes books/codes/row_scale/tenmasks/RHTsigns; whole denominator3353433872. Completedassembly177819; policy-levelstrictreload/actionvalidation pending. Qualityevaluation reconstructs nonserializedBF16densecaches; report theirmemory and fallbacklogicalreads separately, without fusedkernel claims.

E034 allthreeuniformdeploymentsnowpassstrictpolicyreloadandretainedtensoridentity,256heldoutobservations/noise0. LogicalcompressedreadsM3/M2/M1=12,395,449,800/12,001,005,000/11,606,560,200bytes/inference atfixedaccountingobservation; qualityfallbackactuallyuses17,436,048,840logicalparameterbytes and622,854,144bytes densecache(allthree). These are logicaltensormodels, notDRAMmeasurements; no runtimecompressedkernelclaim.

E035/E036 adaptive v2 artifact format stores only deployed conditional subset books, FP16 per-channel affine tensors, codes/row_scale/RHT signs/ten masks. Metadata-inclusive minimum full(c)/(d) is41,030,892bytes =1.0540096398555872expertbpw; original M1 budget is infeasible for those full components. Strict real-layer storage/forward pilot passes; full adaptive policy deployment remains pending.

E038 allreference(a)screeningcomplete: M3/M2/M1=193/191/193 of200, allpass. No user-definedsuccessknee found. M1actualexpert1.030335/whole13.927086; fullmodel remains mostlynative precision. Adaptiveformat/implementationverified, but noadaptivepolicyexperiment yet.


2026-10-03 evening story round: two plain layer-Hessian M2/M3 backbone + uniformM2 expert checkpoints are being fit under models/story_plain_layers; deployment IDs story_plain_bb_M2_ae_M2_fp32 and story_plain_bb_M3_ae_M2_fp32 remain pending assembly. Reduced-step runtime reuses the original checkpoint with independently captured FP32 schedule tables in runs/story_round/tables/n{1,2,5};1-step exact native-action validation passed. Do not treat these as new completed policy results. X0 proves both existing FlowVQ experts were calibrated on BF16-backbone/int8-embedding activations.


Latest user instruction (2026-10-03): do NOT add matched same-GPU control reruns; finish the requested experiments first. This supersedes the immediately preceding approval to rerun controls. No such control job was submitted. Reuse accepted comparators and explicitly label cross-device comparisons; paired episode CIs do not remove hardware/backend confounding. Preserve the running X2 L40S job and its matching accepted references. New rollout jobs use the fastest measured eligible GPU, after that type completes its ten-minute episode-throughput check. Offline work may retain L40S to resume frozen evidence. At most eight GPUs total.

X1 assembled models: hd_srvq_story_plain_bb_M2_ae_M2_fp32 (SHA256 ed2266325ffbaaa0e110a8de8522a374edb600eec8d5d809561efa05c817240f) and M3 counterpart (84fc840cd00d428d4c754616fd06b5ec1dd91767a8f91d1205372add5b987259). Assembly is complete; held-out/rollout validation remains pending. X5 canceled; X6 LIBERO-10/Plus authorized; no extra same-GPU control checkpoints or rollouts.


X1 M3 plain-VQ held-out validation complete: strict tensor reload,256 observations/40 trajectories, fixed noise0, relative MSE0.010938292660258468 onL40S. Medium evaluation is still pending; no policy-quality promotion follows from MSE alone.

X1 plain-VQ medium validation complete, seed7: M2 backbone967/1000, M3 backbone972/1000, with held-out MSE0.03942254808560134 and0.010938292660258468. All nine reduced-step X2 configurations have500 accepted episodes each. Scientific completion audits preserve the two post-completion launcher failures; no checkpoint or inference runtime changed.


## October4 evaluation runtime audit

The accepted evening loader already prepares every expert subset at load time. FlowVQ has persistent BF16 depth1 (steps0–8) and depth2 (step9) reconstructions, with affine alpha included before BF16 rounding and beta added in FP32 before casting the output. The separate `flowvq_cached_runtime.py` evaluation path forbids reconstruction after loading and compiles the existing input Hadamard transform. It does not alter stored checkpoints, protected FP32 tensors or compressed-read accounting. Exact256-observation checks and a fresh L40S episode-throughput preflight gate new rollouts; no acceleration claim is established until that measurement completes. Dense-cache resident bytes are reported separately from serialized compressed bytes.

The final separate evaluation runtime additionally captures the exact ten-step expert operation sequence as a CUDA graph, retaining both weight variants, FP32 tables, static KV/noise buffers and the outside-capture Euler dt tensor. Both M2/M3 compositions passed256/256 direct raw-action comparisons with maximum difference0 on L40S (results/flowvq_runtime_validation.json, source28676f2). Episode throughput remains pending; no compressed-kernel latency claim follows from this dense evaluation optimization.


Evening structure round: H3b uses FPexpert with actual observation-specific quantized-prefix errors (not fulluniformM2expert). H5d retains the uniformM2backbone/expert checkpoint and adds per-layer/type calibration-mean cachedKV subtraction; extra correction bytes are reported separately. Expert-onlyuniformM2(a_M2) is the added fixedPlus comparator. New variants are evaluation-pending, not promoted methods.


## 2026-10-05 calibration-coverage variants (registered, not yet fitted)

`coverage_mixed_backbone`: refit288 backbone Linear layers at2bits using mixed256; retain all126 original-calibration uniformM2 expert layers. `coverage_mixed_both`: identical refitted backbone plus126 uniformM2 expert refits on the same mixed256. Both retain the original embedding/normalization/projection/FP32 table tensors and must match the original deployment payload exactly. Main set128original+64camera+64initial-state, original60 train episodes only. LIBERO-Plus fixed500+500 is primary for method comparisons; LIBERO full/medium remains the headline benchmark. See CALIBRATION_COVERAGE_2026-10-05.md.
