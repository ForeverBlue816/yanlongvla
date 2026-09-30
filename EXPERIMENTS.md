# Experiment ledger (append only)

Entries include UTC date, exact code revision, command, configuration, outcome and conclusion.
External numbers are labelled quoted. Byte accounting is analytic, never measured DRAM traffic.

## E000 — 2026-09-29 UTC — source audit and asset preparation

- Research snapshots: `aa2d8e8` (checkpoint downloader, first committed after launch), `a23436a` (calibration downloader).
- Pinned openpi: `215abfb217dbac7d5f1273282331b9b1866c0479`; LIBERO: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`; ActQuant: `b64791125070652fe6b554e244fe809c79ef5246`.
- Commands from local `research/`: `python3 scripts/download_checkpoint.py --manifest "$VLA_ROOT/models/pi05_libero_manifest.json" --dest "$VLA_ROOT/models/pi05_libero"`; `python3 scripts/download_calibration.py`.
- Measured checkpoint inventory: **16 objects, 12,439,085,481 bytes**, size verified, MD5 verified where supplied, SHA256 recorded locally. This is serialized JAX checkpoint size, not inference memory.
- Measured calibration inventory: **60 episodes, 1,205,270,267 parquet bytes**; dataset revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`; selection seed **20260929**. One training episode for each of **40 tasks**, plus an additional episode for **20** seeded-random tasks. Matches ActQuant's calibration episode count, not a claim of identical episode selection.
- Local artifacts: `models/pi05_libero/download_verified.json`, `datasets/libero-calibration/download_verified.json`.
- Source audit: read pinned `src/openpi/training/config.py`, `models/pi0_config.py`, `models_pytorch/pi0_pytorch.py`, `models_pytorch/gemma_pytorch.py`, `policies/libero_policy.py`, `examples/libero/main.py`, `examples/convert_jax_model_to_pytorch.py`.
- Findings: generic horizon **50** overridden to **10** in `pi05_libero`; model action dimensions **32**, effective dimensions **7**, replan **K=5**, state dimensions **8**. **10** Euler steps at `dt=-0.1`; prefix KV computed before the loop. `discrete_state_input=False`. `torch.no_grad()` decorates sampling; gradient diagnostics require a separate validated unroll.
- Conversion findings: official script ignores load_state_dict incompatibilities and checks `checkpoint_dir.parent/assets` rather than the release's `checkpoint_dir/assets`. Our local wrapper checks active keys and copies the actual assets; preserves FP32-sensitive tensors before native policy loading. Runtime validation pending.
- **Quoted** official success percentages: **98.8 / 98.2 / 98.0 / 92.4**, average **96.85** ([pinned source](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)). G0 planned: **50 episodes/task × 3 seeds** (**7,17,27**); gap threshold **1.5 percentage points**, checked in aggregate and per suite.
- ActQuant code is public ([official source](https://github.com/arashakb/ActQuant/tree/b64791125070652fe6b554e244fe809c79ef5246)); cloned but not run. Its π₀.₅ **3.0 backbone bpw** row is **5.6 whole-model bpw**, **94.8%** success, **2.4 GB**, all **quoted** from [the authors' project page](https://actquant.github.io/).
- Conclusion: assets ready; G0 pending. Use effective LIBERO configuration and count retained-weight overhead before choosing whole-model budgets.

## E001 — 2026-09-29 UTC — isolated environment installation

- Source snapshot: `0e78c96`, `scripts/env.sh` unchanged from `aa2d8e8`.
- Storage root: `/projects/yanlongvla`; model/data/env/cache/temp paths all redirected there.
- openpi: `UV_PROJECT_ENVIRONMENT="$VLA_ROOT/envs/openpi" UV_LINK_MODE=copy uv sync --frozen --python 3.11 --no-dev`, in pinned openpi checkout. Copy mode prevents transformers patches modifying shared cache files.
- LIBERO: `uv venv --python 3.8 "$VLA_ROOT/envs/libero"`; `uv pip sync --python "$VLA_ROOT/envs/libero/bin/python" "$VLA_ROOT/src/openpi/examples/libero/requirements.txt" "$VLA_ROOT/src/openpi/third_party/libero/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match`.
- Result at entry: installation still running; no inference result.

## E001b — 2026-09-29 UTC — installation completed and rendering verified

- Code revision: `18d827e` for simulator; `d9b3969` for setup finalization. Commands: `bash scripts/finish_setup.sh 4145099`; `sbatch slurm/simulator.sbatch`.
- Both isolated environments installed successfully. Applied the pinned upstream transformers replacements to the copied local transformers **4.53.2** package. Server uses Python **3.11** / torch **2.7.1**; client uses Python **3.8** / torch **1.11.0+cu113**. Full package snapshots are local under `runs/`.
- Fixed first LIBERO import attempt by adding the upstream-documented `third_party/libero` path to PYTHONPATH; editable install alone did not expose the package. Four suites enumerate **10 tasks each**.
- Measured simulator check: Slurm job **176286**, actual GPU **NVIDIA L40S**, GPU memory **49,140 MiB**, driver **595.71.05**. **10** dummy steps completed; both camera outputs **256×256×3**, finite robot position. Initialization plus stepping took **13.994649287982611 seconds** (not policy latency or a benchmark score).
- Result: headless EGL simulator check **passed**. Artifact: local `research/results/simulator_smoke.json`.
- Submitted jobs: conversion **176289**, parameter audit/BF16 kernels **176290**, dependent policy smoke **176291**, AQLM kernels **176292**. No complete G0 success evaluation submitted at this entry.
- Additional preparation: frozen **256** observation indices, each with **8** noise seeds (local `datasets/libero-calibration/observation_manifest.json`, script revision `c304385`). No activations collected before G0.
- Validation: **5** coverage tests passed (`python3 -m unittest discover -s tests -v`, source `657f759`); empty G0 run reports **0/6000**, status pending, no success estimate.
- Conclusion: simulator is operational; policy/task performance remains unmeasured until dependent jobs complete.

## E002 — 2026-09-29 UTC — model enumeration and analytic budget audit

- Code revision: `d9b3969`; pinned openpi as E000. Command: `sbatch slurm/audit.sbatch` → `python scripts/audit_model.py`, job **176290**.
- Result: passed; unique instantiated parameter total **3,616,757,520**. SigLIP **412,442,352**; vision projector **2,361,344**; embeddings **526,647,296**; language transformer **1,981,884,416**; expert attention/MLP **311,427,072**; expert AdaRMS **116,505,600**; unused expert lm_head **263,323,648**; action I/O **66,592**; timestep MLP **2,099,200**.
- Method: instantiate pinned model on PyTorch meta device; enumerate unique named_parameters and every expert Linear. This measures constructor inventory, not yet loaded checkpoint inventory. Full map: [MODEL_MAP.md](MODEL_MAP.md); raw inventory is local `research/results/model_audit.json`.
- **Analytic storage lower bound**, not measured device memory: retain all noneligible parameters at idealized **16 bits**, count the unused head as stored. Retained share **25.22843566244939%**; retained-only floor **4.036549705991902 bpw**. Eligible Linear **3 bpw** gives whole-model **6.2796966361184205 bpw**; eligible **2 bpw** gives **5.531980992742914 bpw**. Excludes codebooks/scales and native FP32 overhead, so actual storage can be higher.
- **Analytic one-read/ten-read model** (backbone once, active expert ten times, unused head excluded; not physical DRAM measurements): expert share **59.53480204951186% / 64.39249805645642% / 65.51468975454617%** at eligible **16 / 3 / 2** bits, respectively. Limitations and byte totals in MODEL_MAP.md.
- Conclusion: whole-model targets require explicit retained-weight accounting. Removing a provably unused head, embedding storage choices, and physically stored code planes must be evaluated; per-step decoding masks alone do not reduce stored code planes.

## E003 — 2026-09-29 UTC — BF16 kernel baseline

- Code snapshot: `d9b3969` (benchmark source unchanged since `a23436a`). Job **176290**, command: `python scripts/bench_bf16.py --output results/bf16_kernel.json`.
- Hardware/software: **NVIDIA L40S**, torch **2.7.1+cu126**. Batch **1**. **100** warmups, **20** timing groups, **100** forwards/group; synchronized CUDA events. Each number below is the median of group-average milliseconds.

| Tokens | 1024→4096 BF16 ms | 4096→1024 BF16 ms |
|---:|---:|---:|
| 1 | 0.00788351982831955 | 0.00920575976371765 |
| 10 | 0.007864319980144502 | 0.010762240290641784 |
| 64 | 0.009943040013313293 | 0.012328479886054993 |

- All outputs finite. Synthetic tensors, uncompiled torch.nn.functional.linear; clocks/co-tenancy not controlled. These are kernel microbenchmarks, not end-to-end policy latency. Raw groups local: `research/results/bf16_kernel.json`.
- VPTQ and W4A16 not measured yet. AQLM first attempt recorded below; no VQ speedup conclusion.

## E004 — 2026-09-29 UTC — failed preparation attempts (preserved)

- Code revision: `d9b3969`.
- Conversion: `sbatch slurm/convert.sbatch`, job **176289**. Requested **8 CPUs / 96 GiB**; cluster submission policy actually assigned **24 GiB**, confirmed with `sacct`. Job ended **OUT_OF_MEMORY**. No valid converted checkpoint or policy success result produced. Dependent smoke **176291** never ran.
- AQLM: `sbatch slurm/aqlm.sbatch`, job **176292**; upstream revision `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`. Failed before timing: `RuntimeError: Ninja is required to load C++ extensions`. Attempt artifact preserved locally as `research/results/aqlm_kernel_attempt1.json`. **No measured AQLM rows**.
- Repair: request enough CPU cores to receive required memory under cluster policy; install Ninja in the isolated environment and expose its binary. Retest rather than substitute a non-kernel implementation.

## E004b — 2026-09-29 UTC — preparation repaired

- Revision **182d8eb**. Retried conversion with `sbatch slurm/convert.sbatch`, job **176301**, now assigned **32 CPUs / 96 GiB** under cluster policy. Conversion **completed successfully**. Artifacts: `models/pi05_libero_pytorch/model.safetensors`, copied norm assets, and `research/results/conversion.json`.
- All active keys loaded; the only allowed missing tied/unused heads are recorded in the conversion artifact. Stored tensors use FP32 to preserve retained sensitive tensors before native BF16/FP32 policy loading. This validates conversion bookkeeping, not action equivalence or success rate.
- Replacement policy smoke job **176307** submitted with `afterok:176301` and awaiting/running GPU allocation. Complete G0 still not submitted at this entry.
- AQLM retry **176302** got past Ninja but failed before timing because `CUDA_HOME` was absent. Preserved as `results/aqlm_kernel_attempt2.json`; no measured VQ rows.
- Local build repair: **Ninja 1.13.2** installed; minimal **CUDA 12.6.3** redistributables downloaded from the official NVIDIA index, SHA256 checked, unpacked into project tools. Components: nvcc **12.6.85**, cudart/CCCL **12.6.77**. Command: `python3 scripts/install_cuda_toolkit.py`; manifest local `results/cuda_toolkit_install.json`. System CUDA unchanged.

## E005 — 2026-09-29 UTC — actual AQLM 2×8 kernel measurements

- Eager source revision **cd09a87**; job **176310**, `sbatch slurm/aqlm.sbatch`. CUDA Graph source revision **98091f1**; job **176318**, `sbatch slurm/aqlm_graph.sbatch`. Both completed successfully on **NVIDIA L40S**, torch **2.7.1+cu126**, official AQLM revision `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2`.
- Synthetic weights: **2** codebooks, **256** entries each, input group **8**, BF16 codebooks/activations, batch **1**. Index payload is **2 bpw**, excluding metadata. This is a kernel test, not quantized policy quality.
- Correctness: relative RMSE against dense reconstruction is **0.0033766678534448147 / 0.0034582382068037987** for the two **1-token** shapes, and **0** for the **10/64-token** dequant+GEMM path. Predeclared threshold **0.03** passed. Input/codebook seed **7**.
- Eager times use **50** warmups, **20** groups of **100** forwards. These include CUDA idle gaps from Python/dispatcher launch overhead and must not be described as isolated device kernel latency.
- CUDA Graph comparison uses **100** warmups, **40** operations captured per graph, **10** warmup replays, then **20** groups of **10** replays; elapsed CUDA event time divided by **400**. Dense BF16 reconstructs the same quantized weights. Warm cache, clocks not controlled, not end-to-end inference.

| Tokens | Shape | AQLM eager ms | BF16 graph ms | AQLM graph ms | Graph AQLM/BF16 |
|---:|---|---:|---:|---:|---:|
| 1 | 1024→4096 | 0.049479680061340334 | 0.0058316802978515625 | 0.007506879866123199 | 1.2872584714372621 |
| 1 | 4096→1024 | 0.04858367919921875 | 0.007989760041236877 | 0.0065356802940368655 | 0.8180070816025523 |
| 10 | 1024→4096 | 0.059653601646423346 | 0.00589688003063202 | 0.02354556083679199 | 3.9928844939157444 |
| 10 | 4096→1024 | 0.0642195200920105 | 0.006004119813442231 | 0.023868879079818724 | 3.9754168506731418 |
| 64 | 1024→4096 | 0.05849600076675415 | 0.009233840107917786 | 0.027811839580535888 | 3.0119472782172103 |
| 64 | 4096→1024 | 0.0621726393699646 | 0.010794920325279237 | 0.02905087947845459 | 2.691162009822719 |

- Artifacts local: `results/aqlm_kernel.json`, `results/aqlm_cuda_graph.json`, with all timing groups. No fallback path presented as a custom CUDA result: upstream dispatch selects `code2x8_matmat` for **1** token, `code2x8_matmat_dequant` for **10/64** tokens.
- Conclusion: this AQLM implementation is slower than BF16 at **64** tokens even with graph replay. A **1-token 4096→1024** case is faster. This supports no general latency claim for AQLM on the tested GPU/shapes; VPTQ and W4A16 remain unmeasured. Method memory/bytes-read savings also remain to be validated on a quantized policy.

## E006 — 2026-09-29 UTC — PyTorch policy and closed-loop smoke passed

- Source revision **b61df32** at submission (server/eval implementation unchanged from `0e78c96`); job **176307**, command `sbatch --dependency=afterok:176301 slurm/smoke.sbatch`; job completed with exit **0**.
- Synthetic policy result: finite **10×7** actions, fixed-noise repeat maximum absolute difference **0.0**. Initial inference with warmup **5.611400405992754 s**; torch peak allocated memory **7,644,988,416 bytes**, measured after model loading on **L40S**, compile mode **none**. These are a warmup observation and torch allocation peak, not steady-state latency or total GPU memory.
- True LIBERO smoke: spatial task **0**, seed **7**, episodes **0/1** both succeeded (**2/2**); **91/86** steps including settle, **17/16** policy calls, **6.3587295599863864 / 6.219471549993614 s**. No infrastructure errors. This small check is not the complete benchmark or G0 pass.
- Local artifacts: `results/inference_smoke.json`, `runs/smoke/spatial_seed7.jsonl`.

## E007 — 2026-09-29 UTC — full G0 submitted without reducing coverage

- An initial **12-task** array submission was rejected by `QOSMaxSubmitJobPerUserLimit`; no array/report job was created. Preserved in `results/g0_array_submission_failed.json`. Changed scheduling only, retaining the protocol.
- Successful source revision **35f03fe559e774a2ed1871d7825e23a00defa86c**, command `source scripts/env.sh; python3 scripts/launch_g0.py`.
- Worker jobs **176324 / 176325**, report job **176326** (`afterany` both workers). Each GPU worker reuses one policy server across **6** suite/seed combinations. Max concurrent GPUs **2**; each has a **24-hour** limit. Evaluation still **4 suites × 10 tasks × 50 episodes × 3 seeds = 6000 episodes**, seeds **7,17,27**.
- Per-episode JSONL outputs go to `runs/g0/`. G0 is **pending** until all planned results exist and meet the predefined tolerance. Exceptions abort their worker and are reported as infrastructure errors.
- The dependent report job validates coverage, computes seed-level confidence intervals, and appends only report text to the existing public repository. If the public checkout has local edits or publication fails, validated results remain locally; it does not publish code/data/logs.
- Result at this entry: jobs submitted; no complete G0 success estimate. No diagnostics/quantization stage has been started.

## G0 evaluation update — 2026-09-29T18:31:50.996876+00:00

Status: **passed**; observed **6000 / 6000** planned episodes.

Report generator code: `21c919bf482d6ed8d19c7e96b07ca85236008acb`. Commands: `python scripts/launch_g0.py` (two workers), then `python scripts/publish_g0_summary.py`. Exact evaluation code and worker job IDs are in the launch ledger entry.

Seeds: [7, 17, 27]; 50 episodes/task. two-sided 95% Student t across the three seed-level means; df=2; clipped to [0,100]. Describes seed variation on fixed tasks, not independent rollout uncertainty.

| Suite | Per-seed success % | Mean % | 95% CI % | Absolute gap vs quoted (pp) |
|---|---|---:|---|---:|
| libero_spatial | [98.2, 98.0, 98.6] | 98.266667 | [97.50775005991878, 99.02558327341455] | 0.533333 |
| libero_object | [98.8, 99.2, 98.2] | 98.733333 | [97.48301143741116, 99.9836552292555] | 0.533333 |
| libero_goal | [96.8, 98.8, 97.2] | 97.600000 | [94.97103575680975, 100] | 0.400000 |
| libero_10 | [91.8, 93.4, 93.4] | 92.866667 | [90.57191854404732, 95.16141478928603] | 0.466667 |
| overall | [96.4, 97.35, 96.85] | 96.866667 | [95.68615661219496, 98.04717672113836] | 0.016667 |

G0 passed the predeclared coverage and tolerance check. Diagnostics may begin; no quantization or method gain is claimed here.

## E008 — 2026-09-30 UTC — method-first phase and held-out split

- User phase override: external GPTQ/AWQ/ActQuant baselines and D4 deferred. Accounting -> D1/D2/D3 -> action-expert (a)–(d), then whole model. No additional kernel work before the method table; no KD before PTQ results. Local authority: `PHASE_2026-09-30.md`, decisions/status in `CONFIG.md`.
- G0 complete results remain the FP reference; no full G0 rerun submitted.
- Source `d293cff`: `python3 scripts/prepare_heldout.py`. Pinned dataset revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`; fixed seed **20260930** selected **40** distinct training episodes (one per task) outside all **60** calibration episodes, **256** held-out observations x **8** seeds. Downloaded and SHA256 recorded: **881,632,142 bytes**. Local manifest/checksums: `datasets/libero-heldout/download_verified.json`. Held-out observations are prohibited from diagnostic fitting, allocation, codebook and affine fitting.
- Prepared diagnostic implementations are not measured results. D2 aggregates median commutator over Linear-layer/step-pair units and median of each pair's maximum top-32 principal angle; this conservative angle aggregation was frozen before D2 observations.

## E009 — 2026-09-30 UTC — Step-1 paired pilot, not the full accounting gate

- Source `5cc764813449508f2219a82b1b5edc09231dcb08`; command `sbatch --export=ALL,VLA_OBSERVATIONS=16,VLA_SEEDS=2 slurm/accounting.sbatch`; job **177425**, completed successfully on **L40S**. Samples: first **16** frozen calibration observations x seeds **0,1**, **32** action chunks per configuration. Native BF16/retained-FP32 policy; real LeRobot observations use native training transforms.
- Hook calls: expert lm_head **0**, VLM lm_head **0**. VLM head shares the exact embedding weight storage. Expert head contains **263,323,648** weights. Unique original elements **3,616,757,520**; after head removal **3,353,433,872**. Removed both module heads, retained the single tied embedding storage.
- Native vs heads removed: action max absolute difference **0.0**, executed-window/full-action MSE **0.0** over all **32** pairs. Expert first-layer hooks observed exactly **10** suffix tokens on every call.
- Embedding implementation: signed int8 round(absmax-row-scaled weights), range **[-127,127]**, FP32 per-row scale, BF16 gathered output; only requested rows dequantized. Compared with the same original native actions and same fixed noise.
- Executed **5x7** actions: MSE **1.7387334377437334e-05**, reference mean-square **0.17994315104630054**, relative MSE **9.662681950569744e-05**. Full **10x7** relative MSE **5.745782083636663e-05**. Largest absolute action difference **0.12228983640670776**; maximum per-pair relative MSE **0.003501288320115805**. Aggregate barely meets **1e-4** on the pilot; tail errors are explicitly retained, not averaged away from the report.
- Local evidence: `runs/accounting/177425/result.json`, per-configuration actions/flow states and module-read ledgers. These are a pilot, not proof over the full calibration manifest or a LIBERO success result. No bpw claim is promoted from this pilot.
- Full validation submitted as job **177438**, source **d293cff**, command `sbatch --export=ALL,VLA_OBSERVATIONS=256,VLA_SEEDS=8,VLA_EXPORT_FLAG=--export slurm/accounting.sbatch`. Status at entry: **pending**. Strict head removal equivalence and full-coverage embedding relative MSE gate must pass before the compact model is the default.

## E010 — 2026-09-30 UTC — single-seed resource override and diagnostic validation

- User explicitly changed every subsequent experiment to **one seed** and allowed GPU choice by throughput/memory rather than cheapest type. Source override commits `d57b279` / `62c3efb`; current authority is CONFIG.md. Calibration/offline noise seed **0**; LIBERO seed **7**. Promoted full evaluation becomes **4 suites x 50 episodes/task x 1 seed**. Completed historical G0 results are retained and not rerun.
- Canceled obsolete eight-seed job **177438** and dependent verifier **177448** using `scancel 177438 177448`. The head-removal substage completed before cancellation: both heads unused, **2048** paired chunks, all physical action differences exactly **0**. Its embedding substage was interrupted and has no completed embedding result. Preserved raw artifacts; its exported head-only model is archived as `models/pi05_libero_trimmed_accounting177438`. This partial job is not a complete Step-1 pass.
- Replacement single-seed accounting job **177454**, command `sbatch --export=ALL,VLA_OBSERVATIONS=256,VLA_SEEDS=1,VLA_EXPORT_FLAG=--export slurm/accounting.sbatch`; dependent independent checkpoint reload job **177460**. No result claimed at this entry.
- D3 MMD changes to marginal action distributions across the **256** observation samples at one noise seed. It no longer estimates a conditional distribution over many noise samples for each observation. Bandwidth uses the reference sample alone; paired U-statistic excludes matched-observation cross terms and preserves negative finite-sample estimates. Frames may be episode-correlated; no iid-frame or across-seed interval is claimed.
- Passed numerical/behavioral checks: **5** compact-embedding/single-step-noise tests (roundtrip, error bound, alias storage, target-step-only effect, exact restoration including exception); **1** D1 test comparing every call's local VJP with independently parameterized ten-step autograd and verifying shared-gradient sum; **7** offline-metric tests including identity, distribution shift, executed-window projection, paired seed permutation, reference-only bandwidth, single-seed shape and rejection of accidental multi-seed input. Commands: `python -m unittest discover -s tests -p test_compact_and_noise.py -v`, similarly `test_d1_gradients.py` and `test_offline_metrics.py`. These validate implementation behavior, not the research hypotheses or policy success.

## E011 — 2026-09-30 UTC — full single-seed accounting passed

- Source `d57b279599eb5cd07b214eecdbf409decc7fae23`, job **177454**, exact command in E010. Completed **2026-09-30T04:29:08.717635+00:00** on **L40S**. All **256** frozen observations, noise seed **0** only; **256** paired action chunks in each of native / heads-removed / int8-embedding configurations.
- Expert and VLM lm_head forward calls: **0 / 0**. Tied VLM embedding storage confirmed by pointer identity. Expert head removed **263,323,648** weights; active denominator is **3,353,433,872** original unique weight elements. Removed both head modules, retaining one embedding table. All **256** native/head-removed action pairs are bitwise equal, maximum absolute difference **0.0**. Across all three phases **7,680** first-expert-layer calls had exactly **10** suffix tokens.
- Int8 embedding: executed-window MSE **4.321087437859752e-06**; reference mean square **0.21658506895616192**; relative MSE **1.995099412293451e-05**, below **1e-4**. Full 10x7 relative MSE **1.5887933175410502e-05**. Per-observation relative MSE median / p95 / max: **9.208509597967124e-06 / 2.5045683660898328e-05 / 0.003501288320115805**. Maximum absolute physical action difference **0.12228983640670776**. The aggregate criterion passed; tail deviations are not zero. No LIBERO success claim for int8 embedding yet.
- Quantizable Linear and action-expert quantizable Linear bpw both **16.0** in these accounting references. Whole-model tensor payload / bpw: heads removed **6,946,308,960 bytes / 16.571214403240212**; plus int8 embedding **6,420,690,272 bytes / 15.317290913318478**. Signed-int8 embedding codes plus FP32 scales total **527,675,904 bytes**. Container sizes are separately **6,946,443,712** and **6,420,825,168 bytes**; header/container bytes do not masquerade as tensors.
- Logical reads for the first frozen observation: heads removed **17,434,050,720 bytes/inference**; int8 embedding **17,434,020,060**. The language lookup accesses **15** distinct rows (**200** token positions), vision weights run **3** times including the masked view, language transformer once, expert/time/action modules **10** times. No tensor-owning module is unobserved after head removal. This is an instrumented logical-read model, not measured DRAM traffic; excludes KV/activation traffic and cache effects, and lookup count depends on the prompt.
- Artifacts: `runs/accounting/177454/result.json`, actions/flow-state npz and module-read JSON files; summarized in `research/results/accounting_status.json`. Exported native-inference-dtype models: `models/pi05_libero_trimmed` and `models/pi05_libero_int8embed`, with all norms/time/action precision preserved. Independent strict-loader roundtrip job **177460** is pending at this entry.

### E011 budget implication (analytic, not a quantized result)

From the measured int8-embedding tensor payload and **2,704,306,176** eligible Linear weight elements, retained native-dtype tensors occupy **1,012,077,920 bytes**, or **2.414427619284213** whole-model bpw. Assuming at least one **8-bit** index per **8** weights in every eligible Linear, the ideal one-plane total is **3.2208565751613545** whole-model bpw before VQ codebooks/scales/masks. This explicit-assumption lower bound means the later **3.0/2.5** targets require an additional validated retention/allocation change. It does not block the requested action-expert-only method table, whose backbone remains BF16 with native FP32 exceptions.

## E012 — 2026-09-30 UTC — compact checkpoint verified; single-seed diagnostics launched

- First independent verification job **177460** failed an overly strict comparison of native **float64** unnormalized output to the saved **float32** action array: apparent difference **4.582869994607108e-08**. We did not relax the action tolerance. Instead, comparison now uses the explicitly recorded FP32 reference precision, and additionally checks every loaded tensor against the serialized tensor for exact values AND dtype.
- Retried with source **3905806**, `sbatch --export=ALL,VLA_ACCOUNTING_RESULT=/projects/yanlongvla/runs/accounting/177454/result.json slurm/verify_compact.sbatch`, job **177463**: **passed**. Both trimmed native and int8-embedding checkpoints load strictly; every persistent tensor matches exactly. Each has **8** deterministic real-observation checks, noise seed **0** only, all max differences **0** at the recorded precision. Maximum float64-vs-stored-FP32 rounding differences are **5.7386577045903664e-08 / 5.6587934338381274e-08**.
- Native trimmed checkpoint SHA256: `176031b21186751e49846ca4661c88a881695f290f8332f2bc185752b7b6c5d1`; int8 embedding checkpoint: `ffddb8a0550cedfc168edd7b152329bce0a424cf787eec40da21270d9ae411bf`. Local evidence: `runs/accounting/roundtrip-177463.json`, `research/results/compact_roundtrip.json`. The int8-embedding compact model is now the validated default.
- D1 job **177465**, D2 input-cache job **177466**, both single seed and gated on the passed roundtrip. D1 sampler's differentiable forward matches official physical actions exactly at its first real-observation check; diagnostic result pending. D2 cache payload for 256 observations x one seed x all ten flow steps/tokens is **7,549,747,200 bytes**, using exact BF16 bit representation.
- Cache job 177466 exposed excessive per-observation disk synchronization overhead. Canceled it and its pending downstream screen 177478, retained durable completed observations, and resumed as D2 job **177481** (source **c865b86**), same cache path and validated metadata, batch flushing every 16 observations. Shared q/k/v and gate/up inputs now verify identical storage/view rather than launch redundant device comparisons. Neither sample content nor diagnostic formula changes.
- D3 offline grid job **177477** is queued after D1 releases its GPU. D3 success screening is queued after the resumed D2 job. All use one experiment seed; no external baseline, KD or additional kernel experiment launched.

## E013 — 2026-09-30 UTC — D1 kills coherent score as a contribution

- Job **177465**, source **3905806**; `sbatch --dependency=afterok:177463 --export=ALL,VLA_DIAGNOSTIC=D1,VLA_ACCOUNTING_RESULT=/projects/yanlongvla/runs/accounting/177454/result.json,VLA_ROUNDTRIP_RESULT=/projects/yanlongvla/runs/accounting/roundtrip-177463.json slurm/diagnostic.sbatch`. Complete **256** observations x one noise seed **0**, one fixed-stream Rademacher probe per observation on physical executed **5x7** actions. Every quantizable expert Linear contributes all ten separate flow-step VJPs. Differentiable sampler versus official policy action maximum difference **0.0**.
- **38,928,384** eight-weight groups. Coherent score is mean-probe squared norm of the summed step gradients; incoherent score is mean-probe sum of squared step gradients. Spearman **0.9919422777792508**; top-10% overlap **0.9202330227373904**, top-k **3892839**. Rho min / p05 / median / p95 / max = **[0.02290419456350401, 1.059770317851392, 1.6836658942469627, 2.943397631055688, 9.219170005615961]**; no zero incoherent groups.
- Both predeclared kill thresholds hold (**Spearman >0.95**, overlap **>0.90**). **Decision: use incoherent scores and do not claim coherent scoring as a contribution.** Recorded in CONFIG.md before any Step-3 quantizer fitting. This is a ranking diagnostic, not a quantized LIBERO success result.
- Local evidence: `runs/d1/177465/result.json` and per-layer score npz files; report summary `research/results/d1_status.json`. Elapsed diagnostic compute/analysis **188.0472358830002 s**, excluding initial policy/environment loading; L40S. D2/D3 decisions remain pending.

### E012 follow-up — D3 initial validation stopped before noise experiments

D3 job **177477** stopped at its first clean sampler check for the same float64-output versus FP32-record comparison issue identified in the earlier checkpoint verifier. No weight-noise result or success result was produced. The corrected check repeats the cached sampler and compares native-precision output directly to official native-precision output, then separately verifies the saved FP32 representation. No numerical tolerance is relaxed; rerun required.

## E014 — 2026-09-30 UTC — D2 retains subset decoding; PCA rank audit

- D2 resumed job **177481** completed. All **256** observations x one noise seed **0**, all **10** tokens x **10** flow steps, **72** unique input tensors shared by **126** expert attention/MLP Linear modules. The input cache stores exact BF16 bits, **7,549,747,200 bytes**; all alias views verified. Source **c865b86**. Local cache `runs/expert-inputs/177466/complete.json`; covariance and full per-layer/pair evidence `runs/d2/177481/`.
- Uncentered C_s=E[x_s x_s^T], **2560** token samples per step. FP32 accumulation with TF32 disabled. Median normalized commutator kappa over **5670** Linear-layer/step-pair units: **0.23349341750144958**. This exceeds the **0.05** kill threshold.
- Single-seed inputs can be rank deficient: first-layer q/k/v covariance has numerical ranks **10–17** across flow steps. Audited all covariance spectra using threshold `input_dimension * float32_epsilon * largest_eigenvalue`. **627 / 5670** pair units do not support two identifiable top-32 spaces; their angles are excluded. Median pair-maximum top-32 principal angle among the remaining **5043** valid pair units is **81.04753875732422 degrees**, exceeding **10 degrees**. The unaudited all-pair value **82.54818725585938 degrees** is superseded as the primary angle statistic.
- Rank audit uses existing moments and the already allocated GPU, no new observations or seeds: `srun --jobid=177484 --overlap --ntasks=1 --cpus-per-task=1 .../python -u scripts/audit_d2_rank.py --result .../runs/d2/177481/result.json`. Raw covariance commutators remain defined for rank-deficient inputs; kappa independently prevents the subset-off kill, so excluding null-space angles does not drive the decision.
- **Decision: subset decoding ON; retain variant (d) for actual comparison.** Recorded in CONFIG.md. This supports testing step heterogeneity, not a success-rate or quantization-gain claim. D2 covariance/statistics stage elapsed **55.72510534799994 s**, excluding prior input capture.
- D3 offline repair job **177484** passed exact native-precision cached-sampler comparison and is producing the noise grid. Screening now uses disjoint single-seed worker jobs **177486 / 177487**, respectively clean+even steps and odd steps; obsolete unsharded pending job **177482** canceled. No duplicated conditions, references or experiment seeds; at most two allocated GPUs.

## E015 — 2026-09-30 UTC — complete D3 offline initial grid; informative-amplitude amendment

- Offline job **177484** completed 256 observations x noise seed0 x ten individual flow-step perturbations at relative weight norms **0.01,0.03,0.1** (30 conditions). Cached sampler matches official outputs at native float64 precision. All arrays remain local in `runs/d3/offline-177484`. Independent summary recomputes MSE and MMD from every saved action array and validates the recorded numbers.
- At norm **0.1**, executed-window relative MSE ranges from **7.376259206710607e-7** to **2.4720643562405932e-5** across steps; all are below the preregistered MSE tolerance threshold **1e-3**. MMD is likewise below **0.01** at all measured norms/steps. Thus every initial tolerance is right-censored at amplitude0.1; no allocation weights or final proxy are claimed.
- Before using any perturbed LIBERO success outcomes, amend noise severity using only this completed offline evidence: stop the weak **0.03** screen, extend offline to **0.3/1.0**, then **3.0** if needed; choose the smallest measured norm whose maximum step relative MSE reaches **1e-3** for the ten-condition screening comparison. The previous 0.03/0.1 schedule would consume rollouts at action deviations smaller than the already accepted embedding threshold. This resource-saving design amendment is explicit, not a claimed completed 0.03 screening or a hidden metric/gate change.
- Canceled live screen jobs **177486/177487** and preserve their partial rows. Continue the same clean reference seed7 episodes without repeating completed episode IDs. Noise cases will use the unchanged two-suite x 10eps/task x one-seed protocol; insufficient success spread requires the next measured larger amplitude.
- D3 summarizer tests: **10 passed**, covering exact/partial coverage, duplicates, seed/condition mismatch, boolean results, no tolerance extrapolation, ceiling-limited rejection, correlation direction, no-positive-association reporting and paired bootstrap identity. Implementation: scripts/summarize_d3.py; tests/test_d3_decisions.py. Bootstrap uncertainty conditions on this one evaluation seed and adds no new rollouts.


## E016 — 2026-09-30 UTC — D3 informative amplitude measured; single-seed screen queued

- Offline source **da77d21**, aggregation/figure source **e8148e5**. Extension job **177529** completed relative norms **0.3 / 1.0**, all 256 observations x noise0 x ten flow-step conditions, reusing the original reference actions in artifact177484. Together with E015, the combined grid contains **50** completed conditions. The independent summarizer reads every completed noise JSON and recomputes its metrics from the raw action arrays; the extension-only result.json is not the combined inventory.
- Exact177529 command: `sbatch --export='ALL,VLA_DIAGNOSTIC=D3,VLA_D3_OFFLINE_DIR=/projects/yanlongvla/runs/d3/offline-177484,VLA_D3_MAGNITUDES=0.3 1.0,VLA_ACCOUNTING_RESULT=/projects/yanlongvla/runs/accounting/177454/result.json,VLA_ROUNDTRIP_RESULT=/projects/yanlongvla/runs/accounting/roundtrip-177463.json' slurm/diagnostic.sbatch`.
- Real-policy zero-noise controls at clean, step0 and step9 all give native-precision maximum action difference **0.0**. Realized per-layer norm at requested0.3 spans **[0.2999967634677887, 0.3000122904777527]**, confirming the applied perturbation scale. Native selected precisions and unaffected steps are preserved.
- At0.3, maximum relative action MSE is **0.00027698567335199964**. At1.0, maximum is **0.014214520278667467** (step9). Thus **1.0** is the smallest measured amplitude crossing the predeclared max-step MSE1e-3 screening criterion. This choice uses offline metrics only, not noisy rollout success. No final offline proxy or per-step depth decision is claimed.
- Measured1.0 grid (MSE is physical executed5x7; MMD is paired marginal U-statistic):

| Flow step (0-based) | Relative action MSE | MMD² |
|---|---:|---:|
| 0 | 2.24094453377e-06 | 8.22422865229e-08 |
| 1 | 2.76384750953e-06 | 1.10234276611e-07 |
| 2 | 4.6806219928e-06 | 1.61186139336e-07 |
| 3 | 0.00207001111642 | -2.2138041001e-09 |
| 4 | 0.00210493659764 | 1.15142886354e-07 |
| 5 | 0.00210944688719 | 5.59252605495e-07 |
| 6 | 0.00197881918619 | 1.99935731077e-06 |
| 7 | 0.000119974032663 | 9.23759749993e-06 |
| 8 | 0.00058080793839 | 5.16585603097e-05 |
| 9 | 0.0142145202787 | 0.00285095384247 |

- At this point five MSE tolerances (steps0,1,2,7,8) and all ten MMD tolerances remain right-censored at norm1.0. Measured extension to **3.0**, job **177537**, command `sbatch --dependency=afterok:177529 --export='ALL,VLA_DIAGNOSTIC=D3,VLA_D3_OFFLINE_DIR=/projects/yanlongvla/runs/d3/offline-177484,VLA_D3_MAGNITUDES=3.0,VLA_ACCOUNTING_RESULT=/projects/yanlongvla/runs/accounting/177454/result.json,VLA_ROUNDTRIP_RESULT=/projects/yanlongvla/runs/accounting/roundtrip-177463.json' slurm/diagnostic.sbatch`.
- Informative-amplitude screening jobs **177538** (oddsteps, afterok177537) and **177539** (clean+evensteps, afterok177530), command template `sbatch --dependency=afterok:<prior> --export='ALL,VLA_D3_WORKER=<1|0>,VLA_NOISE_MAGNITUDE=1,VLA_ACCOUNTING_RESULT=/projects/yanlongvla/runs/accounting/177454/result.json,VLA_ROUNDTRIP_RESULT=/projects/yanlongvla/runs/accounting/roundtrip-177463.json' slurm/d3_screen.sbatch`. Each noisy condition retains two suites x10eps/task xseed7; reference job177530 completes the existing clean rows. New screen_1/clean links to that same reference, with explicit provenance; dependency prevents concurrent duplicate clean episodes. No changed episode counts or additional seeds.
- Updated summarizer tests: **12 passed**, adding generic next-measured-amplitude selection, rejection of incomplete offline severities and explicit no-measurement tolerance state. Command `python -m unittest discover -s tests -p test_d3_decisions.py -v`. Gates stay pending until exact paired screening coverage and supported tolerance/proxy decisions.
- Added measured D1/D2 diagnostic-selection figure and complete layer/pair CSV sources, source e8148e5. Rendered alignment and collision audits passed; minimum PDF glyph7pt. No data subsampling; 627 rank-deficient angle units are explicitly retained as unavailable in the source CSV. Figure reports descriptive single-seed diagnostics, not quantization gains. Report environment is isolated from active policy runtimes.
