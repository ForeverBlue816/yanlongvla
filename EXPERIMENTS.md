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
