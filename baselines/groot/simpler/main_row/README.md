# GR00T N1.7 SimplerEnv-Bridge: the baselines on our Hessians `H_D`, and the code of our 2-bit main row

Written for Chamadol (2026-11-12).

## What we ask

Please rerun your Bridge 2-bit baselines (QTIP, QuIP#, VPTQ, AQLM) **on the Hessians our main row is quantized on**, and evaluate
them as before. Nothing else changes: same checkpoint, same method configurations and damping, same evaluation (seed 7, 120 episodes
per task, your `eval_simpler_bridge.py` flags).

Why: your baseline rows were calibrated on the clean Hessians of the 256 Bridge calibration observations. Our main row is calibrated
on a source-balanced Hessian `H_D` that adds three more sources to the clean one (photometric copies, geometric copies, on-policy
states). To compare the quantizers and not the calibration data, the baselines need the same `H_D`.

| Step | Command | Cost |
|---|---|---|
| 1 Settings | edit the first block of `config.sh` (section "Setup" below; the QTIP checkout and the SimplerEnv venv are **not** needed for steps 1 to 3) | |
| 2 Build `H_D` on your GPU | `bash build_hd.sh 0` | 10 min, 33 GB GPU memory, 70 GB disk |
| 3 Convert to your calibration format | `python hd_for_baselines.py --hd $VLA_ROOT/runs/plustrain_20261008/groot/ptq/hessians --like <your clean Bridge calibration> --out <calibration_hd>` | 5 min |
| 4 Quantize | your Bridge commands with `--calibration <calibration_hd>` in place of the clean calibration: `export_layers.py`, then each `quantize_*.py` on `<calibration_hd>/backbone` and `<calibration_hd>/head_bridge`, then `import_external.py` | as your clean rows |
| 5 Evaluate | as your clean rows; name the rows `<method>_hd` | as your clean rows |
| 6 Pair | `simpler/b5_stats.py` of each `<method>_hd` row against FP and against the same method's clean row | |

Notes on the steps:

- **Step 2** runs the FP model on the 256 calibration observations (clean moments), on their 768 perturbed copies and on our 256
  on-policy states (`data/onpolicy_rinit.tar`, unpacked by the script), and mixes the four moments. It uses our states, so your `H_D`
  has the same inputs as ours; the activations are computed on your GPU.
- **Step 3** keeps the layer list, shapes and checkpoint hash of your clean calibration (`--like`) and replaces only the Hessians, so
  the hash check of the external quantizers passes. `tr(H_D) = 2 tr(H_clean)` by construction; the script writes the ratio to your
  clean trace per layer into `<calibration_hd>/summary.json` and stops if a layer is outside 1.6 to 2.4.
  We tested steps 3 and 4a on our machine against a clean calibration made with your `calibrate.py` (`GROOT_PROFILE=bridge`): the ratio is
  2.000000 on all 432 layers, and `export_layers.py` reads the converted directory. No baseline quantizer was run on it here.
- **Step 4**: GPTQ and AWQ (`quantize.py --calibration <calibration_hd>`) can be added in the same way if you have the time.
- **What the baselines do not get:** the fine-tune of our main row (step 4 of "The row" below). It trains our tables and row scales
  and has no counterpart for the other formats. So the `_hd` rows are to be compared with our base `tcq_w2_rht_d1` like for like, and
  with the main row as "PTQ against PTQ + fine-tune".

Please send back, for each `<method>_hd` row: the per-episode files, the paired JSON files, `quant.json` with the bits per weight, the
GPU model and the toolkit commit. If you can, also send the per-episode files of your FP and of your four clean rows, so that we can
recompute every pair here.

## The row

`tcq_raqft_d1_s` is the main row of the LIBERO tables (581 / 1000 on Plus), ported to the Bridge checkpoint without changing the recipe:

1. **Clean row `tcq_w2_rht`.** TCQ (QTIP bitshift trellis, L = 16, K = 2, V = 2, a 2⁹ × 2 FP16 table per layer, 16 × 16 tiles, two-sided
   randomized Hadamard, FP16 row scales) on the Hessians of the 256 Bridge calibration observations (`../data/bridge-calibration`).
2. **Source-balanced Hessian** `H_D = H_clean + Σ_s w_s H_s`, `w_s = tr(H_clean) / (3 tr(H_s))`, with three sources: photometric copies
   (512 observations), geometric copies (256) and on-policy states (256 SimplerEnv rollouts of the clean row).
3. **Base `tcq_w2_rht_d1`.** The same TCQ fit on `H_D`, one round, damping fixed per layer to the clean row's value.
4. **Main row `tcq_raqft_d1_s`.** Stabilised readout fine-tune of the base: 768 updates, Adam, EMA 0.99, 10 % anchor, 40 of the 256
   observations held out. It changes the tables, row scales and norm weights; the codes and every tensor shape stay fixed.

All 432 Linear layers are quantized (216 backbone, 216 action head), the same set as your baselines (`groot_common.is_quantizable`).
Bits per weight: **2.0177 linear, 5.5915 whole model** (payload 590,290,944 B; `reference/tcq_w2_rht_d1.accounting.json`; the fine-tune
does not change the payload).

One thing to keep in mind when you read the results: the on-policy states of step 2 were captured in SimplerEnv, on the same finite set
of layouts that the evaluation runs (the evaluation is exhaustive over them). The policy-noise seeds differ from the evaluation seeds,
and only one policy input per rollout enters a Hessian, but the scenes overlap. Your clean baseline rows saw only the 256 dataset observations; the `_hd` rows see what our row sees.

## Optional: rebuild our row on your GPU

Not part of the request above. `run_main_row.sh` builds the whole row from the official checkpoint (about 22 h on one H100); a row built
on your GPU could be paired with your baselines episode by episode, which our H100 results cannot.

### Setup

| Needed | Where |
|---|---|
| GR00T environment and SimplerEnv venv | as in `../README.md` section 2 (Isaac-GR00T `d2b7e75b937e3ec9aa5dbc798f08b89692c49734`, submodules initialised recursively) |
| QTIP source | `git clone https://github.com/Cornell-RelaxML/qtip $VLA_ROOT/src/qtip`, commit `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`. It is imported, not installed; its CUDA kernels are not used. |
| Official checkpoint | `nvidia/GR00T-N1.7-SimplerEnv-Bridge` @ `940134b3c2948ccfdf8e7393f2d2ca869dc42833`, downloaded with `snapshot_download(..., cache_dir=$HF_HUB_CACHE)`. It must be a hub cache (blob symlinks): the code checks every shard's SHA-256 against its blob name. |
| Calibration and held-out sets | `ln -s <repo>/baselines/groot/simpler/data/bridge-calibration $VLA_ROOT/datasets/bridge-calibration`, and the same for `bridge-heldout` |
| openpi msgpack codec | `OPENPI_CLIENT_MSGPACK=/path/openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py`, as for your `serve.py` |
| Token | `export HF_TOKEN=...` with access to `nvidia/Cosmos-Reason2-2B` |

Edit the first block of `config.sh` (work directory `VLA_ROOT`, the two Python interpreters, the hub cache). `VLA_ROOT` needs about
60 GB: ten Hessian sets of 6 to 7.5 GB, four fits of 2.5 GB, three model directories of 6.5 GB.

### Run

```bash
bash run_main_row.sh 0 1        # GPU ids; with one argument everything runs on that GPU
```

The script skips every step whose output exists, so it can be restarted. Steps, with our times on H100 (the fits shared the GPUs with
other jobs) and the numbers to compare with:

| Step | Command inside the script | Time | Check against |
|---|---|---|---|
| 1 FP model, smoke, clean Hessians | `groot_simpler.py --stage prepare / smoke / calib` | 5 min | `reference/smoke_H100.json` (432 layers, 2,340,421,632 weights) |
| 2 clean row | `groot_simpler.py --stage fit --part head / backbone`, `--stage assemble --row tcq_w2_rht` | 48 + 62 min | `reference/fit/*_tcq_rht_simpler_bridge.complete.json` (relative weight MSE 0.121 head, 0.114 backbone; damping 0.01 on all layers) |
| 3 on-policy states | `onpolicy.sh` (256 SimplerEnv rollouts) | 1 h 40 min | `reference/onpolicy_states.json` (176 of 256 rollouts succeed) |
| 4 per-source moments, `H_D` | `groot_simpler_main.py --stage hsrc`, `--stage hd` | 3 min, 33 GB GPU memory | `reference/hessians/*_hd1_*.complete.json` (mean weights 0.337 / 0.335 / 0.334) |
| 5 base | `groot_simpler_main.py --stage fit`, `--stage assemble --row tcq_w2_rht_d1` | 23 + 28 min | `reference/fit/*_tcq_rht_d1_simpler_bridge.complete.json` |
| 6 fine-tune | `groot_simpler_main.py --stage finetune --smoke`, then `--stage finetune` | 16 h (70 s per update), 25 GB | `reference/finetune_manifest.json`, `reference/finetune_smoke.json` |
| 7 evaluation | `eval_row.sh` (two halves of the task list) | 3.5 h | section "Reference numbers" |

**On-policy states: use ours or capture your own.** For the same `H_D` as ours, unpack our 256 states before step 3:
`mkdir -p $VLA_ROOT/runs/plustrain_20261008/groot/simpler_bridge && tar -xf data/onpolicy_rinit.tar -C $VLA_ROOT/runs/plustrain_20261008/groot/simpler_bridge`
(SHA-256 `bbc02a04de5f19827f204403af1b31d695b9e5642ddd93194c53ce90268eb793`); step 3 is then skipped. A capture on your GPU gives
different states, because the rollouts of the clean row differ between GPU models. Say which one you used.

A rebuilt row will not be bit-identical to ours (the fits and the fine-tune depend on the GPU).

### Evaluation of any model directory

```bash
bash eval_row.sh <row name> <model dir> <gpu> <port> [episodes per task=120] [out name=episodes] [task ...]
```
It starts `scripts/simpler_server.py` (seeded policy noise) and `scripts/simpler_eval.py`. These two are the files your
`../simpler_server.py` and `../eval_simpler_bridge.py` were made from, without your later options; the protocol, seeds and output
records are the same. The output is one JSON list per call; split it per task for `b5_stats.py`.

## Files

| Path | Content |
|---|---|
| `config.sh` | site settings (paths, interpreters) |
| `build_hd.sh` | `H_D` from the FP checkpoint and our on-policy states (steps 1, 4 of the row without the clean fit) |
| `hd_for_baselines.py` | `H_D` in the calibration format of the baseline harness |
| `run_main_row.sh` | the whole row, steps 1 to 7 |
| `onpolicy.sh`, `eval_row.sh` | server + SimplerEnv client for the capture and for an evaluation |
| `scripts/` | 23 Python files: `groot_simpler.py` (clean row), `groot_simpler_main.py` (`H_D`, base, fine-tune port), `groot_tcq.py` (TCQ fit), `groot_tcq_raqft.py` and `groot_tcq_raqft_s.py` (fine-tune), `simpler_onpolicy_client.py`, `simpler_eval.py`, `simpler_server.py`, and the modules they import |
| `data/onpolicy_rinit.tar` | our 256 on-policy states (image 256 × 256 and 8-d state per rollout, with the rollout metadata), 31 MB |
| `reference/` | our records for each step: smoke, Hessian summaries, fit summaries, accounting, fine-tune contract, on-policy metadata |

`scripts/` is a copy of the files that produced our row. Three files differ from our working copy, in path lines only, so that the
root comes from `VLA_ROOT`: `groot_ptq.py` (1 line), `groot_simpler.py` (1 line), `groot_bridge_server.py` (4 lines; also reads
`OPENPI_CLIENT_MSGPACK`). We ran `prepare` and `smoke` from this copy in an empty work directory; the smoke output equals
`reference/smoke_H100.json`. The other steps were run from the working copy.

## Reference numbers (H100, seed 7, 120 episodes per task)

| Task | FP, H100 | FP, yours | main row `tcq_raqft_d1_s`, H100 |
|---|---|---|---|
| spoon_on_towel | 101 | 101 | pending |
| carrot_on_plate | 78 | 75 | pending |
| stack_cube | 42 | 36 | pending |
| put_eggplant_in_basket | 62 | 56 | pending |
| put_eggplant_in_sink | 2 | 0 | pending |
| open_drawer | 120 | 120 | pending |
| close_drawer | 117 | 115 | pending |
| mean (%) | 62.1 | 59.9 | pending |

Our FP per-episode files are `results/lambda_1bit_20261112/bridge_fp.episodes_{a,b}.json` at the repository root. The main-row column,
its per-episode files, the fine-tune curve and the held-out action MSE will be added here when the run finishes.

Two notes on our run, so that nothing surprises you in the records:

- **Fine-tune events.** The fine-tune log shows 20 non-finite steps up to update 672; the stabilised variant skips such a step and
  continues, as on LIBERO, and stops only above its fixed limit of 38 such events.
- **Clean row on our disk.** A second fit worker was started by mistake on the clean backbone fit and rewrote 6 of the 432 layers after
  the clean row had been assembled. The assembled clean row (the policy of the on-policy capture) holds the earlier fits of those 6
  layers; both are fits of the same layer on the same Hessian. Nothing downstream reads the rewritten files except the damping value,
  which is 0.01 for every layer in both.
