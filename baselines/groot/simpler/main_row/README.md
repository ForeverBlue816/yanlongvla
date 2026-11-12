# GR00T N1.7 SimplerEnv-Bridge: our 2-bit main row, for evaluation on your side

Written for Chamadol (2026-11-12). Your four Bridge baselines (QTIP, QuIP#, VPTQ, AQLM, 2-bit) and your FP run are on your GPU; our
2-bit main row `tcq_raqft_d1_s` was built and is being evaluated on a different machine (2 × H100), and episode outcomes differ between
GPU models. Please evaluate our row on your GPU with the harness you used for the baselines, so that all rows are paired episode by
episode. This folder has everything for that.

There are two routes. **Route A is the one we need**; route B is there if you want to rebuild the row yourself.

| Route | What you run | GPU time |
|---|---|---|
| A. Evaluate our checkpoint | one evaluation of a model directory, 840 episodes | as one of your baseline rows |
| B. Rebuild the row from the official checkpoint | `run_main_row.sh` (Hessians, two fits, on-policy capture, fine-tune, evaluation) | about 22 h on one H100 |

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

One thing to keep in mind when you read the result: the on-policy states of step 2 were captured in SimplerEnv, on the same finite set
of layouts that the evaluation runs (the evaluation is exhaustive over them). The policy-noise seeds differ from the evaluation seeds,
and only one policy input per rollout enters a Hessian, but the scenes overlap. Your baselines saw only the 256 dataset observations.

## Route A: evaluate our checkpoint

1. Get the model directory `tcq_raqft_d1_s/simpler_bridge` (6.5 GB: `model.safetensors` with the decoded weights plus the official
   config files; it loads like the FP snapshot). **Location: to be added here when the fine-tune finishes (it is past update 672 of 768
   as this is written, 2026-11-12 04:30 UTC).** The base `tcq_w2_rht_d1` will be placed next to it.
2. Evaluate it exactly as you evaluated the baseline rows: same server, same `eval_simpler_bridge.py` flags, same seed 7, 120 episodes
   per task, one GPU model for the whole row. With your feeder that is a new row whose `MODEL` is the directory above.
3. Pair it with your FP and with each baseline: `python simpler/b5_stats.py --fp <eval>/fp --row <eval>/tcq_raqft_d1_s --out ...`, and
   the same with `--fp` set to each baseline directory.

Please send back: the per-episode files of the row (one JSON per task), the paired JSON files, the GPU model, and, if you can, the
per-episode files of your FP and of the four baselines, so that we can recompute the pairs here.

## Route B: rebuild the row

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

A rebuilt row will not be bit-identical to ours (the fits and the fine-tune depend on the GPU), so for the paired table route A is the
reference; route B tells us how much the result moves when the row is rebuilt.

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
