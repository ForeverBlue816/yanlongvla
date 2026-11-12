# GR00T N1.7 SimplerEnv-Bridge: our 2-bit main row (code and records)

Reference only; nothing here needs to be run. The Bridge comparison is our main row `tcq_raqft_d1_s` (calibrated on the
source-balanced Hessian `H_D`) against the 2-bit baselines as already run with their standard calibration (the 256 Bridge
calibration observations; `../../../RESULTS.md`, SimplerEnv-Bridge table). Our row is evaluated on 2 × H100 with the same protocol and
seeds (7 tasks × 120 episodes, seed 7); since the baselines ran on another GPU model, the comparison is per task and on the 7-task mean,
not episode by episode. This folder documents how the row is built, so that it can be reproduced. Result (section "Reference numbers"): 63.0 %, level with FP.

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
and only one policy input per rollout enters a Hessian, but the scenes overlap. The baselines saw only the 256 dataset observations.

## Rebuilding the row

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
| `run_main_row.sh` | the whole row, steps 1 to 7 |
| `onpolicy.sh`, `eval_row.sh` | server + SimplerEnv client for the capture and for an evaluation |
| `scripts/` | 23 Python files: `groot_simpler.py` (clean row), `groot_simpler_main.py` (`H_D`, base, fine-tune port), `groot_tcq.py` (TCQ fit), `groot_tcq_raqft.py` and `groot_tcq_raqft_s.py` (fine-tune), `simpler_onpolicy_client.py`, `simpler_eval.py`, `simpler_server.py`, and the modules they import |
| `data/onpolicy_rinit.tar` | our 256 on-policy states (image 256 × 256 and 8-d state per rollout, with the rollout metadata), 31 MB |
| `reference/` | our records for each step: smoke, Hessian summaries, fit summaries, accounting, fine-tune contract, on-policy metadata |

`scripts/` is a copy of the files that produced our row. Three files differ from our working copy, in path lines only, so that the
root comes from `VLA_ROOT`: `groot_ptq.py` (1 line), `groot_simpler.py` (1 line), `groot_bridge_server.py` (4 lines; also reads
`OPENPI_CLIENT_MSGPACK`). We ran `prepare` and `smoke` from this copy in an empty work directory; the smoke output equals
`reference/smoke_H100.json`. The other steps were run from the working copy.

## Reference numbers (H100, seed 7, 120 episodes per task; finished 2026-11-12)

| Task | FP, yours | FP, H100 | clean row `tcq_w2_rht` | base `tcq_w2_rht_d1` | main row `tcq_raqft_d1_s` | your QTIP 2-bit, rebuilt on H100 |
|---|---|---|---|---|---|---|
| spoon_on_towel | 101 | 101 | 106 | 100 | 87 | 110 |
| carrot_on_plate | 75 | 78 | 93 | 82 | 68 | 93 |
| stack_cube | 36 | 42 | 68 | 72 | 60 | 72 |
| put_eggplant_in_basket | 56 | 62 | 89 | 68 | 77 | 86 |
| put_eggplant_in_sink | 0 | 2 | 1 | 0 | 2 | 1 |
| open_drawer | 120 | 120 | 120 | 120 | 120 | 119 |
| close_drawer | 115 | 117 | 120 | 118 | 115 | 118 |
| mean (%) | 59.9 | 62.1 | 71.1 | 66.7 | 63.0 | 71.3 |
| vs FP H100, pp [95 %] | | | +8.9 [+4.4, +13.3] | +4.5 [+1.4, +7.7] | +0.8 [−2.9, +4.4] | +9.2 [+5.5, +12.9] |

The H100 columns are paired episode by episode (`../../../simpler/b5_stats.py`). **On Bridge the main row is level with FP and below the
plain 2-bit rows**: main row − base −3.7 [−6.0, −1.3], base − clean row −4.4 [−8.8, +0.1], main row − QTIP rebuilt −8.3 [−12.4, −4.2].
Your finding that 2-bit rows are above FP on Bridge is reproduced with our code (clean row +8.9). Held-out action MSE vs FP (normalised):
clean 0.116, base 0.140, main row 0.152. The fine-tune's held-out readout loss went from 10,347 to 10,196 at the exported update 384.
Full account: `LAMBDA_1BIT_2026-11-12.md` at the repository root; per-episode files `results/lambda_1bit_20261112/bridge_<row>.episodes_{a,b}.json`,
paired statistics `bridge_paired_*.json`, weight shrink per row `bridge_weight_shrink.json`.

The QTIP column is your scripts (`GROOT_PROFILE=bridge`) run on H100 with QTIP's two CUDA extensions replaced by pure-PyTorch stand-ins
(`results/lambda_1bit_20261112/code/run_with_stubs.py`, `qtip_chain.sh`). It differs from your QTIP row on put_eggplant_in_basket
(86 here, 27 in `RESULTS.md`); the cause is open.

Two notes on our run, so that nothing surprises you in the records:

- **Fine-tune events.** The fine-tune log shows 22 non-finite steps in 768 updates; the stabilised variant skips such a step and
  continues, as on LIBERO, and stops only above its fixed limit of 38 such events.
- **Clean row on our disk.** A second fit worker was started by mistake on the clean backbone fit and rewrote 6 of the 432 layers after
  the clean row had been assembled. The assembled clean row (the policy of the on-policy capture) holds the earlier fits of those 6
  layers; both are fits of the same layer on the same Hessian. Nothing downstream reads the rewritten files except the damping value,
  which is 0.01 for every layer in both.
