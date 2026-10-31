# GR00T N1.7 on LIBERO-Plus: 2-bit baseline harness

This folder extends the π₀.₅ baseline harness (`../README.md`) to **GR00T N1.7** (`nvidia/GR00T-N1.7-LIBERO`). The task is
the same: quantize the **whole** policy at **2 bits** with each baseline method, then evaluate it on the **LIBERO-Plus
preregistered 1000-instance subset**. The protocol matches our GR00T rows exactly, so every result can be paired
episode by episode with ours.

**Scope:** 2-bit only, LIBERO-Plus 1000 only.

**Methods:** GPTQ, AWQ, VPTQ, AQLM, QuIP#, QTIP. RTN is not included; it collapses at 2 bits on π₀.₅ (0/292, decision 18).

## What is different from π₀.₅

| Item | GR00T N1.7 |
|---|---|
| Checkpoints | Four fine-tuned checkpoints, one per suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`), revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`. Every suite is evaluated with its own checkpoint. The **backbone is byte-identical** in all four, and `prepare_checkpoints.py` verifies this. Quantize the backbone **once** and the action head **four times**. |
| Code | `github.com/NVIDIA/Isaac-GR00T` at `d2b7e75b937e3ec9aa5dbc798f08b89692c49734` (`Gr00tPolicy`, embodiment `LIBERO_PANDA`). Our environment: Python 3.12, torch 2.9.0+cu128, transformers 4.57.3, flash-attn 2.8.3, diffusers 0.35.1. |
| Backbone config | The loader builds the Qwen3-VL module and its processor from `nvidia/Cosmos-Reason2-2B` (config and processor files only; all backbone weights are in the checkpoint). That repository is **gated**: accept its licence on huggingface.co once and export a read token (`HF_TOKEN`). Our reference rows use this official configuration. |
| Quantized layers | **432 Linear per suite checkpoint.** **Backbone, 216 layers:** Qwen3-VL vision tower 104 (blocks, merger, deepstack mergers) and language-model layers 0–15, 112. **Action head, 216 layers:** `vl_self_attention` 24, and DiT attention + FFN 192. These total 2,340,421,632 weights. |
| Not quantized | DiT AdaLN linears (`norm1.linear`) and `proj_out_*`, because they depend only on the flow timestep. Also `lm_head` (never called), embeddings, norms, the per-embodiment state/action encoders and decoders, and position/timestep embeddings. The rule is `groot_common.is_quantizable`. |
| Inference | 4 flow steps; 16-step action window, of which 8 are executed per policy call (official LIBERO rollout). Inputs are the raw 256 × 256 agent-view and wrist frames flipped `[::-1, ::-1]`, plus an 8-d state. |
| Accounting | Per suite checkpoint: `whole_model_bpw = (linear payload + 1,607,188,736 retained bytes) × 8 / 3,144,016,000`. `quantized_linear_bpw = payload × 8 / 2,340,421,632`. The payload counts every stored tensor of the 432 layers. Retained bytes are every other tensor used by action inference at its stored dtype; `lm_head` is excluded. Whole-model bpw is high for every method (≈ 5.6 at 2-bit linear) because the BF16 embedding and the per-embodiment MLPs are retained. Report both numbers. |

Unchanged from π₀.₅: the 256 calibration observations (`../data/calibration_manifest.json`), the LIBERO-Plus units
(`../data/plus_units_selection_1000.json`: 500 Camera Viewpoints + 500 Robot Initial States, seed 7, init state 0, 10
settle steps, step limits 220/280/300/520, noise seed `SeedSequence([7, suite_index, task, 0])`), the per-layer quantizer
interface, and the paired bootstrap (`../paired_stats.py`). Calibration noise seed: `SeedSequence([7, observation_index])`.

**Smoke-tested 2026-10-30** on our cluster (RTX 6000 Ada): preparation, calibration, export, the import round trip, GPTQ on all four suites, and 5 LIBERO-Plus units on the prepared FP checkpoints, which reproduced our reference episodes exactly (same success, same number of policy calls, same noise seeds).

## Steps

You need two environments: GR00T (Isaac-GR00T at the commit above, plus `openpi-client`) and LIBERO-Plus (as for π₀.₅), and a Hugging Face read token with access to `nvidia/Cosmos-Reason2-2B` (`export HF_TOKEN=...`).
`serve.py` needs openpi-client's msgpack codec. Either install `openpi-client` into the GR00T environment, or set
`OPENPI_CLIENT_MSGPACK=/path/openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py`.

```bash
cd baselines/groot
# 0. checkpoints: merge the official shards into one model.safetensors per suite; checks backbone identity and accounting constants
python prepare_checkpoints.py --out ./checkpoints --hf-cache ./hf
# 1. calibration: backbone Hessians from libero_spatial, head Hessians per suite (≥ 48 GB GPU memory: FP64 accumulators)
python calibrate.py --checkpoints ./checkpoints --out ./calibration            # or --dataset-root /path/local/physical-intelligence-libero
```

**2a. GPTQ and AWQ** (2-bit, group 128, algorithms imported unchanged from `../quantize.py`):
```bash
python quantize.py --method gptq --checkpoints ./checkpoints --calibration ./calibration --out ./ckpt_gptq_w2g128
python quantize.py --method awq  --checkpoints ./checkpoints --calibration ./calibration --out ./ckpt_awq_w2g128
```

**2b. VPTQ, AQLM, QuIP#, QTIP.** These use the existing per-layer quantizers in `..` unchanged, with the same
configurations and damping as the π₀.₅ rows:

| Method | Configuration |
|---|---|
| VPTQ | v4-k256-0 |
| AQLM | 2x8g8, 1 % damping |
| QuIP# | E8P12; main row `--tune-iters 0`, official 10-sweep row in the appendix (decision 29) |
| QTIP | K = 2 |

```bash
python export_layers.py --checkpoints ./checkpoints --calibration ./calibration --out ./layers
# five layer sets: layers/backbone (checkpoint libero_spatial) and layers/head_<suite> (checkpoint <suite>)
for key in backbone head_libero_spatial head_libero_object head_libero_goal head_libero_10; do
  suite=${key#head_}; [ "$key" = backbone ] && suite=libero_spatial
  python ../quantize_vptq.py --layers ./layers/$key --calibration ./calibration/$key --checkpoint ./checkpoints/$suite \
      --vptq-repo /path/VPTQ --bits 2 --out ./vptq_out/$key
done
python import_external.py --checkpoints ./checkpoints --quantized ./vptq_out --out ./ckpt_vptq_w2
```
In `layers.json`, the action-head layers carry `group: "expert"`, the π₀.₅ harness's name for the action part. Any
setting that the π₀.₅ quantizers apply to the expert therefore applies to the GR00T action head.

**3. LIBERO-Plus 1000 and paired statistics.** For each suite, `run_plus.py` starts `serve.py` on that suite's checkpoint
and runs the official simulator client. It resumes where it stopped.
```bash
python run_plus.py --checkpoints ./ckpt_gptq_w2g128 --out ./ckpt_gptq_w2g128/plus_1000.jsonl \
    --groot-python /path/groot-venv/bin/python --plus-python /path/libero-plus-venv/bin/python
python ../paired_stats.py plus --candidate ./ckpt_gptq_w2g128/plus_1000.jsonl --reference data/reference/groot_fp.plus_1000.jsonl
python ../paired_stats.py plus --candidate ./ckpt_gptq_w2g128/plus_1000.jsonl --reference data/reference/groot_2bit_ours_ptq.plus_1000.jsonl
```

**Sanity before a full row.** Use the same gate as π₀.₅ (decision 17). Run the checkpoint check and a short sample, for
example the first 50 units of `libero_spatial`. Stop a row that scores 0, or ≤ 5 % where FP scores ≥ 50 %.

## Reference rows

All reference rows are in `data/reference/`. Episodes are LIBERO-Plus 1000, seed 7, on an RTX 6000 Ada.

| Row | Plus /1000 (camera / robot init) | Linear / whole-model bpw | File |
|---|---|---|---|
| FP (BF16) | 618 (326 / 292) | 16 / 16 | `groot_fp.plus_1000.jsonl` |
| Ours, 2-bit VQ PTQ, backbone + head | 486 (257 / 229) | 2.023 / 5.596 | `groot_2bit_ours_ptq.plus_1000.jsonl`, `groot_2bit_ours_ptq.accounting.json` |
| Ours, 2-bit VQ + fine-tune (row B) | 504 (248 / 256) | 2.023 / 5.596 | `groot_2bit_ours_finetuned.plus_1000.jsonl` |

Pair every baseline against **FP** and against **ours, 2-bit VQ PTQ**. The PTQ row is the like-for-like comparison: pure
post-training quantization, no fine-tuning. Evaluate on any GPU type and record it in the row. Episode outcomes can
differ slightly across GPU models: on GR00T, FP actions differ across devices by a relative MSE of 1.2e-4.

## What to report per row

Per row, report:

- Method and configuration, toolkit commit, and exact command.
- `quant.json` of each suite, with the linear and whole-model bpw. They are identical across suites when the method's
  payload does not depend on the weights.
- Plus 1000 successes, with camera and robot-initial-state counts.
- Paired Δ with 95 % CI vs FP and vs ours (2-bit VQ PTQ).
- GPU model.
- A note that the row is pure PTQ on the shared Hessians.

Record implementation decisions in `../DECISIONS.md` under a GR00T heading, as for π₀.₅.

## Rough cost

| Task | Cost |
|---|---|
| Calibration | about 15 min per suite |
| GPTQ / AWQ | under 1 GPU-hour per row |
| Plus 1000 | ≈ 3 h per row on one RTX 6000 Ada (four model loads included) |
| AQLM | the expensive one; time grows steeply with input width |

Input widths that drive the AQLM cost:

| Width | Layers |
|---|---|
| 6144 | 16 language-model `down_proj`, plus 32 DiT FFN-out per suite |
| 8192 | 4 `vl_self_attention` FFN-out per suite |
| 4096 | vision merger |

Expect tens of GPU-hours for the five layer sets. Build them in parallel, as decision 20 did for π₀.₅.

## Our per-unit Plus files (`results/`)

`results/` holds the per-unit LIBERO-Plus 1000 outcomes of our three GR00T rows, in the format `run_plus.py` writes, so any baseline can be paired against them with `../paired_stats.py plus`:

| File | Row | Plus /1000 |
|---|---|---|
| `results/groot_fp.plus_1000.jsonl` | FP (BF16) | 618 |
| `results/groot_2bit_ours_ptq.plus_1000.jsonl` (+ `.accounting.json`) | ours, 2-bit VQ PTQ, backbone + head (2.023 / 5.596 bpw) | 486 |
| `results/groot_2bit_ours_b2.plus_1000.jsonl` (+ `.accounting.json`) | ours, 2-bit VQ + fine-tune, row B2 (same payload, 2.023 / 5.596 bpw) | 556 |

All three on an RTX 6000 Ada, seed 7, identical noise seeds. `data/reference/` keeps the FP and PTQ files under their earlier names (byte-identical).
