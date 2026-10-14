# Baseline harness: 2-bit VQ / scalar quantization of π₀.₅-LIBERO, evaluated like our method

This directory lets you quantize the **whole** π₀.₅-LIBERO policy (backbone **and** action expert, both 2-bit) with any method, evaluate it on **LIBERO** and **LIBERO-Plus** with exactly the protocol used for our rows, and compare against our published per-episode outcomes with the same paired statistics. Our method's rows (FP, original 2-bit, RAQ-FT, and the final W2A8 row) are in `../results/plus_episodes/` and `../results/libero_full/`.

Everything is evaluated through the **same BF16 dense runtime** (dequantized weights): the comparison measures quantization error only, never kernel speed. Bits-per-weight is counted the same way for every method.

## Fairness rules (please do not deviate)

| Item | Rule |
|---|---|
| Model | openpi PyTorch port of `pi05_libero` (the official checkpoint; `python -m openpi ...` conversion as in the openpi README). Same checkpoint hash for everyone. |
| What is quantized | the 414 `nn.Linear` layers inside the SigLIP encoder (27×6), the Gemma language model (18×7) and the Gemma action expert (18×7). **Both backbone and expert at the same nominal bit-width** (2-bit for the main column; 3-bit as a second column). Embeddings, projector, norms, action/time projections and heads are never quantized. |
| Calibration data | the same 256 observations from 60 LIBERO training episodes (`data/calibration_manifest.json`, dataset `physical-intelligence/libero` at a pinned revision), flow-matching noise seed 0, all 10 flow steps for the expert. `calibrate.py` produces per-layer Hessians `E[x xᵀ]` from them. No evaluation data, no LIBERO-Plus data. |
| Evaluation runtime | fake-quant checkpoint: dequantized weights in BF16 in the unmodified openpi model. |
| LIBERO | 4 suites × 10 tasks × 50 episodes (2000), seed 7, init states in order, 10 settle steps, 10-step chunks executed 5 at a time, suite step limits 220/280/300/520. Also report the 1900-episode column excluding LIBERO-10 tasks 8 and 9. |
| LIBERO-Plus | official LIBERO-Plus simulator, seed 7, init state 0, one episode per instance, same chunking and limits, per-instance noise seed `SeedSequence([7, suite_index, task, 0])`. Report the **preregistered 1000-instance subset** (500 Camera Viewpoints + 500 Robot Initial States, `data/plus_units_selection_1000.json`) and, if you can afford it, the full 10,030 instances (`--units full`). The 2000-unit expansion (`data/plus_units_expansion_2000.json`) is what we use for confirmation. |
| Statistics | paired bootstrap on identical instances with identical noise seeds (`paired_stats.py`), pointwise 95%, 2000 draws, analysis seed 20260930. Single evaluation seed; say so. |
| Accounting | `whole_model_bpw = (linear payload bytes + 542,205,320 retained bytes) × 8 / 3,353,433,872`, where the linear payload counts **every stored tensor** of the 414 quantized Linear weights (codes, codebooks, scales, zero points, permutations) and the retained bytes are the rest of our deployable checkpoint as stored (int8 row-absmax token embedding with FP32 scales 527,675,904 B; SigLIP embeddings/norms/Linear biases, LM norms and projector 9,716,488 B; expert norms, FP32 AdaRMS tables, action/time projections 4,812,928 B; both language heads removed). Also report `quantizable_linear_bpw = payload × 8 / 2,704,306,176`. Our 2-bit rows: linear payload 682,883,676 B = 2.0201 linear bpw, 2.9226 whole-model bpw (check: (682,883,676 + 542,205,320) × 8 / 3,353,433,872 = 2.9226). **Correction 2026-10-14:** the first version of this README gave 1,012,077,920 retained bytes; that was the retained set of the BF16 accounting checkpoint and does not reproduce the published figure. `common.py` carries the corrected constant. |

## Setup

```bash
# openpi (PyTorch), LIBERO, and LIBERO-Plus as in their READMEs. LIBERO-Plus replaces the `libero` package;
# keep two environments (or two venvs) and use the LIBERO one for eval_libero.py, the LIBERO-Plus one for eval_libero_plus.py.
pip install huggingface_hub pyarrow pillow safetensors
export PYTHONPATH=/path/to/this/baselines:$PYTHONPATH
CKPT=/path/to/pi05_libero_pytorch          # official converted checkpoint (model.safetensors + assets/)
```

## Step 1: shared calibration (once)

```bash
python calibrate.py --checkpoint $CKPT --out ./calibration --hf-cache ./hf
# -> calibration/hessians/<layer>.pt (float32 E[x x^T]), calibration/fp_reference_actions.npy, calibration/index.json
```

## Step 2a: built-in scalar baselines (RTN, GPTQ)

```bash
python quantize.py --method rtn  --bits 2 --group 128 --checkpoint $CKPT --out ./ckpt_rtn_w2g128
python quantize.py --method gptq --bits 2 --group 128 --checkpoint $CKPT --calibration ./calibration --out ./ckpt_gptq_w2g128
# repeat with --bits 3 / 4. quant.json in the output directory holds the bpw accounting.
```

## Step 2b: external VQ baselines (AQLM, VPTQ, QuIP#, QTIP, …)

```bash
python export_layers.py --checkpoint $CKPT --calibration ./calibration --out ./layers
# ./layers/<layer>.safetensors: weight (float16, out×in) and hessian (float32, in×in); ./layers/layers.json lists them.
```
Run the toolkit's layer-wise quantizer on each `(weight, hessian)` pair at the target bit-width (2-bit for **both** the backbone and the expert layers; `layers.json` marks which is which), dequantize, and write `<layer>.safetensors` with a `weight` tensor into a directory, plus `accounting.json`:

```json
{"method": "aqlm_2x8", "bits_nominal": 2, "backbone_bits": 2, "expert_bits": 2,
 "layers": {"<layer name>": {"payload_bytes": 123456}, "...": {}}}
```
`payload_bytes` must include codes/indices, codebooks, scales, zero points and any permutation. Then:

```bash
python import_external.py --checkpoint $CKPT --quantized ./aqlm_out --accounting ./aqlm_out/accounting.json --out ./ckpt_aqlm_w2
```
Notes for specific toolkits: AQLM (`aqlm`'s `quantize_layer` / `Hessian` inputs), QuIP# and QTIP (their `ldlq`-based quantizers take the Hessian directly; use the same RHT options for all layers and count the RHT sign vectors in the payload), VPTQ (its `quantize` uses the Hessian and k-means codebooks). Keep the calibration Hessians above; do not re-collect with other data.

## Step 3: evaluation

```bash
# one server per checkpoint (GPU), clients in the matching simulator environment
python serve.py --checkpoint ./ckpt_gptq_w2g128 --port 8105 &
python eval_libero.py --port 8105 --episodes 50 --out ./ckpt_gptq_w2g128/libero/            # LIBERO env
python eval_libero_plus.py --port 8105 --units data/plus_units_selection_1000.json --out ./ckpt_gptq_w2g128/plus_1000.jsonl   # LIBERO-Plus env
python eval_libero_plus.py --port 8105 --units full --out ./ckpt_gptq_w2g128/plus_full.jsonl   # optional, 10,030 instances
```
Both clients are resumable (they skip completed rows). An L40S evaluates ~1000 Plus instances in about 4 hours.

## Step 4: compare with our rows

```bash
python paired_stats.py plus   --candidate ./ckpt_gptq_w2g128/plus_1000.jsonl --reference ../results/plus_episodes/original_2bit.jsonl
python paired_stats.py plus   --candidate ./ckpt_gptq_w2g128/plus_1000.jsonl --reference ../results/plus_episodes/original_fp.jsonl
python paired_stats.py libero --candidate ./ckpt_gptq_w2g128/libero/ --reference ../results/libero_full/original_fp/ --episodes 50
```
Published reference rows (per-episode, identical instances and noise seeds):

| Row | Plus 1000 | LIBERO full 2000 | whole-model bpw |
|---|---:|---:|---:|
| `original_fp` (BF16) | 765 | 1928 | 16 |
| `original_2bit` (2-bit RVQ backbone + 2-bit expert, our base) | 704 | — (medium 1000: 982) | 2.9226 |
| `raqft_a_2bit` (ours: RAQ-FT 2-bit backbone + 2-bit expert) | 739 | — (medium 1000: 974) | 2.9226 |
| `a8_raqft` (ours, final W2A8 row: + INT8 activations, BF16 KV cache) | 740 | 1944 | 2.9226 |

For the table in the paper, report for each baseline: whole-model bpw, Plus 1000 successes with the paired difference and 95% CI vs `original_fp` and vs `original_2bit`, LIBERO 2000 (and the 1900 column), and whether both backbone and expert were quantized at the nominal width. Please also keep `quant.json`, the Hessian index and the exact toolkit commit so the row is reproducible.
