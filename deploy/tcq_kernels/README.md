# Inference kernels for the trellis-coded 2-bit format (π₀.₅ W2A8)

Model: the π₀.₅ LoopQ main row **`tcq_raqft_d1`** (`models/pi05_tcq/tcq_raqft_d1`): bitshift trellis L = 16, 2 bits / weight, V = 2,
16 × 16 tiles, one 512 × 2 lookup table per layer, FP16 row scales, two-sided randomized block-Walsh rotation; 414 quantized Linear
layers (162 SigLIP, 126 Gemma, 126 action expert). 8-bit activations are taken per token in the rotated space.

Nothing stored is refitted or changed: codes, tables, row scales and signs are read as they are. The INT8 runtime derives one thing at
load, an INT8 copy of each table (`t8 = round(tlut · 127 / max|tlut|)`, row scale divided by the same gain; relative weight error
5.2e-5 mean, 5.9e-5 max over the 414 layers).

## One command

```bash
bash deploy/tcq_kernels/run_bench.sh        # inside a GPU job, ~40 min: every row × {10, 2} steps
```

writes `deploy/results.csv`, `deploy/table.md` and one JSON per row and step count under `deploy/results/`. The first run builds
`packed.safetensors` (the 2-bit payload, 0.68 GB) from the fit records and checks every layer.

No build step and no extra packages: the kernels are CUDA C++ compiled at load by NVRTC and launched through the CUDA driver API with
ctypes (`cuda_rt.py`), on PyTorch's context and current stream, so they are captured by `torch.cuda.graph` like PyTorch's own kernels.

## One quantized layer

`y = S_v H_m diag(r) Q H_n S_u x (+ b)`, Q = decoded table values.

| step | prefill (hundreds of tokens) | action expert (10 tokens) |
|---|---|---|
| rotate + quantize | `k_pre`: sign, block Walsh in shared memory, per-token absmax, INT8 — one launch per input tensor | same |
| decode | `k_decode`: all weight pairs in parallel (state = a 16-bit window of the tile's bitstream) into an INT8 scratch buffer, once per call, reused by every token | inside the product kernel |
| product | cuBLASLt INT8 GEMM (`torch._int_mm`), INT32 accumulators | `k_gemm_w8` (INT8 weights) or `k_gemm_tcq` (decode + product), `dp4a`, one warp per output row; token scale and row scale applied in the same launch |
| scale + rotate | `k_post`: `acc · s_token · r_row`, block Walsh, sign, bias, BF16 | same (`k_post` on the scaled accumulators) |

Fusions (all bit-identical to the unfused chain):
- **Shared input rotation**: q / k / v read the same tensor with the same input signs (the sign vector depends only on the input
  width), and so do gate / up: one `k_pre` and one product for the group. 162 of the 414 input rotations disappear (54 per tower).
- **`k_post3`**: the three output rotations of q / k / v in one launch.
- **`k_mid2`** (Gemma and expert MLP): scale + rotate of gate and of up, GELU, product, and the rotate + quantize of the down
  projection's input in one launch (shared memory 96 KB per token at width 16384, opted in through `cuFuncSetAttribute`).
  **`k_mid1`** does the same for SigLIP's `fc1 → GELU → fc2`. GELU is evaluated in the kernel and checked at load against PyTorch on
  all 65,536 BF16 inputs (falls back to a lookup table built by PyTorch if any input differs).
- Walsh passes are radix-16 (4 stages per sweep of shared memory); the first pass uses an XOR-permuted register order so the 32
  threads of a warp hit 32 different shared-memory banks. Stage order and every addition are those of the reference.

Rotations that can be removed exactly: only the shared input rotations above (162 of 414; checked on the stored signs: the input
signs are identical for all layers of one input width, the output signs for all layers of one (input, output) shape). No output
rotation cancels: gate / up share their output signs but feed an elementwise GELU and product; q / k / v feed RoPE and a per-head
attention; o / down (and SigLIP's out_proj / fc2) feed the residual stream, whose norm applies per-channel gains before the next
input rotation, which has different signs. Removing any of them needs a refit with a foldable rotation.

## Operating points

| row | weights | notes |
|---|---|---|
| `w2a8_resident` (i) | every layer decoded once at load, held as INT8 | fastest |
| `w2a8_fly` (ii) | 2-bit stream only; decoded on every call | smallest |
| `w2a8_hybrid` (iii) | backbone on the fly; expert decoded once per chunk into INT8 | |
| `w2_dense` | 2-bit weights decoded at load to dense BF16 in the original basis, FP activations | the evaluated model; FP speed and FP memory |
| `w2_fly`, `w2_hybrid` | as (ii) / (iii) with BF16 activations in the rotated space (BF16 GEMM on decoded tables) | no activation quantization |
| `fp` | official π₀.₅ PyTorch model, BF16, CUDA graphs | the existing deployment table's baseline |
| `fp_ls` | `fp` with the lockstep vision calls of the kernel rows | isolates the call-structure effect (0.5 ms); not bit-identical to `fp` on every GPU (BF16 products) |
| `fp_bv` | `fp` with the cameras batched through the vision encoder (one call, batch 3) | not in the default benchmark |

Vision encoder call structure. The official path calls the vision encoder once per camera image (256 tokens each). The kernel rows
run those calls in lockstep (`runner.lockstep_vision`): the six Linear layers of each encoder block see the tokens of all cameras in
one launch (768 tokens), while PyTorch's own operations (patch embedding, attention, norms, residuals, projector) still run per
image on per-image tensors. Because every kernel works token by token and the INT8 products are exact, the result is bit-identical
to the per-image calls. Simply batching the images (`fp_bv`) is not: PyTorch's BF16 attention arithmetic changes with the batch
size (actions move by about 2e-3 relative for FP and 3e-3 for W2A8).

## Checks (`parity.py`)

- `--stage layers`: every layer on the inputs it sees inside the reference model: decoded INT8 weights, INT8 activations, scales,
  INT32 accumulators (resident and on-the-fly) and BF16 outputs, bit for bit against `tcq_format.reference_linear` (PyTorch ops).
- `--stage actions`: final actions on the 40 held-out observations against the reference W2A8 model.
- `bench.py` also records graph-vs-eager and batched-vs-per-image differences for every row.

## What is self-contained

`cuda_rt.py`, `kernels.py` and `tcq_format.py` need only PyTorch (with its NVRTC / CUDA runtime wheels) and safetensors: the packed
format, the reference arithmetic and every kernel can be used and checked on their own. `modules.py`, `runner.py`, `bench.py` and
`parity.py` plug into this project's π₀.₅ policy loader, checkpoints and held-out observations (`$VLA_ROOT/research/scripts`,
`$VLA_ROOT/models`, default `/projects/yanlongvla`), which are not part of the public report repository.

## Files

`cuda_rt.py` NVRTC + driver runtime · `kernels.py` CUDA sources and launch helpers · `tcq_format.py` packed file, reference arithmetic ·
`modules.py` model integration · `runner.py` loading, stages, timing · `bench.py` benchmark · `parity.py` checks ·
`profile_stages.py` per-stage / per-kernel-class profile · `export_rest.py` non-Linear tensors and sizes on disk · `dev/` tuning scripts.
