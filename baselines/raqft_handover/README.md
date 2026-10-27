# RAQ-FT hand-over (raqft_on_vptq reassigned to Chamadol, 2026-10-28)

Everything needed to run the **frozen `raqft_studentq` fine-tune** on a different 2-bit base (VPTQ) and evaluate it with our paired protocol.
Numbers below are the frozen row's own, for checking a reproduction.

## Files

| File | What |
|---|---|
| `plustrain_raqft.py` | the frozen fine-tune script, byte-identical to the one that produced `raqft_studentq` (job 197014, 2026-10-10) |
| `raqft_studentq_config.json` | the row's configuration (loss, schedule, learning rates, clip, round-trip rules) |
| `raqft_studentq_contract.json` | the run's recorded contract: config, base checkpoint hash, calibration-bank hash, source-file hashes |
| `data/calibration_bank_a_256_observations.json` | calibration bank (a): 256 observations (episode, frame, noise seeds) from 60 LIBERO training episodes, dataset `physical-intelligence/libero` at revision `a4336d58…`; identical to `../data/calibration_manifest.json` |
| `data/heldout_256_observations.json` | the held-out split: 256 frames, one non-calibration training episode per task (40 tasks), selection seed 20260930; validation only, never for fitting |
| `../data/plus_units_selection_1000.json` | Plus selection units (500 Camera Viewpoints + 500 Robot Initial States) — the paired comparison set |
| `../data/plus_units_expansion_2000.json` | the added units used for confirmation (identical to our `confirmation_units.json`) |
| `story_plus_client.py` | the paired-evaluation client that produced every Plus row of ours (seeds, settle, chunking, per-instance noise seed; see below) |
| `../paired_stats.py` | paired bootstrap on identical units and noise seeds |

**Not copied:** `plustrain_raqft.py` imports runtime modules of our private checkout (`plustrain_protocol`, `plustrain_fit`, `causal_h5_common`,
`decisions_h2`, `decisions_runtime`, `calibration_data`, `structure_runtime`, `backbone_policy`, `backbone_rht`, `assemble_evening`, `story_registry`).
They are in `/projects/yanlongvla/research/scripts` on the cluster; the contract lists their SHA-256 at the time of the run. The script also expects the
VQ format of our base (RHT signs, 8-weight groups, 256×8 FP16 codebooks, per-row scale, `codes` as uint8 planes); a VPTQ base needs its own
`BackboneVQLinear`-style module exposing `codebooks`, `codes`, `row_scale`, `signs`, `masks` or an equivalent decode in `install_trainable`.

## What is trained, what is frozen

Trainable (FP32 masters, 723 tensors):

- **Codebooks** of the 288 backbone Linear layers (SigLIP 27×6, Gemma LM 18×7): master `train_books`; **FP16 round trip on every forward**
  (`books = train_books.to(float16).float()`), so the stored FP16 codebook is what the gradient sees.
- **Row scales** of the same 288 layers: master `train_scale`, FP16 round trip on every forward.
- **Backbone norms**: Gemma RMSNorm weights and SigLIP LayerNorm weight + bias, round-tripped through their stored dtype (BF16) on every forward.

Frozen: the **codes** (index planes), the RHT signs, the plane masks, the whole action expert, embeddings, projector, AdaRMS tables, Linear biases.
Because no code, shape or dtype changes, the checkpoint payload is byte-for-byte the same size before and after the fine-tune.

## Loss and schedule (`raqft_studentq`)

- `L = Σ_l (1/18) · mean_{t,h} ‖ W_{O,h} (o^q_{l,t,h} − o^fp_{l,t,h}) ‖² + 0.1 · L_H5b`
  - `o^fp`: attention readout of the **FP** expert's queries over the **FP** prefix KV; `o^q`: the same queries over the **student's** prefix KV.
    `queries = 'student'`: the queries are taken along the quantized model's own denoising trajectory (its expert, its detached prefix KV); all 10
    flow steps × 8 heads × 10 action tokens, all 18 expert layers. The metric `W_O` is the FP expert's output projection in both cases.
  - `L_H5b`: velocity-direction loss of the frozen expert over flow steps 0–5 with the student prefix KV vs the FP prefix KV; `beta = 0.1`.
- Schedule: **3 epochs × 256 observations = 768 updates**, one observation per update, **8 latents** (noise draws, `latent_seed` 20261005) accumulated per
  update; **final iterate exported** (no selection on validation). Validation curves every 64 updates on 300 Plus-train observations are logged only.
- Optimizer: Adam, three groups — codebooks **3e-4**, row scales **2e-5**, norms **3e-4**; **global gradient-norm clip 1.0** over all trainable tensors.
- Base for the frozen row: `bb_M2_ae_M2_fp32` (our 2-bit backbone + 2-bit expert, FP32 AdaRMS tables). The fine-tune only touches the backbone.
- Frozen run: RTX 6000 Ada, torch 2.7.1+cu126, 9.06 h; training readout loss 361.6 → 173.8 (H5b 4.9e-4 → 1.7e-4).

## Evaluation (paired)

`story_plus_client.py` is the exact harness: official LIBERO-Plus simulator; `random/np/torch` seed 7 per unit; `env.seed(7)`, init state 0;
10 settle steps with the dummy action; images flipped `[::-1, ::-1]` and `resize_with_pad` to 224; 10-step chunks executed 5 at a time; suite limits
220/280/300/520; per-unit noise seed `SeedSequence([7, suite_index, task, 0])` sent as `_vla_episode_seed` on the first policy call of the episode
(the server seeds torch with it; later calls continue the stream). Run FP, the base and the fine-tuned row on the same `--units`; compare with
`../paired_stats.py` (2000 draws, seed 20260930). Reference counts on the 1000 selection units: FP **765**, our 2-bit base **704**, `raqft_studentq`
**749**, its W2A8 row `a8_studentq` **741**.

## bpw accounting check

`whole_model_bpw = (linear payload bytes + 542,205,320) × 8 / 3,353,433,872`; `quantizable_linear_bpw = payload × 8 / 2,704,306,176`
(constants in `../common.py`; the payload counts every stored tensor of the 414 quantized Linear weights). The check for a fine-tuned row has two parts:

1. **Payload invariance**: the fine-tune must not change any tensor's shape or dtype. `plustrain_raqft.py` exports with a strict reload of all
   tensors and records `changed_tensors`; the exported `vq.json` must carry the same `tensor_payload_bytes` and the same `whole_model_bpw` as the base.
2. **Recomputation from the file**: recompute the payload from the safetensors offsets of the exported checkpoint (the baseline harness does this in
   `import_external.py` / `common.py`) and compare with the published figure.

Frozen row: payload 682,883,676 B → linear bpw **2.0201**, whole-model bpw **2.9226**; checkpoint SHA-256 `d700b3fc…ca05`; identical bpw to the base.
