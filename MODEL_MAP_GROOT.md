# MODEL_MAP_GROOT — nvidia/GR00T-N1.7-LIBERO (2026-10-15)

Source: checkpoint safetensors headers and `config.json` of the released per-suite checkpoints (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; one fine-tuned model per suite, identical architecture), code in `Isaac-GR00T` (commit d2b7e75, 2026-10-07). Numbers from `results/groot_model_map.json` (`scripts/groot_model_map.py`).

## Components and parameter counts (libero_10 checkpoint; the other three are identical in shape)

| Component | Tensors | Parameters | BF16 bytes | Role |
|---|---:|---:|---:|---|
| backbone: SigLIP2-style vision tower (Qwen3-VL `visual`) | 315 | 407.0 M | 0.81 GB | 24 blocks, width 1024, 16 heads, patch 16; merger to 2048 with deepstack taps at blocks 5/11/17 |
| backbone: Qwen3-VL language model, layers 0–15 only | 176 | 805.4 M | 1.61 GB | `select_layer = 16`: layers 16–27 of the 28-layer Cosmos-Reason2-2B LLM are dropped at load; hidden 2048, 16 q-heads / 8 kv-heads (head 128), MLP 6144 |
| backbone: token embedding | 1 | 311.2 M | 0.62 GB | 151,936 × 2048 |
| backbone: lm_head (stored, never called) | 1 | 311.2 M | 0.62 GB | removed from the deployable accounting, as for π₀.₅ |
| action head: `vlln` + `vl_self_attention` | 66 | 201.4 M | 0.40 GB | LayerNorm(2048) then 4 self-attention blocks (2048, 32 heads × 64, MLP 8192) over the backbone tokens; the output is what every DiT cross-attention block reads |
| action head: DiT (`AlternateVLDiT`), 32 blocks | 448 | 1,082.7 M | 2.17 GB | inner dim 1536 (32 heads × 48); see block table |
| action head: DiT in/out, timestep encoder | 8 | 9.1 M | 0.02 GB | `proj_out_1` (1536→3072 AdaLN-out), `proj_out_2` (1536→1024), timestep MLP |
| action head: state encoder (`CategorySpecificMLP`) | 4 | 54.7 M | 0.11 GB | per-embodiment weights (32 categories × 132 → 1024 → 1536); only the LIBERO category is used |
| action head: action encoder (`MultiEmbodimentActionEncoder`) | 6 | 233.1 M | 0.47 GB | per-embodiment (32 categories), 132-d padded actions + timestep → 1536 |
| action head: action decoder (`CategorySpecificMLP`) | 4 | 37.9 M | 0.08 GB | 1024 → 1024 → 132 per embodiment |
| action head: position embedding | 1 | 1.6 M | — | 1024 positions × 1536 |
| **Total stored** | 1,031 | **3,455.2 M** | 6.91 GB | |
| **Used by action inference** (lm_head removed) | | **3,144.0 M** | 6.29 GB | of which the 31 unused embodiment categories of the three per-embodiment MLPs hold ≈ 316 M parameters that could be dropped at deployment (not counted as quantizable) |

## Quantization candidates (2-D weights, nn.Linear semantics)

| Group | Layers | Parameters | Shapes (out × in) |
|---|---:|---:|---|
| backbone vision tower | 104 | 402.7 M | 24 × {qkv 3072×1024, proj 1024×1024, fc1 4096×1024, fc2 1024×4096}; merger/deepstack 4 × (4096×4096) + 4 × (2048×4096) |
| backbone LLM layers 0–15 | 112 | 805.3 M | 16 × {q 2048×2048, k 1024×2048, v 1024×2048, o 2048×2048, gate 6144×2048, up 6144×2048, down 2048×6144} |
| action head `vl_self_attention` | 24 | 201.3 M | 4 × {q,k,v,o 2048×2048; fc 8192×2048, 2048×8192} |
| action head DiT blocks | 224 | 1,082.1 M | self-attention blocks (odd): q,k,v,o 1536×1536; cross-attention blocks (even): q,o 1536×1536, k,v 1536×2048; every block: FFN 6144×1536 + 1536×6144, AdaLN linear 3072×1536 |
| action head DiT out | 2 | 6.3 M | 3072×1536, 1024×1536 |
| **All Linear** | **466** | **2,497.7 M** | 79% of the inference parameters; backbone 216 layers / 1,208 M, action head 250 layers / 1,290 M |

Not quantized (as for π₀.₅): token embedding (int8 row-absmax at deployment), norms, the per-embodiment MLPs (3-D category weights), position/timestep embeddings. The π₀.₅ analogue of "backbone" is vision tower + LLM (1,208 M quantizable); of "expert" it is `vl_self_attention` + DiT (1,290 M), so the action head is a much larger share than in π₀.₅ (0.31 B of 2.70 B).

## How the DiT reads the backbone

1. Backbone forward (Qwen3-VL, 16 LLM layers): images (two 256×256 views after SmallestMaxSize 256 → center crop 0.95 → resize 256) and the lower-cased, punctuation-stripped instruction through the chat template; `hidden_states[-1]` of layer 15 gives the backbone tokens (S × 2048) with an `image_mask` marking image tokens.
2. `vlln` (LayerNorm) and `vl_self_attention` (4 blocks) refine the tokens; the result is the encoder sequence for the whole flow-matching solve (computed once per inference, 4 flow steps).
3. State (8-d, padded to 132) → `state_encoder`; noisy action window (40 × 132, LIBERO uses 16 valid steps) + timestep → `action_encoder`; concatenated with position embeddings as the DiT token sequence.
4. DiT blocks, `AlternateVLDiT`:

| Block | Attention | Keys/values | AdaLN |
|---|---|---|---|
| 0, 4, 8, …, 28 | cross-attention | non-image backbone tokens (instruction, chat-template tokens) | timestep scale/shift |
| 2, 6, 10, …, 30 | cross-attention | image backbone tokens | timestep scale/shift |
| 1, 3, 5, …, 31 | self-attention | action tokens only | timestep scale/shift |

   Cross-attention positions: 16 blocks (8 over text tokens, 8 over image tokens), interleaved with 16 self-attention blocks. AdaLN (`AdaLayerNorm`: SiLU → Linear(1536→3072) → scale, shift) is conditioned on the flow timestep only (discretized into 1000 buckets); with the fixed 4-step inference schedule the AdaLN outputs are constants per step and can be precomputed per block, exactly as the π₀.₅ AdaRMS tables.
5. Output: `norm_out` with AdaLN-out, `proj_out_2` (→1024), `action_decoder` → velocity for the 132-d padded action; 4 Euler steps (`num_inference_timesteps = 4`); first 16 steps of the window are the LIBERO action (x, y, z, roll, pitch, yaw, gripper), 8 executed per policy call in the official rollout.

Readout for the π₀.₅-style analysis: the "reader" is the cross-attention of the 16 even DiT blocks over the 2048-d backbone tokens (K, V projections 1536×2048 per block); hooks in `scripts/groot_hooks.py` capture, per flow step and block, the attention output before the output projection, the block output, the AdaLN scale/shift, and the encoder tokens. Unused heads: `lm_head` (never called); no text generation path is used.

## Evaluation harness

`scripts/groot_bridge_server.py` (envs/groot, torch 2.9.0+cu128, transformers 4.57.3, flash-attn 2.8.3) serves a suite's checkpoint over the report's websocket protocol; `scripts/groot_libero_client.py` (envs/libero) runs the report's LIBERO protocol (seed 7, init states in order, 10 settle steps, step limits 220/280/300/520) with GR00T's native 256×256 flipped frames and executes 8 of the 16-step window per call; `scripts/plustrain_groot.py` orchestrates the medium-1000 run on one L40S. Backbone config source: the checkpoint stores all backbone weights, but the loader builds the architecture from `config.model_name = nvidia/Cosmos-Reason2-2B`, a gated repo. Gated access is available on the cluster since 2026-10-15 (E115) and the official name is used; the public `Qwen/Qwen3-VL-2B-Instruct` was verified byte-identical in config, tokenizer, chat template and preprocessor, so it remains the documented fallback (`bridge_manifest.json` records which was used).
