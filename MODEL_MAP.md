# π₀.₅ LIBERO model map

## Current deployment accounting — 2026-09-30

Measured hook/action evidence: E011. All new experiments use one seed, overriding the earlier protocol. Both language-output heads are removed; the language embedding remains stored once. Expert head calls = 0, VLM head calls = 0, and all 256 paired single-seed action chunks remain identical at the recorded precision after removal.

| Item | Current value |
|---|---:|
| Active original unique weight elements | 3,353,433,872 |
| Eligible Linear weight elements (all components) | 2,704,306,176 |
| Eligible expert attention/MLP weight elements | 311,427,072 |
| Native-dtype payload after removing heads | 6,946,308,960 bytes |
| Payload after adding row-int8 language embedding | 6,420,690,272 bytes |
| Int8 embedding codes + FP32 row scales | 527,675,904 bytes |
| Whole-model bpw with int8 embedding (before VQ) | 15.317291 |
| Logical tensor reads / action chunk, first frozen observation | 17,434,020,060 bytes |

Storage includes every persistent tensor; container/header bytes are separate. Logical reads count three vision calls, one language pass and ten expert calls, with distinct lookup rows per embedding invocation. They are not DRAM counters and exclude KV/activation/cache effects. Original native FP32 normalization/conditioning and action/time tensors are preserved.

Budget implication, analytically calculated from the measured tensor inventory: retained tensors excluding eligible Linear weights use 1,012,077,920 bytes, a 2.414428 whole-model-bpw floor at the current native dtypes. If every eligible Linear retains at least one 8-bit index per 8 weights, even an ideal one-plane representation needs at least 3.220857 whole-model bpw before codebook/scales/mask overhead. Thus the later 3.0/2.5 whole-model targets need a separately validated retained-tensor or allocation change; they do not follow automatically from int8 embeddings and head removal. This is a budget bound under those explicit assumptions, not a measured quantized-model result. The action-expert-only comparison proceeds first as requested.

## Historical constructor audit (E002; not current deployment denominator)

The following original inventory includes the now-removed expert head. Its old arithmetic tables describe the constructor, and are superseded by the current accounting above for all new results.


Pinned openpi: `215abfb217dbac7d5f1273282331b9b1866c0479`. Evidence: E002 in EXPERIMENTS.md.

## Source configuration

- Active LIBERO action horizon: 10; model action dimension: 32. Generic Pi0Config defaults to horizon 50; the LIBERO override is authoritative.
- Flow: 10 Euler steps, dt = −0.1, t starts at 1 and decreases toward 0.
- LIBERO outputs slice the first 7 dimensions; client replan K = 5. State input has 8 dimensions.
- Prefix image/language KV is built once before the denoising loop; expert uses the cache with use_cache=False.
- Two real views plus one masked padding view pass through the image path. Source-default discrete_state_input=False for this LIBERO config.
- Sampling is decorated with torch.no_grad(): D1 needs a separately validated differentiable unroll.
- Native inference preserves selected FP32 parameters; FP16 is a method storage choice, not the native baseline dtype.

## Unique instantiated parameters (not checkpoint verification yet)

| Component | Parameters | Logical bytes from constructor dtype (not measured allocation) |
|---|---:|---:|
| siglip | 412,442,352 | 826,831,584 |
| vision_projector | 2,361,344 | 4,722,688 |
| language_embeddings | 526,647,296 | 1,053,294,592 |
| language_transformer | 1,981,884,416 | 3,963,920,384 |
| expert_attention_mlp | 311,427,072 | 622,854,144 |
| expert_adarms | 116,505,600 | 466,022,400 |
| unused_expert_lm_head | 263,323,648 | 526,647,296 |
| action_io | 66,592 | 266,368 |
| time_mlp | 2,099,200 | 8,396,800 |
| TOTAL | 3,616,757,520 | 7,472,956,256 |

The expert `lm_head` is constructed but never called by action sampling. Count it as stored overhead until conversion/runtime handling is verified; exclude it from active bytes-read. Tied language embeddings are counted once.

## Analytic bytes-read model (NOT a timing or measured memory-traffic result)

Quantize eligible Linear weights only; retained tensors use an idealized 16-bit representation. Each backbone weight is charged once, each active expert weight ten times. Excludes codebooks, codes metadata, scales, KV/activations, multi-view rereads and hardware cache effects; native BF16/FP32 runtime differs. Full storage accounting must include these overheads before any bpw claim.

| Eligible bits | Backbone once (bytes) | Expert ×10 (bytes) | Expert share |
|---|---:|---:|---:|
| 16 | 5,846,670,816 | 8,601,969,280 | 59.5348% |
| 3 | 1,958,242,272 | 3,541,279,360 | 64.3925% |
| 2 | 1,659,132,384 | 3,151,995,520 | 65.5147% |

## Ideal storage including all instantiated parameters

These are arithmetic lower bounds, excluding codebook/scale metadata. All retained tensors, including the unused expert head, are charged at 16 bits. Their minimum alone can make a whole-model target infeasible. Remove unused tensors and/or change embedding storage only after validating numerical equivalence.

| Eligible bits | Whole-model bpw | Retained share | 16-bit retained floor (bpw) |
|---|---:|---:|---:|
| 16 | 16.000000 | 25.2284% | 4.036550 |
| 3 | 6.279697 | 25.2284% | 4.036550 |
| 2 | 5.531981 | 25.2284% | 4.036550 |

## Every expert Linear

| Name | in → out | Bias | Immediate input / conditioning |
|---|---|---|---|
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.0.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.1.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.1.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.1.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.1.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.2.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.2.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.2.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.2.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.3.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.3.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.3.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.3.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.4.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.4.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.4.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.4.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.5.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.5.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.5.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.5.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.6.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.6.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.6.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.6.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.7.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.7.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.7.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.7.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.8.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.8.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.8.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.8.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.9.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.9.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.9.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.9.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.10.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.10.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.10.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.10.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.11.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.11.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.11.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.11.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.12.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.12.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.12.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.12.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.13.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.13.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.13.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.13.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.14.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.14.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.14.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.14.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.15.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.15.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.15.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.15.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.16.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.16.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.16.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.16.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.q_proj` | 1024 → 2048 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.k_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.v_proj` | 1024 → 256 | False | input_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.self_attn.o_proj` | 2048 → 1024 | False | attention output; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.17.mlp.gate_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.mlp.up_proj` | 1024 → 4096 | False | post_attention_layernorm (AdaRMS) |
| `paligemma_with_expert.gemma_expert.model.layers.17.mlp.down_proj` | 4096 → 1024 | False | gated MLP product; no immediate AdaRMS |
| `paligemma_with_expert.gemma_expert.model.layers.17.input_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.layers.17.post_attention_layernorm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.model.norm.dense` | 1024 → 3072 | True | none: consumes timestep-MLP condition; emits scale/shift/gate |
| `paligemma_with_expert.gemma_expert.lm_head` | 1024 → 257152 | False | unused in action inference |

## Pinned evidence paths

- `src/openpi/training/config.py`: pi05_libero override.
- `src/openpi/models_pytorch/pi0_pytorch.py`: action I/O, timestep MLP, sampling loop.
- `src/openpi/models_pytorch/gemma_pytorch.py`: model construction and prefix/expert routing.
- `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`: AdaRMS and gated residual.
- `src/openpi/policies/libero_policy.py`, `examples/libero/main.py`: action slicing and replan window.


## Reserved tensor audit after D3 — E027

Paper primary rate is quantizable-Linear bpw; whole-model rate is secondary. The complete41-tensor >1Melement inventory is [large_reserved_tensors.csv](results/reserved_floor/large_reserved_tensors.csv), including all37protected FP32 AdaRMS matrices andboth protected FP32 timeMLP weights. [Analytical floors](results/reserved_floor/summary.json) count packed embedding/scales and preserve norms,biases,timeMLP,AdaRMS. Int4embedding and reservedint8 are accounting scenarios, not validated deployments.

E032 uniform VQ deployment artifacts:126expertLinears /311427072originalweights, all other tensors retained from validatedint8embeddingcheckpoint. ActualexpertbpwM3/M2/M1=3.0568505489/2.0435929732/1.0303353974; allLinear14.5094708684/14.3927844360/14.2760980035; whole14.1152850620/14.0211857441/13.9270864262. Includes books/codes/row_scale/tenmasks/RHTsigns; whole denominator3353433872. Completedassembly177819; policy-levelstrictreload/actionvalidation pending. Qualityevaluation reconstructs nonserializedBF16densecaches; report theirmemory and fallbacklogicalreads separately, without fusedkernel claims.

E034 allthreeuniformdeploymentsnowpassstrictpolicyreloadandretainedtensoridentity,256heldoutobservations/noise0. LogicalcompressedreadsM3/M2/M1=12,395,449,800/12,001,005,000/11,606,560,200bytes/inference atfixedaccountingobservation; qualityfallbackactuallyuses17,436,048,840logicalparameterbytes and622,854,144bytes densecache(allthree). These are logicaltensormodels, notDRAMmeasurements; no runtimecompressedkernelclaim.

E035/E036 adaptive v2 artifact format stores only deployed conditional subset books, FP16 per-channel affine tensors, codes/row_scale/RHT signs/ten masks. Metadata-inclusive minimum full(c)/(d) is41,030,892bytes =1.0540096398555872expertbpw; original M1 budget is infeasible for those full components. Strict real-layer storage/forward pilot passes; full adaptive policy deployment remains pending.

E038 allreference(a)screeningcomplete: M3/M2/M1=193/191/193 of200, allpass. No user-definedsuccessknee found. M1actualexpert1.030335/whole13.927086; fullmodel remains mostlynative precision. Adaptiveformat/implementationverified, but noadaptivepolicyexperiment yet.
