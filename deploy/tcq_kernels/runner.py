"""Policy loading, stage functions and CUDA-graph timing shared by profile.py, parity.py and bench.py.

Rows:
  fp          full precision as in the existing deployment table: the official pi0.5 PyTorch model in BF16, CUDA graphs, batch 1
  w2_dense    2-bit weights decoded at load to BF16 dense weights in the original basis, FP activations (the evaluated model)
  w2a8_ref    W2A8 by definition in PyTorch ops (INT8 weights resident)
  w2a8_resident / w2a8_fly / w2a8_hybrid     kernels, operating points (i) / (ii) / (iii)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.environ.get('VLA_ROOT', '/projects/yanlongvla'), 'research/scripts'))      # the project's policy loader and observation sets
sys.path.insert(0, str(Path(__file__).resolve().parent))
import modules
import tcq_format as F

ROOT = F.ROOT
ROWS = dict(fp=None, w2_dense='dense', w2a8_ref='reference', w2a8_resident='resident', w2a8_fly='fly', w2a8_hybrid='hybrid', w2_fly='fly', w2_hybrid='hybrid')
NO_A8 = ('w2_fly', 'w2_hybrid')            # kernel paths with FP (BF16) activations in the rotated space
ROWS['fp_ls'] = None                       # FP with the lockstep vision call structure of the kernel rows (bit-identical to the official path)
ROWS['fp_bv'] = None                       # FP with the camera images batched through the vision encoder (not bit-identical; for reference)
# How the camera images go through the vision encoder:
#   per_image  one call per image (the official path; FP, the dense W2 row, the PyTorch reference)
#   lockstep   the quantized layers see all images' tokens in one launch; PyTorch's own operations (patch embedding, attention,
#              norms, residuals, projector) stay per image, so the result is bit-identical to per_image (kernel rows)
#   batched    one vision-encoder call with batch = number of images (changes PyTorch's BF16 attention arithmetic slightly; fp_bv only)
VISION = {r: 'lockstep' for r in ROWS}
VISION.update(fp='per_image', w2_dense='per_image', w2a8_ref='per_image', fp_bv='batched', fp_ls='lockstep')


class _Abort(Exception):
    pass


class _Capture(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.value = None

    def forward(self, *args, **kwargs):
        self.value = kwargs.get('inputs_embeds', args[0] if args else None)
        raise _Abort


class _Return(torch.nn.Module):
    def __init__(self, value, wrap=False):
        super().__init__(); self.value, self.wrap = [value], wrap

    def forward(self, *args, **kwargs):
        if self.wrap:
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=self.value[0])
        return self.value[0]


class _Tap(torch.nn.Module):
    """Stands in for out_proj: records its input and returns it unchanged."""
    def __init__(self):
        super().__init__(); self.value = None

    def forward(self, x):
        self.value = x
        return x


def lockstep_vision(pwe, images):
    """Per-image vision-encoder calls with the quantized layers batched across the images. Everything that is not one of the six
    Linear layers of an encoder block runs through the original modules, once per image, on per-image tensors."""
    vt = pwe.paligemma.model.vision_tower.vision_model
    encoder, embeddings = vt.encoder, vt.embeddings
    n = len(images)
    cap = _Capture(); cap.layers = encoder.layers; hs = []           # the vision transformer reads encoder.layers[0]...weight.dtype
    vt.encoder = cap
    try:
        for img in images:
            try:
                pwe.embed_image(img)
            except _Abort:
                hs.append(cap.value)
    finally:
        vt.encoder = encoder
    hs0 = list(hs)
    for layer in encoder.layers:
        attn = layer.self_attn
        q_proj, k_proj, v_proj, out_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj
        normed = [layer.layer_norm1(h) for h in hs]
        x = torch.cat(normed, 0)
        q, k, v = q_proj(x).split(1, 0), k_proj(x).split(1, 0), v_proj(x).split(1, 0)
        pre_out = []
        try:
            for i in range(n):
                attn.q_proj, attn.k_proj, attn.v_proj, tap = _Return(q[i]), _Return(k[i]), _Return(v[i]), _Tap()
                attn.out_proj = tap
                attn(hidden_states=normed[i], attention_mask=None, output_attentions=False)
                pre_out.append(tap.value)
        finally:
            attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj = q_proj, k_proj, v_proj, out_proj
        o = out_proj(torch.cat(pre_out, 0)).split(1, 0)
        hs = [h + o[i] for i, h in enumerate(hs)]
        m = layer.mlp(torch.cat([layer.layer_norm2(h) for h in hs], 0)).split(1, 0)
        hs = [h + m[i] for i, h in enumerate(hs)]
    out, first = [], list(hs0)
    try:
        for i, img in enumerate(images):
            ret = _Return(hs[i], wrap=True); ret.layers = encoder.layers
            vt.embeddings, vt.encoder = _Return(first[i]), ret          # embeddings are not recomputed; post-norm and projector run per image
            out.append(pwe.embed_image(img))
    finally:
        vt.embeddings, vt.encoder = embeddings, encoder
    return out


def load(row, steps=10, **kw):
    from story_steps import load_policy
    from story_registry import FP
    policy = load_policy(dict(model=FP, steps=steps))
    policy._model.requires_grad_(False)
    info = dict(row=row, steps=steps)
    if ROWS[row] is not None:
        if row in NO_A8:
            kw = dict(kw, a8=False)
        info.update(modules.install(policy, runtime=ROWS[row], **kw))
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    return policy, info


def heldout40(row=F.ROW):
    """The 40 held-out observations of the fine-tune (indices into the 256 calibration observations, row.json)."""
    from calibration_data import load_observations
    info = json.loads((ROOT / f'models/pi05_tcq/{row}/row.json').read_text())
    idx = set(info['fit_complete']['fine_tune']['contract']['holdout'])
    out = [(i, obs) for i, (_, obs) in enumerate(load_observations()[1]) if i in idx]
    assert len(out) == 40
    return out


def fixed_noise(seed=0):
    from calibration_data import fixed_noise as fn
    return fn(seed)


class Stages:
    """Stage functions on device tensors (no host work inside): embed = image preprocessing + SigLIP + projector + token embedding;
    llm = Gemma prefix pass (KV cache); step = one denoising step of the action expert; chunk = everything for `steps` steps."""
    def __init__(self, policy, steps, vision='per_image'):
        from plustrain_int8_bench import Bench, observation_tensors
        assert vision in ('per_image', 'lockstep', 'batched')
        self.policy, self.steps, self.vision = policy, steps, vision
        self.bench = Bench(policy, steps)
        self.model = policy._model
        self._obs = observation_tensors
        self.hybrid = any(g.mode == 'chunk' for g in getattr(policy, '_tcq_groups', ()))

    def observation(self, obs):
        return self._obs(self.policy, obs)

    def embed(self, observation):
        m = self.model
        images, masks, tokens, token_masks, state = m._preprocess_observation(observation, train=False)
        if self.vision != 'per_image':
            pwe = m.paligemma_with_expert
            assert all(img.shape[0] == 1 for img in images)
            if self.vision == 'batched':                        # one SigLIP call for all cameras
                parts = iter(pwe.embed_image(torch.cat(list(images), 0)).split(1, 0))
            else:
                parts = iter(lockstep_vision(pwe, list(images)))
            pwe.embed_image = lambda img: next(parts)           # embed_prefix picks up the per-image embeddings in order
            try:
                embs, padding, attention = m.embed_prefix(images, masks, tokens, token_masks)
            finally:
                del pwe.embed_image
        else:
            embs, padding, attention = m.embed_prefix(images, masks, tokens, token_masks)
        return embs, padding, attention, state

    def llm(self, embs, padding, attention):
        m = self.model
        mask = m._prepare_attention_masks_4d(self.bench.make_masks(padding, attention))
        positions = torch.cumsum(padding, dim=1) - 1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = 'eager'
        _, kv = m.paligemma_with_expert.forward(attention_mask=mask, position_ids=positions, past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
        return kv

    def prefill(self, observation):
        embs, padding, attention, state = self.embed(observation)
        return state, padding, self.llm(embs, padding, attention)

    def step(self, s, state, padding, kv, x):
        return self.bench.step(s, state, padding, kv, x)

    def expert(self, state, padding, kv, noise):
        if self.hybrid:
            modules.decode_chunk(self.policy)
        return self.bench.unroll(state, padding, kv, noise)

    def chunk(self, observation, noise):
        return self.expert(*self.prefill(observation), noise)


def capture(fn, warm=3):
    from plustrain_int8_bench import capture as cap
    return cap(fn) if warm == 3 else cap(fn)


def host_memo():
    from plustrain_int8_bench import HostTensorMemo
    return HostTensorMemo()


def timed(graph, warmup=20, timed=200):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(timed):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); graph.replay(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end))
    v = np.array(values)
    return dict(median_ms=float(np.median(v)), p95_ms=float(np.percentile(v, 95)), mean_ms=float(v.mean()), min_ms=float(v.min()), n=timed, warmup=warmup)


CLASSES = [
    ('tcq decode', r'k_decode'),
    ('tcq rotate+quantize', r'k_pre'),
    ('tcq scale+rotate', r'k_post'),
    ('tcq decode+product', r'k_gemm_tcq'),
    ('tcq product', r'k_gemm_w8'),
    ('tcq fused', r'k_'),
    ('attention', r'flash|fmha|attention|softmax'),
    ('int8 matrix product', r'igemm|i16832gemm|s8_|int8|_int_mm|Int8|imma|i8'),
    ('matrix product', r'gemm|cutlass|cublas|Gemv|gemv|matmul|bgemm|wgmma|s1688|s16816'),
    ('norm', r'rms|layer_norm|LayerNorm|norm'),
    ('activation fn', r'gelu|Gelu|silu|Silu|swiglu'),
    ('copies / casts', r'copy|cast|Copy|fill|zero|memset|Memset|memcpy|Memcpy'),
    ('other elementwise', r'.*'),
]


def kernel_classes(graph):
    """CUPTI trace of one graph replay: device time per kernel class, launches, wall time and launch gaps."""
    import re
    from collections import Counter, defaultdict
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        start.record(); graph.replay(); end.record(); end.synchronize()
    wall = start.elapsed_time(end)
    by, launches, top = defaultdict(float), Counter(), Counter()
    for e in prof.events():
        if e.device_type.name != 'CUDA':
            continue
        dur = e.time_range.elapsed_us() / 1000
        cls = next(c for c, pat in CLASSES if re.search(pat, e.name))
        by[cls] += dur; launches[cls] += 1; top[e.name[:90]] += dur
    total = sum(by.values())
    return dict(wall_ms=wall, kernel_ms=total, gaps_ms=max(wall - total, 0.0), launches=sum(launches.values()),
                by_class_ms={k: round(v, 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])}, launches_by_class=dict(launches),
                top=[(n, round(d, 3)) for n, d in top.most_common(10)])
