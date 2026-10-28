"""Shared pieces of the baseline harness: model loading, the quantizable-layer rule, fake-quant checkpoints, bpw accounting.

Everything here uses the public openpi PyTorch port of pi0.5-LIBERO. A "fake-quant checkpoint" is the official checkpoint
with the 414 quantizable Linear weights replaced by their dequantized values (BF16). All methods, including ours, are
evaluated through the same BF16 dense runtime, so success rates compare quantization error only, never kernels.
"""
import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

# ---- accounting constants (pi0.5-LIBERO, see MODEL_MAP.md of the report repository) ---------------------------------
ACTIVE_ORIGINAL_ELEMENTS = 3_353_433_872      # all parameters read by action inference (both language heads removed)
BACKBONE_QUANTIZABLE_ELEMENTS = 2_392_879_104  # 288 Linear weights: SigLIP encoder (27 x 6) + Gemma language model (18 x 7)
EXPERT_QUANTIZABLE_ELEMENTS = 311_427_072      # 126 Linear weights: Gemma action expert (18 x 7)
QUANTIZABLE_ELEMENTS = BACKBONE_QUANTIZABLE_ELEMENTS + EXPERT_QUANTIZABLE_ELEMENTS
RETAINED_BYTES = 542_205_320                   # everything else, as stored in our deployable 2-bit checkpoint (safetensors header):
                                               # int8 row-absmax token embedding + FP32 row scales 527,675,904; SigLIP patch/position
                                               # embeddings, norms and Linear biases, LM norms, projector 9,716,488; expert norms,
                                               # FP32 AdaRMS tables, action/time projections 4,812,928. (Corrected 2026-10-14: an
                                               # earlier value, 1,012,077,920, was the retained set of the BF16 accounting checkpoint
                                               # and did not reproduce our published 2.9226 from the 2.0201 linear bpw.)
EXPERT_CALLS_PER_INFERENCE = 10                # flow-matching steps; the backbone prefix is computed once

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
PLUS_LIMITS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)   # max policy steps per suite
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLE_STEPS = 10
REPLAN_STEPS = 5

# The official checkpoint stores the PaliGemma token embedding only under the (tied) lm_head key.
TIED_HEAD_KEY = 'paligemma_with_expert.paligemma.lm_head.weight'
TIED_EMBEDDING_KEY = 'paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def is_quantizable(name, module):
    """The 414 Linear layers quantized by every method (backbone 288 + expert 126); embeddings, projector, heads,
    action/time projections and norms are never quantized."""
    if not isinstance(module, nn.Linear):
        return False
    if name.endswith(('.input_layernorm.dense', '.post_attention_layernorm.dense')):   # expert AdaRMS norms are Linear modules; norms stay unquantized
        return False
    return ('.vision_tower.vision_model.encoder.layers.' in name or '.language_model.layers.' in name or '.gemma_expert.model.layers.' in name)


def quantizable_layers(model):
    layers = {n: m for n, m in model.named_modules() if is_quantizable(n, m)}
    assert len(layers) == 414, len(layers)
    assert sum(m.weight.numel() for m in layers.values()) == QUANTIZABLE_ELEMENTS
    return layers


def native_config():
    from openpi.training import config
    cfg = config.get_config('pi05_libero')
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))


def build_model(checkpoint_dir, device='cuda'):
    """PI0Pytorch with the given checkpoint's weights (official or fake-quant)."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_model
    cfg = native_config()
    model = PI0Pytorch(cfg.model)
    from safetensors import safe_open
    with safe_open(str(Path(checkpoint_dir) / 'model.safetensors'), framework='pt') as f:
        assert TIED_HEAD_KEY in f.keys() or TIED_EMBEDDING_KEY in f.keys(), 'checkpoint has no token embedding (it would load randomly initialized)'
    missing, unexpected = load_model(model, str(Path(checkpoint_dir) / 'model.safetensors'), strict=False)
    assert not unexpected, unexpected[:5]
    assert all(k.endswith('lm_head.weight') for k in missing), missing[:5]   # heads may be absent (never used by action inference)
    model.requires_grad_(False)
    model.eval()
    return cfg, model.to(device)


def load_policy(checkpoint_dir, device='cuda'):
    from openpi import transforms
    from openpi.policies import policy as policy_module
    from openpi.training import checkpoints
    cfg, model = build_model(checkpoint_dir, device)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    norms = checkpoints.load_norm_stats(Path(checkpoint_dir) / 'assets', data.asset_id)
    return policy_module.Policy(model, transforms=[transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs, transforms.Normalize(norms, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs],
                                output_transforms=[*data.model_transforms.outputs, transforms.Unnormalize(norms, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs],
                                sample_kwargs={'num_steps': 10}, metadata=cfg.policy_metadata, is_pytorch=True, pytorch_device=device)


def fixed_noise(seed):
    """The fixed flow-matching noise used for calibration (seed 0) and offline checks (10 x 32 action tokens)."""
    return np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)


def accounting(linear_payload_bytes, method, bits_nominal, extra=None):
    """Whole-model bpw under the shared convention: every stored tensor counts (codes, codebooks, scales, zero points);
    retained tensors are charged at the deployable dtypes of the report repository."""
    result = dict(method=method, bits_nominal=bits_nominal, linear_payload_bytes=int(linear_payload_bytes),
                  quantizable_linear_bpw=linear_payload_bytes * 8 / QUANTIZABLE_ELEMENTS,
                  whole_model_bpw=(linear_payload_bytes + RETAINED_BYTES) * 8 / ACTIVE_ORIGINAL_ELEMENTS,
                  retained_bytes=RETAINED_BYTES, active_original_elements=ACTIVE_ORIGINAL_ELEMENTS,
                  convention='all stored tensors counted; retained tensors at deployable dtypes (int8 embedding, BF16 norms/projections, FP32 tables); denominator = active original elements with both language heads removed')
    if extra:
        result.update(extra)
    return result


def hessian_damping(method):
    """Diagonal damping added to the shared Hessian, as a fraction of its mean diagonal (mentor, 2026-10-26: the same value
    for every Hessian-based row; recorded in each quant.json). GPTQ: percdamp 0.01; VPTQ: percdamp 0.01 on the normalized
    H; QuIP#/QTIP: regularize_H divides H by its mean diagonal and adds sigma_reg 0.01; AQLM: 0.01 (departure from the
    official code, which adds none). AWQ and RTN solve nothing with H (AWQ only scores candidates with it): no damping."""
    if method.startswith(('gptq', 'vptq', 'quip', 'qtip', 'aqlm')):
        return dict(fraction_of_mean_diagonal=0.01)
    return dict(fraction_of_mean_diagonal=None, reason='no inverse or solve of the Hessian')


def write_fakequant_checkpoint(source_dir, out_dir, replaced, info):
    """Copy the source checkpoint, replace the given weights (name -> dequantized tensor, cast to the stored dtype),
    drop the unused language heads, and write quant.json."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(source_dir / 'model.safetensors'))
    if TIED_HEAD_KEY in tensors:   # keep the token embedding under its input key before the heads are dropped
        assert TIED_EMBEDDING_KEY not in tensors or torch.equal(tensors[TIED_EMBEDDING_KEY], tensors[TIED_HEAD_KEY])
        tensors[TIED_EMBEDDING_KEY] = tensors[TIED_HEAD_KEY]
    assert TIED_EMBEDDING_KEY in tensors, 'source checkpoint has no token embedding'
    for k in list(tensors):
        if k.endswith('lm_head.weight'):
            del tensors[k]
    for name, value in replaced.items():
        key = name + '.weight'
        assert key in tensors, key
        assert value.shape == tensors[key].shape, key
        assert torch.isfinite(value).all(), key
        tensors[key] = value.to(tensors[key].dtype).contiguous().cpu()
    save_file(tensors, str(out_dir / 'model.safetensors'))
    if not (out_dir / 'assets').exists():
        shutil.copytree(source_dir / 'assets', out_dir / 'assets')
    for name in ('config.json',):
        if (source_dir / name).exists() and not (out_dir / name).exists():
            shutil.copy(source_dir / name, out_dir / name)
    info = dict(info, hessian_damping=hessian_damping(info.get('method', '')), replaced_layers=len(replaced), checkpoint_sha256=sha256(out_dir / 'model.safetensors'), source_checkpoint_sha256=sha256(source_dir / 'model.safetensors'))
    (out_dir / 'quant.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.partial{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)
