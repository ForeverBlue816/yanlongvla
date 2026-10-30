"""Shared pieces of the GR00T N1.7 baseline harness: checkpoints, the quantizable-layer rule, policy loading, fake-quant
checkpoints and bpw accounting.

GR00T N1.7 for LIBERO is released as four fine-tuned checkpoints, one per suite (nvidia/GR00T-N1.7-LIBERO, subfolders
libero_spatial / libero_object / libero_goal / libero_10). The backbone (Qwen3-VL vision tower + language-model layers 0-15)
is byte-identical in all four (prepare_checkpoints.py verifies this); the action head differs per suite. A method therefore
quantizes the backbone once and the action head four times, and every suite is evaluated with its own checkpoint.

A "fake-quant checkpoint" is a prepared suite checkpoint with the quantizable Linear weights replaced by their dequantized
values (stored dtype BF16). Every method, including ours, runs through the same official Gr00tPolicy in BF16, so success
rates compare quantization error only.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
REPO_ID = 'nvidia/GR00T-N1.7-LIBERO'
REVISION = '2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21'          # snapshot used for every row of the report
ISAAC_GROOT_COMMIT = 'd2b7e75b937e3ec9aa5dbc798f08b89692c49734'  # github.com/NVIDIA/Isaac-GR00T
OFFICIAL_BACKBONE = 'nvidia/Cosmos-Reason2-2B'                  # gated (accept the licence, HF_TOKEN); config + processor only, all backbone weights are in the checkpoint; used by every reference row
PUBLIC_BACKBONE = 'Qwen/Qwen3-VL-2B-Instruct'                   # same architecture; NOT used for the reference rows (the processor config also comes from the gated repo)
EMBODIMENT = 'LIBERO_PANDA'
N_ACTION_STEPS = 8                                              # official LIBERO rollout: 8 of the 16-step action window per call

# ---- the quantizable-layer rule (432 Linear per suite checkpoint) ---------------------------------------------------
BACKBONE_PREFIXES = ('backbone.model.model.visual.', 'backbone.model.model.language_model.layers.')
HEAD_PREFIXES = ('action_head.vl_self_attention.', 'action_head.model.transformer_blocks.')
# Not quantized: the DiT AdaLN linears (norm1.linear) and proj_out_* depend only on the flow timestep (4 fixed steps), so
# they are per-step constants in deployment; lm_head (never called), embeddings, norms, the per-embodiment state/action
# encoders and decoders (3-D category weights) and position/timestep embeddings stay BF16.
EXCLUDE = ('norm1.linear', 'proj_out', 'lm_head', 'embed')
BACKBONE_LAYERS = 216      # vision tower 104 + language model layers 0-15 112
HEAD_LAYERS = 216          # vl_self_attention 24 + DiT attention / FFN 192
QUANTIZED_PARAMETERS = 2_340_421_632   # per suite checkpoint: backbone 216 + action head 216 Linear weights; asserted at load time

# ---- accounting constants (asserted against the prepared checkpoint by accounting()) ---------------------------------
RETAINED_BYTES = 1_607_188_736         # every other tensor used by action inference, at its stored dtype (lm_head excluded)
ACTIVE_PARAMETERS = 3_144_016_000      # all parameters used by action inference (lm_head excluded)


def parent_module(name):
    """Import a module of the π0.5 harness (../<name>.py) by path; this directory has files with the same names."""
    import importlib.util
    import sys
    parent = str(HERE.parent)
    if parent not in sys.path:
        sys.path.append(parent)                     # ../<name>.py imports ../common.py
    spec = importlib.util.spec_from_file_location(f'pi05_{name}', HERE.parent / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def part_of(name):
    if name.startswith(BACKBONE_PREFIXES):
        return 'backbone'
    if name.startswith(HEAD_PREFIXES):
        return 'head'
    return None


def is_quantizable(name, module):
    return isinstance(module, torch.nn.Linear) and part_of(name) is not None and not any(e in name for e in EXCLUDE)


def quantizable_layers(model):
    layers = {n: m for n, m in model.named_modules() if is_quantizable(n, m)}
    assert len(layers) == BACKBONE_LAYERS + HEAD_LAYERS, len(layers)
    assert sum(part_of(n) == 'backbone' for n in layers) == BACKBONE_LAYERS
    assert sum(m.weight.numel() for m in layers.values()) == QUANTIZED_PARAMETERS, sum(m.weight.numel() for m in layers.values())
    return layers


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.partial{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def load_policy(model_dir, device='cuda'):
    """Official Gr00tPolicy on a prepared (or fake-quant) suite checkpoint directory."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    return Gr00tPolicy(embodiment_tag=EMBODIMENT, model_path=str(model_dir), device=device, strict=True)


def groot_observation(obs):
    """Wire format of the evaluation clients -> Gr00tPolicy input. obs: observation/image and observation/wrist_image uint8
    (256, 256, 3), already flipped [::-1, ::-1] as GR00T's own LIBERO wrapper does; observation/state float32 (8,) = eef xyz,
    axis-angle (3), gripper qpos (2); prompt str."""
    import numpy as np
    img = np.asarray(obs['observation/image'], dtype=np.uint8)
    wrist = np.asarray(obs['observation/wrist_image'], dtype=np.uint8)
    state = np.asarray(obs['observation/state'], dtype=np.float32)
    assert img.shape == (256, 256, 3) and wrist.shape == (256, 256, 3) and state.shape == (8,), (img.shape, wrist.shape, state.shape)
    s = lambda sl: state[sl].reshape(1, 1, -1)
    return {'video': {'image': img[None, None], 'wrist_image': wrist[None, None]},
            'state': {'x': s(slice(0, 1)), 'y': s(slice(1, 2)), 'z': s(slice(2, 3)), 'roll': s(slice(3, 4)), 'pitch': s(slice(4, 5)), 'yaw': s(slice(5, 6)), 'gripper': s(slice(6, 8))},
            'language': {'annotation.human.action.task_description': [[str(obs['prompt'])]]}}


def env_action(chunk):
    """GR00T action window (T, 7) in the dataset convention -> LIBERO env actions (gripper [0,1] -> [-1,1], sign, inverted;
    gr00t/eval/sim/LIBERO/libero_env.py)."""
    import numpy as np
    a = np.array(chunk, dtype=np.float32)
    a[..., -1] = -np.sign(2 * a[..., -1] - 1)
    return a


def infer(policy, obs):
    """One policy call: seeds the flow-matching noise from `_vla_episode_seed` (first call of an episode) exactly like the
    report's servers, returns the (16, 7) LIBERO action window."""
    import numpy as np
    obs = dict(obs)
    seed = obs.pop('_vla_episode_seed', None)
    if seed is not None:
        torch.manual_seed(int(seed))
        policy.reset()
    action, _ = policy.get_action(groot_observation(obs))
    get = lambda k: action[f'action.{k}'] if f'action.{k}' in action else action[k]
    chunk = np.concatenate([np.asarray(get(k)[0], dtype=np.float32) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')], axis=-1)
    assert chunk.ndim == 2 and chunk.shape[1] == 7 and np.isfinite(chunk).all(), chunk.shape
    return env_action(chunk)


def read_tensors(model_dir):
    from safetensors.torch import load_file
    return load_file(str(Path(model_dir) / 'model.safetensors'))


def accounting(linear_payload_bytes, method, bits_nominal, extra=None):
    """Same convention as the π0.5 harness and our GR00T rows: every stored tensor of the quantized Linear layers counts
    (codes, codebooks, scales, zero points, signs, permutations); every other tensor used by action inference is charged at
    its stored dtype; lm_head is excluded from both numerator and denominator."""
    result = dict(method=method, bits_nominal=bits_nominal, linear_payload_bytes=int(linear_payload_bytes),
                  quantized_linear_bpw=linear_payload_bytes * 8 / QUANTIZED_PARAMETERS,
                  whole_model_bpw=(linear_payload_bytes + RETAINED_BYTES) * 8 / ACTIVE_PARAMETERS,
                  retained_bytes=RETAINED_BYTES, active_parameters=ACTIVE_PARAMETERS, quantized_parameters=QUANTIZED_PARAMETERS,
                  convention='per suite checkpoint; all stored tensors of the 432 quantized Linear layers counted; other inference tensors at stored dtype (BF16); lm_head excluded')
    if extra:
        result.update(extra)
    return result


def check_constants(model_dir):
    """Recompute retained bytes / active parameters / quantized parameters from a prepared checkpoint and assert the constants."""
    from safetensors import safe_open
    retained = active = quantized = 0
    with safe_open(str(Path(model_dir) / 'model.safetensors'), framework='pt') as f:
        for k in f.keys():
            if 'lm_head' in k:
                continue
            t = f.get_slice(k); n = 1
            for d in t.get_shape():
                n *= d
            size = {'BF16': 2, 'F16': 2, 'F32': 4, 'I64': 8, 'I32': 4, 'U8': 1, 'I8': 1, 'BOOL': 1}[t.get_dtype()]
            active += n
            name = k.removesuffix('.weight')
            if k.endswith('.weight') and part_of(name) and not any(e in name for e in EXCLUDE) and len(t.get_shape()) == 2:
                quantized += n
            else:
                retained += n * size
    assert (quantized, retained, active) == (QUANTIZED_PARAMETERS, RETAINED_BYTES, ACTIVE_PARAMETERS), (quantized, retained, active)
    return dict(quantized_parameters=quantized, retained_bytes=retained, active_parameters=active)


def write_fakequant_checkpoint(source_dir, out_dir, replaced, info):
    """Copy a prepared suite checkpoint, replace the given weights (name -> dequantized tensor, cast to the stored dtype),
    write quant.json. Non-weight files (config, processor, statistics) are copied."""
    from safetensors.torch import save_file
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = read_tensors(source_dir)
    for name, value in replaced.items():
        key = name + '.weight'
        assert key in tensors and tuple(value.shape) == tuple(tensors[key].shape), key
        assert torch.isfinite(value).all(), key
        tensors[key] = value.to(tensors[key].dtype).contiguous().cpu()
    tmp = out_dir / 'model.safetensors.partial'
    save_file(tensors, str(tmp), metadata={'format': 'pt'})
    tmp.replace(out_dir / 'model.safetensors')
    for f in source_dir.iterdir():
        if f.name.startswith('model') or f.name == 'quant.json':
            continue
        dst = out_dir / f.name
        if not dst.exists():
            (shutil.copytree if f.is_dir() else shutil.copy)(f, dst)
    info = dict(info, replaced_layers=len(replaced), checkpoint_sha256=sha256(out_dir / 'model.safetensors'), source_checkpoint_sha256=sha256(source_dir / 'model.safetensors'))
    dump_json(out_dir / 'quant.json', info)
    return info
