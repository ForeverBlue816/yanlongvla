"""Offline checks of a fake-quant checkpoint against the FP policy, for any method. Actions are compared on calibration
observations with the fixed flow-matching noise (seed 0), as relative MSE = mean((a - a_fp)^2) / mean(a_fp^2).

  python jobs/debug_actions.py --checkpoint $CKPT --evaluated <fake-quant ckpt> --hf-cache <libero cache> --out x.json
      [--quick]                 structure + evaluated actions only (run before a row is evaluated; exit 3 if structure fails)
      [--scratch <dir>]         full mode: also write and reload an identity checkpoint through write_fakequant_checkpoint

Structure: the token embedding equals the source checkpoint's tied lm_head, every quantizable layer is finite, and
exactly the quantized layers differ from the source. Full mode adds FP determinism, the identity checkpoint, the
evaluated checkpoint's weights restricted to one layer group (SigLIP / language model / action expert) at a time, and
the layers with the largest relative weight error.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from calibrate import MANIFEST, load_observations
from common import load_policy, quantizable_layers, fixed_noise, write_fakequant_checkpoint, TIED_HEAD_KEY, TIED_EMBEDDING_KEY

GROUPS = {'siglip': '.vision_tower.', 'language_model': '.language_model.', 'expert': '.gemma_expert.'}


def actions(policy, observations):
    return np.stack([np.asarray(policy.infer(o, noise=fixed_noise(0))['actions'], dtype=np.float64) for o in observations])


def rel_mse(a, ref):
    return float(((a - ref) ** 2).mean() / (ref ** 2).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='source (FP) checkpoint')
    p.add_argument('--evaluated', required=True, help='fake-quant checkpoint to check')
    p.add_argument('--hf-cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--observations', type=int, default=32)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--scratch')
    a = p.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    manifest['observations'] = manifest['observations'][:a.observations]
    manifest['observation_count'] = a.observations
    observations = load_observations(manifest, Path(a.hf_cache))
    result = dict(evaluated=a.evaluated, observations=a.observations, noise_seed=0, metric='relative MSE of the 10x7 action chunk vs FP')

    policy = load_policy(a.checkpoint)
    fp = actions(policy, observations)
    layers = quantizable_layers(policy._model)
    original = {n: m.weight.detach().clone() for n, m in layers.items()}
    with safe_open(str(Path(a.checkpoint) / 'model.safetensors'), framework='pt', device='cpu') as f:
        head = f.get_tensor(TIED_HEAD_KEY)
    quantized, errors, structure = {}, {}, {}
    with safe_open(str(Path(a.evaluated) / 'model.safetensors'), framework='pt', device='cpu') as f:
        structure['embedding_equals_source'] = TIED_EMBEDDING_KEY in f.keys() and torch.equal(f.get_tensor(TIED_EMBEDDING_KEY), head)
        for n in layers:
            w = f.get_tensor(n + '.weight')
            quantized[n] = w
            errors[n] = (torch.linalg.norm(w.float() - original[n].float().cpu()) / torch.linalg.norm(original[n].float().cpu())).item()
        structure['nonfinite_layers'] = sum(not torch.isfinite(w).all() for w in quantized.values())
        structure['quantizable_layers_changed'] = sum(e > 0 for e in errors.values())
    quant = json.loads((Path(a.evaluated) / 'quant.json').read_text())
    structure['quantizable_layers_expected'] = quant.get('replaced_layers', len(layers))
    structure['ok'] = (structure['embedding_equals_source'] and structure['nonfinite_layers'] == 0
                       and structure['quantizable_layers_changed'] == structure['quantizable_layers_expected'])
    result['structure'] = structure
    result['relative_weight_error'] = {g: dict(mean=float(np.mean([e for n, e in errors.items() if s in n])),
                                               max=float(np.max([e for n, e in errors.items() if s in n]))) for g, s in GROUPS.items()}

    def with_weights(names):
        with torch.no_grad():
            for n in names:
                layers[n].weight.copy_(quantized[n].to(layers[n].weight.device, layers[n].weight.dtype))
        out = actions(policy, observations)
        with torch.no_grad():
            for n in names:
                layers[n].weight.copy_(original[n])
        return rel_mse(out, fp)

    result['evaluated_relative_mse'] = with_weights(list(layers))   # same weights as the checkpoint, inside the FP model
    if not a.quick:
        result['fp_repeat_relative_mse'] = rel_mse(actions(policy, observations), fp)
        result['group_only_relative_mse'] = {g: with_weights([n for n in layers if s in n]) for g, s in GROUPS.items()}
        result['largest_weight_error_layers'] = sorted(errors.items(), key=lambda kv: -kv[1])[:10]
        del policy
        torch.cuda.empty_cache()
        evaluated = load_policy(a.evaluated)   # the checkpoint as served: also covers everything outside the 414 layers
        result['evaluated_as_loaded_relative_mse'] = rel_mse(actions(evaluated, observations), fp)
        del evaluated
        if a.scratch:
            torch.cuda.empty_cache()
            write_fakequant_checkpoint(a.checkpoint, a.scratch, {}, dict(method='identity', bits_nominal=16))
            identity = load_policy(a.scratch)
            ident = actions(identity, observations)
            result['identity_checkpoint_relative_mse'] = rel_mse(ident, fp)
            result['identity_checkpoint_max_abs_diff'] = float(np.abs(ident - fp).max())
            del identity
            shutil.rmtree(a.scratch)
    Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not structure['ok']:
        sys.exit(3)


if __name__ == '__main__':
    main()
