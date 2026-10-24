"""Shared calibration: the same 256 LIBERO training observations used by every method, per-layer input moments
H = E[x x^T] for all 414 quantizable Linear layers, and the FP reference actions on those observations.

Usage:
  python calibrate.py --checkpoint /path/pi05_libero_pytorch --out /path/calibration [--hf-cache /path]

--manifest and --noise-seed exist only for held-out checks (data/heldout_manifest.json, seed 1); every method uses the
default calibration.

The observations are drawn from the LeRobot dataset physical-intelligence/libero at a pinned revision, by episode
and frame index (data/calibration_manifest.json). Expert layers see all 10 flow steps (fixed noise, seed 0).
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import quantizable_layers, load_policy, fixed_noise, dump_json, sha256

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'data/calibration_manifest.json'


def download(manifest, cache):
    from huggingface_hub import snapshot_download
    patterns = ['meta/*'] + [f"data/chunk-{e['episode_index'] // 1000:03d}/episode_{e['episode_index']:06d}.parquet" for e in manifest['episodes']]
    return Path(snapshot_download(repo_id=manifest['dataset'], repo_type='dataset', revision=manifest['revision'], allow_patterns=patterns, local_dir=cache))


def load_observations(manifest, root):
    import pyarrow.parquet as pq
    tasks = {r['task_index']: r['task'] for r in map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
    cache = {}
    result = []
    for o in manifest['observations']:
        e = o['episode_index']
        if e not in cache:
            table = pq.read_table(root / f'data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet').to_pylist()
            cache[e] = {int(r['frame_index']): r for r in table}
        row = cache[e][o['frame_index']]
        decode = lambda v: np.array(Image.open(io.BytesIO(v['bytes'])).convert('RGB'))
        result.append({'observation/image': decode(row['image']), 'observation/wrist_image': decode(row['wrist_image']),
                       'observation/state': np.asarray(row['state'], dtype=np.float32), 'prompt': tasks[int(row['task_index'])]})
    assert len(result) == manifest['observation_count']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='official pi05_libero PyTorch checkpoint directory')
    p.add_argument('--out', required=True)
    p.add_argument('--hf-cache', default=None)
    p.add_argument('--manifest', default=str(MANIFEST), help='held-out checks only; the methods use the default')
    p.add_argument('--noise-seed', type=int, default=0)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    root = download(manifest, a.hf_cache)
    observations = load_observations(manifest, root)
    policy = load_policy(a.checkpoint)
    layers = quantizable_layers(policy._model)
    moments = {n: torch.zeros((m.in_features, m.in_features), dtype=torch.float64, device='cuda') for n, m in layers.items()}
    counts = {n: 0 for n in layers}
    hooks = []
    for n, m in layers.items():
        def hook(module, args, output, *, n=n):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
            moments[n].addmm_(x.T, x)
            counts[n] += x.shape[0]
        hooks.append(m.register_forward_hook(hook))
    actions = []
    try:
        for i, obs in enumerate(observations):
            actions.append(np.asarray(policy.infer(obs, noise=fixed_noise(a.noise_seed))['actions'], dtype=np.float32))
            if (i + 1) % 32 == 0:
                print(json.dumps(dict(observations=i + 1)), flush=True)
    finally:
        for h in hooks:
            h.remove()
    out = Path(a.out)
    (out / 'hessians').mkdir(parents=True, exist_ok=True)
    index = {}
    for n in layers:
        h = (moments[n] / counts[n]).float().cpu()
        path = out / 'hessians' / (n.replace('.', '_') + '.pt')
        torch.save(dict(name=n, covariance=h, samples=counts[n], note=f"uncentered E[x x^T] over all forward calls of the layer on the {len(observations)} observations of {Path(a.manifest).name} with fixed noise seed {a.noise_seed}"), path)
        index[n] = dict(file=str(path.relative_to(out)), samples=counts[n], in_features=layers[n].in_features, out_features=layers[n].out_features)
    np.save(out / 'fp_reference_actions.npy', np.stack(actions))
    dump_json(out / 'index.json', dict(checkpoint_sha256=sha256(Path(a.checkpoint) / 'model.safetensors'), observations=len(observations), noise_seed=a.noise_seed, manifest=Path(a.manifest).name, dataset=dict(name=manifest['dataset'], revision=manifest['revision']), layers=index))
    print(json.dumps(dict(status='complete', layers=len(index), out=str(out))), flush=True)


if __name__ == '__main__':
    main()
