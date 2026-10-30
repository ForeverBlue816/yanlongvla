"""Step 1: shared calibration. The same 256 LIBERO training observations as every π0.5 row (../data/calibration_manifest.json,
LeRobot dataset physical-intelligence/libero at a pinned revision), each sent through the official policy exactly as the
evaluation clients send it, with flow-matching noise seed SeedSequence([7, index]). Per quantizable Linear layer the
uncentered input moment H = E[x x^T] over every forward call (the DiT layers see all 4 flow steps).

  backbone layers (216): from the libero_spatial checkpoint only (the backbone is identical in all four suites)
  action-head layers (216 per suite): from each suite's own checkpoint

Usage:  python calibrate.py --checkpoints ./checkpoints --out ./calibration [--hf-cache ./hf] [--suites libero_spatial ...]
  -> calibration/backbone/{hessians/*.pt, index.json}, calibration/head_<suite>/{hessians/*.pt, index.json}
index.json carries checkpoint_sha256 of the checkpoint the moments came from (the external quantizers check it).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from groot_common import SUITES, load_policy, quantizable_layers, part_of, infer, sha256, dump_json, parent_module

_pi05_calibrate = parent_module('calibrate')               # ../calibrate.py: the π0.5 harness loader (same observations)
download, load_observations = _pi05_calibrate.download, _pi05_calibrate.load_observations

MANIFEST = HERE.parent / 'data/calibration_manifest.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', required=True, help='output of prepare_checkpoints.py')
    p.add_argument('--out', required=True)
    p.add_argument('--hf-cache', default=None)
    p.add_argument('--dataset-root', default=None, help='an existing local copy of the LeRobot dataset (meta/, data/); skips the download')
    p.add_argument('--suites', nargs='+', default=list(SUITES))
    p.add_argument('--limit', type=int, default=None, help='tests only: first N observations (the methods use all 256)')
    a = p.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    observations = load_observations(manifest, Path(a.dataset_root) if a.dataset_root else download(manifest, a.hf_cache))
    assert len(observations) == 256
    observations = observations[:a.limit] if a.limit else observations
    for suite in a.suites:
        keys = (['backbone'] if suite == 'libero_spatial' else []) + [f'head_{suite}']
        if all((Path(a.out) / k / 'index.json').exists() for k in keys):
            print(json.dumps(dict(suite=suite, skipped='complete')), flush=True); continue
        model_dir = Path(a.checkpoints) / suite
        policy = load_policy(model_dir)
        layers = {n: m for n, m in quantizable_layers(policy.model).items() if part_of(n) == 'head' or suite == 'libero_spatial'}
        moments = {n: torch.zeros((m.in_features, m.in_features), dtype=torch.float64, device='cuda') for n, m in layers.items()}
        counts = {n: 0 for n in layers}
        hooks = []
        for n, m in layers.items():
            def hook(module, args, *, n=n):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
                moments[n].addmm_(x.T, x)
                counts[n] += x.shape[0]
            hooks.append(m.register_forward_pre_hook(hook))
        try:
            with torch.inference_mode():
                for i, obs in enumerate(observations):
                    infer(policy, dict(obs, _vla_episode_seed=int(np.random.SeedSequence([7, i]).generate_state(1)[0])))
                    if (i + 1) % 32 == 0:
                        print(json.dumps(dict(suite=suite, observations=i + 1)), flush=True)
        finally:
            for h in hooks:
                h.remove()
        for key in keys:
            part = 'backbone' if key == 'backbone' else 'head'
            out = Path(a.out) / key
            (out / 'hessians').mkdir(parents=True, exist_ok=True)
            index = {}
            for n in [n for n in layers if part_of(n) == part]:
                path = out / 'hessians' / (n.replace('.', '_') + '.pt')
                torch.save(dict(name=n, covariance=(moments[n] / counts[n]).float().cpu(), samples=counts[n],
                                note='uncentered E[x x^T] over all forward calls on the 256 calibration observations, noise seed SeedSequence([7, index])'), path)
                index[n] = dict(file=str(path.relative_to(out)), samples=counts[n], in_features=layers[n].in_features, out_features=layers[n].out_features)
            dump_json(out / 'index.json', dict(checkpoint_sha256=sha256(model_dir / 'model.safetensors'), suite=suite, part=part, observations=len(observations),
                                               noise_seed='SeedSequence([7, observation_index])', manifest=MANIFEST.name, dataset=dict(name=manifest['dataset'], revision=manifest['revision']),
                                               gpu=torch.cuda.get_device_name(), layers=index))
            print(json.dumps(dict(calibrated=key, layers=len(index))), flush=True)
        del policy, moments
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
