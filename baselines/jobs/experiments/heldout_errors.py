"""Per-layer quantization error on the calibration Hessians and on held-out Hessians (64 observations from episodes
outside the calibration set, noise seed 1; data/heldout_manifest.json). Tests whether error that a method placed in
directions the calibration never exercised shows up on other inputs.

Each source is a fake-quant checkpoint directory (model.safetensors) or a directory of <layer>.safetensors files with a
`weight` tensor (quantize_*.py outputs); layers missing from a layer directory are skipped.

Usage:
  python jobs/experiments/heldout_errors.py --layers $RUN/layers --calibration $RUN/calibration \
      --heldout $RUN/experiments/heldout/calibration --source qtip=$RUN/checkpoints/qtip_w2 --source quip_t0=/dir ... --out out.json
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--heldout', required=True)
    p.add_argument('--source', action='append', required=True, help='label=checkpoint dir or layer dir')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    listing = json.loads((Path(a.layers) / 'layers.json').read_text())['layers']
    cal = json.loads((Path(a.calibration) / 'index.json').read_text())['layers']
    held = json.loads((Path(a.heldout) / 'index.json').read_text())['layers']
    sources = dict(s.split('=', 1) for s in a.source)
    handles = {k: safe_open(str(Path(v) / 'model.safetensors'), 'pt') for k, v in sources.items() if (Path(v) / 'model.safetensors').exists()}
    rel = lambda e, w, h: float(((e @ h) * e).sum() / ((w @ h) * w).sum())
    layers = {}
    for item in listing:
        name = item['name']
        w = load_file(str(Path(a.layers) / item['file']))['weight'].float().cuda()
        hc = torch.load(Path(a.calibration) / cal[name]['file'], map_location='cuda', weights_only=True)['covariance'].float()
        hh = torch.load(Path(a.heldout) / held[name]['file'], map_location='cuda', weights_only=True)['covariance'].float()
        row = {}
        for label, path in sources.items():
            if label in handles:
                q = handles[label].get_tensor(name + '.weight')
            elif (Path(path) / item['file']).exists():
                q = load_file(str(Path(path) / item['file']))['weight']
            else:
                continue
            e = q.float().cuda() - w
            row[label] = dict(weight=float(e.norm() / w.norm()), output_calibration=rel(e, w, hc), output_heldout=rel(e, w, hh))
        if row:
            group = 'siglip' if 'vision_tower' in name else 'language_model' if 'language_model' in name else 'expert'
            layers[name] = dict(group=group, **row)
    summary = {}
    for label in sources:
        for group in ('siglip', 'language_model', 'expert', 'all'):
            rs = [v[label] for v in layers.values() if label in v and group in ('all', v['group'])]
            if rs:
                mean = lambda k: sum(r[k] for r in rs) / len(rs)
                summary.setdefault(label, {})[group] = dict(layers=len(rs), weight=mean('weight'), output_calibration=mean('output_calibration'),
                                                           output_heldout=mean('output_heldout'), heldout_over_calibration=mean('output_heldout') / mean('output_calibration'))
    Path(a.out).write_text(json.dumps(dict(sources=sources, summary=summary, layers=layers), indent=1) + '\n')
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
