"""Task 1 check 4c (GOAL v4): AWQ scale statistic. Our AWQ takes rms(x) from the diagonal of the shared Hessian; the official llm-awq takes the mean |x| per
input channel. For a few GR00T layers (calibration_absmean: the same 256 observations, same noise seed, plus mean |x|), quantize to 2 bits with each
statistic (alpha grid and auto_clip as in quantize.awq) and report the relative output error on the calibration and the held-out Hessian.
  python awq_scale_check.py --calibration DIR --heldout DIR --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'groot'))
from groot_common import parent_module   # noqa: E402

Q = parent_module('quantize')
PICK = ['visual.blocks.20.mlp.linear_fc2', 'visual.blocks.3.attn.qkv', 'language_model.layers.2.mlp.down_proj',
        'language_model.layers.15.self_attn.q_proj', 'language_model.layers.10.mlp.up_proj', 'language_model.layers.12.mlp.gate_proj']


def err(w, q, h):
    e = (q - w).float()
    return float(((e @ h) * e).sum() / ((w @ h) * w).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True)
    p.add_argument('--calibration', required=True, help='calibration_absmean/backbone')
    p.add_argument('--heldout', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    listing = json.loads((Path(a.layers) / 'layers.json').read_text())['layers']
    cal = json.loads((Path(a.calibration) / 'index.json').read_text())['layers']
    held = json.loads((Path(a.heldout) / 'index.json').read_text())['layers']
    res, lines = {}, ['| layer | bits | statistic | alpha | calibration | held-out |', '|---|---|---|---|---|---|']
    for key in PICK:
        item = next(i for i in listing if key in i['name'])
        w = load_file(str(Path(a.layers) / item['file']))['weight'].float().cuda()
        c = torch.load(Path(a.calibration) / cal[item['name']]['file'], map_location='cpu', weights_only=True)
        h, hh = c['covariance'].float().cuda(), torch.load(Path(a.heldout) / held[item['name']]['file'], map_location='cpu', weights_only=True)['covariance'].float().cuda()
        r = res[item['name']] = {}
        clip = 'q_proj' not in item['name'] and 'k_proj' not in item['name'] and 'to_q' not in item['name'] and 'to_k' not in item['name']
        for bits in (2, 3, 4):
            for stat, kw in (('rms (ours)', {}), ('mean abs (official)', dict(act_scale=c['abs_mean']))):
                q, alpha, _ = Q.awq(w, h, bits, 128, clip=clip, **kw)
                r[f'w{bits}_{stat}'] = dict(alpha=alpha, calibration=err(w, q, h), heldout=err(w, q, hh))
                lines.append(f"| {item['name'].split('model.model.')[-1]} | {bits} | {stat} | {alpha:.2f} | {r[f'w{bits}_{stat}']['calibration']:.4f} | {r[f'w{bits}_{stat}']['heldout']:.4f} |")
                print(lines[-1], flush=True)
        (out / 'results.json').write_text(json.dumps(res, indent=1))
    (out / 'tables.md').write_text('Relative output error tr(dW H dW^T)/tr(W H W^T), AWQ with two scale statistics\n\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
