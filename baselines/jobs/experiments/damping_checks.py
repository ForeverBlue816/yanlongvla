"""Task 1 checks 4a and 4b (GOAL v4): does the damping we add handicap AQLM, and what does the QuIP# damping retry do?

  --mode aqlm: official AQEngine.quantize on a few GR00T layers with percdamp 0, 0.001, 0.01 (current, decision 24), 0.1 and with the
               stopping rule at 1% (current) and 0.1% relative improvement; errors on the calibration and the held-out Hessian
               (both undamped, as in vptq_variants.py), plus the relative weight error.
  --mode quip: official quip.quantize (0 sweeps, as the main row) with sigma_reg 0.01 (the first try; a Cholesky failure is recorded, not
               hidden), 0.1 and 1.0 (the retry values of G18), same errors.
Writes OUT/results.json and OUT/tables.md. Run with PYTHONPATH containing the toolkit repo, the source dir and the openpi site-packages.
"""
import argparse
import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PICK = ['visual.blocks.20.mlp.linear_fc2', 'visual.blocks.3.attn.qkv', 'language_model.layers.2.mlp.down_proj',
        'language_model.layers.15.self_attn.q_proj', 'language_model.layers.10.mlp.up_proj']


def errs(w, q, hs):
    e = (q - w).float()
    out = dict(weight=float(e.norm() / w.norm()))
    for k, h in hs.items():
        out[k] = float(((e @ h) * e).sum() / ((w @ h) * w).sum())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('aqlm', 'quip'), required=True)
    p.add_argument('--layers', required=True)
    p.add_argument('--heldout', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pick', nargs='*', default=None, help='only layers whose name contains one of these strings (default: PICK)')
    p.add_argument('--no-tight', nargs='*', default=[], help='layers (substring) for which the 0.1%% stopping rule is skipped (AQLM at 0.1%% never converged on the badly conditioned layer 2 down_proj in 6 h, 5 Nov)')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    dev = torch.device('cuda:0')
    layers = Path(a.layers)
    listing = json.loads((layers / 'layers.json').read_text())['layers']
    held = json.loads((Path(a.heldout) / 'index.json').read_text())['layers']
    res = {}
    if a.mode == 'quip':
        import quantize_quip as qq
        from lib import codebook, utils
        from lib.algo import quip
        cb = codebook.get_codebook('E8P12')
    else:
        import quantize_aqlm as qa
        from aq_engine import AQEngine
        from src.utils import _dequantize_weight
    for key in (a.pick or PICK):
        item = next(i for i in listing if key in i['name'])
        src = load_file(str(layers / item['file']))
        w, h = src['weight'].float(), src['hessian'].float()
        hh = torch.load(Path(a.heldout) / held[item['name']]['file'], map_location='cpu', weights_only=True)['covariance'].float()
        hs = dict(calibration=h.to(dev), heldout=hh.to(dev))
        wd = w.to(dev)
        r = res[item['name']] = {}
        if a.mode == 'aqlm':
            for pd in (0.0, 0.001, 0.01, 0.1):
                for tol in ((0.01,) if any(k in item['name'] for k in a.no_tight) else (0.01, 0.001)):
                    tag = f'damp{pd}_tol{tol}'
                    try:
                        layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device=dev, dtype=torch.float32)
                        layer.weight.copy_(wd)
                        eng = AQEngine(layer)
                        x = h.to(dev, eng.XTX.dtype).clone()
                        x.diagonal().add_(pd * x.diagonal().mean())
                        eng.XTX = x
                        random.seed(0); torch.manual_seed(0)
                        t = time.time()
                        qw = eng.quantize(args=Namespace(devices=[dev], num_codebooks=1, print_frequency=10 ** 9, **dict(qa.SETTINGS, relative_mse_tolerance=tol)), verbose=False)
                        wq = _dequantize_weight(qw.get_codes().long(), qw.get_codebooks().float(), qw.get_scales().float())
                        r[tag] = dict(errs(wd, wq, hs), seconds=round(time.time() - t))
                    except Exception as e:
                        r[tag] = dict(error=repr(e)[:200])
                    print(item['name'], tag, r[tag], flush=True)
        else:
            m, n = w.shape
            scale = w.square().mean().sqrt()
            for sr in (0.01, 0.1, 1.0):
                try:
                    hreg = utils.regularize_H(h.double().clone(), n, sr)
                    torch.manual_seed(0)
                    settings = dict(qq.SETTINGS, sigma_reg=sr, quip_tune_iters=0)
                    hat, _ = quip.quantize(hreg.float().to(dev), (w / scale).to(dev), 0, cb, Namespace(save_pfx='/tmp', **settings), dev)
                    r[f'sigma{sr}'] = errs(wd, hat.float() * scale.to(dev), hs)
                except Exception as e:
                    r[f'sigma{sr}'] = dict(error=repr(e)[:200])
                print(item['name'], sr, r[f'sigma{sr}'], flush=True)
        (out / 'results.json').write_text(json.dumps(res, indent=1))
        torch.cuda.empty_cache()
    tags = list(next(iter(res.values())))
    lines = ['Output error on the calibration / held-out Hessian (weight error in brackets)\n', '| variant | ' + ' | '.join(res) + ' |', '|---|' + '---|' * len(res)]
    for t in tags:
        lines.append(f'| {t} | ' + ' | '.join((f"{v[t]['calibration']:.4f} / {v[t]['heldout']:.4f} ({v[t]['weight']:.3f})" if 'calibration' in v.get(t, {}) else 'fail') for v in res.values()) + ' |')
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
