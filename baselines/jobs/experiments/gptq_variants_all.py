"""GOAL 31 Oct item a2: GPTQ 2-bit variants on every GR00T layer (backbone and the libero_spatial action head), offline.

Variants (official GPTQ options, group 128, asymmetric, exactly 2 bits): base (the collapsed row), actorder, mse (MSE-searched clipping),
damp10 (10% damping), actorder_mse, combo (actorder + mse + 10% damping). Per layer: relative output error tr(dW H dW^T) / tr(W H W^T) on the
shared Hessian. Tables: mean and max per component (vision tower, language model, DiT action head, vl self-attention) and the number of layers
a variant makes worse than the base. Writes results.json and tables.md.

  python gptq_variants_all.py --layers RUN/layers --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'groot'))
from groot_common import parent_module   # baselines/quantize.py in the GR00T environment

Q = parent_module('quantize')
DEFAULT = 'base,ao,mse,d0.1,ao_mse,ao_mse_d0.1'
GRID = 'base,' + ','.join(f'{pre}d{d}' for pre in ('', 'mse_', 'ao_', 'ao_mse_') for d in (0.03, 0.1, 0.3, 1.0)) + ',mse,ao,ao_mse'   # finer damping grid with and without MSE / act-order


def component(name):
    if 'visual.' in name:
        return 'vision tower'
    if 'language_model' in name:
        return 'language model'
    if 'vl_self_attention' in name:
        return 'vl self-attention'
    return 'DiT action head'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--variants', default=DEFAULT, help='comma list: base | ao | mse | d<x> joined by _ ; "grid" = the finer damping grid')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = (GRID if a.variants == 'grid' else a.variants).split(',')
    VARIANTS = {n: Q.gptq_variant(n) for n in names}
    res = {}
    for key in ('backbone', 'head_libero_spatial'):
        listing = json.loads((Path(a.layers) / key / 'layers.json').read_text())['layers']
        for n, item in enumerate(listing, 1):
            src = load_file(str(Path(a.layers) / key / item['file']))
            w, h = src['weight'].float().cuda(), src['hessian'].float().cuda()
            den = ((w @ h) * w).sum()
            rec = dict(component=component(item['name']), shape=[item['out_features'], item['in_features']])
            for v, kw in VARIANTS.items():
                t = time.time()
                try:
                    q = Q.gptq(w, h, 2, 128, **kw)
                    dw = w - q
                    rec[v] = (((dw @ h) * dw).sum() / den).item()
                except Exception as e:   # e.g. a Hessian that does not factor: report it
                    rec[v] = None; rec[v + '_error'] = repr(e)[:120]
            res[item['name']] = rec
            if n % 20 == 0:
                print(key, n, len(listing), flush=True)
                (out / 'results.json').write_text(json.dumps(res))
    (out / 'results.json').write_text(json.dumps(res))
    comps = ['vision tower', 'language model', 'vl self-attention', 'DiT action head']
    lines = ['Relative output error on the shared Hessian, 2 bits, group 128: mean (max) over the layers of a component\n', '| component (layers) | ' + ' | '.join(VARIANTS) + ' |', '|---|' + '---|' * len(VARIANTS)]
    for c in comps + ['all']:
        rs = [r for r in res.values() if c == 'all' or r['component'] == c]
        cells = []
        for v in VARIANTS:
            xs = [r[v] for r in rs if r.get(v) is not None]
            cells.append(f'{sum(xs) / len(xs):.4f} ({max(xs):.3f})')
        lines.append(f'| {c} ({len(rs)}) | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Layers where the variant is worse than base (of all layers), and its worst layer relative to base\n', '| variant | worse than base | worst ratio |', '|---|---|---|']
    for v in list(VARIANTS)[1:]:
        pairs = [(r[v], r['base']) for r in res.values() if r.get(v) is not None and r.get('base')]
        lines.append(f'| {v} | {sum(1 for x, b in pairs if x > b)} of {len(pairs)} | {max(x / b for x, b in pairs):.1f} |')
    best = [min(r[v] for v in VARIANTS if r.get(v) is not None) for r in res.values()]
    lines += ['', f'Per-layer best of all variants by the calibration error (the rule of the row gptq_best): mean {sum(best) / len(best):.4f}, max {max(best):.3f} over {len(best)} layers (base: mean {sum(r["base"] for r in res.values()) / len(res):.4f}).']
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
