"""Task 0 / Task 1: per-layer errors of the 2-bit rows against the FP weights, on the shared calibration Hessians (CPU only).

For every quantizable layer and every row: relative weight error ||W - Q|| / ||W|| and the relative output error on the
calibration inputs, tr(dW H dW^T) / tr(W H W^T), with H the shared uncentered E[x x^T]. Rows: the dequantized layer files each
quantizer wrote (<out>/<layer>.safetensors, tensor `weight`). Writes results.json and tables.md in --out.

  python layer_errors.py --layers RUN/layers --rows vptq_w2=RUN/vptq_layers/vptq_w2 qtip_w2=RUN/qtip_layers/qtip_w2 ... --out DIR
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

torch.set_num_threads(8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True)
    p.add_argument('--rows', nargs='+', required=True, help='row=directory with <layer>.safetensors')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu', help='cuda for the GR00T layer sets (6144-wide Hessians)')
    p.add_argument('--no-output-error', action='store_true', help='weight errors only (fast)')
    a = p.parse_args()
    layers = json.loads((Path(a.layers) / 'layers.json').read_text())['layers']
    rows = dict(r.split('=', 1) for r in a.rows)
    res = {r: {} for r in rows}
    for n, item in enumerate(layers, 1):
        fp = load_file(str(Path(a.layers) / item['file']))
        w = fp['weight'].float().to(a.device)
        h = fp['hessian'].float().to(a.device) if not a.no_output_error else None
        den = (w @ h * w).sum().item() if h is not None else None
        for r, d in rows.items():
            q = load_file(str(Path(d) / item['file']))['weight'].float().to(a.device)
            dw = w - q
            rec = dict(group=item['group'], shape=[item['out_features'], item['in_features']], weight_error=(dw.norm() / w.norm()).item())
            if h is not None:
                rec['output_error'] = ((dw @ h * dw).sum().item() / den)
            res[r][item['name']] = rec
        if n % 25 == 0:
            print(n, len(layers), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(res))
    md = []
    for metric in ('weight_error', 'output_error'):
        if a.no_output_error and metric == 'output_error':
            continue
        md.append(f'\n#### {metric}: mean and max over layers, by group\n')
        # pi0.5 layer names have vision_tower / language_model / gemma_expert; GR00T names have visual.blocks / language_model / vl_self_attention / action_head.model
        groots = any('visual.blocks' in k for k in next(iter(res.values())))
        groups = ((('vision', 'visual.blocks'), ('language_model', '.language_model.'), ('vl_self_attention', 'vl_self_attention'), ('dit', 'action_head.model.')) if groots
                  else (('siglip', '.vision_tower.'), ('language_model', '.language_model.'), ('expert', '.gemma_expert.')))
        md.append('| row | ' + ' | '.join(f'{g} mean | {g} max' for g, _ in groups) + ' |')
        md.append('|---|' + '---|' * (2 * len(groups)))
        for r in rows:
            cells = []
            for g, key in groups:
                v = [x[metric] for k, x in res[r].items() if key in k]
                cells += [f'{sum(v) / len(v):.4f}', f'{max(v):.4f}'] if v else ['-', '-']
            md.append(f'| {r} | ' + ' | '.join(cells) + ' |')
        md.append(f'\n#### {metric}: the 8 worst layers of each row\n')
        for r in rows:
            worst = sorted(res[r].items(), key=lambda kv: -kv[1][metric])[:8]
            md.append(f'**{r}**: ' + '; '.join(f'{k.split("model.")[-1]} {v["shape"]} {v[metric]:.3f}' for k, v in worst) + '\n')
    (out / 'tables.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
