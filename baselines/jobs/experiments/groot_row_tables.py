"""Per-layer tables of GR00T rows for the mentor (31 Oct request): (a) per-layer relative output error on the shared Hessian, from
layer_errors.py results; (b) the layers whose quantizer needed more damping than the default (QuIP# sigma_reg retry, G18), from the quantizer
accounting. Writes DIR/per_layer_output_error.csv, DIR/damping_retries.json and DIR/summary.md.

  python groot_row_tables.py --errors RUN/experiments/layer_errors --q RUN/q --rows qtip_w2 quip_w2 quip10_w2 --out DIR
"""
import argparse
import csv
import json
from pathlib import Path

KEYS = ['backbone', 'head_libero_spatial', 'head_libero_object', 'head_libero_goal', 'head_libero_10']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--errors', required=True)
    p.add_argument('--q', required=True)
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'per_layer_output_error.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer_set', 'layer', 'group', 'out_features', 'in_features'] + [f'{r}_weight_error' for r in a.rows] + [f'{r}_output_error' for r in a.rows])
        for key in KEYS:
            res = json.loads((Path(a.errors) / key / 'results.json').read_text())
            for name, rec in res[a.rows[0]].items():
                w.writerow([key, name, rec['group'], *rec['shape']] + [f"{res[r][name]['weight_error']:.6f}" for r in a.rows] + [f"{res[r][name]['output_error']:.6f}" for r in a.rows])
    retries, lines = {}, ['# Per-layer tables for the mentor', '']
    for r in a.rows:
        for key in KEYS:
            acc = json.loads((Path(a.q) / r / key / 'accounting.json').read_text())
            for name, rec in acc['layers'].items():
                used = rec.get('sigma_reg_used')
                if used is not None and used != 0.01:
                    retries.setdefault(r, {}).setdefault(key, {})[name] = used
    (out / 'damping_retries.json').write_text(json.dumps(retries, indent=1))
    lines.append('QuIP# sigma_reg actually used per layer (default 0.01); layers that needed a retry (10x = 0.1, 100x = 1.0):')
    lines.append(json.dumps(retries, indent=1) if retries else 'none')
    for r in a.rows:
        for key in KEYS:
            res = json.loads((Path(a.errors) / key / 'results.json').read_text())[r]
            xs = sorted(res.items(), key=lambda kv: -kv[1]['output_error'])
            lines.append(f"\n{r} / {key}: mean output error {sum(v['output_error'] for v in res.values()) / len(res):.4f}; worst 5: " + '; '.join(f"{n.split('model.')[-1]} {v['output_error']:.3f}" for n, v in xs[:5]))
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
