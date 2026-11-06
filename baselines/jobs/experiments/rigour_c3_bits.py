"""Rigour C3: linear bits per weight of the stored rows, recomputed from the safetensors headers of the stored layer files (every tensor except the dequantized `weight`
used by the runtime), independently of the quantizer's own payload_bytes; compared with payload_bytes in each layer's json.
Linear bpw of a model for one LIBERO suite = bits of backbone + head_<suite> payload / original (unpadded) weights of the same layers.
  python rigour_c3_bits.py --q RUN/q --layers RUN/layers --out FILE"""
import argparse
import json
from pathlib import Path
from st_header import header


def group_bits(qdir, layers_json):
    listing = {i['name']: i for i in json.loads(Path(layers_json).read_text())['layers']}
    bits = weights = mism = n = 0
    for js in sorted(Path(qdir).glob('*.json')):
        if js.name == 'accounting.json': continue
        meta = json.loads(js.read_text())
        if 'payload_bytes' not in meta or not js.with_suffix('.safetensors').exists(): continue
        h = header(js.with_suffix('.safetensors'))
        stored = sum(v['data_offsets'][1] - v['data_offsets'][0] for k, v in h.items() if k not in ('weight', '__metadata__'))
        mism += stored != meta['payload_bytes']; n += 1
        it = listing[meta['name']]
        bits += 8 * stored; weights += it['out_features'] * it['in_features']
    return bits, weights, mism, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', required=True); ap.add_argument('--layers', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lines = ['| row | suite | layers | layers whose json payload_bytes differs from the stored tensors | stored payload (MB) | original weights (M) | linear bpw (recomputed) |', '|---|---|---|---|---|---|---|']
    for row in sorted(p.name for p in Path(a.q).iterdir() if (p / 'backbone').is_dir() and not p.name.endswith('l2fix')):
        for head in sorted(p.name for p in (Path(a.q) / row).glob('head_*')):
            b1, w1, m1, n1 = group_bits(Path(a.q) / row / 'backbone', Path(a.layers) / 'backbone' / 'layers.json')
            b2, w2, m2, n2 = group_bits(Path(a.q) / row / head, Path(a.layers) / head / 'layers.json')
            lines.append(f'| {row} | {head[5:]} | {n1 + n2} | {m1 + m2} | {(b1 + b2) / 8e6:.1f} | {(w1 + w2) / 1e6:.2f} | {(b1 + b2) / (w1 + w2):.4f} |')
    Path(a.out).write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
