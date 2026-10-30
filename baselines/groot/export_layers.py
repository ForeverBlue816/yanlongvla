"""Step 2b (external VQ toolkits: VPTQ, AQLM, QuIP#, QTIP, ...): export each quantizable Linear weight with its calibration
Hessian, in exactly the per-layer format of the π0.5 harness, so ../quantize_vptq.py, ../quantize_aqlm.py,
../quantize_quip.py and ../quantize_qtip.py run on it unchanged.

Five layer sets: backbone (216 layers, from libero_spatial) and head_<suite> (216 layers each). Each becomes
<out>/<key>/<layer>.safetensors with weight (out, in) float16 and hessian (in, in) float32, plus <out>/<key>/layers.json.

Usage:  python export_layers.py --checkpoints ./checkpoints --calibration ./calibration --out ./layers
Then, for every key, run the toolkit with --layers ./layers/<key> --calibration ./calibration/<key>
--checkpoint ./checkpoints/<suite of that key> (libero_spatial for the backbone), and bring the five outputs back with
import_external.py.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from groot_common import SUITES, quantizable_layers, part_of, read_tensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    keys = [('backbone', 'libero_spatial', 'backbone')] + [(f'head_{s}', s, 'head') for s in SUITES]
    for key, suite, part in keys:
        tensors = read_tensors(Path(a.checkpoints) / suite)
        index = json.loads((Path(a.calibration) / key / 'index.json').read_text())['layers']
        out = Path(a.out) / key
        out.mkdir(parents=True, exist_ok=True)
        listing = []
        for name in sorted(index):
            assert part_of(name) == part, name
            h = torch.load(Path(a.calibration) / key / index[name]['file'], map_location='cpu', weights_only=True)['covariance']
            w = tensors[name + '.weight']
            file = out / (name.replace('.', '_') + '.safetensors')
            save_file(dict(weight=w.half().contiguous(), hessian=h.float().contiguous()), str(file))
            # 'group' is the π0.5 harness's field name (backbone / expert); the action head plays the expert's role
            listing.append(dict(name=name, file=file.name, out_features=int(w.shape[0]), in_features=int(w.shape[1]), group='backbone' if part == 'backbone' else 'expert', suite=suite))
        assert len(listing) == 216, (key, len(listing))
        (out / 'layers.json').write_text(json.dumps(dict(layers=listing, key=key, suite=suite, note='weight float16, hessian float32 (uncentered second moment of the layer input over the shared calibration set)'), indent=1) + '\n')
        print(json.dumps(dict(exported=key, layers=len(listing), out=str(out))), flush=True)
        del tensors


if __name__ == '__main__':
    main()
