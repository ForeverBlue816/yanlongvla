"""Export the 414 quantizable Linear weights and their calibration Hessians for external VQ toolkits
(AQLM, VPTQ, QuIP#, QTIP, ...). Each layer becomes <out>/<layer>.safetensors with tensors
  weight  (out_features, in_features) float16
  hessian (in_features, in_features)  float32   -- uncentered E[x x^T] on the shared 256 observations
plus <out>/layers.json listing names, shapes and which layers belong to the backbone vs the action expert.

Quantize both groups to the same nominal bit-width (e.g. 2-bit backbone + 2-bit expert), dequantize, and bring the
weights back with import_external.py.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from common import quantizable_layers, build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg, model = build_model(a.checkpoint, device='cpu')
    layers = quantizable_layers(model)
    index = json.loads((Path(a.calibration) / 'index.json').read_text())['layers']
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for name, layer in sorted(layers.items()):
        h = torch.load(Path(a.calibration) / index[name]['file'], map_location='cpu', weights_only=True)['covariance']
        file = out / (name.replace('.', '_') + '.safetensors')
        save_file(dict(weight=layer.weight.detach().half().contiguous(), hessian=h.float().contiguous()), str(file))
        listing.append(dict(name=name, file=file.name, out_features=layer.out_features, in_features=layer.in_features, group='expert' if '.gemma_expert.' in name else 'backbone'))
    (out / 'layers.json').write_text(json.dumps(dict(layers=listing, note='weight float16, hessian float32 (uncentered second moment of the layer input over the shared calibration set)'), indent=1) + '\n')
    print(json.dumps(dict(exported=len(listing), out=str(out))), flush=True)


if __name__ == '__main__':
    main()
