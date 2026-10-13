"""Bring externally quantized layers back as a fake-quant checkpoint.

Input: a directory with <layer>.safetensors files (same names as export_layers.py) each holding a tensor `weight`
(dequantized, any float dtype) and a JSON accounting file:
  {"method": "aqlm", "bits_nominal": 2, "layers": {"<layer name>": {"payload_bytes": N}, ...}}
payload_bytes must count every stored tensor of that layer (codes/indices, codebooks, scales, zero points, permutations).

Usage:
  python import_external.py --checkpoint /path/pi05_libero_pytorch --quantized /path/aqlm_out \
      --accounting /path/aqlm_out/accounting.json --out /path/ckpt_aqlm_w2
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import quantizable_layers, build_model, write_fakequant_checkpoint, accounting


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--quantized', required=True)
    p.add_argument('--accounting', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg, model = build_model(a.checkpoint, device='cpu')
    layers = quantizable_layers(model)
    acc = json.loads(Path(a.accounting).read_text())
    replaced = {}
    payload = 0
    for name, layer in layers.items():
        file = Path(a.quantized) / (name.replace('.', '_') + '.safetensors')
        assert file.exists(), f'missing dequantized layer {name}'
        w = load_file(str(file))['weight']
        assert tuple(w.shape) == tuple(layer.weight.shape), name
        replaced[name] = w
        payload += int(acc['layers'][name]['payload_bytes'])
    info = accounting(payload, acc['method'], acc['bits_nominal'], dict(backbone_bits=acc.get('backbone_bits', acc['bits_nominal']), expert_bits=acc.get('expert_bits', acc['bits_nominal']), external_accounting=str(Path(a.accounting))))
    write_fakequant_checkpoint(a.checkpoint, a.out, replaced, info)
    print(json.dumps({k: info[k] for k in ('method', 'bits_nominal', 'quantizable_linear_bpw', 'whole_model_bpw')}), flush=True)


if __name__ == '__main__':
    main()
