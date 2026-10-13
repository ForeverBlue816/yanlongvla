"""Built-in scalar baselines on all 414 quantizable Linear layers (backbone + expert): RTN and GPTQ, W2/W3/W4, group size g.

Usage:
  python quantize.py --method gptq --bits 2 --group 128 --checkpoint /path/pi05_libero_pytorch \
      --calibration /path/calibration --out /path/ckpt_gptq_w2g128

Asymmetric per-group min/max quantization (standard for low-bit GPTQ/AWQ): b-bit codes, FP16 scale and b-bit zero point
per group of g input weights. Payload accounting: b + (16 + b) / g bits per weight. The output is a fake-quant
checkpoint (dequantized BF16 weights) evaluated through the same runtime as every other method.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from common import quantizable_layers, build_model, write_fakequant_checkpoint, accounting, QUANTIZABLE_ELEMENTS


def quant_params(w, bits):
    """Per-row-group asymmetric min/max: w (rows, g) -> scale, zero (rows, 1)."""
    maxq = 2 ** bits - 1
    lo = w.amin(dim=1, keepdim=True)
    hi = w.amax(dim=1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / maxq
    zero = torch.round(-lo / scale).clamp(0, maxq)
    return scale, zero, maxq


def fake_quant(w, scale, zero, maxq):
    return (torch.clamp(torch.round(w / scale) + zero, 0, maxq) - zero) * scale


def rtn(weight, bits, group):
    out, inp = weight.shape
    w = weight.float().reshape(out, inp // group, group)
    scale, zero, maxq = quant_params(w.reshape(-1, group), bits)
    return fake_quant(w.reshape(-1, group), scale, zero, maxq).reshape(out, inp)


def gptq(weight, hessian, bits, group, blocksize=128, percdamp=0.01):
    """GPTQ (Frantar et al. 2022): column-wise quantization with Hessian-based error feedback; groups along the input dim
    quantized in order (no act-order). Reference implementation, float32."""
    W = weight.float().clone()
    out, inp = W.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(inp, device=H.device) * damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    scale = zero = None
    maxq = 2 ** bits - 1
    for i1 in range(0, inp, blocksize):
        i2 = min(i1 + blocksize, inp)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                scale, zero, _ = quant_params(W[:, col:col + group], bits)
            w = W1[:, i:i + 1]
            d = Hinv1[i, i]
            q = fake_quant(w, scale, zero, maxq)
            Q1[:, i:i + 1] = q
            err = (w - q) / d
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            Err1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=('rtn', 'gptq'), required=True)
    p.add_argument('--bits', type=int, choices=(2, 3, 4), required=True)
    p.add_argument('--group', type=int, default=128)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calibration', help='output of calibrate.py (required for gptq)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg, model = build_model(a.checkpoint, device='cuda')
    layers = quantizable_layers(model)
    index = json.loads((Path(a.calibration) / 'index.json').read_text())['layers'] if a.method == 'gptq' else None
    replaced = {}
    payload_bits = 0
    for k, (name, layer) in enumerate(sorted(layers.items())):
        w = layer.weight.detach()
        assert w.shape[1] % a.group == 0, (name, w.shape)
        if a.method == 'rtn':
            q = rtn(w, a.bits, a.group)
        else:
            h = torch.load(Path(a.calibration) / index[name]['file'], map_location='cuda', weights_only=True)['covariance']
            q = gptq(w, h, a.bits, a.group)
        replaced[name] = q
        groups = w.numel() // a.group
        payload_bits += w.numel() * a.bits + groups * (16 + a.bits)
        if (k + 1) % 50 == 0:
            print(json.dumps(dict(layers=k + 1)), flush=True)
    info = accounting(payload_bits / 8, a.method, a.bits, dict(group=a.group, payload_rule=f'{a.bits} bits per weight + FP16 scale + {a.bits}-bit zero point per group of {a.group}', backbone_bits=a.bits, expert_bits=a.bits))
    write_fakequant_checkpoint(a.checkpoint, a.out, replaced, info)
    print(json.dumps({k: info[k] for k in ('method', 'bits_nominal', 'quantizable_linear_bpw', 'whole_model_bpw')}), flush=True)


if __name__ == '__main__':
    main()
