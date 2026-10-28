"""Built-in scalar baselines on all 414 quantizable Linear layers (backbone + expert): RTN, GPTQ and AWQ, W2/W3/W4, group size g.
All pure PTQ (no layer-wise or end-to-end fine-tuning); GPTQ and AWQ use the shared calibration Hessians.

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
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from common import quantizable_layers, build_model, write_fakequant_checkpoint, accounting, sha256, QUANTIZABLE_ELEMENTS


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
    """Groups of g along the input dim; the last group is shorter when g does not divide in_features (SigLIP fc2: 4304)."""
    w = weight.float()
    q = torch.empty_like(w)
    for i in range(0, w.shape[1], group):
        scale, zero, maxq = quant_params(w[:, i:i + group], bits)
        q[:, i:i + group] = fake_quant(w[:, i:i + group], scale, zero, maxq)
    return q


def awq_clip(w, h, bits, group, grid=20, max_shrink=0.5):
    """AWQ auto_clip (official llm-awq): per output row and input group, clip to +-max|w| * (1 - i/grid) for
    i < max_shrink * grid and keep the range with the smallest group output error. The official code measures that error
    on sampled tokens; with the shared second moment it is exactly e^T H_gg e for the group's diagonal block of H.
    i = 0 is plain RTN, so clipping never increases a group's error."""
    q = torch.empty_like(w)
    for i in range(0, w.shape[1], group):
        wg, hg = w[:, i:i + group], h[i:i + group, i:i + group]
        org_max = wg.abs().amax(dim=1, keepdim=True)
        best_err = torch.full((w.shape[0],), float('inf'), device=w.device)
        for k in range(int(max_shrink * grid)):
            limit = org_max * (1 - k / grid)
            clipped = torch.maximum(torch.minimum(wg, limit), -limit)
            scale, zero, maxq = quant_params(clipped, bits)
            qg = fake_quant(clipped, scale, zero, maxq)
            e = qg - wg
            err = ((e @ hg) * e).sum(dim=1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            q[:, i:i + group] = torch.where(better[:, None], qg, q[:, i:i + group]) if k else qg
    return q


def awq(weight, hessian, bits, group, grid=20, clip=True):
    """AWQ (Lin et al. 2024) with the shared Hessian. Scale search: per-input-channel scale s = rms(x)^alpha, alpha on a
    20-point grid in [0, 1), rms(x) from diag(E[x x^T]); quantize W*s with RTN, keep the alpha with the smallest output
    error tr(E H E^T). Then auto_clip on the scaled weight (official order; the official code skips q/k projections, so
    the caller passes clip=False for them). The per-layer scale vector is stored (FP16) and counted in the payload."""
    w = weight.float()
    h = hessian.float()
    rms = torch.sqrt(torch.diagonal(h).clamp_min(0) / torch.diagonal(h).mean()).clamp_min(1e-4)
    best = None
    for alpha in [i / grid for i in range(grid)]:
        s = rms.pow(alpha)
        s = s / torch.sqrt(s.max() * s.min())
        q = rtn(w * s, bits, group) / s
        e = q - w
        loss = ((e @ h) * e).sum().item()
        if best is None or loss < best[2]:
            best = (s, alpha, loss)
    s, alpha, loss = best
    if clip:   # in the scaled space the inputs are x / s, so the second moment is H / (s s^T)
        q = awq_clip(w * s, h / s[:, None] / s[None, :], bits, group) / s
    else:
        q = rtn(w * s, bits, group) / s
    e = q - w
    return q, alpha, ((e @ h) * e).sum().item()


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
                scale, zero, _ = quant_params(W[:, col:min(col + group, inp)], bits)
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
    p.add_argument('--method', choices=('rtn', 'gptq', 'awq'), required=True)
    p.add_argument('--bits', type=int, choices=(2, 3, 4), required=True)
    p.add_argument('--group', type=int, default=128)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calibration', help='output of calibrate.py (required for gptq)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg, model = build_model(a.checkpoint, device='cuda')
    layers = quantizable_layers(model)
    if a.method != 'rtn' and not a.calibration:
        p.error(f'--calibration is required for {a.method}')
    index = json.loads((Path(a.calibration) / 'index.json').read_text())['layers'] if a.method != 'rtn' else None
    replaced = {}
    alphas = {}
    payload_bits = 0
    for k, (name, layer) in enumerate(sorted(layers.items())):
        w = layer.weight.detach()
        if a.method == 'rtn':
            q = rtn(w, a.bits, a.group)
        else:
            h = torch.load(Path(a.calibration) / index[name]['file'], map_location='cuda', weights_only=True)['covariance']
            if a.method == 'gptq':
                q = gptq(w, h, a.bits, a.group)
            else:
                q, alpha, _ = awq(w, h, a.bits, a.group, clip=name.rsplit('.', 1)[-1] not in ('q_proj', 'k_proj'))
                alphas[name] = alpha
                payload_bits += 16 * w.shape[1]   # FP16 per-input-channel AWQ scale
        replaced[name] = q
        groups = w.shape[0] * math.ceil(w.shape[1] / a.group)
        payload_bits += w.numel() * a.bits + groups * (16 + a.bits)
        if (k + 1) % 50 == 0:
            print(json.dumps(dict(layers=k + 1)), flush=True)
    notes = dict(rtn='pure PTQ, no layer-wise or end-to-end fine-tuning; weight-only RTN, no calibration data used',
                 gptq='pure PTQ, no layer-wise or end-to-end fine-tuning; GPTQ error feedback with the shared per-layer Hessians',
                 awq='pure PTQ, no layer-wise or end-to-end fine-tuning; AWQ scale search and per-group clipping search (official auto_clip: 10 shrink steps of 5%, q/k projections not clipped) on the shared per-layer Hessians')[a.method]
    extra = dict(group=a.group, payload_rule=f'{a.bits} bits per weight + FP16 scale + {a.bits}-bit zero point per group of {a.group} (last group shorter when g does not divide in_features)'
                 + (' + FP16 per-input-channel AWQ scale per layer' if a.method == 'awq' else ''),
                 backbone_bits=a.bits, expert_bits=a.bits, notes=notes, exact_command=' '.join(sys.argv),
                 baseline_commit=subprocess.run(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())
    if a.method != 'rtn':
        extra['calibration_index_sha256'] = sha256(Path(a.calibration) / 'index.json')
    if alphas:
        extra['awq_alpha_by_layer'] = alphas
    info = accounting(payload_bits / 8, a.method, a.bits, extra)
    write_fakequant_checkpoint(a.checkpoint, a.out, replaced, info)
    print(json.dumps({k: info[k] for k in ('method', 'bits_nominal', 'quantizable_linear_bpw', 'whole_model_bpw')}), flush=True)


if __name__ == '__main__':
    main()
