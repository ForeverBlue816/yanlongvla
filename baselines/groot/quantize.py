"""Step 2a: built-in scalar baselines GPTQ and AWQ at 2 bits (group 128) on the 432 quantizable Linear layers of every suite
checkpoint, using the shared calibration Hessians. The algorithms are imported unchanged from ../quantize.py (the π0.5
harness). RTN is not part of the GR00T baselines (it collapses at 2 bits on π0.5: 0/292).

The backbone is quantized once (libero_spatial weights and Hessians; identical in all suites) and the action head once per
suite; each suite gets its own fake-quant checkpoint and quant.json.

Usage:  python quantize.py --method gptq --checkpoints ./checkpoints --calibration ./calibration --out ./ckpt_gptq_w2g128
  -> ckpt_gptq_w2g128/<suite>/{model.safetensors, quant.json, ...}
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from groot_common import SUITES, part_of, read_tensors, write_fakequant_checkpoint, accounting, sha256, parent_module

_pi05 = parent_module('quantize')
gptq, awq = _pi05.gptq, _pi05.awq
BITS = 2


def quantize_key(method, tensors, calib_dir, group):
    index = json.loads((calib_dir / 'index.json').read_text())['layers']
    out, bits, alphas = {}, 0, {}
    for k, name in enumerate(sorted(index)):
        w = tensors[name + '.weight'].cuda()
        h = torch.load(calib_dir / index[name]['file'], map_location='cuda', weights_only=True)['covariance']
        if method == 'gptq':
            q = gptq(w, h, BITS, group)
        else:
            q, alpha, _ = awq(w, h, BITS, group, clip=name.rsplit('.', 1)[-1] not in ('q_proj', 'k_proj', 'to_q', 'to_k', 'qkv'))   # official AWQ skips q/k (qkv: fused vision projection)
            alphas[name] = alpha
            bits += 16 * w.shape[1]                       # FP16 per-input-channel AWQ scale
        out[name] = q.cpu()
        bits += w.numel() * BITS + w.shape[0] * math.ceil(w.shape[1] / group) * (16 + BITS)
        if (k + 1) % 50 == 0:
            print(json.dumps(dict(key=calib_dir.name, layers=k + 1)), flush=True)
    return out, bits, alphas, sha256(calib_dir / 'index.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=('gptq', 'awq'), required=True)
    p.add_argument('--group', type=int, default=128)
    p.add_argument('--checkpoints', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    calib = Path(a.calibration)
    bb, bb_bits, bb_alphas, bb_index = quantize_key(a.method, read_tensors(Path(a.checkpoints) / 'libero_spatial'), calib / 'backbone', a.group)
    for suite in SUITES:
        tensors = read_tensors(Path(a.checkpoints) / suite)
        head, head_bits, head_alphas, head_index = quantize_key(a.method, tensors, calib / f'head_{suite}', a.group)
        notes = dict(gptq='pure PTQ, no fine-tuning; GPTQ error feedback (percdamp 0.01, no act-order) with the shared per-layer Hessians',
                     awq='pure PTQ, no fine-tuning; AWQ scale search + official auto_clip (q/k projections not clipped) on the shared per-layer Hessians')[a.method]
        extra = dict(suite=suite, group=a.group, backbone_bits=BITS, head_bits=BITS, notes=notes,
                     payload_rule=f'{BITS} bits per weight + FP16 scale + {BITS}-bit zero point per group of {a.group}' + (' + FP16 per-input-channel AWQ scale per layer' if a.method == 'awq' else ''),
                     calibration_index_sha256=dict(backbone=bb_index, head=head_index), exact_command=' '.join(sys.argv),
                     baseline_commit=subprocess.run(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())
        if a.method == 'awq':
            extra['awq_alpha_by_layer'] = dict(bb_alphas, **head_alphas)
        info = accounting((bb_bits + head_bits) / 8, f'{a.method}_w{BITS}g{a.group}', BITS, extra)
        write_fakequant_checkpoint(Path(a.checkpoints) / suite, Path(a.out) / suite, dict(bb, **head), info)
        print(json.dumps({k: info[k] for k in ('method', 'suite', 'quantized_linear_bpw', 'whole_model_bpw')}), flush=True)
        del tensors, head
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
