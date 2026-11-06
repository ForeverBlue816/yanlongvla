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
import re
import subprocess
import sys
from pathlib import Path

import torch

from groot_common import SUITES, BB_SUITE, part_of, read_tensors, write_fakequant_checkpoint, accounting, sha256, parent_module

_pi05 = parent_module('quantize')
gptq, awq, gptq_variant = _pi05.gptq, _pi05.awq, _pi05.gptq_variant
BITS = 2


SELECT_DEFAULT = ('base', 'mse', 'ao', 'ao_mse', 'd0.1', 'mse_d0.1', 'ao_mse_d0.1', 'mse_d0.3', 'ao_mse_d0.3')   # row gptq_best_w2: per layer, the variant with the smallest calibration output error


def quantize_key(method, tensors, calib_dir, group, actorder=False, mse=False, percdamp=0.01, select=None, chosen=None, only=None, skip=None, overrides=None):
    index = json.loads((calib_dir / 'index.json').read_text())['layers']
    out, bits, alphas = {}, 0, {}
    for k, name in enumerate(sorted(index)):
        if (only and not re.search(only, name)) or (skip and re.search(skip, name)):
            continue   # partial checkpoint (bisection, GOAL 1 Nov 2.2): this layer stays FP
        ov = None
        w = tensors[name + '.weight'].cuda()
        h = torch.load(calib_dir / index[name]['file'], map_location='cuda', weights_only=True)['covariance']
        if method == 'gptq' and select:   # choose per layer by the calibration output error only (no evaluation data is used)
            den = ((w.float() @ h.float()) * w.float()).sum()
            best = None
            for v in select:
                qv = gptq(w, h, BITS, group, **gptq_variant(v))
                dw = w.float() - qv
                e = (((dw @ h.float()) * dw).sum() / den).item()
                if best is None or e < best[0]:
                    best = (e, v, qv)
            q = best[2]; chosen[name] = dict(variant=best[1], output_error=best[0])
            if 'ao' in best[1].split('_'):
                bits += 16 * w.shape[1]                   # column order of act-order layers: a 16-bit index per input column
        elif method == 'gptq':
            ov = next((v for pat, v in (overrides or {}).items() if re.search(pat, name)), None)   # --override REGEX=variant: a different GPTQ variant for matching layers
            q = gptq(w, h, BITS, group, **gptq_variant(ov)) if ov else gptq(w, h, BITS, group, percdamp=percdamp, actorder=actorder, mse=mse)
            if ov and 'ao' in ov.split('_'):
                bits += 16 * w.shape[1]
        else:
            q, alpha, _ = awq(w, h, BITS, group, clip=name.rsplit('.', 1)[-1] not in ('q_proj', 'k_proj', 'to_q', 'to_k', 'qkv'))   # official AWQ skips q/k (qkv: fused vision projection)
            alphas[name] = alpha
            bits += 16 * w.shape[1]                       # FP16 per-input-channel AWQ scale
        out[name] = q.cpu()
        if actorder and not select and not (overrides and ov):
            bits += 16 * w.shape[1]                   # column order (act-order): a 16-bit index per input column
        bits += w.numel() * BITS + w.shape[0] * math.ceil(w.shape[1] / group) * (16 + BITS)
        if (k + 1) % 50 == 0:
            print(json.dumps(dict(key=calib_dir.name, layers=k + 1)), flush=True)
    return out, bits, alphas, sha256(calib_dir / 'index.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=('gptq', 'awq'), required=True)
    p.add_argument('--group', type=int, default=128)
    p.add_argument('--actorder', action='store_true', help='GPTQ act-order (official option)')
    p.add_argument('--mse', action='store_true', help='GPTQ MSE-searched clipping (official option)')
    p.add_argument('--select', default=None, help='comma list of GPTQ variants (e.g. base,mse,ao_mse_d0.1): per layer, keep the one with the smallest calibration output error; "default" = SELECT_DEFAULT')
    p.add_argument('--percdamp', type=float, default=0.01, help='GPTQ damping (fraction of the mean Hessian diagonal)')
    p.add_argument('--only', default=None, help='regex: quantize only the layers whose name matches; the others stay FP (partial checkpoint)')
    p.add_argument('--skip', default=None, help='regex: leave the matching layers FP')
    p.add_argument('--override', nargs='*', default=[], help='REGEX=variant (gptq_variant name, e.g. base, ao_mse): quantize the matching layers with that variant instead')
    p.add_argument('--checkpoints', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    select = (SELECT_DEFAULT if a.select == 'default' else tuple(a.select.split(','))) if a.select else None
    bb_chosen = {}
    overrides = dict(o.split('=', 1) for o in a.override)
    calib = Path(a.calibration)
    bb, bb_bits, bb_alphas, bb_index = quantize_key(a.method, read_tensors(Path(a.checkpoints) / BB_SUITE), calib / 'backbone', a.group, a.actorder, a.mse, a.percdamp, select, bb_chosen, a.only, a.skip, overrides)
    for suite in SUITES:
        tensors = read_tensors(Path(a.checkpoints) / suite)
        head_chosen = {}
        head, head_bits, head_alphas, head_index = quantize_key(a.method, tensors, calib / f'head_{suite}', a.group, a.actorder, a.mse, a.percdamp, select, head_chosen, a.only, a.skip, overrides)
        notes = dict(gptq=f'pure PTQ, no fine-tuning; GPTQ error feedback (percdamp {a.percdamp}, act-order {a.actorder}, MSE clipping {a.mse}) with the shared per-layer Hessians',
                     awq='pure PTQ, no fine-tuning; AWQ scale search + official auto_clip (q/k projections not clipped) on the shared per-layer Hessians')[a.method]
        extra = dict(partial=dict(only=a.only, skip=a.skip, override=overrides) if (a.only or a.skip or overrides) else None, suite=suite, group=a.group, backbone_bits=BITS, head_bits=BITS, notes=notes,
                     payload_rule=f'{BITS} bits per weight + FP16 scale + {BITS}-bit zero point per group of {a.group}' + (' + FP16 per-input-channel AWQ scale per layer' if a.method == 'awq' else '') + (' + 16-bit column index per input column (act-order)' if a.actorder else ''),
                     calibration_index_sha256=dict(backbone=bb_index, head=head_index), exact_command=' '.join(sys.argv),
                     baseline_commit=subprocess.run(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())
        if select:
            extra.update(gptq_select=list(select), gptq_variant_by_layer=dict(bb_chosen, **head_chosen),
                         payload_rule=extra['payload_rule'] + ' + 16-bit column index per input column of the layers that chose act-order')
            extra['notes'] += f'; per layer, the best of {list(select)} by the calibration output error'
        if a.method == 'awq':
            extra['awq_alpha_by_layer'] = dict(bb_alphas, **head_alphas)
        info = accounting((bb_bits + head_bits) / 8, f'{a.method}_w{BITS}g{a.group}' + ('_best' if select else '') + ('_actorder' if a.actorder else '') + ('_mse' if a.mse else '') + (f'_damp{a.percdamp}' if a.percdamp != 0.01 else ''), BITS, extra)
        write_fakequant_checkpoint(Path(a.checkpoints) / suite, Path(a.out) / suite, dict(bb, **head), info)
        print(json.dumps({k: info[k] for k in ('method', 'suite', 'quantized_linear_bpw', 'whole_model_bpw')}), flush=True)
        del tensors, head
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
