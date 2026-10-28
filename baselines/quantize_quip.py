"""QuIP# (Tseng et al. 2024) at 2 or 3 bits per weight on the exported layers (README step 2b). Pure PTQ with the shared Hessians:
the official per-layer quantizer (lib/algo/quip.py:quantize: randomized Hadamard incoherence, block LDL, LDLQ onto the
E8P12 lattice codebook) on H = the shared E[x x^T], regularized as the official quantize_linear does; no block or
end-to-end fine-tuning.

Settings follow the official 2-bit recipe (quantize_llama/example.sh: --codebook E8P12 --scale_override 0.9) and the
script defaults (sigma_reg 1e-2, incoh_mode had, lora_rank 0, quip_tune_iters 10). One 16-bit E8P12 index per 8
weights = 2.000 bits per weight, stored packed four to an int64 as the official code does.

Hadamard sizes: QuIP# supports n = 2^k * K for K in its table. 1152 = 36 * 32 is supported; 4304 = 16 * 269 is not, so
SigLIP fc1 (4304 outputs) and fc2 (4304 inputs) are zero-padded to 4480 = 140 * 32 (zero rows of W, or zero rows and
columns of H plus the official regularization), quantized, and cropped back. The payload counts the padded codes and
both full-length sign vectors (SU per input, SV per output; FP16), and the E8P12 table once for the model.

Usage:
  PYTHONPATH=$QUIP_REPO:$EXT_SITE:$PYTHONPATH python quantize_quip.py --layers ./layers --calibration ./calibration \
      --checkpoint $CKPT --quip-repo $QUIP_REPO --out ./quip_e8p12_w2
"""
import argparse
import json
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from common import dump_json, sha256

PAD_TO = {4304: 4480}   # widths without a QuIP# Hadamard size -> padded width (140 * 32)
# official recipes (README of the QuIP# repository): E8P12 with scale_override 0.9 at 2 bits; E8P12RVQ3B (E8P12 plus a residual
# stage) with the default scales at 3 bits
BITS = {2: dict(codebook='E8P12', scale_override=0.9), 3: dict(codebook='E8P12RVQ3B', scale_override=-1)}
SETTINGS = dict(codebook='E8P12', scale_override=0.9, sigma_reg=1e-2, sigma_reg2=1e-2, incoh_mode='had', lora_rank=0,
                resid_scale_override=-1, quip_tune_iters=10, use_fp64=False, full_svd=False, no_use_buffered=False,
                rescale_WH=False, lowmem_ldlq=False)


def quantize_layer(item, layers_dir, out, cb, scratch, device):
    from lib.algo import quip
    from lib import utils
    src = load_file(str(layers_dir / item['file']))
    w = src['weight'].float()
    h = src['hessian'].double()
    m, n = w.shape
    pm, pn = PAD_TO.get(m, m), PAD_TO.get(n, n)
    wp = torch.zeros(pm, pn, dtype=torch.float32)
    wp[:m, :n] = w
    hp = torch.zeros(pn, pn, dtype=torch.float64)
    hp[:n, :n] = h
    scale = wp.square().mean().sqrt()   # official quantize_linear: quantize W / rms(W), fuse rms(W) into SV
    hp = utils.regularize_H(hp, pn, SETTINGS['sigma_reg'])
    torch.manual_seed(0)
    began = time.time()
    hat, attr = quip.quantize(hp.float().to(device), (wp / scale).to(device), SETTINGS['lora_rank'], cb,
                              Namespace(save_pfx=str(scratch), **SETTINGS), device)
    sv16 = (attr['SV'].float() * scale).to(torch.float16)   # stored sign vector with both scales fused (official layout)
    # The evaluated weight is QuIP#'s own reconstruction; decoding the stored FP16 SV differs from it only by the FP16
    # rounding of the fused per-layer scale (relative ~5e-4), recorded below.
    weight = (hat.float().cpu() * scale)[:m, :n]
    qidxs = attr['Qidxs'].cpu().contiguous()
    su16 = attr['SU'].to(torch.float16).cpu().contiguous()
    e = weight - w
    record = dict(payload_bytes=qidxs.numel() * qidxs.element_size() + 2 * su16.numel() + 2 * sv16.numel(),
                  code_bytes=qidxs.numel() * qidxs.element_size(), sign_vector_bytes=2 * (su16.numel() + sv16.numel()),
                  padded_shape=[pm, pn], fp16_scale_rounding=float(abs(sv16.float().abs().mean() / (attr['SV'].float().abs().mean() * scale) - 1)),
                  relative_weight_error=float(e.norm() / w.norm()),
                  relative_output_error=float(((e @ h.float()) * e).sum() / ((w @ h.float()) * w).sum()),
                  seconds=time.time() - began)
    tmp = out / (item['file'] + '.partial')
    save_file(dict(weight=weight.to(torch.bfloat16).contiguous(), qidxs_packed=qidxs, su_fp16=su16, sv_fp16=sv16.cpu().contiguous()), str(tmp))
    tmp.replace(out / item['file'])
    dump_json(out / (Path(item['file']).stem + '.json'), dict(name=item['name'], group=item['group'], **record))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True, help='output of export_layers.py')
    p.add_argument('--calibration', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--quip-repo', required=True, help='official QuIP# checkout (github.com/Cornell-RelaxML/quip-sharp)')
    p.add_argument('--out', required=True)
    p.add_argument('--only', nargs='+', help='quantize only layers whose name contains one of these strings (tests)')
    p.add_argument('--bits', type=int, choices=(2, 3), default=2, help='2: E8P12 (2.000 bpw); 3: E8P12RVQ3B (3.000 bpw)')
    p.add_argument('--tune-iters', type=int, default=SETTINGS['quip_tune_iters'], help='official default 10; 0 is a diagnostic variant')
    a = p.parse_args()
    SETTINGS['quip_tune_iters'] = a.tune_iters
    SETTINGS.update(BITS[a.bits])
    from lib import codebook
    import lib
    assert Path(a.quip_repo).resolve() in Path(lib.__file__).resolve().parents, f'QuIP# imported from {lib.__file__}'
    device = torch.device('cuda:0')
    layers_dir, out = Path(a.layers), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scratch').mkdir(exist_ok=True)
    index = json.loads((Path(a.calibration) / 'index.json').read_text())
    assert index['checkpoint_sha256'] == sha256(Path(a.checkpoint) / 'model.safetensors'), 'calibration was made from another checkpoint'
    listing = json.loads((layers_dir / 'layers.json').read_text())['layers']
    assert len(listing) == 414
    cb = codebook.get_codebook(SETTINGS['codebook'])
    table_bytes = sum(t.numel() * t.element_size() for t in cb.buffers())   # E8P12 lattice table, shared by all layers
    meta = dict(method=f"quip_sharp_{SETTINGS['codebook'].lower()}", bits_nominal=a.bits, backbone_bits=a.bits, expert_bits=a.bits,
                quip_commit=subprocess.check_output(['git', '-C', a.quip_repo, 'rev-parse', 'HEAD'], text=True).strip(),
                quip_settings=SETTINGS, padding={str(k): v for k, v in PAD_TO.items()}, seed='torch.manual_seed(0) before each layer',
                codebook_table_bytes=table_bytes, calibration_index_sha256=sha256(Path(a.calibration) / 'index.json'),
                payload_rule=f"packed {SETTINGS['codebook']} indices (all stages as stored by the official code; 16 bits per 8 weights per stage at 2 bits) incl. padded rows/columns + FP16 SU and SV sign vectors (scales fused into SV) per layer + the shared lattice tables once",
                notes='pure PTQ, no layer-wise or end-to-end fine-tuning; official QuIP# per-layer quantizer (RHT + LDLQ, E8P12) on the shared per-layer Hessians; no block or end-to-end fine-tuning',
                exact_command=' '.join(sys.argv))
    acc_path = out / 'accounting.json'
    records = {}
    for k, item in enumerate(listing, 1):
        rec_path = out / (Path(item['file']).stem + '.json')
        if (out / item['file']).exists() and rec_path.exists():
            records[item['name']] = json.loads(rec_path.read_text())
            continue
        if a.only and not any(s in item['name'] for s in a.only):
            continue
        records[item['name']] = quantize_layer(item, layers_dir, out, cb, out / 'scratch', device)
        print(json.dumps(dict(layer=k, name=item['name'], **{x: records[item['name']][x] for x in ('payload_bytes', 'relative_output_error', 'seconds')})), flush=True)
        torch.cuda.empty_cache()
    if len(records) < len(listing):
        print(json.dumps(dict(status='incomplete', layers_done=len(records))), flush=True)
        return
    layers = {n: dict(r) for n, r in records.items()}
    first = listing[0]['name']
    layers[first]['payload_bytes'] += table_bytes   # count the shared codebook table once
    dump_json(acc_path, dict(meta, layers=layers, linear_payload_bytes=sum(r['payload_bytes'] for r in layers.values())))
    print(json.dumps(dict(status='complete', linear_payload_bytes=sum(r['payload_bytes'] for r in layers.values()))), flush=True)


if __name__ == '__main__':
    main()
