"""QTIP (Tseng et al. 2024) at 2 bits per weight on the exported layers (README step 2b). Pure PTQ with the shared Hessians:
the official per-layer steps of lib/algo/finetune.py:quantize_finetune_decoder_layer (randomized Hadamard incoherence,
block LDL, LDLQ onto the bitshift trellis codebook, pack_trellis) on H = the shared E[x x^T], regularized with the
official regularize_H; the layer and end-to-end fine-tuning that follows in the official script is not run.

Settings: the official 2-bit recipe (example.sh): --codebook bitshift --L 16 --K 2 --V 2 --td_x 16 --td_y 16
--decode_mode quantlut_sym --tlut_bits 9 --scale_override 0.9, sigma_reg 1e-2. K = 2 trellis bits per weight.

Hadamard sizes as in QuIP#: 4304 has no supported size, so SigLIP fc1 (outputs) and fc2 (inputs) are zero-padded to
4480 = 140 * 32 (a multiple of the 16 x 16 tile), quantized and cropped back. Per layer the payload counts the packed
trellis (incl. padding), FP16 SU and SV sign vectors, the FP32 Wscale and the layer's FP16 lookup table (512 x 2), as
the official code saves them.

Usage:
  PYTHONPATH=$QTIP_REPO:$EXT_SITE:$PYTHONPATH python quantize_qtip.py --layers ./layers --calibration ./calibration \
      --checkpoint $CKPT --qtip-repo $QTIP_REPO --out ./qtip_2bit
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

PAD_TO = {4304: 4480}
SETTINGS = dict(codebook='bitshift', L=16, K=2, V=2, td_x=16, td_y=16, decode_mode='quantlut_sym', tlut_bits=9,
                scale_override=0.9, sigma_reg=1e-2)


def quantize_layer(item, layers_dir, out, cb, has_kernel, device):
    from lib.algo import ldlq
    from lib import utils
    src = load_file(str(layers_dir / item['file']))
    w = src['weight'].float()
    h = src['hessian'].float()
    m, n = w.shape
    pm, pn = PAD_TO.get(m, m), PAD_TO.get(n, n)
    W = torch.zeros(pm, pn, device=device)
    W[:m, :n] = w
    HR = torch.zeros(pn, pn, device=device)
    HR[:n, :n] = h
    torch.manual_seed(0)
    began = time.time()
    SU = (torch.randn(pn, device=device).sign() + 1e-5).sign()
    SV = (torch.randn(pm, device=device).sign() + 1e-5).sign()
    HR = utils.regularize_H(HR, SETTINGS['sigma_reg'])
    Wr = utils.matmul_hadUt(utils.matmul_hadUt(W.T * SV).T * SU)
    HRr = utils.matmul_hadUt(utils.matmul_hadUt(HR * SU).T * SU)
    Wscale = Wr.square().mean().sqrt() / (cb.lut.to(torch.float64).square().mean().sqrt().float() * SETTINGS['scale_override'])
    Wr /= Wscale
    LRr, _ = utils.block_LDL(HRr, SETTINGS['td_y'])
    diag = torch.arange(pn, device=LRr.device)
    LRr[diag, diag] = 0
    args = Namespace(**SETTINGS)
    hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, args, for_kernel=has_kernel)
    td_x, td_y, V = SETTINGS['td_x'], SETTINGS['td_y'], SETTINGS['V']
    packed = cb.pack_trellis(Qidxs.cpu().reshape(pm // td_x, td_x, pn // td_y, td_y // V).transpose(1, 2)
                             .reshape(-1, td_x * td_y // V)).view(torch.int16).contiguous()
    hatWr = hatWr * Wscale
    # inverse of the incoherence transform: Wr = had(had(W^T SV)^T SU)  =>  W = (had_U((had_U(Wr) SU)^T) SV)^T
    hatW = (utils.matmul_hadU((utils.matmul_hadU(hatWr) * SU).T) * SV).T
    roundtrip = (utils.matmul_hadU((utils.matmul_hadU(Wr * Wscale) * SU).T) * SV).T   # transform check on W itself
    weight = hatW[:m, :n].float().cpu()
    e = weight - w
    tlut = cb.tlut.data.to(torch.float16).cpu().contiguous()
    su16, sv16 = SU.to(torch.float16).cpu(), SV.to(torch.float16).cpu()
    record = dict(payload_bytes=2 * packed.numel() + 2 * su16.numel() + 2 * sv16.numel() + 4 + 2 * tlut.numel(),
                  trellis_bytes=2 * packed.numel(), sign_vector_bytes=2 * (su16.numel() + sv16.numel()), wscale_bytes=4,
                  tlut_bytes=2 * tlut.numel(), padded_shape=[pm, pn],
                  transform_roundtrip_max_abs=float((roundtrip - W).abs().max()),
                  relative_weight_error=float(e.norm() / w.norm()),
                  relative_output_error=float(((e @ h) * e).sum() / ((w @ h) * w).sum()), seconds=time.time() - began)
    tmp = out / (item['file'] + '.partial')
    save_file(dict(weight=weight.to(torch.bfloat16).contiguous(), trellis_int16=packed.cpu(), su_fp16=su16, sv_fp16=sv16,
                   wscale_fp32=Wscale.reshape(1).float().cpu(), tlut_fp16=tlut), str(tmp))
    tmp.replace(out / item['file'])
    dump_json(out / (Path(item['file']).stem + '.json'), dict(name=item['name'], group=item['group'], **record))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True, help='output of export_layers.py')
    p.add_argument('--calibration', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--qtip-repo', required=True, help='official QTIP checkout (github.com/Cornell-RelaxML/qtip)')
    p.add_argument('--out', required=True)
    p.add_argument('--only', nargs='+', help='quantize only layers whose name contains one of these strings (tests)')
    a = p.parse_args()
    import lib
    from lib.codebook import bitshift
    from lib import utils
    assert Path(a.qtip_repo).resolve() in Path(lib.__file__).resolve().parents, f'QTIP imported from {lib.__file__}'
    device = torch.device('cuda:0')
    torch.set_grad_enabled(False)
    layers_dir, out = Path(a.layers), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((Path(a.calibration) / 'index.json').read_text())
    assert index['checkpoint_sha256'] == sha256(Path(a.checkpoint) / 'model.safetensors'), 'calibration was made from another checkpoint'
    listing = json.loads((layers_dir / 'layers.json').read_text())['layers']
    assert len(listing) == 414
    s = SETTINGS
    cb = bitshift.bitshift_codebook(L=s['L'], K=s['K'], V=s['V'], tlut_bits=s['tlut_bits'], decode_mode=s['decode_mode']).to(device)
    has_kernel = utils.has_kernel(s['decode_mode'], s['L'], s['K'], s['V'], s['tlut_bits'], s['td_x'], s['td_y'])
    meta = dict(method='qtip_bitshift_l16k2v2', bits_nominal=2, backbone_bits=2, expert_bits=2,
                qtip_commit=subprocess.check_output(['git', '-C', a.qtip_repo, 'rev-parse', 'HEAD'], text=True).strip(),
                qtip_settings=SETTINGS, padding={str(k): v for k, v in PAD_TO.items()}, seed='torch.manual_seed(0) before each layer',
                calibration_index_sha256=sha256(Path(a.calibration) / 'index.json'),
                payload_rule='packed trellis (K=2 bits per weight) incl. padded rows/columns + FP16 SU and SV + FP32 Wscale + FP16 512x2 lookup table, per layer',
                notes='pure PTQ, no layer-wise or end-to-end fine-tuning; official QTIP per-layer quantization (RHT + LDLQ, bitshift trellis) on the shared per-layer Hessians; the official layer and end-to-end fine-tuning are not run',
                exact_command=' '.join(sys.argv))
    records = {}
    for k, item in enumerate(listing, 1):
        rec_path = out / (Path(item['file']).stem + '.json')
        if (out / item['file']).exists() and rec_path.exists():
            records[item['name']] = json.loads(rec_path.read_text())
            continue
        if a.only and not any(x in item['name'] for x in a.only):
            continue
        records[item['name']] = quantize_layer(item, layers_dir, out, cb, has_kernel, device)
        print(json.dumps(dict(layer=k, name=item['name'], **{x: records[item['name']][x] for x in ('payload_bytes', 'relative_output_error', 'transform_roundtrip_max_abs', 'seconds')})), flush=True)
        torch.cuda.empty_cache()
    if len(records) < len(listing):
        print(json.dumps(dict(status='incomplete', layers_done=len(records))), flush=True)
        return
    dump_json(out / 'accounting.json', dict(meta, layers=records, linear_payload_bytes=sum(r['payload_bytes'] for r in records.values())))
    print(json.dumps(dict(status='complete', linear_payload_bytes=sum(r['payload_bytes'] for r in records.values()))), flush=True)


if __name__ == '__main__':
    main()
