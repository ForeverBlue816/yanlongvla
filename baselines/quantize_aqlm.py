"""AQLM (Egiazarian et al. 2024) at 2 / 3 bits per weight on the exported layers (README step 2b). Pure PTQ with the shared
Hessians: the official layer quantizer (AQEngine.quantize: k-means init, Adam on codebooks and scales, beam search on
codes, early stop at 1% relative MSE improvement) with XTX = the shared E[x x^T] plus 1% diagonal damping; no block or
global fine-tuning. The damping departs from the official code: our Hessians come from 256 observations and are far from
full rank (expert down_proj: effective rank ~277 of 4096), so undamped AQLM drifts freely in the null space (relative
weight error 0.86 at output error 2e-5). 1% of the mean diagonal is GPTQ's percdamp, used by every other Hessian method here.

W2 = 2x8g8 (two 8-bit codebooks over groups of 8 input weights: 2.000 bits per weight), W3 = 3x8g8 (3.000). Per layer the
payload counts uint8 codes, FP16 codebooks (num_codebooks x 256 x 8) and one FP16 scale per output row (scale_nbits=0,
the official default). Every width here is a multiple of 8, so nothing is padded.

Several processes may run this on the same --out at once (one per GPU job): each claims a layer with a lock file, skips
layers that are done or claimed, and saves <layer>.safetensors (dequantized `weight` plus the stored codes, codebooks
and scales) with <layer>.json. The process that finds all 414 layers done writes accounting.json for import_external.py.

Usage:
  PYTHONPATH=$AQLM_REPO:$PYTHONPATH python quantize_aqlm.py --layers ./layers --calibration ./calibration \
      --checkpoint $CKPT --aqlm-repo $AQLM_REPO --bits 2 --out ./aqlm_2x8g8_w2
"""
import argparse
import fcntl
import json
import random
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from common import dump_json, sha256

IN_GROUP = 8
PERCDAMP = 0.01
NBITS = 8
SETTINGS = dict(out_group_size=1, in_group_size=IN_GROUP, nbits_per_codebook=NBITS, codebook_value_nbits=16,
                codebook_value_num_groups=1, scale_nbits=0, init_max_iter=100, init_max_points_per_centroid=None,
                lr=1e-4, max_epochs=1000, steps_per_epoch=100, relative_mse_tolerance=0.01, beam_size=1)   # official defaults + README recipe


def quantize_layer(item, layers_dir, out, codebooks_per_layer, device):
    from aq_engine import AQEngine
    from src.utils import _dequantize_weight
    src = load_file(str(layers_dir / item['file']))
    layer = torch.nn.Linear(item['in_features'], item['out_features'], bias=False, device=device, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(src['weight'].float())
    engine = AQEngine(layer)
    xtx = src['hessian'].to(device=device, dtype=engine.XTX.dtype)   # shared E[x x^T] (= official XTX / nsamples)
    xtx.diagonal().add_(PERCDAMP * xtx.diagonal().mean())
    engine.XTX = xtx
    random.seed(0)
    torch.manual_seed(0)
    began = time.time()
    qw = engine.quantize(args=Namespace(devices=[device], num_codebooks=codebooks_per_layer, print_frequency=10 ** 9, **SETTINGS), verbose=False)
    with torch.no_grad():
        codes = qw.get_codes().to(torch.uint8).contiguous().cpu()
        codebooks = qw.get_codebooks().to(torch.float16).contiguous().cpu()
        scales = qw.get_scales().to(torch.float16).contiguous().cpu()
        runtime = qw().float().cpu()
        weight = _dequantize_weight(codes.long(), codebooks.float(), scales.float())   # exactly what the payload encodes
        ref = src['weight'].float()
        h = src['hessian'].float()
        e = weight - ref
        record = dict(payload_bytes=codes.numel() + 2 * codebooks.numel() + 2 * scales.numel(),
                      code_bytes=codes.numel(), codebook_bytes=2 * codebooks.numel(), scale_bytes=2 * scales.numel(),
                      relative_weight_error=float(e.norm() / ref.norm()),
                      relative_output_error=float(((e @ h) * e).sum() / ((ref @ h) * ref).sum()),
                      max_abs_diff_payload_vs_runtime=float((weight - runtime).abs().max()), seconds=time.time() - began)
    tmp = out / (item['file'] + '.partial')
    save_file(dict(weight=weight.to(torch.bfloat16).contiguous(), codes_u8=codes, codebooks_fp16=codebooks, scales_fp16=scales), str(tmp))
    tmp.replace(out / item['file'])
    dump_json(out / (Path(item['file']).stem + '.json'), dict(name=item['name'], group=item['group'], **record))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True, help='output of export_layers.py')
    p.add_argument('--calibration', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--aqlm-repo', required=True, help='official AQLM checkout (github.com/Vahe1994/AQLM)')
    p.add_argument('--bits', type=int, choices=(2, 3), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-layers', type=int, help='stop after this many layers (tests)')
    p.add_argument('--only', nargs='+', help='quantize only layers whose name contains one of these strings (tests)')
    a = p.parse_args()
    import aq_engine
    assert Path(a.aqlm_repo).resolve() in Path(aq_engine.__file__).resolve().parents, f'AQLM imported from {aq_engine.__file__}'
    device = torch.device('cuda:0')
    layers_dir, out = Path(a.layers), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((Path(a.calibration) / 'index.json').read_text())
    assert index['checkpoint_sha256'] == sha256(Path(a.checkpoint) / 'model.safetensors'), 'calibration was made from another checkpoint'
    listing = json.loads((layers_dir / 'layers.json').read_text())['layers']
    assert len(listing) == 414 and all(i['in_features'] % IN_GROUP == 0 for i in listing)
    meta = dict(method=f'aqlm_{a.bits}x8g8', bits_nominal=a.bits, backbone_bits=a.bits, expert_bits=a.bits,
                structure=dict(num_codebooks=a.bits, nbits_per_codebook=NBITS, in_group_size=IN_GROUP, out_group_size=1, scales='FP16 per output row'),
                aqlm_commit=subprocess.check_output(['git', '-C', a.aqlm_repo, 'rev-parse', 'HEAD'], text=True).strip(),
                aqlm_settings=SETTINGS, percdamp=PERCDAMP, seed='random.seed(0) and torch.manual_seed(0) before each layer',
                calibration_index_sha256=sha256(Path(a.calibration) / 'index.json'),
                payload_rule='uint8 code per codebook per 8 input weights + FP16 codebooks + FP16 per-row scale, all counted',
                notes='pure PTQ, no layer-wise or end-to-end fine-tuning; official AQLM layer quantizer (AQEngine.quantize) with the shared per-layer Hessians (+1% diagonal damping) as XTX; no block or global fine-tuning',
                exact_command=' '.join(sys.argv))
    meta_path = out / 'meta.json'
    if meta_path.exists():
        assert json.loads(meta_path.read_text()) == meta | dict(exact_command=json.loads(meta_path.read_text())['exact_command']), 'existing output was made with other settings; use a new --out'
    else:
        dump_json(meta_path, meta)

    done_now = 0
    for k, item in enumerate(listing, 1):
        if (out / item['file']).exists() or (a.only and not any(s in item['name'] for s in a.only)):
            continue
        lock = open(out / (item['file'] + '.lock'), 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # another job is quantizing this layer
        except OSError:
            continue
        if not (out / item['file']).exists():
            record = quantize_layer(item, layers_dir, out, a.bits, device)
            print(json.dumps(dict(layer=k, name=item['name'], **{x: record[x] for x in ('payload_bytes', 'relative_output_error', 'seconds')})), flush=True)
            done_now += 1
        lock.close()
        torch.cuda.empty_cache()
        if a.max_layers and done_now >= a.max_layers:
            break

    records = {}
    for item in listing:
        path = out / (Path(item['file']).stem + '.json')
        if (out / item['file']).exists() and path.exists():
            records[item['name']] = json.loads(path.read_text())
    if len(records) < len(listing):
        print(json.dumps(dict(status='incomplete', layers_done=len(records), quantized_by_this_process=done_now)), flush=True)
        return
    acc = dict(meta, layers=records, linear_payload_bytes=sum(r['payload_bytes'] for r in records.values()))
    dump_json(out / 'accounting.json', acc)
    print(json.dumps(dict(status='complete', linear_payload_bytes=acc['linear_payload_bytes'])), flush=True)


if __name__ == '__main__':
    main()
