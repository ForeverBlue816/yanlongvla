"""VPTQ (Liu et al. 2024) at a true 2 / 3 bits per weight on the exported layers (README step 2b). Pure PTQ with the shared
Hessians; no layer-wise or end-to-end fine-tuning.

W2 = v4-k256-0, W3 = v4-k256-16 (same main codebook plus a 16-entry residual codebook, 4-bit residual index per vector).

Each layer: vectors of 4 consecutive output rows in one input column, one 256-entry k-means codebook per layer
(Hessian-weighted, official microsoft/VPTQ `algorithm` branch), no outliers, no norm/permutation.
Index cost is log2(256) / 4 = 2 bits per weight (+ log2(16) / 4 = 1 for W3); the FP16 codebooks (256 x 4 x 2 B = 2 KiB,
+128 B residual, per layer) are added to the payload, so quantizable_linear_bpw is slightly above 2 / 3. Writes
<layer>.safetensors (dequantized `weight` plus the stored codebooks and codes; W3 residual codes packed two per byte) and
accounting.json for import_external.py. Resumable: completed layers are skipped.

Usage:
  PYTHONPATH=$VPTQ_REPO:$PYTHONPATH python quantize_vptq.py --layers ./layers --calibration ./calibration \
      --checkpoint $CKPT --vptq-repo $VPTQ_REPO --bits 2 --out ./vptq_v4k256_w2
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from common import dump_json, sha256

VECTOR_LEN = 4
CENTROIDS = 256
RESIDUAL_CENTROIDS = {2: 0, 3: 16}
PERCDAMP = 0.01
KMEANS_ITER = 100
KMEANS_TOL = 1e-5


def prepare_hessian(raw, device):
    """VPTQ normalizes H by its mean diagonal and needs the upper Cholesky factor of the damped inverse.
    The shared H is already the uncentered E[x x^T], so no mean term is added."""
    h = raw.to(device=device, dtype=torch.float32)
    h = (h + h.T) / 2
    dead = torch.diagonal(h) == 0
    h = h / torch.diagonal(h).mean()
    h[dead, :] = 0
    h[:, dead] = 0
    damped = h.clone()
    damped.diagonal().add_(PERCDAMP)
    damped[dead, dead] = 1
    inv = torch.cholesky_inverse(torch.linalg.cholesky(damped))
    return h, torch.linalg.cholesky(inv, upper=True)


def pack_u4(x):
    x = x.reshape(-1).to(torch.uint8)
    x = torch.cat((x, x.new_zeros(x.numel() % 2)))
    return (x[0::2] | (x[1::2] << 4)).contiguous().cpu()


def unpack_u4(packed, count):
    return torch.stack((packed & 15, packed >> 4), dim=1).reshape(-1)[:count]


def decode(codebook, codes, out_features, in_features, res_codebook=None, res_packed=None):
    """Stored payload -> (out, in) weight: codes[g, j] picks the 4-vector for output rows 4g..4g+3 of input column j."""
    groups = (out_features + VECTOR_LEN - 1) // VECTOR_LEN
    assert tuple(codes.shape) == (groups, in_features), codes.shape
    v = codebook.float()[codes.long()]
    if res_codebook is not None:
        v = v + res_codebook.float()[unpack_u4(res_packed, codes.numel()).long().reshape(codes.shape)]
    w = v.permute(1, 0, 2).reshape(in_features, groups * VECTOR_LEN)
    return w[:, :out_features].T.contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True, help='output of export_layers.py')
    p.add_argument('--calibration', required=True, help='output of calibrate.py (for hashes and sample counts)')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--vptq-repo', required=True, help='microsoft/VPTQ algorithm-branch checkout')
    p.add_argument('--bits', type=int, choices=(2, 3), required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    res = RESIDUAL_CENTROIDS[a.bits]
    import vptq
    from vptq.quantizer import NPVectorQuantizer
    from vptq.vptq import VPTQ
    assert Path(a.vptq_repo).resolve() in Path(vptq.__file__).resolve().parents, f'vptq imported from {vptq.__file__}'
    device = torch.device('cuda:0')
    log = logging.getLogger('vptq')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    index = json.loads((Path(a.calibration) / 'index.json').read_text())
    assert index['checkpoint_sha256'] == sha256(Path(a.checkpoint) / 'model.safetensors'), 'calibration was made from another checkpoint'
    listing = json.loads((Path(a.layers) / 'layers.json').read_text())['layers']
    assert len(listing) == 414, len(listing)
    meta = dict(method=f'vptq_v4_k256_r{res}', bits_nominal=a.bits, backbone_bits=a.bits, expert_bits=a.bits,
                structure=dict(vector_len=VECTOR_LEN, centroids=CENTROIDS, residual_centroids=res, outlier_percent=0, codebooks_per_layer=1),
                vptq_commit=subprocess.check_output(['git', '-C', a.vptq_repo, 'rev-parse', 'HEAD'], text=True).strip(),
                kmeans=dict(mode='hessian', init='random', seed=0, iterations=KMEANS_ITER, tol=KMEANS_TOL), percdamp=PERCDAMP,
                calibration_index_sha256=sha256(Path(a.calibration) / 'index.json'),
                payload_rule='uint8 index per 4-weight vector (2 bits/weight) + FP16 256x4 codebook per layer'
                             + ('; + packed 4-bit residual index per vector (1 bit/weight) + FP16 16x4 residual codebook' if res else '') + '; padded output rows counted',
                notes='pure PTQ, no layer-wise or end-to-end fine-tuning; Hessian-weighted k-means and VPTQ error feedback with the shared per-layer Hessians',
                exact_command=' '.join(sys.argv))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    acc_path = out / 'accounting.json'
    acc = json.loads(acc_path.read_text()) if acc_path.exists() else dict(meta, layers={})
    assert all(acc.get(k) == v for k, v in meta.items()), 'existing output was made with other settings; use a new --out'

    for k, item in enumerate(listing, 1):
        name, file = item['name'], out / item['file']
        if name in acc['layers'] and file.exists():
            continue
        src = load_file(str(Path(a.layers) / item['file']))
        linear = torch.nn.Linear(item['in_features'], item['out_features'], bias=False, device=device, dtype=src['weight'].dtype)
        with torch.no_grad():
            linear.weight.copy_(src['weight'].to(device))
        h, inv_factor = prepare_hessian(src['hessian'], device)
        quantizer = NPVectorQuantizer(layer_name=name, logger=log, vector_lens=[-1, VECTOR_LEN], num_centroids=[-1, CENTROIDS],
                                      num_res_centroids=[-1, res if res else -1], npercent=0, group_size=-1, group_num=1, kmeans_mode='hessian',
                                      kmeans_seed=0, iter=KMEANS_ITER, tol=KMEANS_TOL, enable_norm=False, norm_dim=0, enable_perm=False, debug=False)
        VPTQ(linear, quantizer=quantizer, hessian=h, inv_hessian=inv_factor, perm=None, zero_idx=None, logger=log, collect_act=False,
             layer_name=name, block_size=128, step=1, percdamp=PERCDAMP, enable_perm=False, enable_norm=False, norm_dim=0, debug=False).fast_vector_quant()
        codebook = quantizer.centroids[1].to(torch.float16).contiguous().cpu()
        codes = quantizer.indices[1].to(torch.uint8).contiguous().cpu()
        assert tuple(codebook.shape) == (CENTROIDS, VECTOR_LEN), codebook.shape
        stored = dict(codebook_fp16=codebook, codes_u8=codes)
        if res:
            stored.update(res_codebook_fp16=quantizer.res_centroids[1].to(torch.float16).contiguous().cpu(), res_codes_u4=pack_u4(quantizer.res_indices[1]))
            assert tuple(stored['res_codebook_fp16'].shape) == (res, VECTOR_LEN)
        weight = decode(codebook, codes, item['out_features'], item['in_features'], stored.get('res_codebook_fp16'), stored.get('res_codes_u4')).to(torch.bfloat16)
        assert torch.isfinite(weight).all(), name
        nbytes = {k: v.numel() * v.element_size() for k, v in stored.items()}
        save_file(dict(stored, weight=weight), str(file))
        acc['layers'][name] = dict(payload_bytes=sum(nbytes.values()), codebook_bytes=sum(v for k, v in nbytes.items() if 'codebook' in k),
                                   stored_bytes=nbytes, samples=index['layers'][name]['samples'], group=item['group'])
        dump_json(acc_path, acc)
        print(json.dumps(dict(layer=k, name=name, payload_bytes=acc['layers'][name]['payload_bytes'])), flush=True)
        torch.cuda.empty_cache()

    assert len(acc['layers']) == 414
    payload = sum(v['payload_bytes'] for v in acc['layers'].values())
    acc['linear_payload_bytes'] = payload
    acc['codebook_bytes'] = sum(v['codebook_bytes'] for v in acc['layers'].values())
    dump_json(acc_path, acc)
    print(json.dumps(dict(status='complete', linear_payload_bytes=payload, codebook_bytes=acc['codebook_bytes'])), flush=True)


if __name__ == '__main__':
    main()
