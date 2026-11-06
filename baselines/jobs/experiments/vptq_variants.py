"""Task 0: is VPTQ W2's large action error (0.102, SigLIP layers) a property of the fixed configuration (v4-k256-0) or of how we run it?

Re-quantizes a few layers with VPTQ under several settings and measures, per layer, the relative weight error and the relative
output error tr(dW H dW^T) / tr(W H W^T) on the shared calibration Hessian and on the held-out Hessian (experiments/heldout).
Layers: the worst of the row (SigLIP mlp.fc2 of encoder layers 20-23, q/k of layers 0-3) and a few controls where VPTQ is fine.
Variants (decision D8):
  base       the row's settings: v4, 256 centroids, 100 k-means iterations, 1% damping, no norm / perm / outliers
  iter500    500 k-means iterations
  damp10     10% damping of the error-feedback Hessian
  norm       per-row normalization (VPTQ enable_norm)
  perm       column permutation by Hessian diagonal (VPTQ enable_perm)
  normperm   both
  outlier1   1% outlier columns kept in 16 bit (VPTQ npercent=1; counted as extra bits, shown in the table)
  v8k65536   the official 2-bit configuration of VPTQ (8-weight vectors, 65536 centroids), for reference
No result of this script enters a table of rows; it only explains one.

  PYTHONPATH=$SRC:$VPTQ_REPO:... python vptq_variants.py --layers RUN/layers --calibration RUN/calibration \
      --heldout RUN/experiments/heldout/calibration --vptq-repo $VPTQ_REPO --out DIR
"""
import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from quantize_vptq import prepare_hessian, decode

LAYERS = [
    'encoder.layers.20.mlp.fc2', 'encoder.layers.21.mlp.fc2', 'encoder.layers.22.mlp.fc2', 'encoder.layers.23.mlp.fc2',
    'encoder.layers.0.self_attn.k_proj', 'encoder.layers.2.self_attn.k_proj', 'encoder.layers.2.self_attn.q_proj', 'encoder.layers.3.self_attn.k_proj',
    # controls: layers where VPTQ is close to the other rows
    'encoder.layers.10.mlp.fc2', 'encoder.layers.10.self_attn.v_proj', 'paligemma.model.language_model.layers.5.mlp.down_proj', 'gemma_expert.model.layers.5.mlp.down_proj',
]
VARIANTS = dict(
    base=dict(),
    iter500=dict(iter=500),
    iter2000=dict(iter=2000),
    iter5000=dict(iter=5000),
    damp10=dict(percdamp=0.1),
    norm=dict(enable_norm=True),
    perm=dict(enable_perm=True),
    normperm=dict(enable_norm=True, enable_perm=True),
    outlier1=dict(npercent=1),
    v8k65536=dict(vector_len=8, centroids=65536),
    # GOAL 31 Oct item c.2: other structures; kept as candidates only when the codebook-counted bits per weight stay at 2.0 +- 0.02 (see bpw())
    v3k64=dict(vector_len=3, centroids=64),
    v5k1024=dict(vector_len=5, centroids=1024),
    v6k2048=dict(vector_len=6, centroids=2048),
    v6k4096=dict(vector_len=6, centroids=4096),
    v8k4096=dict(vector_len=8, centroids=4096),
)


def variant_cfg(name):
    """A fixed name from VARIANTS or a generated one: v<L>k<K> (L-weight vectors, K centroids), iter<N> / iter<N>t0 (N k-means iterations;
    t0 = tolerance 1e-30 (cuml k-means refuses tol <= 0), so that practically all N iterations run)."""
    if name in VARIANTS:
        return VARIANTS[name]
    m = re.fullmatch(r'v(\d+)k(\d+)', name)
    if m:
        return dict(vector_len=int(m[1]), centroids=int(m[2]))
    m = re.fullmatch(r'iter(\d+)(t0)?', name)
    if m:
        return dict(iter=int(m[1]), **(dict(tol=1e-30) if m[2] else {}))
    raise KeyError(name)


def bpw(out_f, in_f, vector_len, centroids):
    """Bits per weight of one layer with a single FP16 codebook: ceil(log2 k)-bit index per vector (output rows padded to a multiple of
    the vector length) + the codebook."""
    groups = -(-out_f // vector_len)
    return (groups * in_f * math.log2(centroids) + centroids * vector_len * 16) / (out_f * in_f)   # log2 k bits per index: mixed-radix packing for k not a power of two (assumption A6)


def run(item, src, h_cal, device, log, v):
    from vptq.quantizer import NPVectorQuantizer
    from vptq.vptq import VPTQ
    cfg = dict(vector_len=4, centroids=256, iter=100, tol=1e-5, percdamp=0.01, enable_norm=False, enable_perm=False, npercent=0)
    cfg.update(v)
    linear = torch.nn.Linear(item['in_features'], item['out_features'], bias=False, device=device, dtype=src['weight'].dtype)
    with torch.no_grad():
        linear.weight.copy_(src['weight'].to(device))
    import quantize_vptq as q
    q.PERCDAMP = cfg['percdamp']
    h, inv_factor = prepare_hessian(src['hessian'], device)
    quantizer = NPVectorQuantizer(layer_name=item['name'], logger=log, vector_lens=[-1, cfg['vector_len']], num_centroids=[-1, cfg['centroids']],
                                  num_res_centroids=[-1, -1], npercent=cfg['npercent'], group_size=-1, group_num=1, kmeans_mode='hessian', kmeans_seed=0,
                                  iter=cfg['iter'], tol=cfg['tol'], enable_norm=cfg['enable_norm'], norm_dim=0, enable_perm=cfg['enable_perm'], debug=False)
    obj = VPTQ(linear, quantizer=quantizer, hessian=h, inv_hessian=inv_factor, perm=None, zero_idx=None, logger=log, collect_act=False, layer_name=item['name'],
               block_size=128, step=1, percdamp=cfg['percdamp'], enable_perm=cfg['enable_perm'], enable_norm=cfg['enable_norm'], norm_dim=0, debug=False)
    obj.fast_vector_quant()
    return obj.qweight.detach().float(), quantizer, cfg


def errors(w, qw, hessians):
    dw = (w - qw).float()
    out = dict(weight_error=(dw.norm() / w.norm()).item())
    for name, h in hessians.items():
        out[f'output_error_{name}'] = ((dw @ h * dw).sum() / (w @ h * w).sum()).item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--heldout', default=None, help='held-out Hessians (pi0.5 only); omitted for GR00T')
    p.add_argument('--quant-calibration', default=None, help='quantize with the Hessians of this calibration directory instead of --calibration (a larger set, GOAL 1 Nov 1.3); errors are still reported on --calibration and --heldout')
    p.add_argument('--layer-substr', nargs='+', default=LAYERS, help='layers whose name ends with one of these')
    p.add_argument('--all-layers-json', nargs='+', default=None, help='layers.json used for the whole-set bpw of every structure')
    p.add_argument('--vptq-repo', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--variants', nargs='+', default=list(VARIANTS))
    a = p.parse_args()
    import vptq
    assert Path(a.vptq_repo).resolve() in Path(vptq.__file__).resolve().parents
    device = torch.device('cuda:0')
    log = logging.getLogger('vptq'); log.setLevel(logging.WARNING)
    log.addHandler(logging.StreamHandler(sys.stdout))
    listing = json.loads((Path(a.layers) / 'layers.json').read_text())['layers']
    quant_index = json.loads((Path(a.quant_calibration) / 'index.json').read_text())['layers'] if a.quant_calibration else None
    held_index = json.loads((Path(a.heldout) / 'index.json').read_text())['layers'] if a.heldout else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / 'results.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for item in listing:
        if not any(item['name'].endswith(s) for s in a.layer_substr):
            continue
        src = load_file(str(Path(a.layers) / item['file']))
        w = src['weight'].float().to(device)
        h_cal = src['hessian'].float().to(device)
        h_held = torch.load(Path(a.heldout) / held_index[item['name']]['file'], map_location=device, weights_only=True)['covariance'].float() if a.heldout else None
        h_quant = torch.load(Path(a.quant_calibration) / quant_index[item['name']]['file'], map_location=device, weights_only=True)['covariance'].float() if a.quant_calibration else h_cal
        for vname in a.variants:
            key = f"{item['name'].split('model.')[-1]}|{vname}"
            if key in res:
                continue
            t = time.time()
            try:
                qw, quantizer, cfg = run(item, src, h_quant, device, log, variant_cfg(vname))
            except Exception as e:   # a variant that the toolkit cannot run on this layer is reported, not fatal
                res[key] = dict(error=repr(e)[:300]); res_path.write_text(json.dumps(res, indent=1)); print(key, 'failed', repr(e)[:200], flush=True)
                continue
            rec = errors(w, qw.to(device), dict(calibration=h_cal, **(dict(heldout=h_held) if h_held is not None else {}), **(dict(quant_set=h_quant) if a.quant_calibration else {})))
            rec.update(seconds=round(time.time() - t, 1), config={k: v for k, v in cfg.items()}, bpw_this_layer=bpw(item['out_features'], item['in_features'], cfg['vector_len'], cfg['centroids']))
            res[key] = rec
            res_path.write_text(json.dumps(res, indent=1))
            print(key, json.dumps(rec), flush=True)
            torch.cuda.empty_cache()
    names = sorted({k.split('|')[0] for k in res})
    lines = ['| layer | ' + ' | '.join(a.variants) + ' |', '|---|' + '---|' * len(a.variants)]
    for n in names:
        lines.append(f'| {n} | ' + ' | '.join((f"{res[f'{n}|{v}']['output_error_calibration']:.4f}" + (f" / {res[f'{n}|{v}']['output_error_heldout']:.4f}" if 'output_error_heldout' in res[f'{n}|{v}'] else '') if f'{n}|{v}' in res and 'error' not in res[f'{n}|{v}'] else 'n/a') for v in a.variants) + ' |')
    extra = ''
    if a.all_layers_json:   # whole-set bits per weight (codebooks counted) of every structure
        allj = [i for f in a.all_layers_json for i in json.loads(Path(f).read_text())['layers']]
        rows = ['| variant | bits per weight over the layer set |', '|---|---|']
        for v in a.variants:
            c = {**dict(vector_len=4, centroids=256), **variant_cfg(v)}
            tot = sum(bpw(i['out_features'], i['in_features'], c['vector_len'], c['centroids']) * i['out_features'] * i['in_features'] for i in allj)
            rows.append(f"| {v} | {tot / sum(i['out_features'] * i['in_features'] for i in allj):.4f} |")
        extra = '\n\n' + '\n'.join(rows)
    (out / 'tables.md').write_text('Output error on the calibration' + (' / the held-out' if a.heldout else '') + ' Hessian\n\n' + '\n'.join(lines) + extra + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
