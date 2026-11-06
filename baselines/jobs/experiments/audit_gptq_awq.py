"""Task 1 audit of GPTQ and AWQ on GR00T layers (GOAL 31 Oct, items a and b).

For a few layers (vision, language model, action head) and the same shared Hessian:
  1. my GPTQ (baselines/quantize.py) against the official GPTQ (IST-DASLab/gptq, GPTQ.fasterquant on a layer whose Hessian is set directly):
     relative weight error and relative output error tr(dW H dW^T) / tr(W H W^T) must agree;
  2. my GPTQ, RTN and AWQ at 4, 3 and 2 bits (errors must fall steeply with the bits);
  3. 2-bit GPTQ variants: act-order, MSE-searched clipping, 10% damping, and combinations;
  4. AWQ: my pseudo-quantizer against the official one (llm-awq pseudo_quantize_tensor), and my clipping search against the
     official auto_clip_layer on activations sampled from N(0, H) (the Hessian is the only activation statistic we keep).
Writes results.json and tables.md in --out.

  python audit_gptq_awq.py --layers RUN/layers --gptq-repo TOOLKITS/gptq --awq-repo TOOLKITS/llm-awq --out DIR
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'groot'))
from groot_common import parent_module   # loads baselines/quantize.py (the π0.5 harness) in the GR00T environment

Q = parent_module('quantize')

PICK = [('backbone', 'visual.blocks.20.mlp.linear_fc2'), ('backbone', 'visual.blocks.3.attn.qkv'),
        ('backbone', 'language_model.layers.2.mlp.down_proj'), ('backbone', 'language_model.layers.15.self_attn.q_proj'),
        ('head_libero_spatial', 'transformer_blocks.14.ff.net.0.proj'), ('head_libero_spatial', 'transformer_blocks.8.attn1.to_k')]


def errors(w, q, h):
    dw = (w - q).float()
    return dict(weight_error=(dw.norm() / w.norm()).item(), output_error=(((dw @ h) * dw).sum() / ((w @ h) * w).sum()).item())


def official_gptq(w, h, bits, group, gptq_repo, percdamp=0.01, actorder=False, mse=False):
    sys.path.insert(0, str(gptq_repo))
    import gptq as og
    import quant as oq
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False).cuda()
    layer.weight.data = w.clone().float()
    g = og.GPTQ(layer)
    g.quantizer = oq.Quantizer()
    g.quantizer.configure(bits, perchannel=True, sym=False, mse=mse)
    g.H = h.clone().float()
    g.fasterquant(blocksize=128, percdamp=percdamp, groupsize=group, actorder=actorder, static_groups=False)
    return layer.weight.data.float()


def load_awq(awq_repo):
    """Load the two official files (llm-awq awq/quantize/{quantizer,auto_clip}.py) without importing the awq package, whose __init__ needs the model zoo:
    auto_clip.py does `from .quantizer import pseudo_quantize_tensor`, which is pointed at the already loaded quantizer module."""
    d = Path(awq_repo) / 'awq' / 'quantize'
    src = (d / 'quantizer.py').read_text()
    start = src.index('def pseudo_quantize_tensor')
    nxt = [i for i in (src.find('\n@', start), src.find('\ndef ', start)) if i > 0]
    oq = type(sys)('awq_quantizer')
    exec(compile('import torch\nimport torch.nn as nn\n' + src[start:min(nxt) if nxt else len(src)], 'quantizer.py', 'exec'), oq.__dict__)   # only the official function; the rest of the file needs the model zoo
    oc = type(sys)('awq_auto_clip')
    sys.modules['awq_quantizer'] = oq
    exec(compile((d / 'auto_clip.py').read_text().replace('from .quantizer import', 'from awq_quantizer import'), 'auto_clip.py', 'exec'), oc.__dict__)
    return oq, oc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True)
    p.add_argument('--gptq-repo', required=True)
    p.add_argument('--awq-repo', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for key, sub in PICK:
        listing = json.loads((Path(a.layers) / key / 'layers.json').read_text())['layers']
        item = next(i for i in listing if sub in i['name'])
        src = load_file(str(Path(a.layers) / key / item['file']))
        w, h = src['weight'].float().cuda(), src['hessian'].float().cuda()
        name = item['name'].split('model.model.')[-1]
        r = res[name] = dict(shape=[item['out_features'], item['in_features']])
        for bits in (4, 3, 2):
            t = time.time(); q = Q.gptq(w, h, bits, 128); r[f'my_gptq_w{bits}'] = dict(errors(w, q, h), seconds=round(time.time() - t, 1))
            r[f'rtn_w{bits}'] = errors(w, Q.rtn(w, bits, 128), h)
            qa, alpha, _ = Q.awq(w, h, bits, 128, clip=True); r[f'my_awq_w{bits}'] = dict(errors(w, qa, h), alpha=alpha)
            if bits in (4, 2):
                r[f'official_gptq_w{bits}'] = errors(w, official_gptq(w, h, bits, 128, a.gptq_repo), h)
        for vname, kw in dict(actorder=dict(actorder=True), mse=dict(mse=True), damp10=dict(percdamp=0.1), actorder_mse=dict(actorder=True, mse=True),
                              actorder_mse_damp10=dict(actorder=True, mse=True, percdamp=0.1), damp1=dict(percdamp=0.01)).items():
            r[f'my_gptq_w2_{vname}'] = errors(w, Q.gptq(w, h, 2, 128, **kw), h)
        r['official_gptq_w2_actorder'] = errors(w, official_gptq(w, h, 2, 128, a.gptq_repo, actorder=True), h)
        r['official_gptq_w2_mse'] = errors(w, official_gptq(w, h, 2, 128, a.gptq_repo, mse=True), h)
        # AWQ pieces against the official code
        try:
            oq, oc = load_awq(a.awq_repo)
            qo = oq.pseudo_quantize_tensor(w.clone(), n_bit=2, zero_point=True, q_group_size=128)
            r['awq_pseudo_quantize_vs_mine_max_abs_diff'] = (qo - Q.rtn(w, 2, 128)).abs().max().item()
            x = torch.randn(4096, w.shape[1], device='cuda') @ torch.linalg.cholesky(h + 1e-6 * h.diag().mean() * torch.eye(h.shape[0], device='cuda')).T
            mx = oc.auto_clip_layer(w.clone(), x[None], n_bit=2, q_config=dict(zero_point=True, q_group_size=128), n_grid=20, max_shrink=0.5, n_sample_token=512)
            co, ci = w.shape
            wc = torch.maximum(torch.minimum(w.reshape(co, -1, 128), mx.reshape(co, -1, 1)), -mx.reshape(co, -1, 1)).reshape(co, ci)
            r['awq_clip_official_w2'] = errors(w, oq.pseudo_quantize_tensor(wc, n_bit=2, zero_point=True, q_group_size=128), h)
            r['awq_clip_mine_w2'] = errors(w, Q.awq_clip(w, h, 2, 128), h)   # same clip search, exact Hessian error instead of 512 sampled tokens; no AWQ scale in both
            r['rtn_w2_check'] = errors(w, oq.pseudo_quantize_tensor(w.clone(), n_bit=2, zero_point=True, q_group_size=128), h)
        except Exception as e:   # report, do not hide
            r['awq_official_error'] = repr(e)[:300]
        print(name, json.dumps({k: (round(v['output_error'], 5) if isinstance(v, dict) and 'output_error' in v else v) for k, v in r.items()}), flush=True)
        (out / 'results.json').write_text(json.dumps(res, indent=1))
        del w, h
        torch.cuda.empty_cache()
    cols = [k for k in next(iter(res.values())) if k != 'shape' and isinstance(next(iter(res.values()))[k], dict)]
    lines = ['Relative output error tr(dW H dW^T)/tr(W H W^T) on the shared Hessian\n', '| variant | ' + ' | '.join(res) + ' |', '|---|' + '---|' * len(res)]
    for c in cols:
        lines.append(f'| {c} | ' + ' | '.join(f"{v[c]['output_error']:.4f}" if c in v else 'n/a' for v in res.values()) + ' |')
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
