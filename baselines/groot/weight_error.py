"""Per-tensor weight error of a fake-quant checkpoint against its source (rms error, rms weight, mean error), for a matched random-perturbation control. Usage: weight_error.py <source ckpt dir> <fake-quant ckpt dir> <out.json>"""
import sys, json, torch
from safetensors import safe_open
A, B, out = sys.argv[1:4]
fa = safe_open(A + '/model.safetensors', 'pt'); fb = safe_open(B + '/model.safetensors', 'pt')
res = {}
for k in fa.keys():
    a = fa.get_tensor(k); b = fb.get_tensor(k)
    if torch.equal(a, b): continue
    d = (b.float() - a.float())
    res[k] = dict(shape=list(a.shape), err_rms=d.pow(2).mean().sqrt().item(), w_rms=a.float().pow(2).mean().sqrt().item(), err_mean=d.mean().item())
json.dump(res, open(out, 'w'), indent=0)
r = [v['err_rms'] / v['w_rms'] for v in res.values()]
print(len(res), 'tensors; relative rms error min %.3f median %.3f max %.3f' % (min(r), sorted(r)[len(r)//2], max(r)))
