"""X5 / X6 (GOAL v7 Q3 / Q4): a checkpoint that mixes the FP weights W and a 2-bit row's weights Wq on the 432 quantized linear weights.

W' = W + alpha * (Wq - W) for every tensor in the error json whose name starts with one of --prefix (all of them if no prefix);
every other tensor is the FP tensor. alpha 1 with prefix backbone. = the 2-bit backbone with an FP action head; alpha 0.5 without a prefix
= half of the 2-bit error everywhere. Stored in the source dtype (bf16). Writes mix.json and renames the .partial folder at the end.
Usage: mix.py --source <FP ckpt dir> --other <2-bit ckpt dir> --error-json <json> [--prefix backbone.] [--alpha 1.0] --out <dir>
"""
import argparse, json, re, shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--other', required=True)
p.add_argument('--error-json', required=True)
p.add_argument('--prefix', nargs='*', default=[])
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--parallel', action='store_true', help='X12: use only the part of (Wq - W) along W, per tensor: c W with c = <Wq - W, W> / <W, W>')
p.add_argument('--scale', type=float, help='H8: W\' = scale W on the targets (no --other tensor used)')
p.add_argument('--orth', action='store_true', help='W7: use only the part of (Wq - W) orthogonal to W, per tensor: E - c W')
p.add_argument('--post-scale', type=float, default=1.0, help='W7: multiply the result by this on the targets (0.95 with alpha 1 = 0.95 Wq)')
p.add_argument('--match', help='W2: keep only targets whose name matches this regex (re.search), after --prefix')
p.add_argument('--out', required=True)
a = p.parse_args()

src, oth, out = Path(a.source), Path(a.other), Path(a.out)
part = out.with_name(out.name + '.partial')
shutil.rmtree(part, ignore_errors=True)
part.mkdir(parents=True)
for f in src.iterdir():
    if f.name != 'model.safetensors':
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, part / f.name)
targets = [k for k in json.loads(Path(a.error_json).read_text()) if (not a.prefix or k.startswith(tuple(a.prefix))) and (not a.match or re.search(a.match, k))]
tensors, done = {}, 0
with safe_open(src / 'model.safetensors', 'pt') as fs, safe_open(oth / 'model.safetensors', 'pt') as fo:
    meta = fs.metadata()
    for k in sorted(fs.keys()):
        w = fs.get_tensor(k)
        if k in targets:
            if a.scale is not None:
                tensors[k] = (w.double() * a.scale).to(w.dtype).contiguous()
                done += 1
                continue
            wq = fo.get_tensor(k)
            assert wq.shape == w.shape, k
            e = wq.double() - w.double()
            if a.parallel or a.orth:
                par = (e * w.double()).sum() / (w.double() ** 2).sum() * w.double()
                e = par if a.parallel else e - par
            w = ((w.double() + a.alpha * e) * a.post_scale).to(w.dtype)
            done += 1
        tensors[k] = w.contiguous()
assert done == len(targets) > 0, (done, len(targets))
save_file(tensors, str(part / 'model.safetensors'), metadata=meta)
(part / 'mix.json').write_text(json.dumps(dict(source=str(src), other=str(oth), prefix=a.prefix, alpha=a.alpha, parallel=a.parallel, orth=a.orth, post_scale=a.post_scale, match=a.match, scale=a.scale, tensors=done), indent=1))
shutil.rmtree(out, ignore_errors=True)
part.rename(out)
print(json.dumps(dict(status='complete', tensors=done, prefix=a.prefix, match=a.match, alpha=a.alpha, scale=a.scale, orth=a.orth, post_scale=a.post_scale)))
