"""H4 control (RIGOUR_2026-11-06.md): the prepared FP checkpoint plus Gaussian noise on the linear weights that the 2-bit rows quantize.

std per tensor = rel x rms of that tensor's FP weight; the tensor list and rms come from weight_error.py's json. Seeded; stored in the
source dtype (bf16), like the fake-quant rows. Every other file of the checkpoint is copied unchanged.
Usage: perturb.py --source <ckpt dir> --error-json <json> --rel 0.06 --seed 0 --out <dir>
"""
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--error-json', required=True)
p.add_argument('--rel', type=float, required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
a = p.parse_args()

src, out = Path(a.source), Path(a.out)
part = out.with_name(out.name + '.partial')
shutil.rmtree(part, ignore_errors=True)
part.mkdir(parents=True)
for f in src.iterdir():
    if f.name != 'model.safetensors':
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, part / f.name)
targets = json.loads(Path(a.error_json).read_text())
g = torch.Generator().manual_seed(a.seed)
tensors, done, rel_err = {}, 0, []
with safe_open(src / 'model.safetensors', 'pt') as fs:
    meta = fs.metadata()
    for k in sorted(fs.keys()):
        w = fs.get_tensor(k)
        if k in targets:
            wf = w.float()
            noise = torch.randn(wf.shape, generator=g) * (a.rel * wf.pow(2).mean().sqrt())
            w = (wf + noise).to(w.dtype)
            rel_err.append(((w.float() - wf).pow(2).mean().sqrt() / wf.pow(2).mean().sqrt()).item())
            done += 1
        tensors[k] = w.contiguous()
assert done == len(targets), (done, len(targets))
save_file(tensors, str(part / 'model.safetensors'), metadata=meta)
(part / 'perturb.json').write_text(json.dumps(dict(source=str(src), rel=a.rel, seed=a.seed, tensors=done,
                                                   realized_rel_rms_median=sorted(rel_err)[len(rel_err) // 2]), indent=1))
shutil.rmtree(out, ignore_errors=True)
part.rename(out)
print(json.dumps(dict(status='complete', tensors=done, rel=a.rel, realized_rel_rms_median=sorted(rel_err)[len(rel_err) // 2])))
