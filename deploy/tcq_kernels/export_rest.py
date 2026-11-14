"""Everything outside the packed Linear payload as one file (models/pi05_tcq/<row>/rest.safetensors): INT8 token embedding, norms (fine-tuned),
biases, positional / projection / action-head tensors at their deployed precision. packed.safetensors + rest.safetensors = the weights on disk."""
import json
import torch
from safetensors.torch import save_file
import runner, modules, tcq_format as F

policy, info = runner.load('w2a8_fly')
model = policy._model
sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
for g in policy._tcq_groups:
    for L in g.layers:
        if L.bias is not None:
            sd[L.name + '.bias'] = L.bias.cpu()
by = {}
for k, v in sd.items():
    by[str(v.dtype)] = by.get(str(v.dtype), 0) + v.numel() * v.element_size()
p = F.ROOT / f'models/pi05_tcq/{F.ROW}/rest.safetensors'
save_file(sd, str(p))
from story_registry import FP, checkpoint
fp = checkpoint(dict(model=FP)) / 'model.safetensors'
size = dict(row=F.ROW, packed_bytes=F.packed_path().stat().st_size, rest_bytes=p.stat().st_size, rest_tensors=len(sd), rest_bytes_by_dtype=by,
            fp_checkpoint=str(fp), fp_bytes=fp.stat().st_size, evaluated_dense_bytes=(F.ROOT / f'models/pi05_tcq/{F.ROW}/dense.safetensors').stat().st_size)
(F.ROOT / f'models/pi05_tcq/{F.ROW}/deploy_size.json').write_text(json.dumps(size, indent=1))
print(json.dumps(size))
