"""Step 0: download the official per-suite GR00T N1.7 LIBERO checkpoints at the pinned revision and write one prepared
directory per suite with a single model.safetensors (the sharded weights merged, nothing changed), the official config with
the backbone config name set, and the processor / statistics files. Verifies that the backbone weights are identical in the
four suites (so the backbone is quantized once) and that the accounting constants hold.

Usage:  python prepare_checkpoints.py --out ./checkpoints [--hf-cache ./hf]
  -> checkpoints/<suite>/{model.safetensors, config.json, processor_config.json, statistics.json, embodiment_id.json, ...}
     checkpoints/prepared.json

Backbone config: the N1.7 loader builds the Qwen3-VL backbone module and its processor from nvidia/Cosmos-Reason2-2B (config
and processor files only; all backbone weights are stored in the checkpoint). That repository is gated: accept its licence
on huggingface.co once and provide a read token (HF_TOKEN). Our reference rows use the official name. The checkpoint's
config.json is copied unchanged; --backbone exists only to record a deliberate substitution in prepared.json.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from groot_common import SUITES, REPO_ID, REVISION, OFFICIAL_BACKBONE, PUBLIC_BACKBONE, part_of, check_constants, sha256, dump_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--hf-cache', default=None)
    p.add_argument('--backbone', default=OFFICIAL_BACKBONE, choices=(OFFICIAL_BACKBONE, PUBLIC_BACKBONE))
    a = p.parse_args()
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(REPO_ID, revision=REVISION, cache_dir=a.hf_cache))
    out = Path(a.out)
    record = dict(repo=REPO_ID, revision=REVISION, backbone_config=a.backbone, suites={})
    backbone_ref = None
    for suite in SUITES:
        src = snap / suite
        dst = out / suite
        dst.mkdir(parents=True, exist_ok=True)
        index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
        tensors = {}
        for shard in sorted(set(index.values())):
            with safe_open(str(src / shard), framework='pt') as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
        backbone = {k: v for k, v in tensors.items() if part_of(k.removesuffix('.weight').removesuffix('.bias')) == 'backbone'}
        if backbone_ref is None:
            backbone_ref = backbone
        else:
            assert backbone.keys() == backbone_ref.keys() and all(torch.equal(backbone[k], backbone_ref[k]) for k in backbone), f'{suite}: backbone differs from libero_spatial'
        if not (dst / 'model.safetensors').exists():
            save_file(tensors, str(dst / 'model.safetensors.partial'), metadata={'format': 'pt'})
            (dst / 'model.safetensors.partial').replace(dst / 'model.safetensors')
        for f in src.iterdir():
            if f.name.startswith('model') or f.name == 'config.json':
                continue
            target = dst / f.name
            if not target.exists():
                if f.is_dir():
                    import shutil; shutil.copytree(f, target)
                else:
                    target.write_bytes(f.read_bytes())
        cfg = json.loads((src / 'config.json').read_text())
        cfg['model_name'] = a.backbone
        (dst / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        record['suites'][suite] = dict(checkpoint_sha256=sha256(dst / 'model.safetensors'), constants=check_constants(dst), shards=sorted(set(index.values())))
        print(json.dumps({suite: record['suites'][suite]}), flush=True)
        del tensors, backbone
    record['backbone_identical_across_suites'] = True
    dump_json(out / 'prepared.json', record)


if __name__ == '__main__':
    main()
