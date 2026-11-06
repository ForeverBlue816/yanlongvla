"""Task 5: the official Fractal checkpoint (s5/checkpoints/fractal, two shards) as one prepared checkpoint directory like the LIBERO suites:
one model.safetensors (shards merged, nothing changed), config and processor files copied, training-state files left out.
Asserts the quantized-parameter / retained-byte / active-parameter constants of groot_common (the Fractal checkpoint has the same 432 Linear layers).

  python prepare_fractal.py --source S5/checkpoints/fractal --out RUN/checkpoints
  -> RUN/checkpoints/fractal/{model.safetensors, config.json, processor_config.json, ...}
"""
import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from groot_common import check_constants, sha256, dump_json

SKIP = ('model-', 'model.safetensors.index.json', 'rng_state', 'scheduler.pt', 'training_args.bin', 'trainer_state.json', 'latest', 'zero_to_fp32.py', 'wandb_config.json', 'global_step', 'README.md', 'model-architecture.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    src, dst = Path(a.source), Path(a.out) / 'fractal'
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(json.loads((src / 'model.safetensors.index.json').read_text())['weight_map'].values())
    if not (dst / 'model.safetensors').exists():
        tensors = {}
        for s in sorted(set(shards)):
            tensors.update(load_file(str(src / s)))
        save_file(tensors, str(dst / 'model.safetensors.partial'), metadata={'format': 'pt'})
        (dst / 'model.safetensors.partial').replace(dst / 'model.safetensors')
    for f in src.iterdir():
        if f.name.startswith(SKIP) or f.name in SKIP or (dst / f.name).exists():
            continue
        (shutil.copytree if f.is_dir() else shutil.copy)(f, dst / f.name)
    dump_json(dst / 'prepared.json', dict(source=str(src), shards=sorted(set(shards)), checkpoint_sha256=sha256(dst / 'model.safetensors'), constants=check_constants(dst)))
    print(json.dumps(json.loads((dst / 'prepared.json').read_text())['constants']))


if __name__ == '__main__':
    main()
