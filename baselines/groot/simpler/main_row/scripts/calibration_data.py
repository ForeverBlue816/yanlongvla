"""Frozen real observations from the training calibration set (never eval rollouts)."""
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(os.environ.get('VLA_ROOT', '/projects/yanlongvla'))
DATA = ROOT / 'datasets/libero-calibration'


def load_observations(limit=None, data_dir=DATA):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'observation_manifest.json').read_text())
    tasks = {r['task_index']: r['task'] for r in map(json.loads, (data_dir / 'meta/tasks.jsonl').read_text().splitlines())}
    rows = manifest['observations'][:limit]
    cached = {}
    result = []
    for index, sample in enumerate(rows):
        episode = sample['episode_index']
        if episode not in cached:
            path = data_dir / f'data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet'
            table = pq.read_table(path).to_pylist()
            cached[episode] = {int(r['frame_index']): r for r in table}
        row = cached[episode][sample['frame_index']]
        def decode(value):
            assert value['bytes'] is not None, 'Expected self-contained image bytes'
            return np.array(Image.open(io.BytesIO(value['bytes'])).convert('RGB'))
        # These are already RLDS/LeRobot training images. Do not apply the raw
        # simulator's 180-degree rotation again. Native model transform resizes.
        observation = {
            'observation/image': decode(row['image']),
            'observation/wrist_image': decode(row['wrist_image']),
            'observation/state': np.asarray(row['state'], dtype=np.float32),
            'prompt': tasks[int(row['task_index'])],
        }
        assert observation['observation/state'].shape == (8,)
        result.append((dict(sample, observation_index=index, task_index=int(row['task_index'])), observation))
    return manifest, result


def fixed_noise(seed):
    return np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
