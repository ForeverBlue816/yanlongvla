"""Task 5 calibration observations from the Fractal fine-tuning data (IPEC-COMMUNITY/fractal20220817_data_lerobot, downloaded by s5__data).

A manifest observation is {'episode_index', 'frame_index'}; the image is that frame of the episode's AV1 video (decoded with the static ffmpeg of the
SimplerEnv environment, runs/toolkits/bin/ffmpeg), the state is observation.state (x y z rx ry rz rw gripper) of that frame, the prompt is the episode's
instruction (meta/episodes.jsonl). Same 256x320 image and state layout as GoogleFractalEnv, so the calibration inputs match what the evaluation sends.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = os.environ.get('FFMPEG', '/scratch/d250009/pi05-handover/runs/toolkits/bin/ffmpeg')
H, W = 256, 320


def decode(path, need):
    """Frames 0..need of the video as uint8 (n, H, W, 3)."""
    raw = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-frames:v', str(need + 1), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def load_observations(manifest, root):
    import pandas as pd
    root = Path(root)
    tasks = {}
    for line in (root / 'meta/episodes.jsonl').read_text().splitlines():
        e = json.loads(line)
        tasks[e['episode_index']] = e['tasks'][0]
    out, cache = [], {}
    for o in manifest['observations']:
        ep, fr = o['episode_index'], o['frame_index']
        c = ep // 1000
        if ep not in cache:
            cache.clear()
            df = pd.read_parquet(root / f'data/chunk-{c:03d}/episode_{ep:06d}.parquet')
            cache[ep] = (np.stack(df['observation.state'].to_numpy()).astype(np.float32), decode(root / f'videos/chunk-{c:03d}/observation.images.image/episode_{ep:06d}.mp4', max(x['frame_index'] for x in manifest['observations'] if x['episode_index'] == ep)))
        states, frames = cache[ep]
        assert fr < len(frames) and fr < len(states), (ep, fr, len(frames), len(states))
        out.append(dict(image=frames[fr], state=states[fr], prompt=tasks[ep]))
    return out
