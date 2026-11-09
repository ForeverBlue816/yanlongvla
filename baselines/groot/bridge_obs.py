"""GOAL v6 2c: Bridge calibration / held-out observations from groot/simpler/data/bridge-{calibration,heldout} (built by
groot/simpler/build_bridge_calibration.py from IPEC-COMMUNITY/bridge_orig_lerobot, the checkpoint's own fine-tuning data).
observations.parquet rows: image (PNG bytes, 256x256 RGB), state (x y z roll pitch yaw pad gripper), task_index (meta/tasks.jsonl).
Same image size and state layout as WidowXBridgeEnv, so the calibration inputs match what the evaluation sends.
"""
import io
import json
from pathlib import Path

import numpy as np


def load_observations(folder):
    import pandas as pd
    from PIL import Image
    folder = Path(folder)
    tasks = {}
    for line in (folder / 'meta/tasks.jsonl').read_text().splitlines():
        t = json.loads(line)
        tasks[t['task_index']] = t['task']
    df = pd.read_parquet(folder / 'observations.parquet')
    out = []
    for _, r in df.iterrows():
        img = r['image']
        img = img['bytes'] if isinstance(img, dict) else img
        out.append(dict(image=np.asarray(Image.open(io.BytesIO(img)).convert('RGB'), dtype=np.uint8),
                        state=np.asarray(r['state'], dtype=np.float32), prompt=tasks[int(r['task_index'])]))
    return out
