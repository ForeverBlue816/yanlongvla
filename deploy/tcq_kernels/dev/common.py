"""Shared helpers of the dev / benchmark scripts: policy loading, held-out observations, graph runner."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/projects/yanlongvla/research/scripts')
import modules, tcq_format as F

ROOT = F.ROOT


def load_policy(steps=10):
    from story_steps import load_policy as lp
    from story_registry import FP
    policy = lp(dict(model=FP, steps=steps))
    policy._model.requires_grad_(False)
    return policy


def heldout40(row=F.ROW):
    """The 40 held-out observations of the fine-tune (indices into the 256 calibration observations, row.json)."""
    from calibration_data import load_observations
    info = json.loads((ROOT / f'models/pi05_tcq/{row}/row.json').read_text())
    idx = set(info['fit_complete']['fine_tune']['contract']['holdout'])
    return [(i, obs) for i, (_, obs) in enumerate(load_observations()[1]) if i in idx]


def noise(seed=0):
    from calibration_data import fixed_noise
    return fixed_noise(seed)
