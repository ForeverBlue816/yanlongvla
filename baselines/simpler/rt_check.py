"""Renderer check for the Fractal drawer gap (E6, hypothesis H1).
Which renderer does each Google Robot task use on this GPU, and is its picture repeatable?
For each task: reset the environment twice with the same seed and compare the two camera pictures (ray tracing is random,
rasterization is not); then reset with another seed. Pictures are saved as PNG next to the report.
Run with the SimplerEnv interpreter from the Isaac-GR00T directory:  python rt_check.py <out dir>
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
report = {}
for task in ('google_robot_pick_coke_can', 'google_robot_open_drawer', 'google_robot_close_drawer'):
    env = simpler_env.make(task)
    u = env.unwrapped
    info = {'shader_dir': getattr(u, '_shader_dir', None) or getattr(u, 'shader_dir', None),
            'render_config': {k: str(v) for k, v in getattr(u, '_render_config', {}).items()} if hasattr(u, '_render_config') else None,
            'renderer_type': getattr(u, '_renderer_type', None)}
    imgs = {}
    for tag, seed in (('a', 7), ('b', 7), ('c', 8)):
        obs, _ = env.reset(seed=seed)
        imgs[tag] = get_image_from_maniskill2_obs_dict(env, obs).astype(np.float32)
        cv2.imwrite(str(out / f'{task}_{tag}.png'), cv2.cvtColor(imgs[tag].astype(np.uint8), cv2.COLOR_RGB2BGR))
    d = lambda x, y: dict(mean_abs=float(np.abs(imgs[x] - imgs[y]).mean()), max_abs=float(np.abs(imgs[x] - imgs[y]).max()),
                          share_pixels_differing=float((np.abs(imgs[x] - imgs[y]).max(axis=-1) > 0).mean()))
    info['same_seed_twice'] = d('a', 'b')
    info['seed7_vs_seed8'] = d('a', 'c')
    report[task] = info
    print(task, json.dumps(info), flush=True)
    env.close()
(out / 'rt_check.json').write_text(json.dumps(report, indent=1))
print('done')
