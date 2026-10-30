"""LIBERO-Plus client for GR00T (runs in the LIBERO-Plus environment, where LIBERO-Plus is importable as `libero`).
Same units, seeds, init state, settle steps and step limits as the π0.5 rows of the report; GR00T-native inputs (raw
256 x 256 agent-view and wrist frames flipped [::-1, ::-1], 8-d state) and the official GR00T execution of 8 actions of the
16-step window per policy call. One JSONL row per unit; resumable.

Usage (one suite's units against one serve.py; run_plus.py drives all four suites):
  python eval_libero_plus.py --port 8205 --units units_libero_spatial.json --out plus_1000.jsonl
"""
import argparse
import collections
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
LIMITS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLE_STEPS = 10
RESOLUTION = 256
N_ACTION_STEPS = 8


def quat2axisangle(quat):
    """openpi examples/libero/main.py (robosuite convention)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--units', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--host', default='127.0.0.1')
    a = p.parse_args()
    import torch
    from openpi_client import websocket_client_policy
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    units = json.loads(a.units.read_text())
    units = units['units'] if isinstance(units, dict) else units
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r['suite'], r['task']))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    suites = {name: benchmark.get_benchmark_dict()[name](task_order_index=0) for name in {u['suite'] for u in units}}
    client = None
    for entry in units:
        if (entry['suite'], entry['task']) in done:
            continue
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        suite = suites[entry['suite']]; task = suite.get_task(entry['task']); assert task.name == entry['name'], (task.name, entry['name'])
        noise_seed = int(np.random.SeedSequence([7, SUITES.index(entry['suite']), entry['task'], 0]).generate_state(1)[0])
        env = None; t = calls = 0; success = False; start = time.perf_counter()
        try:
            bddl = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)   # Plus parses viewpoint / robot parameters from it
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RESOLUTION, camera_widths=RESOLUTION)
            env.seed(7); states = suite.get_task_init_states(entry['task'])
            env.reset(); obs = env.set_init_state(states[0]); plan = collections.deque()
            while t < LIMITS[entry['suite']] + SETTLE_STEPS:
                if t < SETTLE_STEPS:
                    obs, _, _, _ = env.step(DUMMY_ACTION); t += 1; continue
                if not plan:
                    element = {'observation/image': np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
                               'observation/wrist_image': np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]),
                               'observation/state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])).astype(np.float32),
                               'prompt': str(task.language)}
                    if calls == 0:
                        element['_vla_episode_seed'] = noise_seed
                    if client is None:
                        client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
                    actions = np.asarray(client.infer(element)['actions'], dtype=np.float32)
                    assert actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
                    calls += 1; plan.extend(actions[:N_ACTION_STEPS])
                obs, _, success, _ = env.step(plan.popleft().tolist()); t += 1
                if success:
                    break
            row = dict(entry, noise_seed=noise_seed, success=bool(success), policy_calls=calls, n_action_steps=N_ACTION_STEPS, steps_including_settle=t, seconds=time.perf_counter() - start)
            with a.out.open('a') as handle:
                handle.write(json.dumps(row) + '\n'); handle.flush(); os.fsync(handle.fileno())
            print(json.dumps(row), flush=True)
        except Exception as exc:
            with a.out.with_suffix('.errors.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(entry, error=repr(exc))) + '\n')
            raise
        finally:
            if env is not None:
                env.close()


if __name__ == '__main__':
    main()
