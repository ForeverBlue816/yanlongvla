"""LIBERO (standard) evaluation client, identical to the protocol of the report repository.

Four suites x 10 tasks x N episodes (N = 50 for "full", 25 for "medium"), seed 7, LIBERO init states in order,
10 settle steps, 10-step action chunks executed 5 at a time (replan 5), per-suite step limits. One JSONL row per
episode; resumable. Run against serve.py.

Usage:  python eval_libero.py --port 8105 --episodes 50 --out /path/ckpt_name/libero/
Requires the standard LIBERO package (not LIBERO-Plus) importable as `libero`.
"""
import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np

from common import SUITES, PLUS_LIMITS, LIBERO_DUMMY_ACTION, SETTLE_STEPS, REPLAN_STEPS

RESOLUTION = 256


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def observation(obs, prompt):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    return {'observation/image': image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
            'observation/wrist_image': image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
            'observation/state': np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])),
            'prompt': str(prompt)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8105)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--suites', nargs='+', default=list(SUITES))
    p.add_argument('--out', required=True, help='directory; one <suite>.jsonl per suite')
    a = p.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', a.port)
    for suite_name in a.suites:
        np.random.seed(a.seed)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        path = out / f'{suite_name}.jsonl'
        done_keys = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done_keys.add((r['task'], r['episode']))
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            assert a.episodes <= len(states)
            env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file), camera_heights=RESOLUTION, camera_widths=RESOLUTION)
            env.seed(a.seed)
            try:
                for episode in range(a.episodes):
                    if (task_id, episode) in done_keys:
                        continue
                    noise_seed = int(np.random.SeedSequence([a.seed, task_id, episode]).generate_state(1)[0])
                    began = time.perf_counter()
                    env.reset()
                    obs = env.set_init_state(states[episode])
                    t = calls = 0
                    success = False
                    plan = collections.deque()
                    while t < PLUS_LIMITS[suite_name] + SETTLE_STEPS:
                        if t < SETTLE_STEPS:
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue
                        if not plan:
                            element = observation(obs, task.language)
                            if calls == 0:
                                element['_vla_episode_seed'] = noise_seed
                            chunk = client.infer(element)['actions']
                            assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                            calls += 1
                            plan.extend(chunk[:REPLAN_STEPS])
                        obs, _, success, _ = env.step(plan.popleft().tolist())
                        t += 1
                        if success:
                            break
                    row = dict(suite=suite_name, seed=a.seed, task=task_id, episode=episode, success=bool(success), steps_including_settle=t, policy_calls=calls, noise_seed=noise_seed, seconds=time.perf_counter() - began)
                    with path.open('a') as f:
                        f.write(json.dumps(row) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    print(json.dumps(row), flush=True)
            finally:
                env.close()


if __name__ == '__main__':
    main()
