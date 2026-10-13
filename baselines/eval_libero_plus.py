"""LIBERO-Plus evaluation client (official simulator), identical to the protocol of the report repository.

Units: either the preregistered 1000-instance comparison subset (data/plus_units_selection_1000.json), the 2000-unit
expansion (data/plus_units_expansion_2000.json), or the full benchmark (--units full: every task of the four
LIBERO-Plus suites, 10,030 instances). Each unit: seed 7, init state 0, one episode, 10 settle steps, 10-step chunks
executed 5 at a time, per-suite step limits, noise seed SeedSequence([7, suite_index, task, 0]). One JSONL row per
unit; resumable. Run against serve.py with LIBERO-Plus importable as `libero` (it replaces the standard package).

Usage:  python eval_libero_plus.py --port 8105 --units data/plus_units_selection_1000.json --out /path/ckpt_name/plus_1000.jsonl
"""
import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np

from common import SUITES, PLUS_LIMITS, LIBERO_DUMMY_ACTION, SETTLE_STEPS, REPLAN_STEPS
from eval_libero import observation, RESOLUTION

HERE = Path(__file__).resolve().parent


def full_units(benchmark):
    """All LIBERO-Plus tasks; category/difficulty from the benchmark's task_classification.json when present."""
    import importlib
    classification = {}
    try:
        import libero.libero.benchmark as b
        path = Path(b.__file__).parent / 'task_classification.json'
        if path.exists():
            for suite, rows in json.loads(path.read_text()).items():
                for r in rows:
                    classification[(suite, r['name'])] = r
    except Exception:
        pass
    units = []
    for suite_name in SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            c = classification.get((suite_name, task.name), {})
            base = task.name.split('_view_')[0] if '_view_' in task.name else task.name
            units.append(dict(suite=suite_name, task=i, name=task.name, base_task=base, category=c.get('category', 'unknown'), difficulty_level=c.get('difficulty_level'), episode=0, seed=7))
    return units


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8105)
    p.add_argument('--units', required=True, help='JSON file with a "units" list, or "full"')
    p.add_argument('--out', required=True, help='JSONL output (resumable)')
    a = p.parse_args()
    import random
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    units = full_units(benchmark) if a.units == 'full' else json.loads(Path(a.units).read_text())['units']
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done_keys.add((r['suite'], r['task']))
    suites = {s: benchmark.get_benchmark_dict()[s](task_order_index=0) for s in {u['suite'] for u in units}}
    client = None
    for u in units:
        if (u['suite'], u['task']) in done_keys:
            continue
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        suite = suites[u['suite']]
        task = suite.get_task(u['task'])
        assert task.name == u['name'], (task.name, u['name'])
        noise_seed = int(np.random.SeedSequence([7, SUITES.index(u['suite']), u['task'], 0]).generate_state(1)[0])
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file), camera_heights=RESOLUTION, camera_widths=RESOLUTION)
        began = time.perf_counter()
        try:
            env.seed(7)
            states = suite.get_task_init_states(u['task'])
            env.reset()
            obs = env.set_init_state(states[0])
            t = calls = 0
            success = False
            plan = collections.deque()
            while t < PLUS_LIMITS[u['suite']] + SETTLE_STEPS:
                if t < SETTLE_STEPS:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                if not plan:
                    element = observation(obs, task.language)
                    if calls == 0:
                        element['_vla_episode_seed'] = noise_seed
                    if client is None:
                        client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', a.port)
                    chunk = client.infer(element)['actions']
                    assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                    calls += 1
                    plan.extend(chunk[:REPLAN_STEPS])
                obs, _, success, _ = env.step(plan.popleft().tolist())
                t += 1
                if success:
                    break
            row = dict(u, noise_seed=noise_seed, success=bool(success), policy_calls=calls, steps_including_settle=t, seconds=time.perf_counter() - began)
            with out.open('a') as f:
                f.write(json.dumps(row) + '\n')
                f.flush()
                os.fsync(f.fileno())
            print(json.dumps(row), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
