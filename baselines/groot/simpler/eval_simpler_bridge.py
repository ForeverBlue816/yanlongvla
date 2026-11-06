"""Exhaustive, seeded SimplerEnv-Bridge (WidowX) evaluation of a GR00T policy server, one JSON record per episode, so that two policies
(the FP baseline and a 2-bit row) are paired episode by episode on the same initial scene and the same flow-matching noise.

Runs in the SimplerEnv island (gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python; gr00t importable via its .pth); the policy runs in
a separate process (envs/groot): simpler_server.py --model-path <dir> --port <port> (= the official run_gr00t_server.py with the
SIMPLER_ENV_WIDOWX sim wrapper plus a per-episode noise seed taken from reset options; the official server also works, then the policy
noise is unseeded).

Protocol: 7 WidowX tasks x 120 episodes. The 5 object tasks have a finite set of SimplerEnv "visual matching" layouts (object xy grid x
orientations: spoon_on_towel 24, carrot_on_plate 24, stack_cube 24, put_eggplant_in_basket 8, put_eggplant_in_sink 1; robot init fixed);
episode k uses layout k mod N (every layout the same number of times) and noise seed SeedSequence([seed, t, k]). The two drawer tasks draw
their robot / cabinet pose and texture from the RNGs (ManiSkill2 episode RNG + numpy's global RNG), both seeded with the same episode seed.
300 env steps per episode (NVIDIA's README), 8-step chunk of which 4 are executed per call, no early termination; success = any step
reported success (gr00t/eval/rollout_policy.py rule). Resumable: episodes already in --out are skipped.
"""
import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

TASKS = ('widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube', 'widowx_put_eggplant_in_basket', 'widowx_put_eggplant_in_sink',
         'widowx_open_drawer', 'widowx_close_drawer')
LAYOUTS = {'widowx_spoon_on_towel': 24, 'widowx_carrot_on_plate': 24, 'widowx_stack_cube': 24, 'widowx_put_eggplant_in_basket': 8, 'widowx_put_eggplant_in_sink': 1,
           'widowx_open_drawer': None, 'widowx_close_drawer': None}      # len(xy_configs) x len(quat_configs) of the ManiSkill2_real2sim task; None = RNG-driven
EPISODES = 120


def episode_seed(seed, task_index, k):
    return int(np.random.SeedSequence([seed, task_index, k]).generate_state(1)[0] % (2 ** 31 - 1))


def any_success(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.any(np.asarray(value, dtype=bool)))
    return bool(value)


def patch_bridge_env():
    """gr00t's WidowXBridgeEnv.reset drops `options`; forward them so obj_init_options.episode_id (the layout) reaches ManiSkill2."""
    from gr00t.eval.sim.SimplerEnv import simpler_env as gse

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=int(seed) if seed is not None else None, options=options)
        observation = self._process_observation(observation)
        info['success'] = False
        return observation, info
    gse.WidowXBridgeEnv.reset = reset


def run_task(policy, task, task_index, episodes, seed, max_episode_steps, n_action_steps, video_dir, out, records):
    from gr00t.eval.rollout_policy import create_eval_env, WrapperConfigs, MultiStepConfig, VideoConfig
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=n_action_steps)
    video = VideoConfig(video_dir=str(Path(video_dir) / task), max_episode_steps=max_episode_steps, fps=5) if video_dir else VideoConfig()
    wrappers = WrapperConfigs(multistep=MultiStepConfig(contract=contract, max_episode_steps=max_episode_steps, terminate_on_success=False), video=video)
    env = gym.vector.SyncVectorEnv([partial(create_eval_env, env_name=f'simpler_env_widowx/{task}', env_idx=0, total_n_envs=1, wrapper_configs=wrappers)])
    done_keys = {(r['task'], r['episode']) for r in records}
    n_layouts = LAYOUTS[task]
    try:
        for k in range(episodes):
            if (task, k) in done_keys:
                continue
            s = episode_seed(seed, task_index, k)
            layout = k % n_layouts if n_layouts else None
            t0 = time.time()
            np.random.seed(s)          # the drawer tasks draw robot qpos / cabinet pose / overlay brightness from numpy's global RNG
            random.seed(s)
            options = {'obj_init_options': {'episode_id': layout}} if n_layouts else None
            obs, info = env.reset(seed=[s], options=options)
            policy.reset(options={'noise_seed': s})
            success, calls = False, 0
            while True:
                actions, _ = policy.get_action(obs)
                obs, rewards, terminations, truncations, infos = env.step(actions)
                calls += 1
                if 'success' in infos:
                    success |= any_success(infos['success'][0])
                if 'final_info' in infos and infos['final_info'][0] is not None and 'success' in infos['final_info'][0]:
                    success |= any_success(infos['final_info'][0]['success'])
                if terminations[0] or truncations[0]:
                    break
            rec = dict(task=task, task_index=task_index, episode=k, layout=layout, seed=s, success=bool(success), policy_calls=calls, env_steps=calls * n_action_steps,
                       seconds=round(time.time() - t0, 1))
            records.append(rec)
            Path(out).write_text(json.dumps(records, indent=0) + '\n')
            print(json.dumps(rec), flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='+', default=list(TASKS), choices=TASKS)
    p.add_argument('--episodes', type=int, default=EPISODES, help='per task; a multiple of 24 covers every layout of the object tasks equally')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--max-episode-steps', type=int, default=300)
    p.add_argument('--n-action-steps', type=int, default=4)
    p.add_argument('--video-dir', default=None)
    p.add_argument('--out', required=True, help='JSON list of per-episode records (resumable)')
    a = p.parse_args()
    patch_bridge_env()
    from gr00t.policy.server_client import PolicyClient
    policy = PolicyClient(host=a.host, port=a.port)
    out = Path(a.out)
    records = json.loads(out.read_text()) if out.exists() else []
    out.parent.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        run_task(policy, task, TASKS.index(task), a.episodes, a.seed, a.max_episode_steps, a.n_action_steps, a.video_dir, out, records)
    by_task = {t: [r['success'] for r in records if r['task'] == t] for t in a.tasks}
    summary = {t: dict(successes=sum(v), episodes=len(v), rate=round(sum(v) / max(len(v), 1), 4)) for t, v in by_task.items()}
    summary['average'] = round(float(np.mean([v['rate'] for v in summary.values()])), 4) if summary else None
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
