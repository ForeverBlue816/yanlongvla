"""Per-episode seeded SimplerEnv-Bridge (WidowX) evaluation of a GR00T policy server, with one JSON record per episode so that two
policies (e.g. the FP baseline and a 2-bit row) can be paired episode by episode (paired_stats.py).

Runs in the SimplerEnv island (gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python; gr00t importable via its .pth); the policy runs
in a separate process (envs/groot): python gr00t/eval/run_gr00t_server.py --model-path <dir> --embodiment-tag SIMPLER_ENV_WIDOWX
--use-sim-policy-wrapper --port <port>.

Protocol (= examples/SimplerEnv/README.md, made reproducible): 7 WidowX tasks, N episodes per task, max 300 env steps per episode,
8-step action chunk of which 4 are executed per policy call, no early termination on success (success = any step reported success,
as gr00t/eval/rollout_policy.py), one environment at a time, episode k of task t reset with seed SeedSequence([seed, t, k]) -> ManiSkill2
main + episode RNG (object layout, robot init). Records: task, episode, seed, success, env_steps, policy_calls, seconds.
Resumable: episodes already in --out are skipped.
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

TASKS = ('widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube', 'widowx_put_eggplant_in_basket', 'widowx_put_eggplant_in_sink',
         'widowx_open_drawer', 'widowx_close_drawer')


def episode_seed(seed, task_index, k):
    return int(np.random.SeedSequence([seed, task_index, k]).generate_state(1)[0] % (2 ** 31 - 1))


def any_success(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.any(np.asarray(value, dtype=bool)))
    return bool(value)


def run_task(policy, task, task_index, episodes, seed, max_episode_steps, n_action_steps, video_dir, out, records):
    from gr00t.eval.rollout_policy import create_eval_env, WrapperConfigs, MultiStepConfig, VideoConfig
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=n_action_steps)
    video = VideoConfig(video_dir=str(Path(video_dir) / task), max_episode_steps=max_episode_steps, fps=5) if video_dir else VideoConfig()
    wrappers = WrapperConfigs(multistep=MultiStepConfig(contract=contract, max_episode_steps=max_episode_steps, terminate_on_success=False), video=video)
    env = gym.vector.SyncVectorEnv([partial(create_eval_env, env_name=f'simpler_env_widowx/{task}', env_idx=0, total_n_envs=1, wrapper_configs=wrappers)])
    done_keys = {(r['task'], r['episode']) for r in records}
    try:
        for k in range(episodes):
            if (task, k) in done_keys:
                continue
            s = episode_seed(seed, task_index, k)
            t0 = time.time()
            obs, info = env.reset(seed=[s])
            policy.reset()
            success, env_steps, calls = any_success(info.get('success', [False])[0]) if 'success' in info else False, 0, 0
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
            lengths = infos.get('final_info', [None])[0] or {}
            env_steps = int(lengths.get('episode_length', [calls * n_action_steps])[-1]) if isinstance(lengths, dict) and 'episode_length' in lengths else calls * n_action_steps
            rec = dict(task=task, task_index=task_index, episode=k, seed=s, success=bool(success), env_steps=env_steps, policy_calls=calls, seconds=round(time.time() - t0, 1))
            records.append(rec)
            Path(out).write_text(json.dumps(records, indent=0) + '\n')
            print(json.dumps(rec), flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='+', default=list(TASKS), choices=TASKS)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--max-episode-steps', type=int, default=300)
    p.add_argument('--n-action-steps', type=int, default=4)
    p.add_argument('--video-dir', default=None)
    p.add_argument('--out', required=True, help='JSON list of per-episode records (resumable)')
    a = p.parse_args()
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
