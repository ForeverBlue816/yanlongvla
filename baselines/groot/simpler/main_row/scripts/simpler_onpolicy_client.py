"""On-policy states of GR00T N1.7 SimplerEnv-Bridge for the source-balanced Hessian H_D (the Bridge port of groot_onpolicy_client.py,
user 2026-11-11). Runs in the SimplerEnv island against simpler_server.py serving the 2-bit PTQ row tcq_w2_rht (Bridge calibration).

256 rollouts, one per calibration-observation slot i (as on LIBERO). The Bridge calibration observations come from the real Bridge
dataset, so their instructions are not SimplerEnv tasks: slot i runs task TASKS[i mod 7] (37 / 37 / 37 / 37 / 36 / 36 / 36 rollouts).
  rollout seed   r_i = SeedSequence([20261107, i])                       (the LIBERO arm-D rollout seed)
  scene          env.reset(seed = r_i mod (2^31 - 1)), numpy / random seeded with it (drawer tasks); object tasks: layout drawn by
                 default_rng(SeedSequence([r_i])).integers(0, N) of the task's N visual-matching layouts (the LIBERO rule for the init state)
  policy noise   SeedSequence([7, task_index, 2000 + i]) mod (2^31 - 1)  (the LIBERO rule; evaluation uses SeedSequence([7, task_index, k]), k < 120)
  protocol       simpler_eval.py's: 300 env steps, 4 of 8 actions per call, the env ends the episode at the first success
The policy input of every call is recorded; one call is drawn uniformly among the first ceil(0.4 * C) calls (rng SeedSequence([20261103,
0, i, 4]), as on LIBERO) and saved as <out>/obs_<i>.npz (image 256 x 256 x 3 uint8, state (8,) = x, y, z, roll, pitch, yaw, pad, gripper:
the calibration format) + obs_<i>.json. Resumable.
NOTE: the layouts are SimplerEnv's evaluation layouts (the evaluation is exhaustive over them); recorded in complete.json."""
import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

from simpler_eval import TASKS, LAYOUTS, any_success, patch_bridge_env

STATE_KEYS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'pad', 'gripper')
SEED = 20261103
M31 = 2 ** 31 - 1


def policy_input(obs):
    image = np.asarray(obs['video.image_0'])
    image = image.reshape(-1, *image.shape[-3:])[-1]
    state = np.array([float(np.asarray(obs[f'state.{k}']).reshape(-1)[-1]) for k in STATE_KEYS], dtype=np.float32)
    prompt = obs['annotation.human.action.task_description']
    while isinstance(prompt, (list, tuple, np.ndarray)):
        prompt = prompt[0]
    assert image.shape == (256, 256, 3) and image.dtype == np.uint8, (image.shape, image.dtype)
    return dict(image=np.ascontiguousarray(image), state=state, prompt=str(prompt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=256)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--max-episode-steps', type=int, default=300)
    p.add_argument('--n-action-steps', type=int, default=4)
    a = p.parse_args()
    patch_bridge_env()
    from gr00t.policy.server_client import PolicyClient
    from gr00t.eval.rollout_policy import create_eval_env, WrapperConfigs, MultiStepConfig, VideoConfig
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    policy = PolicyClient(host='127.0.0.1', port=a.port)
    a.out.mkdir(parents=True, exist_ok=True)
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=a.n_action_steps)
    for task_index, task in enumerate(TASKS):
        todo = [i for i in range(a.count) if i % len(TASKS) == task_index and not (a.out / f'obs_{i:03d}.json').exists()]
        if not todo:
            continue
        wrappers = WrapperConfigs(multistep=MultiStepConfig(contract=contract, max_episode_steps=a.max_episode_steps, terminate_on_success=False), video=VideoConfig())
        env = gym.vector.SyncVectorEnv([partial(create_eval_env, env_name=f'simpler_env_widowx/{task}', env_idx=0, total_n_envs=1, wrapper_configs=wrappers)])
        try:
            for i in todo:
                began = time.time()
                rollout_seed = int(np.random.SeedSequence([20261107, i]).generate_state(1)[0])
                s = rollout_seed % M31
                n_layouts = LAYOUTS[task]
                layout = int(np.random.default_rng(np.random.SeedSequence([rollout_seed])).integers(0, n_layouts)) if n_layouts else None
                noise_seed = int(np.random.SeedSequence([a.seed, task_index, 2000 + i]).generate_state(1)[0] % M31)
                np.random.seed(s); random.seed(s)
                obs, info = env.reset(seed=[s], options={'obj_init_options': {'episode_id': layout}} if n_layouts else None)
                policy.reset(options={'noise_seed': noise_seed})
                calls, success = [], False
                while True:
                    calls.append(policy_input(obs))
                    actions, _ = policy.get_action(obs)
                    obs, rewards, terminations, truncations, infos = env.step(actions)
                    if 'success' in infos:
                        success |= any_success(infos['success'][0])
                    if 'final_info' in infos and infos['final_info'][0] is not None and 'success' in infos['final_info'][0]:
                        success |= any_success(infos['final_info'][0]['success'])
                    if terminations[0] or truncations[0]:
                        break
                first = max(1, math.ceil(0.4 * len(calls)))
                c = int(np.random.default_rng(np.random.SeedSequence([SEED, 0, i, 4])).integers(0, first))
                el = calls[c]
                np.savez_compressed(a.out / f'obs_{i:03d}.npz', image=el['image'], state=el['state'])
                meta = dict(observation_index=i, task=task, task_index=task_index, prompt=el['prompt'], layout=layout, rollout_seed=rollout_seed, env_seed=s, noise_seed=noise_seed, call=c, calls=len(calls),
                            first_40pct_calls=first, steps_before_call=c * a.n_action_steps, phase=c / max(len(calls), 1), success=bool(success), seconds=round(time.time() - began, 1))
                (a.out / f'obs_{i:03d}.json').write_text(json.dumps(meta))
                print(json.dumps(meta), flush=True)
        finally:
            env.close()
    metas = [json.loads((a.out / f'obs_{i:03d}.json').read_text()) for i in range(a.count)]
    (a.out / 'complete.json').write_text(json.dumps(dict(status='complete', states=len(metas), successes=sum(m['success'] for m in metas), phase_mean=sum(m['phase'] for m in metas) / len(metas),
                                                         per_task={t: sum(m['task'] == t for m in metas) for t in TASKS}, policy='tcq_w2_rht (Bridge calibration, clean H)',
                                                         protocol='SimplerEnv-Bridge evaluation protocol (300 env steps, 4 of 8 actions per call); slot i -> task i mod 7; layout / scene by the rollout seed',
                                                         note='the scenes are SimplerEnv evaluation layouts (the evaluation is exhaustive over them); policy noise seeds differ from the evaluation seeds'), indent=1) + '\n')
    print(json.dumps(dict(onpolicy='complete', states=len(metas), successes=sum(m['success'] for m in metas))), flush=True)


if __name__ == '__main__':
    main()
