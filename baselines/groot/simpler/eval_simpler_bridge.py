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
300 env steps per episode (NVIDIA's README), 8-step chunk of which 4 are executed per call, no early termination requested (but the env itself ends the episode at the first success, see below); success = any step
reported success (gr00t/eval/rollout_policy.py rule). Resumable: episodes already in --out are skipped.

Why a successful episode still ends early although the wrapper gets terminate_on_success=False (verified in the Isaac-GR00T checkout
and its ManiSkill2_real2sim submodule): the flag only controls MultiStepWrapper's OWN extra rule (multistep_wrapper.py:359-361). The inner
ManiSkill2 env already reports terminated = success: BaseEnv.get_done returns bool(info["success"]) (sapien_env.py:579-580) and step()
returns it as `terminated` (sapien_env.py:554-555); no subclass overrides get_done. MultiStepWrapper.step then (a) appends that done to
self.done (multistep_wrapper.py:326), (b) stops executing the remaining actions of the chunk (:309-311) and (c) returns done = max(self.done)
(:331), so the episode ends at the first success step, mid-chunk if need be, and the vector env resets. The reported success is therefore
"first-success" by construction and success_at_last_step is True for every successful episode (the success step is the last step).
--step-log (below) records the per-step flags to make this visible.
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
LAYOUTS = {'widowx_spoon_on_towel': 24, 'widowx_carrot_on_plate': 24, 'widowx_stack_cube': 24, 'widowx_put_eggplant_in_basket': 24, 'widowx_put_eggplant_in_sink': 3,
           'widowx_open_drawer': None, 'widowx_close_drawer': None}      # len(xy_configs) x len(quat_configs) of the ManiSkill2_real2sim task (put_on_in_scene.py: spoon 12x2, carrot 12x2, stack 24x1, basket 8x3, sink 1x3); None = RNG-driven
# T1 (12 Nov): the table before 12 Nov had basket 8 and sink 1 (len(xy_configs) only, the 3 quat configs were missed): episode ids 0..7 gave basket
# xy configs 0-2 only and sink yaw 0 only. --layout-table v1 reproduces it for paired re-checks of old rows.
LAYOUTS_V1 = dict(LAYOUTS, widowx_put_eggplant_in_basket=8, widowx_put_eggplant_in_sink=1)
EPISODES = 120


def episode_seed(seed, task_index, k):
    return int(np.random.SeedSequence([seed, task_index, k]).generate_state(1)[0] % (2 ** 31 - 1))


def any_success(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.any(np.asarray(value, dtype=bool)))
    return bool(value)


def patch_bridge_env(step_log=False, no_env_termination=False):
    """gr00t's WidowXBridgeEnv.reset drops `options`; forward them so obj_init_options.episode_id (the layout) reaches ManiSkill2.
    With step_log, also wrap WidowXBridgeEnv.step (read-only) to add the executed 7-d action and the source / target object positions to
    the per-step info (they travel through MultiStepWrapper's per-step info lists)."""
    from gr00t.eval.sim.SimplerEnv import simpler_env as gse

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=int(seed) if seed is not None else None, options=options)
        observation = self._process_observation(observation)
        info['success'] = False
        return observation, info
    gse.WidowXBridgeEnv.reset = reset
    if not (step_log or no_env_termination):
        return
    orig_step = gse.WidowXBridgeEnv.step

    def step(self, action):
        observation, reward, done, truncated, info = orig_step(self, action)
        gripper = float(2.0 * (float(np.ravel(action['action.gripper'])[0]) > 0.5) - 1.0)      # = WidowXBridgeEnv._postprocess_gripper
        info['executed_action'] = [float(np.ravel(action[k])[0]) for k in ('action.x', 'action.y', 'action.z', 'action.roll', 'action.pitch', 'action.yaw')] + [gripper]
        env = self.env.unwrapped
        try:        # put_on_in_scene tasks only (MoveNearInSceneEnv properties); the drawer tasks have no source / target object
            info['src_obj_xyz'] = [float(x) for x in env.source_obj_pose.p]
            info['tgt_obj_xyz'] = [float(x) for x in env.target_obj_pose.p]
        except Exception:
            pass
        if no_env_termination:   # X3: keep the episode running after the first success (ManiSkill2 sets done = success); success flags stay in info
            done = False
        return observation, reward, done, truncated, info
    gse.WidowXBridgeEnv.step = step


STEP_LOG_SKIP = {'success', 'states', 'model', 'actions', 'rewards', 'dones', 'n_env_steps', 'episode_stats', 'intermediate_signals'}


def jsonable(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        return [jsonable(y) for y in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x if isinstance(x, (str, type(None))) else str(x)


def chunk_step_records(infos, terminations, truncations, n_steps_before, n_action_steps, max_episode_steps):
    """One dict per executed env step of the chunk just returned by env.step. The per-step info lists of MultiStepWrapper hold the last
    n_action_steps entries; only the last n_env_steps of them belong to this chunk (a chunk is cut short when the env reports done).
    A terminating chunk's info is in final_info[0] (gymnasium 0.29 inline autoreset); the top-level info then describes the reset env."""
    fi = infos.get('final_info')
    def env0(v):   # value of env 0; dict-valued infos (e.g. episode stats keyed by name) are skipped
        try:
            return v[0]
        except (KeyError, IndexError, TypeError):
            return None
    src = fi[0] if fi is not None and fi[0] is not None else {k: env0(v) for k, v in infos.items() if not k.startswith('_')}
    src = {k: v for k, v in src.items() if v is not None}
    n = int(np.ravel(src['n_env_steps'])[0]) if 'n_env_steps' in src else len(src['success'])

    def last(v):
        v = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
        return v[-n:] if len(v) >= n else None
    cols = {k: last(v) for k, v in src.items() if k not in STEP_LOG_SKIP and not isinstance(v, dict)}
    cols = {k: v for k, v in cols.items() if v is not None and not any(isinstance(y, dict) for y in v)}
    succ, dones = last(src['success']), last(src['dones']) if 'dones' in src else [False] * n
    rows = []
    for i in range(n):
        idx = n_steps_before + i
        row = dict(step=idx, success=any_success(succ[i]), **{('action' if k == 'executed_action' else k): jsonable(v[i]) for k, v in cols.items()})
        row['terminated'] = bool(dones[i])
        row['truncated'] = bool(idx + 1 >= max_episode_steps and not dones[i])     # MultiStepWrapper horizon cut (multistep_wrapper.py:321-325)
        rows.append(row)
    return rows


def run_task(policy, task, task_index, episodes, seed, max_episode_steps, n_action_steps, video_dir, out, records, step_log=None, fresh_env=False):
    from gr00t.eval.rollout_policy import create_eval_env, WrapperConfigs, MultiStepConfig, VideoConfig
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=n_action_steps)
    video = VideoConfig(video_dir=str(Path(video_dir) / task), max_episode_steps=max_episode_steps, fps=5) if video_dir else VideoConfig()
    wrappers = WrapperConfigs(multistep=MultiStepConfig(contract=contract, max_episode_steps=max_episode_steps, terminate_on_success=False), video=video)
    make = lambda: gym.vector.SyncVectorEnv([partial(create_eval_env, env_name=f'simpler_env_widowx/{task}', env_idx=0, total_n_envs=1, wrapper_configs=wrappers)])
    env = make()
    done_keys = {(r['task'], r['episode']) for r in records}
    n_layouts = LAYOUTS[task]
    try:
        for k in range(episodes):
            if (task, k) in done_keys:
                continue
            s = episode_seed(seed, task_index, k)
            layout = k % n_layouts if n_layouts else None
            t0 = time.time()
            if fresh_env and k > 0:   # X3 / A12 (9 Nov): a new env per episode, so no simulator or renderer state carries over from the previous episode
                env.close()
                env = make()
            np.random.seed(s)          # the drawer tasks draw robot qpos / cabinet pose / overlay brightness from numpy's global RNG
            random.seed(s)
            options = {'obj_init_options': {'episode_id': layout}} if n_layouts else None
            obs, info = env.reset(seed=[s], options=options)
            policy.reset(options={'noise_seed': s})
            success, calls = False, 0
            step_rows = []
            while True:
                actions, _ = policy.get_action(obs)
                obs, rewards, terminations, truncations, infos = env.step(actions)
                calls += 1
                if 'success' in infos:
                    success |= any_success(infos['success'][0])
                if 'final_info' in infos and infos['final_info'][0] is not None and 'success' in infos['final_info'][0]:
                    success |= any_success(infos['final_info'][0]['success'])
                if step_log:
                    try:   # logging must never stop the evaluation: on an unexpected info layout, log the error once per chunk and go on
                        rows = chunk_step_records(infos, terminations, truncations, len(step_rows), n_action_steps, max_episode_steps)
                    except Exception as err:
                        rows = [dict(step=len(step_rows), success=False, step_log_error=repr(err)[:300], info_keys=sorted(map(str, infos.keys())))]
                    step_rows += rows
                if terminations[0] or truncations[0]:
                    break
            rec = dict(task=task, task_index=task_index, episode=k, layout=layout, seed=s, success=bool(success), policy_calls=calls, env_steps=calls * n_action_steps,
                       seconds=round(time.time() - t0, 1))
            if step_log:
                flags = [r['success'] for r in step_rows]
                rec.update(first_success_step=flags.index(True) if True in flags else None, success_at_last_step=bool(flags and flags[-1]), n_success_steps=sum(flags))
                d = Path(step_log) / task
                d.mkdir(parents=True, exist_ok=True)
                (d / f'ep{k}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in step_rows))
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
    p.add_argument('--no-env-termination', action='store_true', help='X3 (8 Nov): do not end the episode at the first success; run to --max-episode-steps (use with --step-log to score the last step)')
    p.add_argument('--step-log', default=None, help='optional dir: write <dir>/<task>/ep<k>.jsonl, one record per env step (flags, info, executed action); off by default')
    p.add_argument('--fresh-env', action='store_true', help='X3 / A12 (9 Nov): create a new env for every episode (no state carried over)')
    p.add_argument('--layout-table', choices=('fixed', 'v1'), default='fixed', help='v1 = the wrong pre-12-Nov table (basket 8, sink 1), only to reproduce old rows')
    p.add_argument('--out', required=True, help='JSON list of per-episode records (resumable)')
    a = p.parse_args()
    if a.layout_table == 'v1':
        LAYOUTS.update(LAYOUTS_V1)
    print(f'layout table {a.layout_table}: {LAYOUTS}', flush=True)
    patch_bridge_env(step_log=bool(a.step_log), no_env_termination=a.no_env_termination)
    from gr00t.policy.server_client import PolicyClient
    policy = PolicyClient(host=a.host, port=a.port)
    out = Path(a.out)
    records = json.loads(out.read_text()) if out.exists() else []
    out.parent.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        run_task(policy, task, TASKS.index(task), a.episodes, a.seed, a.max_episode_steps, a.n_action_steps, a.video_dir, out, records, a.step_log, a.fresh_env)
    by_task = {t: [r['success'] for r in records if r['task'] == t] for t in a.tasks}
    summary = {t: dict(successes=sum(v), episodes=len(v), rate=round(sum(v) / max(len(v), 1), 4)) for t, v in by_task.items()}
    summary['average'] = round(float(np.mean([v['rate'] for v in summary.values()])), 4) if summary else None
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
