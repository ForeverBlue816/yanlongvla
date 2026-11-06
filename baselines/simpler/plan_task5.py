"""Task 5 (GR00T N1.7 on SimplerEnv Fractal): planning code.

  python simpler/plan_task5.py manifest   builds data/fractal_calibration_manifest.json (256 observations) and
                                          data/fractal_heldout_manifest.json (64) from the dataset's meta/episodes.jsonl
  python simpler/plan_task5.py dag        prints the task list with dependencies, outputs and done / not done
Rules (TASK5_SIMPLERENV_FRACTAL_PLAN.md): whole trajectories, same number of episodes per task family, 4 to 5 observations per
episode at even positions, unperturbed (the dataset is clean), a held-out set of other episodes, one seed.
Families are keyword rules on the instruction text (FAMILIES); they approximate the six SimplerEnv Google Robot tasks.
"""
import argparse
import collections
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
GRUN = Path(os.environ.get('GRUN', '/scratch/d250009/pi05-handover/runs/groot_baselines'))
S5 = GRUN / 's5'
DATASET, DATASET_REVISION = 'IPEC-COMMUNITY/fractal20220817_data_lerobot', None   # revision is recorded from the download (s5__data)
SEED = 20261102
N_CAL, N_HELD, EPS_PER_FAMILY, HELD_PER_FAMILY = 256, 64, 10, 3
FAMILIES = [   # first match wins; SimplerEnv task -> rule on the lower-cased instruction
    ('place_in_closed_drawer', r'^place .* into .*drawer'),
    ('open_drawer', r'^open .*drawer'),
    ('close_drawer', r'^close .*drawer'),
    ('pick_coke_can', r'^pick coke can'),
    ('move_near', r'^move .* near '),
    ('pick_object', r'^pick '),
]
ENV_TASKS = {f: f'simpler_env_google/google_robot_{f}' for f, _ in FAMILIES}
N_EVAL_EPISODES = 100   # per task, as in the GR00T README table (nvidia/GR00T-N1.7-SimplerEnv-Fractal: 72.5% average)


def family(text):
    t = text.lower()
    for name, rx in FAMILIES:
        if re.search(rx, t):
            return name
    return None


def frames(length, n):
    """n frame indices at even positions across the episode (centres of n equal bins)."""
    return [min(length - 1, int((i + 0.5) * length / n)) for i in range(n)]


def build_manifest():
    eps = [json.loads(l) for l in (S5 / 'data/meta/episodes.jsonl').read_text().splitlines()]
    by = collections.defaultdict(list)
    for e in eps:
        f = family(e['tasks'][0])
        if f and e['length'] >= 5:
            by[f].append(e)
    rng = random.Random(SEED)
    cal, held = [], []
    for f, _ in FAMILIES:
        pool = by[f][:]
        rng.shuffle(pool)
        cal += pool[:EPS_PER_FAMILY]
        held += pool[EPS_PER_FAMILY:EPS_PER_FAMILY + HELD_PER_FAMILY]

    def observations(episodes, total):
        n = len(episodes)
        per = [total // n + (1 if i < total % n else 0) for i in range(n)]   # 4 or 5 per episode when total/n is in [4, 5]
        out = []
        for e, k in zip(episodes, per):
            out += [dict(episode_index=e['episode_index'], frame_index=i) for i in frames(e['length'], k)]
        return out
    for name, episodes, total in (('calibration', cal, N_CAL), ('heldout', held, N_HELD)):
        obs = observations(episodes, total)
        m = dict(dataset=DATASET, revision=DATASET_REVISION, selection_seed=SEED,
                 selection=f'{EPS_PER_FAMILY if name == "calibration" else HELD_PER_FAMILY} episodes per task family (random, seeded), 4 to 5 observations per episode at the centres of equal bins; calibration and held-out episodes are disjoint',
                 families={f: r for f, r in FAMILIES}, episodes=[dict(episode_index=e['episode_index'], tasks=e['tasks'], length=e['length'], family=family(e['tasks'][0])) for e in episodes],
                 observation_count=len(obs), noise_seed=0 if name == 'calibration' else 1, observations=obs)
        assert len(obs) == total and len({e['episode_index'] for e in cal} & {e['episode_index'] for e in held}) == 0
        (BASE / 'data' / f'fractal_{name}_manifest.json').write_text(json.dumps(m))
        c = collections.Counter(family(e['tasks'][0]) for e in episodes)
        print(f'{name}: {len(episodes)} episodes, {len(obs)} observations, per family {dict(c)}')
    print('pool sizes (episodes with a family):', {f: len(by[f]) for f, _ in FAMILIES}, ' unmatched:', sum(1 for e in eps if not family(e['tasks'][0])))


def dag():
    """name -> (dependencies, estimated hours, output that marks it done). Estimates for the new kinds are guesses until measured."""
    t = {}
    t['s5__ckpt'] = ([], 0.5, S5 / 'checkpoints/fractal/config.json')
    t['s5__env'] = ([], 1.0, S5 / 'env_ok')
    t['s5__data'] = ([], 0.5, S5 / 'data/done')
    for f, _ in FAMILIES:
        t[f's5__fp_{f}'] = (['s5__ckpt', 's5__env'], 1.0, S5 / f'eval/fp/{f}.log')
    return t


def main():
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=('manifest', 'dag')); a = p.parse_args()
    if a.cmd == 'manifest':
        build_manifest()
    else:
        print('| task | depends on | est. h | done marker | done |\n|---|---|---|---|---|')
        for n, (d, h, o) in dag().items():
            print(f'| {n} | {", ".join(d) or "-"} | {h} | `{o}` | {"yes" if o.exists() else "no"} |')


if __name__ == '__main__':
    main()
