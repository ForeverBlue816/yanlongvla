"""GOAL v6 2c: a Bridge row paired with FP on (task, episode) (same seeded layout and noise seed per episode, asserted).
Paired bootstrap clustered by (task, layout), stratified by task (2000 draws, seed 20260930), pointwise 95%; mean over the 7 tasks.
  python simpler/b5_stats.py --fp s5b/eval/fp --row s5b/eval/<row> --out s5bq/paired/<row>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ('widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube', 'widowx_put_eggplant_in_basket',
         'widowx_put_eggplant_in_sink', 'widowx_open_drawer', 'widowx_close_drawer')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fp', required=True)
    p.add_argument('--row', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tasks', default=','.join(TASKS), help='comma-separated subset (H4 uses 3 tasks)')
    a = p.parse_args()
    tasks = tuple(a.tasks.split(','))
    per_task, clusters = {}, {}
    for t in tasks:
        f = {r['episode']: r for r in json.loads((Path(a.fp) / f'{t}.json').read_text())}
        c = {r['episode']: r for r in json.loads((Path(a.row) / f'{t}.json').read_text())}
        assert set(f) == set(c), (t, len(f), len(c))
        for e in f:
            assert (f[e]['layout'], f[e]['seed']) == (c[e]['layout'], c[e]['seed']), (t, e, 'not a paired evaluation')
        per_task[t] = dict(episodes=len(f), fp=sum(bool(r['success']) for r in f.values()), row=sum(bool(r['success']) for r in c.values()),
                           row_only=sum(bool(c[e]['success']) and not f[e]['success'] for e in f), fp_only=sum(bool(f[e]['success']) and not c[e]['success'] for e in f))
        cl = {}
        for e in f:
            cl.setdefault(f[e]['layout'], []).append((float(c[e]['success']), float(f[e]['success'])))
        clusters[t] = [np.array(v) for v in cl.values()]
    rate = lambda which: float(np.mean([per_task[t][which] / per_task[t]['episodes'] for t in tasks]) * 100)
    rng = np.random.default_rng(20260930)
    diffs = np.empty(2000)
    for d in range(2000):
        task_diffs = []
        for t in tasks:
            arrs = clusters[t]
            pick = np.concatenate([arrs[k] for k in rng.integers(0, len(arrs), len(arrs))])
            task_diffs.append(pick[:, 0].mean() - pick[:, 1].mean())
        diffs[d] = np.mean(task_diffs) * 100
    out = dict(per_task=per_task, fp_mean_pct=rate('fp'), row_mean_pct=rate('row'), difference_pp=rate('row') - rate('fp'),
               paired_difference_ci95_pp=np.quantile(diffs, [.025, .975]).tolist(),
               method=f'mean over {len(tasks)} tasks; paired bootstrap clustered by (task, layout), stratified by task, 2000 draws, seed 20260930, pointwise 95%')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('fp_mean_pct', 'row_mean_pct', 'difference_pp', 'paired_difference_ci95_pp')}))


if __name__ == '__main__':
    main()
