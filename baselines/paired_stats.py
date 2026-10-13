"""Paired statistics used by the report repository (reference implementation).

  LIBERO-Plus: paired suite-stratified base-task / perturbation-instance bootstrap (2000 draws, analysis seed
               20260930); every model is evaluated on identical instances with identical noise seeds.
  LIBERO:      paired suite-stratified task / episode bootstrap with two task scopes (all 40 tasks; excluding
               LIBERO-10 tasks 8 and 9).
Intervals are pointwise 95% and are conditional on evaluation seed 7; they are not multiplicity-adjusted.

Usage:
  python paired_stats.py plus  --candidate ckpt/plus_1000.jsonl --reference results/plus_episodes/original_2bit.jsonl
  python paired_stats.py libero --candidate ckpt/libero/ --reference results/libero_full/fp/ --episodes 50
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import SUITES


def read_rows(path):
    path = Path(path)
    files = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
    rows = []
    for f in files:
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def plus_identity(r):
    return (r['suite'], r['task'])


def plus_paired(candidate_rows, reference_rows, draws=2000, seed=20260930):
    ref = {plus_identity(r): r for r in reference_rows}
    rows = [r for r in candidate_rows if plus_identity(r) in ref]
    missing = [plus_identity(r) for r in candidate_rows if plus_identity(r) not in ref]
    assert rows, 'no common instances'
    for r in rows:
        assert r.get('noise_seed') == ref[plus_identity(r)].get('noise_seed'), 'noise seeds differ: not a paired evaluation'
    c = np.array([float(r['success']) for r in rows])
    f = np.array([float(ref[plus_identity(r)]['success']) for r in rows])
    rng = np.random.default_rng(seed)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r['suite'], {}).setdefault(r.get('base_task', r['task']), []).append(i)
    samples = np.empty((draws, 2))
    for d in range(draws):
        total = np.zeros(2)
        for suite, bases in groups.items():
            arrays = [np.array(v) for v in bases.values()]
            picked = rng.integers(0, len(arrays), len(arrays))
            idx = []
            for k in picked:
                ids = arrays[k]
                idx.extend(ids[rng.integers(0, len(ids), len(ids))])
            weight = sum(len(v) for v in arrays) / len(rows)
            total += np.array([c[idx].mean(), f[idx].mean()]) * weight * 100
        samples[d] = total
    diff = samples[:, 0] - samples[:, 1]
    return dict(instances=len(rows), unmatched_candidate_instances=len(missing), candidate_successes=int(c.sum()), reference_successes=int(f.sum()),
                difference_pp=float((c - f).mean() * 100), paired_difference_ci95_pp=np.quantile(diff, [.025, .975]).tolist(),
                paired_improvements=int(((c == 1) & (f == 0)).sum()), paired_regressions=int(((c == 0) & (f == 1)).sum()),
                per_category={cat: dict(instances=int(sum(1 for r in rows if r.get('category') == cat)), candidate=int(sum(r['success'] for r in rows if r.get('category') == cat)), reference=int(sum(ref[plus_identity(r)]['success'] for r in rows if r.get('category') == cat))) for cat in sorted({r.get('category', 'unknown') for r in rows})},
                method='paired suite-stratified base-task/instance bootstrap, 2000 draws, seed 20260930, pointwise 95%, conditional on evaluation seed 7, no multiplicity adjustment')


def libero_cube(rows, episodes):
    cube = np.full((len(SUITES), 10, episodes), np.nan)
    for r in rows:
        if r['episode'] < episodes:
            cube[SUITES.index(r['suite']), r['task'], r['episode']] = float(r['success'])
    assert not np.isnan(cube).any(), 'incomplete LIBERO rows'
    return cube


def libero_paired(candidate_rows, reference_rows, episodes, draws=2000, seed=20260930):
    cand = libero_cube(candidate_rows, episodes)
    ref = libero_cube(reference_rows, episodes)
    result = {}
    for label, counts in (('all_tasks', [10, 10, 10, 10]), ('excluding_libero10_tasks8_9', [10, 10, 10, 8])):
        rng = np.random.default_rng(seed)
        samples = np.zeros((draws, 2))
        total = sum(counts)
        values = [np.concatenate([a[s, :n].reshape(-1) for s, n in enumerate(counts)]) for a in (cand, ref)]
        for d in range(draws):
            for s, n in enumerate(counts):
                tasks = rng.integers(0, n, n)
                eps = rng.integers(0, episodes, (n, episodes))
                for i, a in enumerate((cand, ref)):
                    samples[d, i] += a[s, tasks[:, None], eps].mean() * n / total * 100
        result[label] = dict(candidate_successes=int(values[0].sum()), reference_successes=int(values[1].sum()), episodes=len(values[0]),
                             difference_pp=float((values[0] - values[1]).mean() * 100), paired_difference_ci95_pp=np.quantile(samples[:, 0] - samples[:, 1], [.025, .975]).tolist())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=('plus', 'libero'))
    p.add_argument('--candidate', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--episodes', type=int, default=50)
    a = p.parse_args()
    c, r = read_rows(a.candidate), read_rows(a.reference)
    print(json.dumps(plus_paired(c, r) if a.kind == 'plus' else libero_paired(c, r, a.episodes), indent=2))


if __name__ == '__main__':
    main()
