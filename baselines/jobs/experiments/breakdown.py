"""Breakdown of the calibration set and of the evaluation sets into their smallest buckets.

Question: are the LIBERO-Plus 1000, the added 1000, their 2000 union, plain LIBERO and the calibration set stratified, and how
do they compare with the full LIBERO-Plus pool (10,030 instances)? Reads only manifests and the pool classification; no GPU.

  python breakdown.py --out DIR [--libero-eval DIR_with_four_suite_jsonl]
Writes DIR/results.json and DIR/tables.md.
"""
import argparse
import collections
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
POOL_FILE = Path('/home/d250009/projects/pi05-handover/runs/toolkits/LIBERO-plus/libero/libero/benchmark/task_classification.json')
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
SCENE = re.compile(r'^[A-Z_]+SCENE\d+_')
VIEW = re.compile(r'_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_(\d+)$')


def base_of(name):
    return name.split('_view_')[0] if '_view_' in name else name


def norm_task(text):
    """Language of a task, lower case with spaces, scene prefix removed."""
    return SCENE.sub('', text).replace('_', ' ').strip().lower()


def chi2_sf(x, dof):
    """Survival function of chi-square, regularized upper incomplete gamma (series / continued fraction)."""
    if dof <= 0:
        return float('nan')
    a, x = dof / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gof(observed, expected_share, n):
    """Chi-square goodness of fit of observed counts against the expected shares; returns (chi2, dof, p)."""
    keys = [k for k in expected_share if expected_share[k] > 0]
    chi = sum((observed.get(k, 0) - n * expected_share[k]) ** 2 / (n * expected_share[k]) for k in keys)
    dof = len(keys) - 1
    return chi, dof, chi2_sf(chi, dof)


def share(counter):
    t = sum(counter.values())
    return {k: v / t for k, v in counter.items()}


def table(rows, header):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--libero-eval', default=None, help='directory with libero_spatial.jsonl ... (plain LIBERO episodes)')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, res = [], {}

    # ---- pool
    raw = json.loads(POOL_FILE.read_text())
    pool = []
    for suite, rows in raw.items():
        for r in rows:
            m = VIEW.search(r['name'])
            pool.append(dict(suite=suite, name=r['name'], base=base_of(r['name']), category=r['category'], difficulty=r['difficulty_level'], view=(m.group(1), m.group(2)) if m else None, init=int(m.group(6)) if m else None))
    assert len(pool) == 10030
    base_to_suite = collections.defaultdict(set)
    for r in pool:
        base_to_suite[norm_task(r['base'])].add(r['suite'])
    ambiguous = {k: sorted(v) for k, v in base_to_suite.items() if len(v) > 1}

    md.append('# Pool: LIBERO-Plus, 10,030 instances\n')
    cats = sorted({r['category'] for r in pool})
    rows = [[c] + [sum(1 for r in pool if r['category'] == c and r['suite'] == s) for s in SUITES] + [sum(1 for r in pool if r['category'] == c)] for c in cats]
    rows.append(['all'] + [sum(1 for r in pool if r['suite'] == s) for s in SUITES] + [len(pool)])
    md.append(table(rows, ['category'] + SUITES + ['total']) + '\n')
    res['pool_suite_by_category'] = {r[0]: r[1:] for r in rows}

    # ---- evaluation sets
    sel = json.loads((HERE / 'data/plus_units_selection_1000.json').read_text())['units']
    exp = json.loads((HERE / 'data/plus_units_expansion_2000.json').read_text())['units']
    sel_names = {u['name'] for u in sel}
    add = [u for u in exp if u['name'] not in sel_names]
    sets = {'Plus 1000': sel, 'Added 1000': add, 'Plus 2000 (union)': exp}
    res['sizes'] = {k: len(v) for k, v in sets.items()}
    res['overlap_selection_added'] = len(sel_names & {u['name'] for u in add})
    res['duplicates_in_2000'] = len(exp) - len({u['name'] for u in exp})
    pool_by_name = {(r['suite'], r['name']): r for r in pool}

    def enrich(u):
        r = pool_by_name.get((u['suite'], u['name']))
        m = VIEW.search(u['name'])
        return dict(suite=u['suite'], base=u['base_task'], category=u['category'], difficulty=u['difficulty_level'], in_pool=r is not None, view=(m.group(1), m.group(2)) if m else None, init=int(m.group(6)) if m else None)

    for sname, units in sets.items():
        e = [enrich(u) for u in units]
        md.append(f'\n# {sname} ({len(e)} instances)\n')
        res.setdefault(sname, {})['not_in_pool'] = sum(1 for x in e if not x['in_pool'])
        # category x suite, against the pool shares of that category
        rows, tests = [], {}
        for c in sorted({x['category'] for x in e}):
            ec = [x for x in e if x['category'] == c]
            obs = collections.Counter(x['suite'] for x in ec)
            pool_share = share(collections.Counter(r['suite'] for r in pool if r['category'] == c))
            chi, dof, pv = gof(obs, pool_share, len(ec))
            tests[f'{c}: suite vs pool'] = dict(chi2=round(chi, 2), dof=dof, p=round(pv, 4))
            uniform = {s: 0.25 for s in SUITES}
            chi_u, dof_u, pv_u = gof(obs, uniform, len(ec))
            tests[f'{c}: suite vs equal quarters'] = dict(chi2=round(chi_u, 2), dof=dof_u, p=round(pv_u, 4))
            rows.append([c] + [f'{obs.get(s, 0)} ({pool_share[s] * len(ec):.0f})' for s in SUITES] + [len(ec)])
        md.append('Counts per suite, with the count a pool-proportional draw would give in brackets.\n')
        md.append(table(rows, ['category'] + SUITES + ['n']) + '\n')
        # difficulty x category
        rows = []
        for c in sorted({x['category'] for x in e}):
            ec = [x for x in e if x['category'] == c]
            obs = collections.Counter(x['difficulty'] for x in ec)
            pool_share = share(collections.Counter(r['difficulty'] for r in pool if r['category'] == c and r['difficulty'] is not None))
            chi, dof, pv = gof({k: v for k, v in obs.items() if k is not None}, pool_share, sum(v for k, v in obs.items() if k is not None))
            tests[f'{c}: difficulty vs pool'] = dict(chi2=round(chi, 2), dof=dof, p=round(pv, 4))
            rows.append([c] + [f'{obs.get(d, 0)} ({pool_share.get(d, 0) * len(ec):.0f})' for d in (1, 2, 3, 4, 5)] + [obs.get(None, 0), len(ec)])
        md.append('Counts per difficulty level, pool-proportional count in brackets.\n')
        md.append(table(rows, ['category', 'd1', 'd2', 'd3', 'd4', 'd5', 'no level', 'n']) + '\n')
        # per base task
        rows, per_task = [], {}
        for s in SUITES:
            for c in sorted({x['category'] for x in e}):
                cnt = collections.Counter(x['base'] for x in e if x['suite'] == s and x['category'] == c)
                n_tasks_pool = len({r['base'] for r in pool if r['suite'] == s and r['category'] == c})
                vals = list(cnt.values())
                per_task[f'{s}/{c}'] = dict(tasks_hit=len(cnt), tasks_in_pool=n_tasks_pool, min=min(vals), max=max(vals), counts=sorted(vals))
                rows.append([s, c, f'{len(cnt)}/{n_tasks_pool}', min(vals), max(vals), f'{sum(vals) / max(len(cnt), 1):.1f}'])
        md.append('Instances per base task (task) inside each suite and category.\n')
        md.append(table(rows, ['suite', 'category', 'tasks hit / in pool', 'min per task', 'max per task', 'mean']) + '\n')
        # task balance: equal allocation (40 task cells per category) and pool-proportional allocation
        for c in sorted({x['category'] for x in e}):
            ec = [x for x in e if x['category'] == c]
            obs = collections.Counter((x['suite'], x['base']) for x in ec)
            pool_c = collections.Counter((r['suite'], r['base']) for r in pool if r['category'] == c)
            cells = sorted(pool_c)
            chi, dof, pv = gof({k: obs.get(k, 0) for k in cells}, {k: 1 / len(cells) for k in cells}, len(ec))
            tests[f'{c}: task vs equal allocation ({len(cells)} tasks)'] = dict(chi2=round(chi, 2), dof=dof, p=round(pv, 6))
            chi, dof, pv = gof({k: obs.get(k, 0) for k in cells}, share(pool_c), len(ec))
            tests[f'{c}: task vs pool'] = dict(chi2=round(chi, 2), dof=dof, p=round(pv, 6))
            ranked = sorted(cells, key=lambda k: obs.get(k, 0))
            extremes = [(k[0], k[1][:60], obs.get(k, 0), pool_c[k]) for k in ranked[:3] + ranked[-2:]]
            res[sname].setdefault('task_extremes', {})[c] = extremes
            md.append(f'{c}: fewest and most instances per task (suite, task, in this set, in pool): ' + '; '.join(f'{a} / {b} / {n} / {m}' for a, b, n, m in extremes) + '\n')
        # smallest bucket: suite x category x difficulty x task
        pool_buckets = collections.Counter((r['suite'], r['category'], r['difficulty'], r['base']) for r in pool if r['category'] in {x['category'] for x in e})
        obs_buckets = collections.Counter((x['suite'], x['category'], x['difficulty'], x['base']) for x in e)
        multi = sum(1 for v in obs_buckets.values() if v > 1)
        md.append(f'Smallest bucket = suite x category x difficulty x task: pool has {len(pool_buckets)} non-empty buckets for these categories; this set fills {len(obs_buckets)} of them ({multi} with more than one instance). Largest bucket holds {max(obs_buckets.values())} instances.\n')
        # camera views and robot init states
        extra = {}
        cam = [x for x in e if x['category'] == 'Camera Viewpoints']
        rob = [x for x in e if x['category'] == 'Robot Initial States']
        pool_views = collections.Counter(r['view'] for r in pool if r['category'] == 'Camera Viewpoints')
        pool_init = collections.Counter(r['init'] for r in pool if r['category'] == 'Robot Initial States')
        views = collections.Counter(x['view'] for x in cam)
        inits = collections.Counter(x['init'] for x in rob)
        extra['camera'] = dict(n=len(cam), distinct_views=len(views), views_in_pool=len(pool_views), max_repeat=max(views.values()) if views else 0)
        extra['robot_init'] = dict(n=len(rob), distinct_initstate_ids=len(inits), ids_in_pool=len(pool_init), max_repeat=max(inits.values()) if inits else 0, id_min=min(inits), id_max=max(inits), pool_id_max=max(pool_init))
        md.append(f"Camera Viewpoints: {extra['camera']['n']} instances use {extra['camera']['distinct_views']} distinct (view_x, view_y) settings of the {extra['camera']['views_in_pool']} in the pool. Robot Initial States: {extra['robot_init']['n']} instances use {extra['robot_init']['distinct_initstate_ids']} distinct init-state ids of {extra['robot_init']['ids_in_pool']} in the pool.\n")
        res[sname].update(tests=tests, per_task=per_task, buckets=dict(pool_nonempty=len(pool_buckets), filled=len(obs_buckets), multi=multi, max_in_bucket=max(obs_buckets.values())), extra=extra)
        md.append('Chi-square tests (small p means the set departs from the reference shares):\n')
        md.append(table([[k, v['chi2'], v['dof'], v['p']] for k, v in tests.items()], ['test', 'chi2', 'dof', 'p']) + '\n')

    # ---- plain LIBERO
    if a.libero_eval:
        md.append('\n# Plain LIBERO (evaluation)\n')
        rows, lib = [], {}
        for s in SUITES:
            ep = [json.loads(l) for l in (Path(a.libero_eval) / f'{s}.jsonl').read_text().splitlines() if l.strip()]
            per_task = collections.Counter(x['task'] for x in ep)
            rows.append([s, len(ep), len(per_task), min(per_task.values()), max(per_task.values()), len({x['episode'] for x in ep})])
            lib[s] = dict(episodes=len(ep), tasks=len(per_task), min=min(per_task.values()), max=max(per_task.values()))
        md.append(table(rows, ['suite', 'episodes', 'tasks', 'min per task', 'max per task', 'distinct episode ids per task']) + '\n')
        md.append('Plain LIBERO has no perturbation; every task is run with the same number of episodes.\n')
        res['LIBERO'] = lib

    # ---- calibration
    cal = json.loads((HERE / 'data/calibration_manifest.json').read_text())
    md.append('\n# Calibration set (256 observations from 60 trajectories of physical-intelligence/libero)\n')
    ep_info = {}
    for i, e in enumerate(cal['episodes']):
        ep_info[e['episode_index']] = dict(task=norm_task(e['tasks'][0]), length=e['length'])
    unmapped = sorted({v['task'] for v in ep_info.values() if v['task'] not in base_to_suite})
    for v in ep_info.values():
        s = base_to_suite.get(v['task'])
        v['suite'] = sorted(s)[0] if s and len(s) == 1 else ('ambiguous' if s else 'unmapped')
    obs = cal['observations']
    ep_obs = collections.Counter(o['episode_index'] for o in obs)
    res['calibration'] = dict(episodes=len(ep_info), observations=len(obs), unmapped_tasks=unmapped, ambiguous_tasks=ambiguous)
    rows = []
    cal_suite = {}
    for s in SUITES + ['ambiguous', 'unmapped']:
        eps = [k for k, v in ep_info.items() if v['suite'] == s]
        tasks = {ep_info[k]['task'] for k in eps}
        n_obs = sum(ep_obs.get(k, 0) for k in eps)
        pool_tasks = 10
        cal_suite[s] = dict(episodes=len(eps), tasks=len(tasks), observations=n_obs)
        rows.append([s, len(eps), f'{len(tasks)}/{pool_tasks}' if s in SUITES else len(tasks), n_obs, f'{100 * n_obs / len(obs):.1f}%'])
    md.append(table(rows, ['suite', 'trajectories', 'tasks hit / tasks in suite', 'observations', 'share of 256']) + '\n')
    res['calibration']['by_suite'] = cal_suite
    # per task
    task_obs = collections.Counter()
    task_eps = collections.Counter()
    for k, v in ep_info.items():
        task_obs[(v['suite'], v['task'])] += ep_obs.get(k, 0)
        task_eps[(v['suite'], v['task'])] += 1
    rows = [[s, t, task_eps[(s, t)], task_obs[(s, t)]] for (s, t) in sorted(task_obs)]
    md.append('Per task (40 tasks; two trajectories were drawn for 20 of them, one for the other 20):\n')
    md.append(table(rows, ['suite', 'task', 'trajectories', 'observations']) + '\n')
    vals = list(task_obs.values())
    res['calibration']['obs_per_task'] = dict(min=min(vals), max=max(vals), counts=sorted(vals))
    # per episode and phase
    per_ep = sorted(ep_obs.values())
    res['calibration']['obs_per_trajectory'] = dict(min=per_ep[0], max=per_ep[-1], median=per_ep[len(per_ep) // 2], zero=len(ep_info) - len(ep_obs))
    md.append(f"Observations per trajectory: min {per_ep[0]}, median {per_ep[len(per_ep) // 2]}, max {per_ep[-1]}; trajectories with none: {len(ep_info) - len(ep_obs)}.\n")
    bins = collections.Counter()
    for o in obs:
        L = ep_info[o['episode_index']]['length']
        bins[min(int(5 * o['frame_index'] / L), 4)] += 1
    md.append(table([[f'{20 * i}-{20 * i + 20}%', bins.get(i, 0), f'{100 * bins.get(i, 0) / len(obs):.1f}%'] for i in range(5)], ['position in trajectory', 'observations', 'share']) + '\n')
    res['calibration']['phase_quintiles'] = [bins.get(i, 0) for i in range(5)]
    chi, dof, pv = gof({i: bins.get(i, 0) for i in range(5)}, {i: 0.2 for i in range(5)}, len(obs))
    res['calibration']['phase_chi2_vs_even'] = dict(chi2=round(chi, 2), dof=dof, p=round(pv, 4))
    # suite balance tests
    chi, dof, pv = gof({s: cal_suite[s]['observations'] for s in SUITES}, {s: 0.25 for s in SUITES}, len(obs))
    res['calibration']['suite_chi2_vs_equal'] = dict(chi2=round(chi, 2), dof=dof, p=round(pv, 4))
    md.append(f"Suite balance of observations against equal quarters: chi2 {chi:.2f}, dof {dof}, p {pv:.4f}. Phase balance against even fifths: chi2 {res['calibration']['phase_chi2_vs_even']['chi2']}, p {res['calibration']['phase_chi2_vs_even']['p']}.\n")
    md.append('Perturbations in the calibration set: none. The trajectories are unperturbed training demonstrations, so no camera-viewpoint, robot-initial-state or other LIBERO-Plus category is represented.\n')

    (out / 'results.json').write_text(json.dumps(res, indent=1, default=str))
    (out / 'tables.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
