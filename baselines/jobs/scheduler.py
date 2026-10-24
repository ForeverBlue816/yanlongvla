"""PBS feeder for the README baseline matrix.

The plan is a priority-ordered list of parts. It starts with one 'sanity' part per row (build the checkpoint, then a
100-episode sample kept out of the results); a row's evaluation parts are packed only once its sample passed (judged in
monitor()). Parts are packed in plan order into bundles of up to
BUDGET_HOURS of estimated work; each bundle is one 12-hour job (jobs/evaluate.pbs) that runs its parts back to back and
builds any missing calibration or fake-quant checkpoint first. The feeder keeps MAX_WAITING bundles queued (state Q),
topping up as they start running, and lets only one in-flight bundle build a given checkpoint (or the calibration).
When a bundle leaves PBS, each unfinished part goes back to the pool; a part that was started but not finished counts as
an attempt (every step resumes), up to MAX_ATTEMPTS. After each pass, completed parts are merged and paired_stats.py
writes the README step-4 comparisons; STATUS.md summarizes every row.

  python jobs/scheduler.py init <row> [<row> ...]   # freeze the source, write the Plus shards and plan.json under $RUN
  python jobs/scheduler.py enable <row> [...]        # un-park rows whose quantizer was added later (new source snapshot)
  python jobs/scheduler.py stop <row> <reason>       # stop a row: no new parts, running clients killed, debug job submitted
  python jobs/scheduler.py refresh                   # freeze the current source for bundles submitted from now on
  python jobs/scheduler.py reset <row> <reason>      # quantizer changed: delete the row's checkpoint, sample and
                                                     # evaluation outputs and restart it from its sanity part
  python jobs/scheduler.py add-sanity                # put a sanity part for every row at the front of an existing plan
  touch $RUN/STOP                                    # pause all submissions (the loop keeps tracking and collecting)
  python jobs/scheduler.py loop                      # every 5 minutes: step + collect (started by jobs/watchdog.sh)

Per row: Plus 1000 (4 parts of 250) -> the 1000 units the expansion adds (4 parts) -> LIBERO (one part per suite, 50
episodes per task). Plus 2000 = the Plus 1000 episodes + the added units: the expansion file lists the selection first,
with identical seeds and protocol, and asks for the added 1000 to be reported separately.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUN = Path(os.environ.get('RUN', '/home/d250009/projects/pi05-handover/runs/readme_baselines'))
HERE = Path(__file__).resolve().parent.parent
PY = '/home/d250009/projects/openpi/.venv/bin/python'
REFS = Path('/home/d250009/yanlongvla-clean/results')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
PARTS = 4
MAX_WAITING = 5
MAX_ATTEMPTS = 4
NOT_YET_PORTED = ('aqlm', 'quip', 'qtip')   # rows of these methods are parked until their quantizer is in this repository
BUDGET_HOURS = 11.0                         # estimated work per 12-hour job
# Early stop (collapse): after at least MIN_EPISODES finished episodes, a row with success rate <= COLLAPSE_RATE while the
# reference rows succeed on >= REFERENCE_RATE of the same episodes is stopped and gets one debug job (jobs/debug.pbs).
# Two collapsed rows pause all submissions (touch $RUN/STOP): that points at a shared bug, not a quantizer.
MIN_EPISODES = 50
COLLAPSE_RATE = 0.05
REFERENCE_RATE = 0.5
# Conservative hours per part, from earlier runs of this runtime (Plus 1000 ~5.1 h, LIBERO 2000 ~9.3 h, failure-heavy rows)
PART_HOURS = dict(build=2.6, sanity=0.5, plus=1.35, libero_spatial=1.9, libero_object=2.3, libero_goal=2.4, libero_10=3.2)
SANITY_UNITS = 50   # Plus units in the sanity sample (with 5 episodes per LIBERO spatial task: 100 episodes)
CALIBRATION_HOURS = 1.0
QUANTIZE_HOURS = dict(rtn=0.2, gptq=1.5, awq=1.0, vptq=3.5, aqlm=2.6, quip=1.0, qtip=2.0)   # aqlm: the sanity job's share next to the helpers
AQLM_HELPERS = 4   # extra jobs that quantize AQLM layers alongside the sanity job (~106 s per layer, 414 layers)


def now():
    return datetime.now().isoformat(timespec='seconds')


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(RUN / 'scheduler.log', 'a') as f:
        f.write(line + '\n')


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1) + '\n')
    tmp.replace(path)


def keys_in(path, fields):
    if not path.exists():
        return set()
    return {tuple(json.loads(l)[f] for f in fields) for l in path.read_text().splitlines() if l.strip()}


# ---- plan -------------------------------------------------------------------------------------------------------------
def snapshot():
    """Freeze the current baselines source (plus git HEAD/diff/status and the Plus shard files) under $RUN."""
    src = RUN / f"source_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    src.mkdir(parents=True)
    for f in HERE.glob('*.py'):
        shutil.copy(f, src)
    shutil.copytree(HERE / 'data', src / 'data')
    shutil.copytree(HERE / 'jobs', src / 'jobs', ignore=shutil.ignore_patterns('__pycache__'))
    git = lambda *a: subprocess.run(['git', '-C', str(HERE), *a], capture_output=True, text=True).stdout
    (src / 'GIT_HEAD').write_text(git('rev-parse', 'HEAD'))
    (src / 'GIT_DIFF.patch').write_text(git('diff', 'HEAD', '--', '.'))
    (src / 'GIT_STATUS').write_text(git('status', '--porcelain', '--', '.'))
    units = json.loads((src / 'data/plus_units_expansion_2000.json').read_text())['units']
    selection = json.loads((src / 'data/plus_units_selection_1000.json').read_text())['units']
    assert [(u['suite'], u['task']) for u in units[:1000]] == [(u['suite'], u['task']) for u in selection]
    shards = src / 'data/shards'
    shards.mkdir()
    for name, block in (('p1000', units[:1000]), ('add1000', units[1000:])):
        n = len(block) // PARTS
        for i in range(PARTS):
            (shards / f'{name}_{i}.json').write_text(json.dumps(dict(units=block[i * n:(i + 1) * n])) + '\n')
    (shards / 'sanity50.json').write_text(json.dumps(dict(units=selection[:SANITY_UNITS])) + '\n')
    return src


def init(rows):
    src = snapshot()
    items = sanity_items(rows)
    for r in rows:
        for name in ('p1000', 'add1000'):
            for i in range(PARTS):
                items.append(dict(name=f'{r}:{name}_{i}', kind='plus', row=r, part=f'{name}_{i}'))
        for s in SUITES:
            items.append(dict(name=f'{r}:{s}', kind='libero', row=r, suite=s))
    parked = [r for r in rows if r.split('_w')[0] in NOT_YET_PORTED]
    save(RUN / 'plan.json', dict(source=str(src), rows=rows, parked=parked, created=now(), items=items))
    save(RUN / 'state.json', {})
    log(f'init: {len(items)} jobs for rows {rows}; parked {parked}; source {src}')


def sanity_items(rows):
    """Per row: AQLM build helpers (one job each), then the sanity part."""
    items = []
    for r in rows:
        if r.startswith('aqlm'):
            items += [dict(name=f'{r}:build_{i}', kind='build', row=r, part=f'build_{i}') for i in range(AQLM_HELPERS)]
        items.append(dict(name=f'{r}:sanity', kind='sanity', row=r, part='sample'))
    return items


def add_sanity():
    plan = load(RUN / 'plan.json', None)
    have = {i['name'] for i in plan['items']}
    new = [i for i in sanity_items(plan['rows']) if i['name'] not in have]
    for item in new:   # in front of the row's sanity part if that exists, else at the front
        at = next((k for k, i in enumerate(plan['items']) if i['name'] == f"{item['row']}:sanity"), 0)
        plan['items'].insert(at, item)
    plan['source'] = str(snapshot())
    save(RUN / 'plan.json', plan)
    log(f"added sanity parts for {plan['rows']}; bundles submitted from now on use {plan['source']}")


def sanity_passed(row):
    return (RUN / 'sanity' / row / 'passed.json').exists()


def enable(rows):
    plan = load(RUN / 'plan.json', None)
    src = snapshot()
    for item in plan['items']:
        if item['row'] in rows:
            item['source'] = str(src)
    plan['parked'] = [r for r in plan['parked'] if r not in rows]
    save(RUN / 'plan.json', plan)
    log(f'enabled {rows} from {src}')


def stop(row, reason, debug=True):
    """Stop a row: running jobs kill its client within a minute ($RUN/stopped/<row>) and skip its parts, queued bundles
    holding only stopped rows are deleted, and one debug job is submitted for the row."""
    plan, state = load(RUN / 'plan.json', None), load(RUN / 'state.json', {})
    plan.setdefault('stopped', {})[row] = dict(reason=reason, at=now())
    save(RUN / 'plan.json', plan)
    (RUN / 'stopped').mkdir(exist_ok=True)
    if not (RUN / 'stopped' / row).exists():
        (RUN / 'stopped' / row).write_text(reason + '\n')
    log(f'stopped {row}: {reason}')
    jobs = my_jobs()
    for bundle, b in state.get('_bundles', {}).items():
        rows = {state['_items'][n].get('row') or n.split(':')[0] for n in b['items'] if state['_items'][n].get('status') != 'done'}
        if jobs.get(b['job'].split('.')[0]) in ('Q', 'R') and rows <= set(plan['stopped']):
            subprocess.run(['qdel', b['job']], capture_output=True)
            log(f'deleted {bundle} ({b["job"]}): only stopped rows left in it')
    if debug and (RUN / 'checkpoints' / row / 'quant.json').exists():
        src = str(snapshot())
        tag = f"{row}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        job = subprocess.run(['qsub', '-p', '1023', '-o', str(RUN / 'logs'), '-N', f'bl_dbg_{row}'[:15], '-v',
                              f'RUN={RUN},SRC={src},ROW={row},TAG={tag}', f'{src}/jobs/debug.pbs'],
                             capture_output=True, text=True, check=True).stdout.strip()
        plan = load(RUN / 'plan.json', None)
        plan['stopped'][row]['debug'] = dict(job=job, results=str(RUN / 'debug' / tag))
        save(RUN / 'plan.json', plan)
        log(f'submitted debug job {job} for {row}: results in {RUN / "debug" / tag}')
    unexplained = [r for r, v in plan['stopped'].items() if v['reason'].startswith('collapse') and not v.get('diagnosis')]
    if len(unexplained) >= 2 and not (RUN / 'STOP').exists():
        (RUN / 'STOP').write_text('two rows collapsed: pausing all submissions until the debug results are reviewed\n')
        log('PAUSED all submissions: two rows collapsed')


def reset(row, reason):
    """Restart a row from scratch after its quantizer changed. Bundles still holding its parts must be gone first, or
    their (older) source would rebuild the checkpoint."""
    plan, state = load(RUN / 'plan.json', None), load(RUN / 'state.json', {})
    jobs = my_jobs()
    holding = [b for b, v in state.get('_bundles', {}).items() if v['job'].split('.')[0] in jobs
               and any(n.split(':')[0] == row and state['_items'][n].get('status') != 'done' for n in v['items'])]
    if holding:
        sys.exit(f'{row}: bundles {holding} still hold its parts; qdel them first')
    for path in (RUN / 'checkpoints' / row, RUN / 'sanity' / row, RUN / 'evaluation' / row):
        shutil.rmtree(path, ignore_errors=True)
    (RUN / 'stopped' / row).unlink(missing_ok=True)
    plan.get('stopped', {}).pop(row, None)
    save(RUN / 'plan.json', plan)
    for name in state.get('_items', {}):
        if name.split(':')[0] == row:
            state['_items'][name] = dict(attempts=0)
    save(RUN / 'state.json', state)
    log(f'reset {row}: {reason}')


def refresh():
    plan = load(RUN / 'plan.json', None)
    plan['source'] = str(snapshot())
    save(RUN / 'plan.json', plan)
    log(f"bundles submitted from now on use {plan['source']}")


REFERENCES = dict(plus=('plus_episodes/original_fp.jsonl', 'plus_episodes/original_2bit_expanded_2000.jsonl'), libero='libero_full/original_fp')


def monitor(plan):
    """Stop rows that collapse (see MIN_EPISODES), and adopt rows a job stopped itself (failed sanity check)."""
    if not hasattr(monitor, 'refs'):
        rows = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        plus = {}
        for f in reversed(REFERENCES['plus']):   # FP where it exists, the original 2-bit expansion for the added units
            plus.update({(r['suite'], r['task']): r['success'] for r in rows(REFS / f)})
        libero = {(r['suite'], r['task'], r['episode']): r['success'] for f in (REFS / REFERENCES['libero']).glob('*.jsonl') for r in rows(f)}
        monitor.refs = (plus, libero)
    plus, libero = monitor.refs
    for path in sorted((RUN / 'stopped').glob('*')) if (RUN / 'stopped').exists() else []:
        if path.name not in plan.get('stopped', {}):
            stop(path.name, path.read_text().strip())
            plan = load(RUN / 'plan.json', None)
    for row in plan['rows']:   # judge finished sanity samples
        sample = RUN / 'sanity' / row
        if row in plan.get('stopped', {}) or row in plan['parked'] or sanity_passed(row):
            continue
        if not done(dict(kind='sanity', row=row), plan['source']):
            continue
        cand, ref = [], []
        for l in (sample / 'plus.jsonl').read_text().splitlines():
            if l.strip():
                r = json.loads(l); cand.append(r['success']); ref.append(plus[(r['suite'], r['task'])])
        for l in (sample / 'libero/libero_spatial.jsonl').read_text().splitlines():
            if l.strip():
                r = json.loads(l); cand.append(r['success']); ref.append(libero[(r['suite'], r['task'], r['episode'])])
        verdict = dict(episodes=len(cand), successes=sum(cand), reference_successes=sum(ref), judged=now())
        if sum(cand) == 0 or (sum(cand) <= COLLAPSE_RATE * len(cand) and sum(ref) >= REFERENCE_RATE * len(ref)):
            stop(row, f"collapse in the sanity sample: {sum(cand)}/{len(cand)} successes, reference {sum(ref)}/{len(ref)} on the same episodes")
            plan = load(RUN / 'plan.json', None)
        else:
            save(sample / 'passed.json', verdict)
            log(f"sanity passed {row}: {sum(cand)}/{len(cand)} successes, reference {sum(ref)}/{len(ref)}; full evaluation can start")
    for row in plan['rows']:
        if row in plan.get('stopped', {}) or row in plan['parked']:
            continue
        out, cand, ref = RUN / 'evaluation' / row, [], []
        for f in sorted((out / 'plus_parts').glob('*.jsonl')) if (out / 'plus_parts').exists() else []:
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    if (r['suite'], r['task']) in plus:
                        cand.append(r['success']); ref.append(plus[(r['suite'], r['task'])])
        for f in sorted((out / 'libero').glob('*.jsonl')) if (out / 'libero').exists() else []:
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    cand.append(r['success']); ref.append(libero[(r['suite'], r['task'], r['episode'])])
        if len(cand) >= MIN_EPISODES and sum(cand) <= COLLAPSE_RATE * len(cand) and sum(ref) >= REFERENCE_RATE * len(ref):
            stop(row, f'collapse: {sum(cand)}/{len(cand)} successes, reference {sum(ref)}/{len(ref)} on the same episodes')
            plan = load(RUN / 'plan.json', None)
    return plan


def done(item, src):
    if item['kind'] == 'build':
        return ((RUN / 'aqlm_layers' / item['row'] / 'accounting.json').exists()
                or (RUN / 'checkpoints' / item['row'] / 'quant.json').exists())
    if item['kind'] == 'sanity':
        sample = RUN / 'sanity' / item['row']
        return (len(keys_in(sample / 'libero/libero_spatial.jsonl', ('task', 'episode'))) == 50
                and len(keys_in(sample / 'plus.jsonl', ('suite', 'task'))) == SANITY_UNITS)
    out = RUN / 'evaluation' / item['row']
    if item['kind'] == 'plus':
        need = {(u['suite'], u['task']) for u in json.loads((Path(src) / f"data/shards/{item['part']}.json").read_text())['units']}
        return need <= keys_in(out / f"plus_parts/{item['part']}.jsonl", ('suite', 'task'))
    return len(keys_in(out / f"libero/{item['suite']}.jsonl", ('task', 'episode'))) == 500


def qsub(bundle, lines):
    (RUN / 'bundles').mkdir(exist_ok=True)
    (RUN / 'bundles' / f'{bundle}.tsv').write_text(''.join('\t'.join(l) + '\n' for l in lines))
    args = ['qsub', '-o', str(RUN / 'logs'), '-N', f'bl_{bundle}'[:15], '-v', f'RUN={RUN},BUNDLE={bundle}',
            str(Path(lines[0][4]) / 'jobs/evaluate.pbs')]
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def my_jobs():
    out = subprocess.run(['qstat', '-u', os.environ['USER']], capture_output=True, text=True, check=True).stdout
    jobs = {}
    for line in out.splitlines():
        f = line.split()
        if f and f[0][0].isdigit() and len(f) >= 10:
            jobs[f[0].split('.')[0]] = f[9]
    return jobs


def progress(bundle):
    """Last recorded status per part of a bundle (started / finished / deferred / skipped / failed(rc))."""
    path = RUN / 'bundles' / f'{bundle}.progress'
    last = {}
    for l in (path.read_text().splitlines() if path.exists() else []):
        item, status, _ = l.split('\t')
        last[item] = status
    return last


def step():
    plan, state = load(RUN / 'plan.json', None), load(RUN / 'state.json', {})
    bundles = state.setdefault('_bundles', {})
    items = state.setdefault('_items', {})
    jobs = my_jobs()
    for item in plan['items']:
        st = items.setdefault(item['name'], dict(attempts=0))
        if st.get('status') not in ('done', 'failed') and done(item, item.get('source', plan['source'])):
            st.update(status='done', finished=now())
            log(f"done {item['name']}")
    for bundle, b in list(bundles.items()):   # bundles that left PBS: return unfinished parts to the pool
        if b['job'].split('.')[0] in jobs:
            continue
        last = progress(bundle)
        for name in b['items']:
            st = items[name]
            st.pop('bundle', None)
            if st.get('status') == 'done':
                continue
            if last.get(name, 'not reached') in ('started',) or last.get(name, '').startswith('failed'):
                st['attempts'] += 1
                log(f"{name}: unfinished in {bundle} ({last.get(name)}), attempt {st['attempts']}/{MAX_ATTEMPTS}")
                if st['attempts'] >= MAX_ATTEMPTS:
                    st['status'] = 'failed'
                    log(f'FAILED {name}')
        log(f"{bundle} ({b['job']}) left PBS: " + ', '.join(f'{n}={last.get(n, "not reached")}' for n in b['items']))
        del bundles[bundle]

    save(RUN / 'state.json', state)
    plan = monitor(plan)
    state = load(RUN / 'state.json', {})
    bundles, items = state['_bundles'], state['_items']
    calibrated = (RUN / 'calibration/index.json').exists() and (RUN / 'layers/layers.json').exists()
    quantized = lambda row: (RUN / 'checkpoints' / row / 'sanity.json').exists()   # built and sanity-checked
    building = {r for b in bundles.values() for r in b.get('builds', [])}   # rows / 'calibration' owned by in-flight bundles
    waiting = sum(1 for s in jobs.values() if s in ('Q', 'H', 'W', 'T'))
    while waiting < MAX_WAITING and not (RUN / 'STOP').exists():
        # Strict plan order. A part whose checkpoint (or the calibration it needs) another in-flight bundle is building runs
        # last in its job and waits for that build (or does the build itself if no job has started it yet).
        lines, names, builds, hours = [], [], [], 0.0
        for item in plan['items']:
            name, row = item['name'], item['row']
            st = items[name]
            if row in plan['parked'] or row in plan.get('stopped', {}) or st.get('status') in ('done', 'failed') or st.get('bundle'):
                continue
            if item['kind'] not in ('build', 'sanity') and not sanity_passed(row):
                continue   # full evaluation only after the row's 100-episode sample passed
            if item['kind'] == 'build' and lines:
                continue   # a build helper gets a job of its own, so helpers run in parallel
            cost = PART_HOURS[item['suite'] if item['kind'] == 'libero' else item['kind']]
            need = []
            if item['kind'] != 'build' and not quantized(row) and row not in building and row not in builds:
                need.append(row)
                cost += QUANTIZE_HOURS[row.split('_w')[0]]
                if not row.startswith('rtn') and not calibrated and 'calibration' not in builds and 'calibration' not in building:
                    need.append('calibration')
                    cost += CALIBRATION_HOURS
            if hours + cost > BUDGET_HOURS and lines:
                break
            lines.append((name, row, item['kind'], item.get('part') or item['suite'], item.get('source', plan['source'])))
            names.append(name)
            builds += need
            hours += cost
            if item['kind'] == 'build':
                break
        if not lines:
            break
        bundle = f"b{state.setdefault('_next', 1):03d}"
        state['_next'] += 1
        job = qsub(bundle, lines)
        bundles[bundle] = dict(job=job, items=names, builds=builds, hours=round(hours, 2), submitted=now())
        for n in names:
            items[n]['bundle'] = bundle
        building |= set(builds)
        waiting += 1
        log(f"submitted {bundle} as {job}: {len(names)} parts, ~{hours:.1f} h" + (f", builds {builds}" if builds else '') + f": {', '.join(names)}")
    save(RUN / 'state.json', state)


# ---- merging and statistics (README step 4) -------------------------------------------------------------------------------
def stats(row, kind, candidate, reference, label, extra=()):
    dest = RUN / 'evaluation' / row / 'paired_stats' / f'{label}.json'
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = load(RUN / 'plan.json', {})['source']
    r = subprocess.run([PY, 'paired_stats.py', kind, '--candidate', str(candidate), '--reference', str(reference), *extra],
                       cwd=src, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=src))
    if r.returncode:
        log(f'paired_stats failed for {row} {label}: {r.stderr[-500:]}')
        return
    dest.write_text(r.stdout)
    log(f'stats {row} {label}')


def merge(row, parts, dest):
    out = RUN / 'evaluation' / row
    rows = {}
    for p in parts:
        for l in (out / f'plus_parts/{p}.jsonl').read_text().splitlines():
            if l.strip():
                rows.setdefault((json.loads(l)['suite'], json.loads(l)['task']), l)
    (out / dest).write_text('\n'.join(rows.values()) + '\n')
    return len(rows)


def collect():
    plan, state = load(RUN / 'plan.json', None), load(RUN / 'state.json', {})
    items = state.get('_items', {})
    ok = lambda n: items.get(n, {}).get('status') == 'done'
    fmt = lambda d: f"{d['difference_pp']:+.1f} [{d['paired_difference_ci95_pp'][0]:+.1f}, {d['paired_difference_ci95_pp'][1]:+.1f}]" if d else '—'
    lines = [f'# README baselines — status {now()}', '', f"source snapshot: `{plan['source']}`", '',
             'Successes with paired difference in pp and 95% CI (README paired bootstrap). All rows pure PTQ with the shared calibration Hessians.', '',
             '| row | linear bpw | whole bpw | Plus 1000 | vs FP | vs orig 2-bit | Plus 2000 | vs orig 2-bit | added 1000 vs orig 2-bit | LIBERO 2000 | LIBERO 1900 | vs FP (2000) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in plan['rows']:
        out = RUN / 'evaluation' / row
        p1 = [f'p1000_{i}' for i in range(PARTS)]
        add = [f'add1000_{i}' for i in range(PARTS)]
        if all(ok(f'{row}:{p}') for p in p1) and merge(row, p1, 'plus_1000.jsonl') == 1000:
            stats(row, 'plus', out / 'plus_1000.jsonl', REFS / 'plus_episodes/original_fp.jsonl', 'plus_1000_vs_original_fp')
            stats(row, 'plus', out / 'plus_1000.jsonl', REFS / 'plus_episodes/original_2bit.jsonl', 'plus_1000_vs_original_2bit')
        if all(ok(f'{row}:{p}') for p in p1 + add) and merge(row, add, 'plus_added_1000.jsonl') == 1000 and merge(row, p1 + add, 'plus_2000.jsonl') == 2000:
            stats(row, 'plus', out / 'plus_2000.jsonl', REFS / 'plus_episodes/original_2bit_expanded_2000.jsonl', 'plus_2000_vs_original_2bit_expanded')
            stats(row, 'plus', out / 'plus_added_1000.jsonl', REFS / 'plus_episodes/original_2bit_expanded_2000.jsonl', 'plus_added_1000_vs_original_2bit_expanded')
        if all(ok(f'{row}:{s}') for s in SUITES):
            stats(row, 'libero', out / 'libero', REFS / 'libero_full/original_fp', 'libero_vs_original_fp', ('--episodes', '50'))
        q = load(RUN / 'checkpoints' / row / 'quant.json', {})
        s = lambda label: load(out / 'paired_stats' / f'{label}.json', None)
        a, b, c, d, l = (s(x) for x in ('plus_1000_vs_original_fp', 'plus_1000_vs_original_2bit', 'plus_2000_vs_original_2bit_expanded',
                                        'plus_added_1000_vs_original_2bit_expanded', 'libero_vs_original_fp'))
        count = lambda parts: sum(len(keys_in(out / f'plus_parts/{p}.jsonl', ('suite', 'task'))) for p in parts)
        nlib = sum(len(keys_in(out / f'libero/{x}.jsonl', ('task', 'episode'))) for x in SUITES)
        bpw = f"{q['quantizable_linear_bpw']:.4f} | {q['whole_model_bpw']:.4f}" if q else ('parked (quantizer not yet ported)' if row in plan['parked'] else 'not quantized yet') + ' | —'
        sanity = load(RUN / 'checkpoints' / row / 'sanity.json', None)
        if sanity:
            bpw += f" (action rel. MSE vs FP {sanity['evaluated_relative_mse']:.3g})"
        sample = load(RUN / 'sanity' / row / 'passed.json', None)
        if sample:
            bpw += f" (sanity sample {sample['successes']}/{sample['episodes']}, reference {sample['reference_successes']})"
        if row in plan.get('stopped', {}):
            v = plan['stopped'][row]
            bpw += f" (stopped: {v['reason']}" + (f"; diagnosis: {v['diagnosis']}" if v.get('diagnosis') else f"; debug results in {v['debug']['results']}" if v.get('debug') else '') + ')'
        lines.append(f'| {row} | {bpw} | ' + (f"{a['candidate_successes']}/1000" if a else f'{count(p1)}/1000 run') + f' | {fmt(a)} | {fmt(b)} | '
                     + (f"{c['candidate_successes']}/2000" if c else f'{count(p1 + add)}/2000 run') + f' | {fmt(c)} | {fmt(d)} | '
                     + (f"{l['all_tasks']['candidate_successes']}/2000 | {l['excluding_libero10_tasks8_9']['candidate_successes']}/1900 | {fmt(l['all_tasks'])}" if l else f'{nlib}/2000 run | — | —') + ' |')
    counts = {}
    for item in plan['items']:
        st = items.get(item['name'], {})
        key = 'parked' if item['row'] in plan['parked'] else 'stopped' if item['row'] in plan.get('stopped', {}) else st.get('status') or ('in a queued/running bundle' if st.get('bundle') else 'not yet submitted')
        counts[key] = counts.get(key, 0) + 1
    lines += ['', 'parts: ' + ', '.join(f'{k} {v}' for k, v in counts.items()), '',
              'bundles in PBS: ' + (', '.join(f"{k} ({v['job'].split('.')[0]}, {len(v['items'])} parts, ~{v['hours']} h)" for k, v in state.get('_bundles', {}).items()) or 'none')]
    (RUN / 'STATUS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'init':
        init(sys.argv[2:])
    elif cmd == 'enable':
        enable(sys.argv[2:])
    elif cmd == 'stop':
        stop(sys.argv[2], ' '.join(sys.argv[3:]))
    elif cmd == 'refresh':
        refresh()
    elif cmd == 'reset':
        reset(sys.argv[2], ' '.join(sys.argv[3:]))
    elif cmd == 'add-sanity':
        add_sanity()
    elif cmd == 'step':
        step()
    elif cmd == 'collect':
        collect()
    elif cmd == 'loop':
        import fcntl
        lock = open(RUN / 'scheduler.lock', 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # exactly one feeder at a time
        except OSError:
            sys.exit('another scheduler loop is running')
        lock.write(f'{os.getpid()} {os.uname().nodename}\n')
        lock.flush()
        log(f'scheduler loop started (pid {os.getpid()})')
        while True:
            try:
                step()
                collect()
            except Exception as e:   # PBS hiccups must not kill the feeder
                log(f'error: {e!r}')
            time.sleep(300)
