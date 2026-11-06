"""PBS feeder for the GR00T N1.7 2-bit baseline rows (LIBERO-Plus 1000).

Keeps QUOTA jobs waiting in my queue (state Q; the running ones are not counted): each pass it collects finished jobs, packs the
unclaimed tasks, in sequence where prerequisites allow, into jobs of at most BUDGET hours of estimated work and submits them (groot/jobs/groot.pbs). A task is
kind__row[__key], see groot.pbs. Every task resumes, so a job that hits its walltime is simply followed by another one.

  python gsched.py step            one pass
  python gsched.py loop            a pass every 5 minutes (one instance, lock file)
  python gsched.py status          print the state of every task and row
  python gsched.py stop <row> <reason>      stop a row (no new tasks for it)
  python gsched.py reset <task> [<task> ...]  forget attempts and the done marker of tasks (e.g. after fixing a quantizer)
  touch $GRUN/STOP                 pause submissions
"""
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

GRUN = Path(os.environ.get('GRUN', '/scratch/d250009/pi05-handover/runs/groot_baselines'))   # big files belong on scratch (decision G11)
REPO = Path('/home/d250009/yanlongvla-clean/baselines')
QUOTA = int(os.environ.get('GROOT_QUOTA', 8))
BUDGET = 13.75
MAX_CKPT = 2      # fake-quant checkpoints (26 GB each) that may exist at once: the scratch quota was exceeded three times (G22)
MAX_ROWS = 3      # new rows (not the finished baselines) that may hold a quantizer output (13 GB) or a checkpoint at once
MAX_RUNNING = 12  # no cap on running jobs (user, 5 Nov); independent ready tasks each get their own job until this many jobs run or wait. The disk limits above stay
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
KEYS = ('backbone',) + tuple(f'head_{s}' for s in SUITES)
BASE_ROWS = ('gptq_w2', 'awq_w2', 'vptq_w2', 'qtip_w2', 'quip_w2', 'aqlm_w2', 'quip10_w2')
NO_GATE = ('gptq_w2',)   # GOAL 31 Oct a3: the original GPTQ row goes through the full Plus 1000 although the gate had flagged it (0/50)


def rows():
    """The fixed rows plus the new rows listed in $GRUN/enabled_rows.txt (one row name per line; written once the offline layer test
    has shown the variant to be worth a sanity gate): gptq_<variant>_w2 (variants: actorder, mse, aomse = actorder + mse, damp10, combo = actorder + mse + damp10), awq_<variant>_w2,
    vptq_iter<N>_w2."""
    p = GRUN / 'enabled_rows.txt'
    extra = [l.split('#')[0].strip() for l in p.read_text().splitlines()] if p.exists() else []
    return BASE_ROWS + tuple(r for r in extra if r and r not in BASE_ROWS)


ROWS = BASE_ROWS
MIX = ([(c, 'gptq_w2') for c in ('vision', 'language', 'head', 'vis_0_3', 'vis_4_7', 'vis_8_11')] + [(c, 'aqlm_w2') for c in ('vision', 'language', 'head')]
       + [(c, 'qtip_w2') for c in ('vision', 'language', 'head')] + [('base_l2down', 'gptq_actorder_w2'), ('l2down', 'gptq_actorder_w2')] + [(c, 'gptq_actorder_w2') for c in ('vision', 'language', 'head')]
       + [(c, 'awq_w2') for c in ('vision', 'language', 'head')]
       + [(c, r) for r in ('gptq_mse_w2', 'gptq_aomse_w2', 'gptq_best_w2') for c in ('vision', 'language')]   # intervention for E1: do the better GPTQ variants restore success where plain GPTQ collapses?
       + [(c, r) for r in ('gptq_w2', 'aqlm_w2', 'qtip_w2') for c in ('vision_a', 'vision_b', 'lang_a', 'lang_b', 'head_dit', 'head_vl')])   # halves of the groups (run before the verdict on which half breaks)   # (component, row); components are defined in groot.pbs mx_sel
MIX += [('l2fix', 'aqlm_w2')]   # AQLM with the outlier layer re-quantized at 0.1% damping
# MIX += [('all', 'qtip_alt_w2')]   # deferred with cal__alt256   # GOAL v4 1.6: QTIP quantized with the Hessians of 256 other observations, 200 units, paired against the first QTIP row
F5_ROWS = ('qtip_w2', 'quip10_w2', 'quip_w2', 'vptq_w2', 'gptq_best_w2', 'aqlm_w2', 'gptq_w2', 'awq_w2')
F5_MAX_ROWS = 5   # Fractal toolkit rows in flight at once (6 GB of quantizer output each)
F5_KEYS = ('backbone', 'head_fractal')
F5_FAMILIES = ('pick_coke_can', 'pick_object', 'move_near', 'open_drawer', 'close_drawer', 'place_in_closed_drawer')
EST = dict(mx=1.6, prepare=0.8, calibrate=1.6, q=1.0, qk=1.5, imp=0.3, sn=0.5, ev=3.5)
AQLM_EST = dict(backbone=8.0, head_libero_spatial=6.0, head_libero_object=6.0, head_libero_goal=6.0, head_libero_10=6.0)


def log(msg):
    line = f'{datetime.now().isoformat(timespec="seconds")} {msg}'
    print(line, flush=True)
    try:
        with open(GRUN / 'scheduler.log', 'a') as f:
            f.write(line + '\n')
    except OSError:   # a full disk must not kill the feeder; the line is still on stdout
        pass


def build_tasks():
    t = {}

    def add(name, deps, est, workers=1, attempts=4):
        t[name] = dict(deps=list(deps), est=est, workers=workers, attempts=attempts)

    add('prepare', [], EST['prepare'])
    add('calibrate', ['prepare'], EST['calibrate'])
    add('ev__fp', ['prepare'], EST['ev'])
    add('rr__fp', ['ev__fp'], EST['ev'])             # noise floor (GOAL 1 Nov item 0): second FP run on our GPU
    add('rr__qtip_w2', ['ev__qtip_w2'], EST['ev'] + 0.3)   # second QTIP run (rebuilds the checkpoint from q/qtip_w2)
    for r in ('fp', 'qtip_w2', 'quip_w2'):   # rigour R3 (GOAL v5): evaluation seed 8 on the same 1000 units
        add(f'sd__{r}', [f'ev__{r}'], EST['ev'] + (0 if r == 'fp' else 0.3))
    for r in ('fp', 'qtip_w2', 'quip_w2'):   # rigour R7: 500 units outside the Plus 1000 selection
        add(f'dj__{r}', [f'ev__{r}'], 2.0 + (0 if r == 'fp' else 0.3))
    add('cal__heldout', ['prepare'], 0.5)       # Hessians of 64 held-out observations (data/heldout_manifest.json, noise seed 1)
    add('cal__1024', ['prepare'], 2.0)          # Hessians of 1024 observations (data/calibration_manifest_1024.json), libero_spatial only
    # cal__alt256 and qk__qtip_alt_w2__* (GOAL v4 1.6, QTIP on the Hessians of 256 other observations) are deferred: they need about 60 GB of scratch and the quota was exceeded again on 5 Nov
    add('cal__absmean', ['prepare'], 0.8)    # GOAL v4 4c: the 256 observations again, with mean |x| per channel
    add('xp__awqscale', ['calibrate', 'cal__heldout', 'cal__absmean'], 0.5)
    add('xp__damp_quip', ['calibrate', 'cal__heldout'], 0.7)    # GOAL v4 4b: QuIP# damping retry on layers
    add('xp__damp_aqlm', ['calibrate', 'cal__heldout'], 3.0)    # GOAL v4 4a: AQLM damping and stopping rule (stalled on layer 2 down_proj; its first two layers are kept)
    add('xp__aqlm_l2fix', ['calibrate'], 0.3)    # layer 2 down_proj of AQLM at 0.1% damping
    add('xp__damp_aqlm2', ['calibrate', 'cal__heldout'], 2.0)   # ... the remaining layers, without the 0.1% rule on layer 2 down_proj
    add('xp__heldout', ['calibrate', 'cal__heldout'], 3.5)                  # GOAL 1 Nov 1.3: VPTQ structures, calibration vs held-out error
    add('xp__cal1024', ['calibrate', 'cal__heldout', 'cal__1024'], 3.5)     # ... quantized with the 1024-observation Hessians
    for kb in ('math', 'efficient'):
        add(f'kn__{kb}', ['ev__fp'], 0.9)   # GOAL 1 Nov 2.6: FP with one SDPA kernel on the 200 units
    # Task 5 (GOAL 30 Oct): GR00T N1.7 on SimplerEnv Fractal. Steps and estimates: simpler/plan_task5.py, simpler/s5.sh. Estimates are guesses until the first run is timed.
    add('s5__ckpt', [], 0.5)
    add('s5__env', [], 1.0)
    add('s5__data', [], 0.5)
    for fam in ('place_in_closed_drawer', 'open_drawer', 'close_drawer', 'pick_coke_can', 'move_near', 'pick_object'):
        add(f's5__fp_{fam}', ['s5__ckpt', 's5__env'], 1.0)
    # Task 5, quantized rows on Fractal (GOAL v4 1c): order QTIP, QuIP# (10 sweeps main, 0 sweeps appendix), VPTQ, GPTQ best, AQLM, plain GPTQ, AWQ
    add('f5__prep', ['s5__ckpt'], 0.4)
    add('f5__cal', ['f5__prep', 's5__data'], 1.2)
    for frow in F5_ROWS:
        meth = frow.split('_')[0]
        if meth in ('gptq', 'awq'):
            first = f'f5__q__{frow}'; add(first, ['f5__cal'] + (['f5__fin__aqlm_w2'] if frow == 'awq_w2' else []), EST['q'])   # AWQ is the last row: its 26 GB checkpoint must not hold a slot while the toolkit rows wait
        else:
            for k in F5_KEYS:
                add(f'f5__qk__{frow}__{k}', ['f5__cal'], (AQLM_EST['backbone'] if k == 'backbone' else AQLM_EST['head_libero_spatial']) if meth == 'aqlm' else EST['qk'] * (2 if k == 'backbone' else 1), attempts=8 if meth == 'aqlm' else 4)
            first = f'f5__imp__{frow}'; add(first, [f'f5__qk__{frow}__{k}' for k in F5_KEYS], EST['imp'])
        add(f'f5__ev__{frow}__pick_coke_can', [first, 's5__env'], 0.8)
        for fam in F5_FAMILIES[1:]:
            add(f'f5__ev__{frow}__{fam}', [f'f5__ev__{frow}__pick_coke_can'], 0.8)
        add(f'f5__fin__{frow}', [f'f5__ev__{frow}__{fam}' for fam in F5_FAMILIES], 0.1)
    for tag, fam in (('fprr', 'open_drawer'), ('fpseed8', 'open_drawer'), ('fpseed9', 'open_drawer'), ('fpseed8', 'close_drawer'), ('fpseed9', 'close_drawer')):
        add(f's5__{tag}_{fam}', ['s5__ckpt', 's5__env'], 0.6)   # FP again: is the gap to the model card (open_drawer 48 against 65, close_drawer 63 against 69) the noise of 100 episodes?
    add('xp__layererr', ['calibrate'], 0.7)    # per-layer tables of QTIP and QuIP# for the mentor
    add('xp__gptqgrid', ['calibrate'], 3.0)    # GPTQ damping / clipping / act-order grid on every layer
    add('xp__vptqgrid', ['calibrate'], 3.5)    # VPTQ structures and k-means settings, bits counted
    # bisection / component tests (GOAL 1 Nov 2.2, 2.3, 2.1): FP except one component quantized with the method of a row; 200 units
    for comp, row in MIX:
        deps = ['ev__fp', 'calibrate'] if row.split('_')[0] in ('gptq', 'awq') else [f'qk__{row}__{k}' for k in KEYS]
        add(f'mx__{comp}__{row}', deps, EST['mx'])
    t['mx__l2fix__aqlm_w2']['deps'].append('xp__aqlm_l2fix')
    for row in rows():
        method = row.split('_')[0]
        if method in ('gptq', 'awq'):
            add(f'q__{row}', ['calibrate'], EST['q'])
            ready = f'q__{row}'
        else:
            for key in KEYS:
                if method == 'aqlm':
                    add(f'qk__{row}__{key}', ['calibrate'], AQLM_EST[key], workers=1, attempts=8)
                else:
                    slow = max(1.0, int(re.match(r'vptq_iter(\d+)', row).group(1)) / 100 * 0.6) if row.startswith('vptq_iter') else 1.0   # more k-means iterations take longer
                    add(f'qk__{row}__{key}', ['calibrate'], EST['qk'] * slow * (2 if key == 'backbone' else 1))
            add(f'imp__{row}', [f'qk__{row}__{k}' for k in KEYS], EST['imp'])
            ready = f'imp__{row}'
        if row in NO_GATE:
            add(f'ev__{row}', [ready], EST['ev'])
        else:
            add(f'sn__{row}', [ready], EST['sn'])
            add(f'ev__{row}', [f'sn__{row}'], EST['ev'])
    return t


def row_of(task):
    parts = task.split('__')
    if parts[0] == 'f5':   # Task 5 rows on Fractal: f5__<step>__<row>[__<key>]; stopped/f5_<row> is written by the gate
        return 'f5_' + parts[2] if len(parts) > 2 else None
    return parts[1] if len(parts) > 1 else None


def done(task):
    return (GRUN / 'tasks' / f'{task}.done').exists()


def attempts(task):
    p = GRUN / 'tasks' / f'{task}.attempts'
    return int(p.read_text().strip() or 0) if p.exists() else 0   # an empty file (a write cut off by the home quota) counts as 0


def stopped(row):
    return row is not None and (GRUN / 'stopped' / row).exists()


def my_jobs():
    out = subprocess.run(['qstat', '-u', os.environ['USER']], capture_output=True, text=True).stdout.splitlines()
    jobs = {}
    for line in out:
        f = line.split()
        if len(f) >= 10 and f[0][0].isdigit():
            jobs[f[0].split('.')[0]] = f[9]
    return jobs


def snapshot():
    """Freeze the source for jobs submitted now (reused when nothing changed)."""
    h = hashlib.sha1()
    for p in sorted(REPO.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.pbs', '.sh', '.json', '.jsonl'):
            h.update(str(p.relative_to(REPO)).encode()); h.update(p.read_bytes())
    tag = h.hexdigest()[:12]
    dest = GRUN / f'source_{tag}'
    if not dest.exists():
        shutil.copytree(REPO, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', '.'], capture_output=True, text=True).stdout.strip()
        (dest / 'GIT_HEAD').write_text(head + ('+uncommitted' if dirty else '') + '\n')
        for p in dest.rglob('*.sh'):   # plus_python.sh is executed directly by run_plus.py
            p.chmod(0o755)
        log(f'new source snapshot {dest.name}')
    return dest


def collect(live):
    """Jobs that left PBS: count an attempt for every task that started and did not finish."""
    for f in sorted((GRUN / 'jobs').glob('*.json')):
        jid = f.stem
        if jid in live or (GRUN / 'jobs' / f'{jid}.collected').exists():
            continue
        rec = json.loads(f.read_text())
        progress = {}
        pf = next(iter((GRUN / 'jobs').glob(f'{jid}.*.progress')), None)
        if pf:
            for line in pf.read_text().splitlines():
                name, what, _ = (line.split('\t') + ['', ''])[:3]
                progress[name] = what
        for task in rec['tasks']:
            if done(task):
                continue
            what = progress.get(task, 'never started')
            if what.startswith('start') or what.startswith('failed'):
                n = attempts(task) + 1
                (GRUN / 'tasks' / f'{task}.attempts').write_text(str(n))
                log(f'{task}: unfinished in job {jid} ({what}), attempt {n}')
        (GRUN / 'jobs' / f'{jid}.collected').write_text('')


def running_claims(live):
    claims = {}
    for f in (GRUN / 'jobs').glob('*.json'):
        if f.stem in live:
            for task in json.loads(f.read_text())['tasks']:
                claims[task] = claims.get(task, 0) + 1
    return claims


def priority(tasks):
    """Order of the ready tasks: what unblocks and validates the pipeline first (prepare, calibrate, the FP evaluation, sanity samples,
    evaluations, imports, GPTQ / AWQ), then the layer sets: AQLM backbone early (the longest chain), the other methods, the AQLM heads."""
    order = list(tasks)
    rows = {r: i for i, r in enumerate(globals()['rows']())}
    def key(n):
        kind = n.split('__')[0]
        if kind == 'f5':
            step = n.split('__')[1]
            cls = {'prep': 1, 'cal': 1, 'ev': 2, 'fin': 2, 'imp': 2, 'q': 3, 'qk': 5}[step]
            return (cls, 100 + F5_ROWS.index(n.split('__')[2]) if len(n.split('__')) > 2 else 99, order.index(n))
        if kind in ('prepare', 'calibrate'):
            cls = 0
        elif n == 'ev__fp' or kind in ('xp', 'rr', 'sd', 'dj', 'mx', 'cal', 'kn', 's5'):
            cls = 1
        elif kind in ('sn', 'ev', 'imp'):
            cls = 2
        elif kind == 'q':
            cls = 3
        elif n.startswith('qk__aqlm') and n.endswith('__backbone'):
            cls = 4
        elif n.startswith('qk__aqlm'):
            cls = 6
        else:
            cls = 5
        return (cls, rows.get(row_of(n), -1), order.index(n))
    return key


def plan(tasks, live, slots, spread=0):
    """Pack the tasks that nobody has claimed into at most `slots` jobs of at most BUDGET hours of estimated work, in sequence: a job starts
    with the highest-priority ready tasks, independent ready tasks fill it up to the budget, and a task whose prerequisites are all done
    or earlier in the same job (sanity after a quantization, evaluation after the sanity, import after its five layer sets) follows
    them, so that one 15-hour job does as much as fits."""
    claimed = set()
    for f in (GRUN / 'jobs').glob('*.json'):
        if f.stem in live:
            claimed.update(json.loads(f.read_text())['tasks'])
    key = priority(tasks)
    pending = sorted((n for n, t in tasks.items() if not done(n) and n not in claimed and not stopped(row_of(n)) and attempts(n) < t['attempts']), key=key)
    bins, where = [], {}
    live_ckpts = sum(len(list(d.iterdir())) for d in (GRUN / 'ckpt', GRUN / 's5q' / 'ckpt') if d.exists())   # LIBERO and Fractal fake-quant checkpoints (26 GB each)
    making = [0]   # checkpoints this pass would start

    def makes_ckpt(n):
        return n.split('__')[0] in ('imp', 'q') or n.startswith(('f5__imp__', 'f5__q__')) or n in ('rr__qtip_w2', 'sd__qtip_w2', 'sd__quip_w2', 'dj__qtip_w2', 'dj__quip_w2') or n.startswith('mx__')   # these write ckpt/<row>

    claimed_rows = {row_of(c) for c in claimed}
    inflight = {r for r in rows() if r not in BASE_ROWS and not done('ev__' + r) and not stopped(r)
                and ((GRUN / 'q' / r).exists() or (GRUN / 'ckpt' / r).exists() or r in claimed_rows)}
    started = set()
    f5_inflight = {r for r in F5_ROWS if r.split('_')[0] not in ('gptq', 'awq') and not done(f'f5__fin__{r}') and not stopped('f5_' + r) and ((GRUN / 's5q' / 'q' / r).exists() or (GRUN / 's5q' / 'ckpt' / r).exists() or 'f5_' + r in claimed_rows)}   # toolkit rows only (6 GB of quantizer output each); GPTQ / AWQ rows write checkpoints, limited by the flock slots
    f5_started = set()

    def ckpt_ok(n):
        if n.split('__')[0] == 'qk' and row_of(n) not in BASE_ROWS and row_of(n) not in inflight | started and len(inflight | started) >= MAX_ROWS:
            return False   # too many new rows in flight: their quantizer outputs (13 GB each) would exceed the scratch quota
        if n.startswith('f5__qk__') and len(f5_inflight | f5_started) >= F5_MAX_ROWS and n.split('__')[2] not in f5_inflight | f5_started:
            return False   # Fractal quantizer outputs (about 13 GB per row) wait while MAX_ROWS rows are in flight
        if n.startswith('mx__') or n in ('rr__qtip_w2', 'sd__qtip_w2', 'sd__quip_w2', 'dj__qtip_w2', 'dj__quip_w2'):
            return True   # the jobs serialize these through two flock slots (groot.pbs ckpt_slot): at most 2 such checkpoints exist at once, whatever the number of jobs
        return not makes_ckpt(n) or live_ckpts + making[0] < MAX_CKPT

    def place(n, target):
        """Put n into the job and straight after it every pending task that now has all its prerequisites done or in this job."""
        target['tasks'].append(n); target['est'] += tasks[n]['est']; where[n] = target
        making[0] += makes_ckpt(n) and not (n.startswith('mx__') and any(t.startswith('mx__') for t in target['tasks'][:-1]))   # mx tasks delete their checkpoint before the next one of the same job starts
        if n.split('__')[0] == 'qk':
            started.add(row_of(n))
        if n.startswith('f5__qk__'):
            f5_started.add(n.split('__')[2])
        for m in pending:
            if m in where or n not in tasks[m]['deps'] or not ckpt_ok(m):
                continue
            if m.startswith('f5__ev__') and not m.endswith('__pick_coke_can'):
                continue   # Fractal families run in parallel jobs once the gate family has passed (the project runs about 4 jobs at once; a chain would use one)
            open_deps = [d for d in tasks[m]['deps'] if not done(d)]
            if all(d in where and where[d] is target for d in open_deps) and target['est'] + tasks[m]['est'] <= BUDGET:
                place(m, target)

    for n in pending:   # ready tasks, highest priority first; each starts or fills a job together with the chain behind it
        if n in where or any(not done(d) for d in tasks[n]['deps']) or not ckpt_ok(n):
            continue
        target = next((b for b in bins if b['est'] + tasks[n]['est'] <= BUDGET), None) if len(bins) >= spread else None   # spread: while fewer jobs than the GPUs we may use run, independent ready tasks get their own job
        if target is None:
            if len(bins) >= slots:
                continue
            target = dict(tasks=[], est=0.0)
            bins.append(target)
        place(n, target)
    return [[b['tasks'], b['est']] for b in bins if b['tasks']]


def step():
    for d in ('tasks', 'jobs', 'logs', 'stopped'):
        (GRUN / d).mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    live = my_jobs()
    collect(live)
    waiting = sum(1 for st in live.values() if st in ('Q', 'H', 'W'))   # QUOTA counts jobs waiting in the queue; running jobs are not counted
    slots = QUOTA - waiting
    if (GRUN / 'STOP').exists() or slots <= 0:
        log(f'queue: {waiting} waiting of {QUOTA}, {len(live) - waiting} running: nothing submitted')
        write_status(tasks, live)
        return
    running = len(live)   # running and waiting jobs both will occupy a GPU
    bins = plan(tasks, live, slots, spread=max(0, MAX_RUNNING - running))   # GPUs left among the ~3 the project allows
    if not bins:
        log(f'queue: {waiting} waiting of {QUOTA}, {len(live) - waiting} running: nothing to submit')
        write_status(tasks, live)
        return
    src = snapshot()
    for names, est in bins:
        out = subprocess.run(['qsub', '-N', 'gr_' + names[0][:12].replace('__', '_'), '-o', str(GRUN / 'logs') + '/',
                              '-v', f'GRUN={GRUN},TASKS={"+".join(names)},SRC={src}', str(src / 'groot/jobs/groot.pbs')], capture_output=True, text=True)
        if out.returncode != 0:
            log(f'qsub failed: {out.stderr.strip()}')
            continue
        jid = out.stdout.strip().split('.')[0]
        (GRUN / 'jobs' / f'{jid}.json').write_text(json.dumps(dict(tasks=names, est=est, src=str(src), submitted=time.time())))
        log(f'submitted {jid}: {", ".join(names)} (~{est:.1f} h)')
    write_status(tasks, my_jobs())


def write_status(tasks, live):
    lines = [f'# GR00T baselines status, {datetime.now().isoformat(timespec="minutes")}', f'queue: {sum(1 for st in live.values() if st in ("Q", "H", "W"))} waiting of {QUOTA}, {sum(1 for st in live.values() if st == "R")} running', '',
             '| row | prerequisites / layers | sanity | Plus 1000 | stopped |', '|---|---|---|---|---|']
    for row in ('fp',) + globals()['rows']():
        mine = [n for n in tasks if row_of(n) == row]
        d = sum(done(n) for n in mine if not n.startswith(('sn', 'ev')))
        k = sum(1 for n in mine if not n.startswith(('sn', 'ev')))
        san = GRUN / 'sanity' / f'{row}.json'
        ev = GRUN / 'eval' / row / 'plus_1000.jsonl'
        n_ev = sum(1 for _ in open(ev)) if ev.exists() else 0
        san_s = json.loads(san.read_text()) if san.exists() else None
        lines.append(f'| {row} | {d}/{k} | {"%d/%d" % (san_s["successes"], san_s["units"]) if san_s else "-"} | {n_ev}/1000 {"done" if done("ev__" + row) else ""} | {(GRUN / "stopped" / row).read_text().strip() if stopped(row) else ""} |')
    (GRUN / 'STATUS.md').write_text('\n'.join(lines) + '\n')


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'step'
    GRUN.mkdir(parents=True, exist_ok=True)
    if cmd == 'step':
        step()
    elif cmd == 'loop':
        lock = open(GRUN / 'scheduler.lock', 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit('another gsched loop is running')
        while True:
            try:
                step()
            except Exception as e:   # keep the loop alive; the next pass retries
                log(f'error: {e!r}')
            time.sleep(300)
    elif cmd == 'ready':   # exit 0 when the prerequisites of a task are done (groot.pbs asks before every task of a chained job)
        sys.exit(0 if all(done(d) for d in build_tasks()[sys.argv[2]]['deps']) else 3)
    elif cmd == 'status':
        tasks = build_tasks(); live = my_jobs(); write_status(tasks, live)
        print((GRUN / 'STATUS.md').read_text())
        for n, t in tasks.items():
            print(f'{"done " if done(n) else "todo "} {n}  attempts {attempts(n)}')
    elif cmd == 'stop':
        (GRUN / 'stopped').mkdir(parents=True, exist_ok=True)
        (GRUN / 'stopped' / sys.argv[2]).write_text(' '.join(sys.argv[3:]) + '\n')
        log(f'stopped {sys.argv[2]}: {" ".join(sys.argv[3:])}')
    elif cmd == 'reset':
        for n in sys.argv[2:]:
            for suffix in ('done', 'attempts'):
                (GRUN / 'tasks' / f'{n}.{suffix}').unlink(missing_ok=True)
            log(f'reset {n}')
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
