"""X3 caveat: first step where fpnt (no env termination) and fpsl (stored seeded run) differ. Per episode and task."""
import json, numpy as np
from pathlib import Path
E = Path('/scratch/d250009/pi05-handover/runs/groot_baselines/s5b/eval')
def steps(p): return [json.loads(l) for l in open(p)]
for t in ('widowx_carrot_on_plate', 'widowx_stack_cube'):
    res = []
    n_ident = 0
    for f in sorted((E/'quipsl/steps'/t).glob('ep*.jsonl'), key=lambda p: int(p.stem[2:])):
        g = E/'quipnt/steps'/t/f.name
        if not g.exists(): continue
        a, b = steps(f), steps(g)
        n = min(len(a), len(b)); first = None
        for i in range(n):
            if a[i]['action'] != b[i]['action'] or a[i]['src_obj_xyz'] != b[i]['src_obj_xyz']:
                first = i; break
        if first is None: n_ident += 1; continue
        da = max(abs(x - y) for x, y in zip(a[first]['action'], b[first]['action']))
        res.append((int(f.stem[2:]), first, da, len(a), len(b), a[first]['action'] == b[first]['action']))
    print(t, 'episodes compared', len(list((E/'quipsl/steps'/t).glob('ep*.jsonl'))), 'identical over common prefix', n_ident, 'diverge', len(res))
    for r in res[:40]: print('  ep%d first diff step %d max|d action|=%.2e (len fpsl %d fpnt %d; action equal=%s)' % r)
    if res:
        fs = [r[1] for r in res]; print('  first-diff step min/median/max', min(fs), int(np.median(fs)), max(fs))
