"""X3 caveat: does divergence of episode k (fpnt vs fpsl) depend on how episode k-1 ended in fpsl (early success vs 300 steps)?"""
import json
from pathlib import Path
E = Path('/scratch/d250009/pi05-handover/runs/groot_baselines/s5b/eval')
def steps(p): return [json.loads(l) for l in open(p)]
for t in ('widowx_carrot_on_plate', 'widowx_stack_cube'):
    L, D = {}, {}
    for k in range(120):
        a, b = steps(E/'fpsl/steps'/t/f'ep{k}.jsonl'), steps(E/'fpnt/steps'/t/f'ep{k}.jsonl')
        L[k] = len(a)
        n = min(len(a), len(b))
        D[k] = any(a[i]['action'] != b[i]['action'] or a[i]['src_obj_xyz'] != b[i]['src_obj_xyz'] for i in range(n))
    tab = {}
    for k in range(1, 120):
        key = 'prev ended early in fpsl (so fpnt prev ran longer)' if L[k-1] < 300 else 'prev ran 300 in both'
        tab.setdefault(key, [0, 0]); tab[key][0] += 1; tab[key][1] += D[k]
    print(t, 'episode 0 diverges:', D[0])
    for k, v in tab.items(): print(f'  {k}: {v[1]} diverge of {v[0]}')
    # run lengths of divergent episodes
    print('  divergent episodes:', [k for k in range(120) if D[k]])
