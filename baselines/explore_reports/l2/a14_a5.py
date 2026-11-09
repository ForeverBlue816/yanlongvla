import json, statistics as st, math, sys
sys.path.insert(0, '.')
from pathlib import Path
G = Path('/scratch/d250009/pi05-handover/runs/groot_baselines')
def libero(p):
    return {(r['suite'], r['task'], r['name']): bool(r['success']) for r in map(json.loads, open(p))}
rows = ['vptq_w2','vptq_v5k1024_w2','vptq_iter500_w2','vptq_iter500t0_w2','vptq_v6k3072_w2','vptq_v8k12288_w2']
D = {r: libero(G/f'eval/{r}/plus_1000.jsonl') for r in rows}
print('VPTQ variants: count, flips vs vptq_w2 (up, down), flips/100')
for r in rows[1:]:
    a, b = D['vptq_w2'], D[r]; up = sum(b[u] and not a[u] for u in a); dn = sum(a[u] and not b[u] for u in a)
    print(f'  {r:20s} {sum(b.values())}  up {up} down {dn} flips/100 {(up+dn)/10:.1f}  net {sum(b.values())-sum(a.values()):+d}')
c = [sum(D[r].values()) for r in rows]
print('counts', c, 'mean %.1f sd %.1f max-mean %.1f' % (st.mean(c), st.stdev(c), max(c)-st.mean(c)))
# eval-noise sigma of a single run count from seed-8 data (flips f -> var of net difference f -> sigma_run = sqrt(f/2))
for n, f in (('fp seed7 vs seed8', 167), ('qtip', 185), ('quip0', 169), ('mentor stack', 128)):
    print(f'  sigma_run from {n}: {math.sqrt(f/2):.1f}')
s = math.sqrt(172/2)
print('sigma_run used: %.1f' % s)
for k, e in ((2, .564), (3, .846), (4, 1.029), (5, 1.163), (6, 1.267)):
    print(f'  E[max of {k}] - mean (std normal) = {e}:  {e*s:.1f} successes = {e*s/10:.2f} pp')
# in-budget 3 near variants
c3 = c[:3]; print('3 same-structure variants', c3, 'sd %.1f; expected sd from eval noise only %.1f' % (st.stdev(c3), s))
print('chi2 (5 df) for 6 variants vs eval noise: %.1f' % (sum((x-st.mean(c))**2 for x in c)/s**2))
print()
# Bridge FP numerics
for t, runs in (('stack_cube', dict(fp=36, fpfa=43, fps8=44, fps9=39, fps10=42, fps11=39, fpfas8=41, fpfas9=46, fpfas10=44, fpfas11=47, pbdb5=41, pbdb5s8=35, pbdb5s9=45)),
                ('carrot_on_plate', dict(fp=75, fpfa=72, fps8=74, pbdb5=81))):
    v = list(runs.values()); print(t, 'n', len(v), 'mean %.1f sd %.1f min %d max %d; fp (seed7 SDPA) minus mean %.1f' % (st.mean(v), st.stdev(v), min(v), max(v), runs['fp']-st.mean(v)))
