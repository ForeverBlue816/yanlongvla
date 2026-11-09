"""GOAL v8 Part I (L2): read-only analysis of stored per-episode files. Run: python part_i.py > part_i_out.txt"""
import json, statistics as st, math
from pathlib import Path
G = Path('/scratch/d250009/pi05-handover/runs/groot_baselines')
def libero(p):
    d = {}
    for l in open(p):
        r = json.loads(l); d[(r['suite'], r['task'], r['name'])] = bool(r['success'])
    return d
def flips(a, b):
    k = set(a) & set(b); up = sum(b[u] and not a[u] for u in k); dn = sum(a[u] and not b[u] for u in k)
    return len(k), sum(a[u] for u in k), sum(b[u] for u in k), up, dn
print('== LIBERO GR00T flips (A=reference, B=other); n, A, B, up, down, flips per 100')
fp = libero(G/'eval/fp/plus_1000.jsonl')
mentor = libero('/home/d250009/yanlongvla-clean/baselines/groot/data/reference/groot_fp.plus_1000.jsonl')
pairs = [('fp vs fp_rerun (same stack)', fp, libero(G/'eval/fp_rerun/plus_1000.jsonl')),
         ('fp vs mentor fp (other stack)', fp, mentor),
         ('fp vs fp_seed8 (eval seed)', fp, libero(G/'eval/fp_seed8/plus_1000.jsonl')),
         ('qtip vs qtip_seed8', libero(G/'eval/qtip_w2/plus_1000.jsonl'), libero(G/'eval/qtip_w2_seed8/plus_1000.jsonl')),
         ('quip0 vs quip0_seed8', libero(G/'eval/quip_w2/plus_1000.jsonl'), libero(G/'eval/quip_w2_seed8/plus_1000.jsonl')),
         ('fp vs math kernel (200)', fp, libero(G/'kernel/math.plus_200.jsonl')),
         ('fp vs efficient kernel (200)', fp, libero(G/'kernel/efficient.plus_200.jsonl'))]
for n, a, b in pairs:
    k, sa, sb, up, dn = flips(a, b); print(f'{n:34s} n={k} A={sa} B={sb} up={up} down={dn} flips/100={100*(up+dn)/k:.1f} net/100={100*(sb-sa)/k:+.1f}')
print('\n== LIBERO Plus 1000 totals per row dir')
for d in sorted((G/'eval').iterdir()):
    f = d/'plus_1000.jsonl'
    if f.exists():
        x = libero(f); print(f'{d.name:24s} {len(x)} {sum(x.values())}')
print('\n== Bridge: carrot / stack_cube successes of all FP no-op variants (seed 7 unless named)')
def b(r, t):
    f = G/f's5b/eval/{r}/widowx_{t}.json'
    return {x['episode']: bool(x['success']) for x in json.loads(f.read_text())}
for t in ('carrot_on_plate', 'stack_cube'):
    base = b('fp', t)
    print(t)
    for r in ('fp','fprr','fpsl','fpfa','fpnt','fps8','fps9','fps10','fps11','fpfas8','fpfas9','fpfas10','fpfas11','pbdb5','pbdb5s8','pbdb5s9'):
        try: x = b(r, t)
        except FileNotFoundError: continue
        k, sa, sb, up, dn = flips(base, x)
        print(f'  {r:9s} {sb:3d}/120  flips vs fp: up {up} down {dn} = {100*(up+dn)/k:.1f}/100')
