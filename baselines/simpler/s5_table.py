"""Markdown table of the Task 5 rows (success per family, paired difference with FP) from RUN/s5q/paired/<row>.json and RUN/s5/eval/fp logs.
  python s5_table.py RUN > table.md"""
import ast
import json
import sys
from pathlib import Path

FAM = ('pick_coke_can', 'pick_object', 'move_near', 'open_drawer', 'close_drawer', 'place_in_closed_drawer')
ROWS = [('qtip_w2', 'QTIP'), ('quip10_w2', 'QuIP# 10 sweeps (main)'), ('quip_w2', 'QuIP# 0 sweeps (appendix)'), ('vptq_w2', 'VPTQ'), ('aqlm_w2', 'AQLM'), ('gptq_best_w2', 'GPTQ best'),
        ('gptq_w2', 'GPTQ plain'), ('awq_w2', 'AWQ')]
run = Path(sys.argv[1])
fp = {f: sum(ast.literal_eval(l.split('results:', 1)[1].strip())[1][:100]) for f in FAM for l in (run / f's5/eval/fp/{f}.log').read_text().splitlines() if l.startswith('results:')}
print('| row | ' + ' | '.join(FAM) + ' | all (of 600) | difference to FP [95% interval] |')
print('|---|' + '---|' * (len(FAM) + 2))
print('| FP | ' + ' | '.join(str(int(fp[f])) for f in FAM) + f' | {int(sum(fp.values()))} | - |')
for key, name in ROWS:
    p = run / f's5q/paired/{key}.json'
    if not p.exists():
        print(f'| {name} | not finished |')
        continue
    d = json.loads(p.read_text())
    a = d['all']
    print(f'| {name} | ' + ' | '.join(str(d[f]['row']) if f in d else '-' for f in FAM) + f" | {a['row']} | {a['diff_pp']:+.1f} pp [{a['ci95_pp'][0]:+.1f}, {a['ci95_pp'][1]:+.1f}] |")
