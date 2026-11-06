"""Rigour C4 (Task 6 b, c) on the stored per-unit files of Plus 1000.
(b) per suite x perturbation category (and difficulty level): successes of FP and each main row, paired difference with a base-task-clustered 95% interval.
(c) units FP fails but a row solves (gains) against units FP solves but the row fails (losses): under pure numerical noise gains ~ losses; and the units FP fails that EVERY main row solves,
    against what independence of the rows would give (product of the per-row gain rates among FP-failed units).
  python rigour_c4_units.py --groot DIR --out FILE"""
import argparse
import json
import math
from pathlib import Path
from rigour_c1_c2 import load, CLUSTER, paired

ROWS = ('qtip_w2', 'quip10_w2', 'quip_w2', 'vptq_w2', 'aqlm_w2', 'gptq_best_w2')


def records(p):
    return {(r['suite'], r['name']): r for r in map(json.loads, (Path(p) / 'plus_1000.jsonl').read_text().splitlines()) if True}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--groot', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fp = load(Path(a.groot) / 'fp'); rows = {r: load(Path(a.groot) / r) for r in ROWS}
    meta = records(Path(a.groot) / 'fp')
    out = []
    # (b)
    for field, title in (('category', 'perturbation category'), ('difficulty_level', 'difficulty level')):
        out.append(f'### (b) FP and rows by suite x {title}: successes of units (n), difference to FP in pp [95% interval clustered by base task]\n')
        groups = sorted({(k[0], meta[k][field]) for k in fp}, key=str)
        out.append('| suite | ' + title + ' | n | FP | ' + ' | '.join(ROWS) + ' |'); out.append('|---|---|---|---|' + '---|' * len(ROWS))
        for g in groups:
            ks = [k for k in fp if k[0] == g[0] and meta[k][field] == g[1]]
            cells = []
            for r in ROWS:
                sub_f = {k: fp[k] for k in ks}; sub_r = {k: rows[r][k] for k in ks}
                v = paired(sub_r, sub_f) if len(ks) > 3 else None
                cells.append(f"{sum(sub_r.values())} ({v['diff_pp']:+.0f} [{v['lo']:+.0f}, {v['hi']:+.0f}])" if v else '-')
            out.append(f'| {g[0]} | {g[1]} | {len(ks)} | {sum(fp[k] for k in ks)} | ' + ' | '.join(cells) + ' |')
        out.append('')
    # (c)
    keys = sorted(fp)
    ffail = [k for k in keys if not fp[k]]
    out.append(f'### (c) units FP fails ({len(ffail)}) that a row solves, against units FP solves ({len(keys) - len(ffail)}) that the row fails\n')
    out.append('| row | gains (FP fails, row solves) | losses (FP solves, row fails) | gain rate among FP-failed | loss rate among FP-solved | net |'); out.append('|---|---|---|---|---|---|')
    rate = {}
    for r in ROWS:
        g = sum(1 for k in ffail if rows[r][k]); l = sum(1 for k in keys if fp[k] and not rows[r][k])
        rate[r] = g / len(ffail)
        out.append(f'| {r} | {g} | {l} | {100 * g / len(ffail):.1f}% | {100 * l / (len(keys) - len(ffail)):.1f}% | {g - l:+d} |')
    main_rows = ('qtip_w2', 'quip10_w2', 'quip_w2', 'vptq_w2', 'aqlm_w2')
    allg = [k for k in ffail if all(rows[r][k] for r in main_rows)]
    exp = len(ffail) * math.prod(rate[r] for r in main_rows)
    out.append(f'\nFP-failed units solved by all five main rows (QTIP, QuIP# x2, VPTQ, AQLM): **{len(allg)}**; expected if the rows were independent given FP fails: {exp:.1f}.')
    # by base task / category
    cnt = {}
    for k in allg:
        cnt[(k[0], meta[k]['category'])] = cnt.get((k[0], meta[k]['category']), 0) + 1
    out.append('By suite and category: ' + ', '.join(f'{s} / {c}: {n}' for (s, c), n in sorted(cnt.items())) + '.')
    bt = {}
    for k in allg: bt[meta[k]['base_task']] = bt.get(meta[k]['base_task'], 0) + 1
    out.append('Base tasks with the most such units: ' + '; '.join(f'{t[:50]} ({n})' for t, n in sorted(bt.items(), key=lambda x: -x[1])[:5]) + '.')
    Path(a.out).write_text('\n'.join(out) + '\n'); print('\n'.join(out))


if __name__ == '__main__':
    main()
