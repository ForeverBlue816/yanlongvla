"""Task 2 (GR00T part) and Task 6: success of the 2-bit rows on LIBERO-Plus 1000 by suite, by perturbation, by both and by difficulty level,
and a look at the actual instances: how many rows solve each unit, which units FP solves but every 2-bit row fails, which base tasks lose
most, and whether the units that fail for GR00T also fail for pi0.5 (the Plus 1000 selection is shared; units are joined on suite + name).

  python instances.py --groot RUN/groot_baselines/eval --pi05 RUN/readme_baselines/evaluation --out DIR
Writes DIR/tables.md and DIR/units.csv (one line per unit with the success of every row).
"""
import argparse
import collections
import csv
import json
from pathlib import Path

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
CATS = ['Camera Viewpoints', 'Robot Initial States']
G_ROWS = [('fp', 'FP'), ('qtip_w2', 'QTIP'), ('quip_w2', 'QuIP# 0 sweeps'), ('quip10_w2', 'QuIP# 10 sweeps'), ('vptq_w2', 'VPTQ'), ('aqlm_w2', 'AQLM'),
          ('gptq_w2', 'GPTQ'), ('gptq_mse_w2', 'GPTQ MSE'), ('gptq_aomse_w2', 'GPTQ act-order+MSE'), ('gptq_combo_w2', 'GPTQ combo'), ('vptq_v5k1024_w2', 'VPTQ v5 k1024'),
          ('gptq_best_w2', 'GPTQ best'), ('vptq_iter500_w2', 'VPTQ 500 iter'), ('vptq_iter500t0_w2', 'VPTQ 500 iter t0'), ('vptq_v6k4096_w2', 'VPTQ v6 k4096'),
          ('vptq_v6k3072_w2', 'VPTQ v6 k3072'), ('vptq_v8k12288_w2', 'VPTQ v8 k12288')]
P_ROWS = [('quip_w2', 'pi0.5 QuIP# 10 sweeps'), ('qtip_w2', 'pi0.5 QTIP'), ('vptq_w2', 'pi0.5 VPTQ'), ('aqlm_w2', 'pi0.5 AQLM'), ('gptq_w2', 'pi0.5 GPTQ'), ('gptq_best_w2', 'pi0.5 GPTQ best')]


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {(r['suite'], r['name']): r for r in rows} if len(rows) >= 1000 else None   # only finished rows


def pct(k, n):
    return f'{k}/{n} ({100 * k / n:.0f}%)' if n else '-'


def md_table(title, header, body):
    return f'\n#### {title}\n\n| | ' + ' | '.join(header) + ' |\n|---|' + '---|' * len(header) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in body) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groot', required=True)
    p.add_argument('--pi05', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    g = {r: load(Path(a.groot) / r / 'plus_1000.jsonl') for r, _ in G_ROWS}
    g = {r: d for r, d in g.items() if d}
    labels = dict(G_ROWS)
    fp = g['fp']
    units = list(fp)
    md = ['# GR00T N1.7 on LIBERO-Plus 1000: success by group and a look at the instances\n', f'Rows with 1000 finished episodes: {", ".join(labels[r] for r in g)}.\n']

    def succ(r, u):
        return bool(g[r][u]['success'])

    def grouped(title, keyf, keys):
        body = []
        for r in g:
            cells = []
            for k in keys:
                us = [u for u in units if keyf(fp[u]) == k]
                cells.append(pct(sum(succ(r, u) for u in us), len(us)))
            body.append([labels[r]] + cells)
        md.append(md_table(title, [str(k) for k in keys], body))

    grouped('By suite', lambda x: x['suite'], SUITES)
    grouped('By perturbation', lambda x: x['category'], CATS)
    grouped('By perturbation and suite', lambda x: (x['category'], x['suite']), [(c, s) for c in CATS for s in SUITES])
    grouped('By difficulty level (the unit field difficulty_level)', lambda x: (x['category'], x['difficulty_level']), [(c, d) for c in CATS for d in range(1, 6)])

    # Task 6: the instances
    quant = [r for r in g if r != 'fp']
    main_rows = [r for r in ('qtip_w2', 'quip10_w2', 'vptq_w2', 'aqlm_w2') if r in g]
    md.append(f'\n## Instances (main rows: {", ".join(labels[r] for r in main_rows)})\n')
    dist = collections.Counter((succ('fp', u), sum(succ(r, u) for r in main_rows)) for u in units)
    md.append(md_table('Units by FP result and by the number of main 2-bit rows that solve them', [f'{k} of {len(main_rows)} solve' for k in range(len(main_rows) + 1)],
                       [[f'FP {"solves" if f else "fails"}'] + [str(dist[(f, k)]) for k in range(len(main_rows) + 1)] for f in (True, False)]))
    lost = [u for u in units if succ('fp', u) and not any(succ(r, u) for r in main_rows)]
    gained = [u for u in units if not succ('fp', u) and all(succ(r, u) for r in main_rows)]
    md.append(f'FP solves and no main row solves: {len(lost)} units. FP fails and every main row solves: {len(gained)} units (the noise level of the comparison).\n')
    bt = collections.defaultdict(list)
    for u in units:
        bt[(fp[u]['suite'], fp[u]['base_task'])].append(u)
    rowsb = sorted(((sum(succ('fp', u) for u in us) - sum(sum(succ(r, u) for r in main_rows) / len(main_rows) for u in us), k, len(us),
                     sum(succ('fp', u) for u in us), sum(sum(succ(r, u) for r in main_rows) / len(main_rows) for u in us)) for k, us in bt.items()), reverse=True)
    md.append(md_table('Base tasks that lose most (FP successes minus the mean main-row successes)', ['units', 'FP solves', 'main rows solve (mean)', 'loss'],
                       [[f'{k[0]} {k[1][:70]}', str(n), str(f), f'{m:.1f}', f'{loss:.1f}'] for loss, k, n, f, m in rowsb[:12]]))
    md.append(md_table('Base tasks with no loss or a gain (the 5 best)', ['units', 'FP solves', 'main rows solve (mean)', 'loss'],
                       [[f'{k[0]} {k[1][:70]}', str(n), str(f), f'{m:.1f}', f'{loss:.1f}'] for loss, k, n, f, m in rowsb[-5:]]))
    md.append('\n### The units FP solves and no main row solves (first 25)\n\n| suite | perturbation | level | unit |\n|---|---|---|---|\n' +
              '\n'.join(f'| {fp[u]["suite"]} | {fp[u]["category"]} | {fp[u]["difficulty_level"]} | {u[1][:110]} |' for u in lost[:25]) + '\n')

    # agreement with pi0.5 on the shared units
    pi = {r: load(Path(a.pi05) / r / 'plus_1000.jsonl') for r, _ in P_ROWS}
    pi = {r: d for r, d in pi.items() if d}
    if pi and main_rows:
        both, g_only, p_only, none = 0, 0, 0, 0
        for u in units:
            if not all(u in d for d in pi.values()):
                continue
            gs = sum(succ(r, u) for r in main_rows) >= len(main_rows) / 2
            ps = sum(bool(d[u]['success']) for d in pi.values()) >= len(pi) / 2
            both += gs and ps; g_only += gs and not ps; p_only += ps and not gs; none += not gs and not ps
        md.append(f'\n### Agreement with pi0.5 on the shared units (majority of the main rows solves the unit)\n\n| both models solve | only GR00T | only pi0.5 | neither |\n|---|---|---|---|\n| {both} | {g_only} | {p_only} | {none} |\n')

    (out / 'tables.md').write_text('\n'.join(md))
    with open(out / 'units.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['suite', 'category', 'level', 'name'] + [labels[r] for r in g])
        for u in units:
            w.writerow([fp[u]['suite'], fp[u]['category'], fp[u]['difficulty_level'], u[1]] + [int(succ(r, u)) for r in g])
    print('\n'.join(md))


if __name__ == '__main__':
    main()
