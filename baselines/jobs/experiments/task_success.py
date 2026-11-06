"""Task 2: task success of the current 2-bit rows grouped (1) by task suite, (2) by perturbation, (3) by perturbation and suite,
for LIBERO and for LIBERO-Plus (Plus 1000, the added 1000 and the 2000 together).

  python task_success.py --eval DIR_with_row_folders --out DIR
Writes DIR/tables.md and DIR/results.json. Reads evaluation/<row>/{plus_1000.jsonl,plus_added_1000.jsonl,plus_2000.jsonl,libero/<suite>.jsonl}.
Plain LIBERO has no perturbation, so it only has the by-suite arrangement.
"""
import argparse
import collections
import json
from pathlib import Path

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
CATS = ['Camera Viewpoints', 'Robot Initial States']
ROWS = [('fp_w16', 'FP'), ('quipt0_w2', 'QuIP# W2, 0 sweeps'), ('qtip_w2', 'QTIP W2'), ('vptq_w2', 'VPTQ W2'), ('quip_w2', 'QuIP# W2, official 10 sweeps'),
        ('aqlm_w2', 'AQLM W2'), ('gptq_w2', 'GPTQ W2'), ('awq_w2', 'AWQ W2'), ('gptq_best_w2', 'GPTQ best W2')]


RESULTS = Path('/home/d250009/yanlongvla-clean/results')


def fp_rows(fname, ev):
    """FP has no evaluation/<row>/plus_*.jsonl: its Plus 1000 is the reference episodes in results/, its added 1000 the parts under
    evaluation/fp_w16/plus_parts, LIBERO the reference folder."""
    if fname == 'plus_1000.jsonl':
        return rows_of(RESULTS / 'plus_episodes/original_fp.jsonl')
    added = [r for f in sorted((ev / 'fp_w16/plus_parts').glob('add1000_*.jsonl')) for r in rows_of(f)]
    if fname == 'plus_added_1000.jsonl':
        return added
    return rows_of(RESULTS / 'plus_episodes/original_fp.jsonl') + added


def rows_of(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else None


def cell(rows):
    n = len(rows)
    k = sum(r['success'] for r in rows)
    return f'{k}/{n} ({100 * k / n:.1f}%)' if n else '-'


def table(title, header, body):
    out = [f'\n#### {title}\n', '| row | ' + ' | '.join(header) + ' |', '|---|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(r) + ' |' for r in body]
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    ev, out = Path(a.eval), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, res = [], {}
    for label, fname in (('LIBERO-Plus 1000', 'plus_1000.jsonl'), ('LIBERO-Plus added 1000', 'plus_added_1000.jsonl'), ('LIBERO-Plus 2000 (both)', 'plus_2000.jsonl')):
        md.append(f'\n### {label}\n')
        data = {r: (fp_rows(fname, ev) if r == 'fp_w16' else rows_of(ev / r / fname)) for r, _ in ROWS}
        have = [(r, name) for r, name in ROWS if data[r]]
        md.append(table('By task suite', SUITES + ['all'], [[name] + [cell([x for x in data[r] if x['suite'] == s]) for s in SUITES] + [cell(data[r])] for r, name in have]))
        md.append(table('By perturbation', CATS + ['all'], [[name] + [cell([x for x in data[r] if x['category'] == c]) for c in CATS] + [cell(data[r])] for r, name in have]))
        header = [f'{c.split()[0]} / {s}' for c in CATS for s in SUITES]
        md.append(table('By perturbation and task suite', header, [[name] + [cell([x for x in data[r] if x['category'] == c and x['suite'] == s]) for c in CATS for s in SUITES] for r, name in have]))
        res[label] = {r: dict(n=len(data[r]), successes=sum(x['success'] for x in data[r]),
                              by_suite={s: [sum(x['success'] for x in data[r] if x['suite'] == s), sum(1 for x in data[r] if x['suite'] == s)] for s in SUITES},
                              by_perturbation={c: [sum(x['success'] for x in data[r] if x['category'] == c), sum(1 for x in data[r] if x['category'] == c)] for c in CATS},
                              by_perturbation_suite={f'{c} / {s}': [sum(x['success'] for x in data[r] if x['category'] == c and x['suite'] == s), sum(1 for x in data[r] if x['category'] == c and x['suite'] == s)] for c in CATS for s in SUITES})
                      for r, _ in have}
    md.append('\n### LIBERO (2000 episodes, 50 per task; no perturbation)\n')
    lib = {}
    for r, name in ROWS:
        suites = {s: rows_of((RESULTS / 'libero_full/original_fp' if r == 'fp_w16' else ev / r / 'libero') / f'{s}.jsonl') for s in SUITES}
        if all(suites.values()):
            lib[r] = suites
    md.append(table('By task suite', SUITES + ['all'], [[name] + [cell(lib[r][s]) for s in SUITES] + [cell([x for s in SUITES for x in lib[r][s]])] for r, name in ROWS if r in lib]))
    res['LIBERO'] = {r: {s: [sum(x['success'] for x in lib[r][s]), len(lib[r][s])] for s in SUITES} for r in lib}
    (out / 'tables.md').write_text('\n'.join(md) + '\n')
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
