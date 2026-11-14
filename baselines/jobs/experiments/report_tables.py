"""Final pi0.5 tables : LIBERO and LIBERO-Plus (1st 1000, 2nd 1000, 2000) for every evaluated row, with bpw
from checkpoints/<row>/accounting.json and the paired statistics the scheduler wrote (evaluation/<row>/paired_stats). Read-only;
prints markdown. A cell is '-' when the part is missing or incomplete."""
import glob, json, os, sys

RUN = sys.argv[1] if len(sys.argv) > 1 else '/home/d250009/projects/pi05-handover/runs/readme_baselines'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
NON_OFFICIAL = {'qtip_w1', 'vptq_w1', 'quip_w1', 'vptq_v4k16_w1'}   # no official 1-bit recipe: closest-to-official setting
LABEL = {'qtip': 'QTIP', 'quip': 'QuIP#', 'vptq': 'VPTQ', 'aqlm': 'AQLM', 'gptq': 'GPTQ', 'awq': 'AWQ', 'rtn': 'RTN', 'fp': 'FP'}
STRUCT = {'vptq_w1': 'v8 k256', 'vptq_w2': 'v4 k256', 'vptq_w3': 'v4 k256 + r16', 'vptq_v8k14080_w2': 'v8 k14080',
          'vptq_v8k13824r256_w3': 'v8 k13824 + r256', 'vptq_v4k16_w1': 'v4 k16', 'aqlm_w1': '1x8g8', 'aqlm_w2': '2x8g8',
          'quip_w1': 'E81B (256-point E8)', 'qtip_w1': 'K=1', 'quip_w2': '10 sweeps', 'quip_w3': '10 sweeps'}
ROWS = ['qtip_w1', 'aqlm_w1', 'quip_w1', 'vptq_w1', 'vptq_v4k16_w1',
        'qtip_w2', 'vptq_w2', 'vptq_v8k14080_w2', 'quip_w2', 'aqlm_w2',
        'qtip_w3', 'vptq_w3', 'vptq_v8k13824r256_w3', 'quip_w3', 'aqlm_w3']


def count(path, n):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return sum(bool(r.get('success')) for r in rows) if len(rows) >= n else None


def stat(row, name, key='all_tasks'):
    p = f'{RUN}/evaluation/{row}/paired_stats/{name}.json'
    if not os.path.exists(p):
        return '-'
    d = json.load(open(p))
    d = d.get(key, d)
    lo, hi = d['paired_difference_ci95_pp']
    return f"{d['difference_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def bpw(row):
    for p in glob.glob(f'{RUN}/checkpoints/{row}/quant.json') + glob.glob(f'{RUN}/checkpoints/{row}/accounting.json'):
        d = json.load(open(p))
        return d.get('quantizable_linear_bpw'), d.get('whole_model_bpw')
    return None, None


def name(row):
    m = row.split('_')[0]
    s = f"{LABEL.get(m, m)} {row.rsplit('_w', 1)[1]}-bit, {STRUCT.get(row, 'default recipe')}"
    return s + (' (NON-OFFICIAL)' if row in NON_OFFICIAL else '')


def f(x, d=4):
    return '-' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


print('### pi0.5 on LIBERO (500 episodes per suite; difference vs FP over 2000 episodes, paired 95% CI)\n')
print('| row | linear bpw | whole bpw | spatial | object | goal | LIBERO-10 | total /2000 | vs FP (pp) |')
print('|---|---|---|---|---|---|---|---|---|')
for row in ROWS:
    if not os.path.isdir(f'{RUN}/evaluation/{row}'):
        continue
    c = [count(f'{RUN}/evaluation/{row}/libero/{s}.jsonl', 500) for s in SUITES]
    lb, wb = bpw(row)
    tot = sum(c) if None not in c else None
    print(f'| {name(row)} | {f(lb)} | {f(wb)} | ' + ' | '.join(f(x) for x in c) + f' | {f(tot)} | {stat(row, "libero_vs_original_fp")} |')

print('\n### pi0.5 on LIBERO-Plus (1st 1000 = selection units, 2nd 1000 = added units; paired 95% CI)\n')
print('| row | 1st 1000 | vs FP 765 (pp) | 2nd 1000 | 2000 | 2000 vs original_2bit (pp) |')
print('|---|---|---|---|---|---|')
for row in ROWS:
    if not os.path.isdir(f'{RUN}/evaluation/{row}'):
        continue
    e = f'{RUN}/evaluation/{row}'
    a, b, t = count(f'{e}/plus_1000.jsonl', 1000), count(f'{e}/plus_added_1000.jsonl', 1000), count(f'{e}/plus_2000.jsonl', 2000)
    print(f'| {name(row)} | {f(a)} | {stat(row, "plus_1000_vs_original_fp")} | {f(b)} | {f(t)} | {stat(row, "plus_2000_vs_original_2bit_expanded")} |')
