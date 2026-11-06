"""Rigour C1 (numbers audit of the Plus 1000 counts) and C2 (which differences are real; Holm-corrected paired tests).
Per-unit files: GR00T RUN/groot_baselines/eval/<row>/plus_1000.jsonl, pi0.5 RUN/readme_baselines/evaluation/<row>/plus_1000.jsonl (first 1000 units, joined on suite + name).
  python rigour_c1_c2.py --groot DIR --pi05 DIR --results RESULTS.md --out DIR
"""
import argparse
import itertools
import json
import math
import re
from pathlib import Path


CLUSTER = {}   # unit -> (suite, base task); units of one base task are not independent


def load(p):
    p = Path(p)
    p = p if p.suffix == '.jsonl' else p / 'plus_1000.jsonl'
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    d = {(r['suite'], r['name']): bool(r['success']) for r in rows}
    for r in rows:
        CLUSTER[(r['suite'], r['name'])] = (r['suite'], r.get('base_task', r['name']))
    return d if len(d) >= 1000 else None


NOISE_SD_PP = 100 * math.sqrt(0.13 * 1000) / 1000   # E0: a tiny numerical change flips ~13% of the 1000 units; net-count scale sqrt(130) = 11.4 successes = 1.14 pp
FLOOR_PP = 2.0   # E0: differences below about 20 of 1000 successes are within what a numerical perturbation alone can produce


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = [int(a[k]) - int(b[k]) for k in keys]
    n = len(d)
    m = sum(d) / n
    groups = {}
    for k, x in zip(keys, d):
        g = groups.setdefault(CLUSTER[k], [0, 0]); g[0] += x; g[1] += 1
    G = len(groups)
    se = math.sqrt(G / (G - 1) * sum((g[0] - m * g[1]) ** 2 for g in groups.values())) / n or 1e-9   # cluster-robust (by base task) standard error of the mean paired difference
    z = 100 * m / math.sqrt((100 * se) ** 2 + NOISE_SD_PP ** 2)   # widened by the E0 perturbation noise, in quadrature
    p = math.erfc(abs(z) / math.sqrt(2))
    return dict(n=n, a=sum(a[k] for k in keys), b=sum(b[k] for k in keys), diff_pp=100 * m, lo=100 * (m - 1.96 * se), hi=100 * (m + 1.96 * se), p=p,
                flips=sum(1 for x in d if x))


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ok = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            ok[i] = True
        else:
            break
    return ok


def model(name, root, rows, reruns, out, extra=None, only=None):
    data = {r: load(Path(root) / r) for r in rows}
    data.update({k: load(v) for k, v in (extra or {}).items()})
    data = {r: d for r, d in data.items() if d and (only is None or r in only)}
    names = list(data)
    floor = FLOOR_PP
    res = {pair: paired(data[pair[0]], data[pair[1]]) for pair in itertools.combinations(names, 2)}
    ok = holm([v['p'] for v in res.values()])
    lines = [f'### {name}: {len(names)} rows, {len(res)} comparisons, Holm at 0.05; E0 floor = {floor:.1f} pp; p-values use the paired se widened by the E0 perturbation noise ({NOISE_SD_PP:.2f} pp) in quadrature; the interval column is the paired interval with a standard error clustered by base task\n',
             '| row A | row B | A | B | A-B (pp) | 95% interval | p | Holm | beyond floor | real |', '|---|---|---|---|---|---|---|---|---|---|']
    for (pair, v), h in zip(res.items(), ok):
        beyond = abs(v['diff_pp']) > floor
        lines.append(f"| {pair[0]} | {pair[1]} | {v['a']} | {v['b']} | {v['diff_pp']:+.1f} | [{v['lo']:+.1f}, {v['hi']:+.1f}] | {v['p']:.3g} | {'yes' if h else 'no'} | {'yes' if beyond else 'no'} | {'**yes**' if h and beyond else 'no'} |")
    (out / f'c2_{name}{"_main" if only else ""}.md').write_text('\n'.join(lines) + '\n')
    return {r: sum(d.values()) for r, d in data.items()}, floor


KEYS = {'QTIP W2': 'qtip_w2', 'VPTQ W2': 'vptq_w2', 'AQLM W2': 'aqlm_w2', 'GPTQ W2': 'gptq_w2', 'AWQ W2': 'awq_w2', 'QuIP# W2 (10 sweeps, official)': 'QUIP_OFFICIAL',
        '*QuIP# W2, 0 sweeps (appendix)* ¹': 'QUIP_ZERO'}


def c1(results, groot, pi05, ref, out):
    """Recompute successes and difference to FP of each Plus 1000 row in RESULTS.md; list mismatches (count differs, difference differs by > 0.06 pp, interval ends by > 0.8 pp: the document uses a stratified bootstrap)."""
    fp_g, fp_p = load(groot / 'fp'), load(ref / 'original_fp.jsonl')
    sec, lines, n = None, [], 0
    for l in Path(results).read_text().splitlines():
        if l.startswith('## π0.5'): sec = 'p'
        elif l.startswith('## GR00T N1.7 ('): sec = 'g'
        elif l.startswith('## GR00T N1.7 on'): sec = None
        cells = [c.strip() for c in l.split('|')[1:-1]]
        if not sec or len(cells) < 5: continue
        import re
        m = [re.fullmatch(r'([+-]?\d+\.\d) \(([+-]?\d+\.\d), ([+-]?\d+\.\d)\)', c) for c in cells]
        idx = next((i for i, x in enumerate(m) if x), None)
        cnt = next((c for c in reversed(cells[:idx]) if c.isdigit()), None) if idx else None
        if idx is None or cnt is None: continue
        name, key = cells[0], KEYS.get(cells[0])
        if key == 'QUIP_OFFICIAL': key = 'quip_w2' if sec == 'p' else 'quip10_w2'
        if key == 'QUIP_ZERO': key = 'quipt0_w2' if sec == 'p' else 'quip_w2'
        if 'best variant per layer' in name: key = 'gptq_best_w2'
        mm = re.search(r'\((G\d+)', name)
        VARIANTS = {'G34': 'vptq_v5k1024_w2', 'G37': 'vptq_iter500_w2', 'G39': 'vptq_v6k4096_w2', 'G40': 'vptq_iter500t0_w2', 'G41': None}
        if sec == 'g' and 'VPTQ' in name and name != 'VPTQ W2':
            key = {'5-weight': 'vptq_v5k1024_w2', '500 k-means iterations (G37)': 'vptq_iter500_w2', '6-weight vectors, 4096': 'vptq_v6k4096_w2', 'tolerance': 'vptq_iter500t0_w2',
                   '3072': 'vptq_v6k3072_w2', '12288': 'vptq_v8k12288_w2'}
            key = next((v for k, v in key.items() if k in name), None)
        if 'GPTQ W2 + MSE clipping' in name: key = 'gptq_mse_w2'
        elif 'act-order + MSE + 10%' in name: key = 'gptq_combo_w2'
        elif 'act-order + MSE' in name: key = 'gptq_aomse_w2'
        if not key: continue
        cand = load((groot / 'eval' if False else groot) / key) if sec == 'g' else load(pi05 / key)
        if cand is None: continue
        v = paired(cand, fp_g if sec == 'g' else fp_p)
        d, lo, hi = (float(x) for x in m[idx].groups())
        n += 1
        bad = v['a'] != int(cnt) or abs(v['diff_pp'] - d) > 0.06 or abs(v['lo'] - lo) > 0.8 or abs(v['hi'] - hi) > 0.8
        lines.append(f"| {sec} | {name[:60]} | {cnt} | {v['a']} | {d:+.1f} | {v['diff_pp']:+.1f} | ({lo:+.1f}, {hi:+.1f}) | ({v['lo']:+.1f}, {v['hi']:+.1f}) | {'**MISMATCH**' if bad else 'ok'} |")
    head = ['| model | row | doc count | recomputed | doc diff | recomputed | doc interval | recomputed (normal) | |', '|---|---|---|---|---|---|---|---|---|']
    (out / 'c1_plus1000.md').write_text('\n'.join(head + lines) + '\n')
    print(f'C1: {n} Plus 1000 rows checked;', sum('MISMATCH' in x for x in lines), 'mismatches')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groot', required=True)
    ap.add_argument('--pi05', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    g_rows = [p.name for p in sorted(Path(a.groot).iterdir()) if (p / 'plus_1000.jsonl').exists()]
    p_rows = [p.name for p in sorted(Path(a.pi05).iterdir()) if (p / 'plus_1000.jsonl').exists() and p.name.endswith('_w2')]
    ref = Path(a.pi05).parents[2] / 'results' / 'plus_episodes'
    gc, gf = model('groot', a.groot, g_rows, [('fp', 'fp_rerun'), ('qtip_w2', 'qtip_w2_rerun')], out)
    pc, pf = model('pi05', a.pi05, p_rows, [], out, extra=dict(fp_w16=ref / 'original_fp.jsonl', original_2bit=ref / 'original_2bit.jsonl'))
    model('groot', a.groot, g_rows, [], out, only={'fp', 'qtip_w2', 'quip10_w2', 'quip_w2', 'vptq_w2', 'aqlm_w2', 'gptq_best_w2'})
    model('pi05', a.pi05, p_rows, [], out, extra=dict(fp_w16=ref / 'original_fp.jsonl', original_2bit=ref / 'original_2bit.jsonl'), only={'fp_w16', 'original_2bit', 'qtip_w2', 'quip_w2', 'quipt0_w2', 'vptq_w2', 'aqlm_w2', 'gptq_best_w2'})
    c1(a.results, Path(a.groot), Path(a.pi05), ref, out)
    json.dump(dict(groot=gc, pi05=pc, groot_floor=gf), open(out / 'counts.json', 'w'), indent=1)
    print('GR00T counts', gc)
    print('pi0.5 counts', pc)


if __name__ == '__main__':
    main()
