"""Task 5: per family success of a quantized row against FP on SimplerEnv Fractal (100 episodes per family, seed 7).
Reads the logs of s5_run (line `results: (env, [bool x 100], {...})`), pairs episodes by index (same seed list; episode order is the order the rollout
collects them: the pairing is therefore approximate, so the unpaired half-width is stored too), writes the per-family table with the paired difference and a 95% interval (normal approximation on the per-episode differences).

  python s5_stats.py --fp S5/eval/fp --row RUN/s5q/eval/<row> --out RUN/s5q/paired/<row>.json
"""
import argparse
import ast
import json
import math
from pathlib import Path

FAMILIES = ('pick_coke_can', 'pick_object', 'move_near', 'open_drawer', 'close_drawer', 'place_in_closed_drawer')


def successes(log):
    for line in Path(log).read_text().splitlines():
        if line.startswith('results:'):
            return [bool(x) for x in ast.literal_eval(line.split('results:', 1)[1].strip())[1]][:100]   # the rollout sometimes returns 101 episodes (the last batch of 5 environments overshoots); the first 100 are used
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fp', required=True)
    p.add_argument('--row', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    res, tot = {}, dict(fp=0, row=0, n=0, d=[])
    for fam in FAMILIES:
        f, r = successes(Path(a.fp) / f'{fam}.log'), successes(Path(a.row) / f'{fam}.log')
        if f is None or r is None or len(f) != len(r):
            continue
        d = [int(y) - int(x) for x, y in zip(f, r)]
        n, m = len(d), sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
        p1, p2 = sum(f) / n, sum(r) / n
        res[fam] = dict(fp=sum(f), row=sum(r), n=n, ci95_unpaired_halfwidth_pp=196 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n), diff_pp=100 * m, ci95_pp=[100 * (m - 1.96 * sd / math.sqrt(n)), 100 * (m + 1.96 * sd / math.sqrt(n))],
                        fp_only=sum(1 for x, y in zip(f, r) if x and not y), row_only=sum(1 for x, y in zip(f, r) if y and not x))
        tot['fp'] += sum(f); tot['row'] += sum(r); tot['n'] += n; tot['d'] += d
    if tot['d']:
        n, m = tot['n'], sum(tot['d']) / tot['n']
        sd = math.sqrt(sum((x - m) ** 2 for x in tot['d']) / (n - 1))
        p1, p2 = tot['fp'] / n, tot['row'] / n
        res['all'] = dict(fp=tot['fp'], row=tot['row'], n=n, ci95_unpaired_halfwidth_pp=196 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n), diff_pp=100 * m, ci95_pp=[100 * (m - 1.96 * sd / math.sqrt(n)), 100 * (m + 1.96 * sd / math.sqrt(n))])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    for k, v in res.items():
        print(f"{k:24s} FP {v['fp']:4d}  row {v['row']:4d}  of {v['n']}  diff {v['diff_pp']:+.1f} pp  [{v['ci95_pp'][0]:+.1f}, {v['ci95_pp'][1]:+.1f}]")


if __name__ == '__main__':
    main()
