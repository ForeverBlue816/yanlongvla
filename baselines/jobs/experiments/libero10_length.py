"""Task 6a / GOAL v3 2.5: does the loss of the 2-bit rows on libero_10 follow episode length?
For every row, the units FP solves are split into terciles by FP's number of policy calls (episode length); per tercile the share of units the row loses.
Also: failed rollouts of the row that end at the step limit (policy_calls at the maximum of the suite) against ones that end earlier.
  python libero10_length.py --eval RUN/eval --out out.md
"""
import argparse
import json
from pathlib import Path

ROWS = ['qtip_w2', 'quip_w2', 'quip10_w2', 'vptq_w2', 'aqlm_w2', 'gptq_best_w2', 'gptq_combo_w2', 'gptq_mse_w2']
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']


def load(p):
    return {(r['suite'], r['name']): r for r in (json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())} if Path(p).exists() else None


def main():
    a = argparse.ArgumentParser(); a.add_argument('--eval', required=True); a.add_argument('--out', required=True); a = a.parse_args()
    fp = load(Path(a.eval) / 'fp/plus_1000.jsonl')
    out = ['# libero_10 against episode length (FP policy calls as the length)\n']
    for suite in SUITES:
        solved = sorted((r['policy_calls'], k) for k, r in fp.items() if k[0] == suite and r['success'])
        n = len(solved); cut = [solved[n // 3][0], solved[2 * n // 3][0]]
        out += [f'\n## {suite}: FP solves {n} units; FP policy calls: short <= {cut[0]}, middle <= {cut[1]}, long above\n', '| row | lost in short | lost in middle | lost in long | all lost | lost with step limit reached |', '|---|---|---|---|---|---|']
        limit = max(r['policy_calls'] for r in fp.values() if r['suite'] == suite)
        for row in ROWS:
            d = load(Path(a.eval) / row / 'plus_1000.jsonl')
            if not d:
                continue
            b = [[0, 0], [0, 0], [0, 0]]
            lim = lost = 0
            for calls, k in solved:
                t = 0 if calls <= cut[0] else 1 if calls <= cut[1] else 2
                b[t][1] += 1
                if not d[k]['success']:
                    b[t][0] += 1; lost += 1; lim += d[k]['policy_calls'] >= limit
            out.append(f'| {row} | ' + ' | '.join(f'{x}/{y} ({100 * x / max(y, 1):.0f}%)' for x, y in b) + f' | {lost}/{n} | {lim}/{lost} |')
    Path(a.out).write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
