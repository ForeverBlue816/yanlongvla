"""Rigour C5: overlap of the solved sets of the main rows (agreement, Jaccard) and what a rerun of the same row would give (E0 floor: ~13% of units flip under a tiny numerical change).
  python rigour_c5_overlap.py --groot DIR --pi05 DIR --out FILE"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rigour_c1_c2 import load


def table(title, data):
    names = list(data)
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    out = [f'### {title} ({len(keys)} units)\n', '| row A | row B | solved A | solved B | both | only A | only B | neither | agreement | Jaccard (solved) |', '|---|---|---|---|---|---|---|---|---|---|']
    for x, y in itertools.combinations(names, 2):
        a, b = [data[x][k] for k in keys], [data[y][k] for k in keys]
        both = sum(i and j for i, j in zip(a, b)); oa = sum(i and not j for i, j in zip(a, b)); ob = sum(j and not i for i, j in zip(a, b)); nei = len(keys) - both - oa - ob
        out.append(f'| {x} | {y} | {sum(a)} | {sum(b)} | {both} | {oa} | {ob} | {nei} | {100 * (both + nei) / len(keys):.1f}% | {both / (both + oa + ob):.3f} |')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groot', required=True); ap.add_argument('--pi05', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ref = Path(a.pi05).parents[2] / 'results' / 'plus_episodes'
    g = {r: load(Path(a.groot) / r) for r in ('fp', 'qtip_w2', 'quip10_w2', 'quip_w2', 'vptq_w2', 'aqlm_w2', 'gptq_best_w2')}
    p = {'fp': load(ref / 'original_fp.jsonl')}
    p.update({r: load(Path(a.pi05) / r) for r in ('qtip_w2', 'quip_w2', 'quipt0_w2', 'vptq_w2', 'aqlm_w2', 'gptq_best_w2')})
    Path(a.out).write_text(table('GR00T', g) + '\n' + table('pi0.5', p))


if __name__ == '__main__':
    main()
