"""Sanity gate for one GR00T row (same rule as the π0.5 rows, decision 17): the first 50 libero_spatial units of the Plus 1000.
A row is stopped when it scores 0, or 5 percent or less where FP scores 50 percent or more on the same units.

  python gate.py --row gptq_w2 --candidate sanity/gptq_w2.plus.jsonl --reference data/reference/groot_fp.plus_1000.jsonl \
      --stopped-dir stopped --out sanity/gptq_w2.json
Writes --out; on failure also writes <stopped-dir>/<row> with the reason. Always exits 0 (the scheduler reads the files).
"""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--row', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--stopped-dir', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cand = {(r['suite'], r['task']): r['success'] for r in rows(a.candidate)}
    ref = {(r['suite'], r['task']): r['success'] for r in rows(a.reference)}
    keys = sorted(cand)
    n = len(keys)
    got = sum(cand[k] for k in keys)
    fp = sum(ref[k] for k in keys if k in ref)
    rate, fp_rate = got / n, fp / n
    failed = got == 0 or (rate <= 0.05 and fp_rate >= 0.5)
    rec = dict(row=a.row, units=n, successes=got, fp_successes_same_units=fp, passed=not failed)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    if failed:
        d = Path(a.stopped_dir); d.mkdir(parents=True, exist_ok=True)
        (d / a.row).write_text(f'sanity sample {got}/{n} against FP {fp}/{n} on the same units\n')


if __name__ == '__main__':
    main()
