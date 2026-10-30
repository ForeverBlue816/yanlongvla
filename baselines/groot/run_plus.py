"""Run LIBERO-Plus for one GR00T row: for each suite, start serve.py on that suite's checkpoint (GR00T environment), run
eval_libero_plus.py on that suite's units (LIBERO-Plus environment), stop the server. Resumable (finished units are skipped).

Usage:
  python run_plus.py --checkpoints ./ckpt_gptq_w2g128 --units ../data/plus_units_selection_1000.json \
      --out ./ckpt_gptq_w2g128/plus_1000.jsonl --groot-python /path/groot-venv/bin/python --plus-python /path/libero-plus-venv/bin/python
  python ../paired_stats.py plus --candidate ./ckpt_gptq_w2g128/plus_1000.jsonl --reference data/reference/groot_fp.plus_1000.jsonl
(--checkpoints ./checkpoints evaluates the prepared FP checkpoints; our reference rows are in data/reference/.)
"""
import argparse
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def free_port(start):
    for port in range(start, start + 500):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port)); return port
            except OSError:
                continue
    raise RuntimeError('no free port')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', required=True, help='directory with one subdirectory per suite')
    p.add_argument('--units', default=str(HERE.parent / 'data/plus_units_selection_1000.json'))
    p.add_argument('--out', required=True)
    p.add_argument('--groot-python', required=True)
    p.add_argument('--plus-python', required=True)
    p.add_argument('--port', type=int, default=None, help='first port to try (default: derived from the job id)')
    a = p.parse_args()
    units = json.loads(Path(a.units).read_text())
    units = units['units'] if isinstance(units, dict) else units        # ../data/plus_units_*.json carry a 'units' list
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)['suite'], json.loads(l)['task']) for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    for suite in SUITES:
        mine = [u for u in units if u['suite'] == suite]
        if not mine or all((u['suite'], u['task']) in done for u in mine):
            continue
        units_file = out.with_name(f'{out.stem}.units_{suite}.json'); units_file.write_text(json.dumps(mine))
        port = free_port(a.port or 20000 + os.getpid() % 20000)
        log = open(out.with_name(f'{out.stem}.server_{suite}.log'), 'a')
        server = subprocess.Popen([a.groot_python, '-u', str(HERE / 'serve.py'), '--checkpoint', str(Path(a.checkpoints) / suite), '--port', str(port)], stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
        try:
            for _ in range(900):
                if server.poll() is not None:
                    raise RuntimeError(f'server exited, see {log.name}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1):
                        break
                except Exception:
                    time.sleep(2)
            subprocess.run([a.plus_python, '-u', str(HERE / 'eval_libero_plus.py'), '--units', str(units_file), '--out', str(out), '--port', str(port)], check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
            log.close()
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    keys = {(r['suite'], r['task']) for r in rows}
    assert keys == {(u['suite'], u['task']) for u in units}, 'incomplete'
    print(json.dumps(dict(units=len(rows), successes=sum(r['success'] for r in rows))), flush=True)


if __name__ == '__main__':
    main()
