"""Standard LIBERO full (four suites x 10 tasks x 50 episodes, seed 7: the protocol of plustrain_libero.py --stage full) for a pi0.5
TCQ row at the current rate (user 2026-11-12: the 1-bit rows, after their Plus stage). Same server as the Plus stage
(pi05_tcq.py --stage serve), same client as every LIBERO-full row (decisions_bookkeeping_client.py in the libero env).

  source scripts/env.sh; TCQ_BITS=1 PI05_TCQ_SKELETON=fp PLUSTRAIN_ANY_GPU=1 JAX_PLATFORMS=cpu PYTHONPATH=$PYTHONPATH:scripts \
      CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 envs/openpi/bin/python lambda/pi05_libero.py --row tcq_raqft_d1 --port 24100

Output: runs/plustrain_20261008/pi05_tcq<_w1>/libero/<row>/full/{<suite>_task<t>.jsonl, complete.json}. Resumable (per-task files).
No paired statistics here: the per-episode FP rows (runs/g0/<suite>_seed7.jsonl) are not on this machine."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/projects/yanlongvla')
S = ROOT / 'research/scripts'
sys.path.insert(0, str(S))
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
EPISODES = 50


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def start_server(T, row, port, log):
    """pi05_tcq.start_server with an explicit port, and a check that the process answering on the port is this row's server.
    (2026-11-12 03:57 UTC: two rows started together both chose port 23000; one server failed to bind and its client was served by the
    other row's server. Those partial outputs are kept as full_INVALID_port_clash_20261112_0357 and are not used.)"""
    import time
    import urllib.request
    port = T.free_port(port)
    start = log.tell()
    server = subprocess.Popen([sys.executable, '-u', str(S / 'pi05_tcq.py'), '--stage', 'serve', '--row', row, '--port', str(port)], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(1800):
        if server.poll() is not None:
            raise RuntimeError(f'server exited: {log.name}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1):
                break
        except Exception:                                  # noqa: BLE001  (starting, or a non-HTTP reply)
            time.sleep(2)
    else:
        server.kill(); raise TimeoutError(log.name)
    time.sleep(5)
    text = Path(log.name).read_text()[start:]
    assert server.poll() is None and f'"serve": "{row}"' in text and 'address already in use' not in text, f'the server on port {port} is not this row\'s: {log.name}'
    return server, port


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--row', required=True)
    p.add_argument('--port', type=int, required=True, help='first port to try; give every concurrent row its own range')
    a = p.parse_args()
    import pi05_tcq as T
    directory = T.P5 / 'libero' / a.row / 'full'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        print((directory / 'complete.json').read_text()); return
    gpu = T.gpu_name()
    if (directory / 'gpu.txt').exists():
        assert (directory / 'gpu.txt').read_text().strip() == gpu, 'one GPU model per row'
    (directory / 'gpu.txt').write_text(gpu + '\n')
    for path in directory.glob('*.errors.jsonl'):
        assert not path.stat().st_size, path
    if a.row in T.ROWS:
        T.stage_assemble(a.row)
    paths = {(s, t): directory / f'{s}_task{t}.jsonl' for s in SUITES for t in range(10)}
    if any(len(read(q)) < EPISODES for q in paths.values()):
        with (directory / 'server.log').open('a') as log:
            server, port = start_server(T, a.row, a.port, log)
            try:
                for (suite, task), path in paths.items():
                    if len(read(path)) < EPISODES:
                        subprocess.run([str(ROOT / 'envs/libero/bin/python'), '-u', str(S / 'decisions_bookkeeping_client.py'), '--port', str(port), '--suite', suite, '--task-ids', str(task),
                                        '--seed', '7', '--episodes', str(EPISODES), '--output', str(path)], check=True)
            finally:
                T.stop(server)
    rows = {k: read(q) for k, q in paths.items()}
    assert all(len(r) == EPISODES and sorted(x['episode'] for x in r) == list(range(EPISODES)) for r in rows.values())
    per_suite = {s: sum(int(x['success']) for t in range(10) for x in rows[(s, t)]) for s in SUITES}
    per_task = {f'{s}_task{t}': sum(int(x['success']) for x in rows[(s, t)]) for s in SUITES for t in range(10)}
    record = dict(status='complete', row=a.row, bits=os.environ.get('TCQ_BITS'), protocol='LIBERO full: 4 suites x 10 tasks x 50 episodes, seed 7', episodes=40 * EPISODES,
                  successes=sum(per_suite.values()), per_suite=per_suite, per_task=per_task,
                  successes_1900_excluding_libero_10_tasks_8_9=sum(per_suite.values()) - per_task['libero_10_task8'] - per_task['libero_10_task9'], gpu=gpu)
    (directory / 'complete.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
