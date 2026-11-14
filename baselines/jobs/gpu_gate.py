"""GPU gate: at most MAX_RUN of OUR jobs (names bl_* = pi0.5 scheduler, gr_* = GR00T gsched) run at a time.
Both feeders submit with `qsub -h`; every minute this loop releases (qrls) the oldest held job of ours while fewer than MAX_RUN of
ours run or are already released and queued. Jobs with other names (another project of this account) are never touched.
  nohup python3 jobs/gpu_gate.py >> /scratch/d250009/pi05-handover/runs/gpu_gate.log 2>&1 &"""
import os, re, subprocess, time
from datetime import datetime

MAX_RUN = int(os.environ.get('GATE_MAX_RUN', 3))   # GPUs this study may occupy at once
OURS = re.compile(r'^(bl_|gr_)')


def jobs():
    out = subprocess.run(['qstat', '-u', os.environ.get('USER', 'd250009'), '-w'], capture_output=True, text=True).stdout.splitlines()
    rows = []
    for l in out:
        f = l.split()
        if len(f) >= 10 and f[0][0].isdigit():
            rows.append((f[0].split('.')[0], f[3], f[9]))   # id, name, state
    return rows


while True:
    js = [j for j in jobs() if OURS.match(j[1])]
    active = [j for j in js if j[2] in ('R', 'Q', 'E', 'B')]   # running, or released and waiting for a node
    held = sorted((j for j in js if j[2] == 'H'), key=lambda j: (not j[1].startswith('bl_'), int(j[0])))   # pi0.5 (bl_*) first, then oldest
    held = [j for j in held if 'u' in subprocess.run(['qstat', '-f', j[0]], capture_output=True, text=True).stdout.split('Hold_Types = ')[-1][:3]]   # only user holds; a job held by its dependency (Hold_Types s) would block a slot
    free = MAX_RUN - len(active)
    for jid, name, _ in held[:max(free, 0)]:
        r = subprocess.run(['qrls', jid], capture_output=True, text=True)
        print(f'{datetime.now().isoformat(timespec="seconds")} released {jid} {name} ({len(active)} active of {MAX_RUN}) {r.stderr.strip()}', flush=True)
    time.sleep(60)
