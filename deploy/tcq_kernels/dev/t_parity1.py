"""First end-to-end check: reference W2A8 (PyTorch) vs the kernel runtimes on a few held-out observations, eager (no graph)."""
import sys, json, time
sys.path.insert(0, 'dev')
import numpy as np, torch
from common import *
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
obs = heldout40()[:n]; nz = noise(0)
out = {}
for rt in ('dense', 'reference', 'resident', 'fly', 'hybrid'):
    torch.cuda.empty_cache(); t0 = time.time()
    policy = load_policy(); info = modules.install(policy, runtime=rt)
    acts = []
    for i, o in obs:
        if rt == 'hybrid': modules.decode_chunk(policy)
        acts.append(np.asarray(policy.infer(o, noise=nz)['actions']))
    out[rt] = np.stack(acts); print(rt, info, 'seconds', round(time.time() - t0, 1), dict(modules.STATS), 'mem GB', round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)
    del policy
ref = out['reference']
for rt in out:
    d = out[rt] - ref
    print(f'{rt:10s} vs reference: max|dA| {np.abs(d).max():.3e}  rel err {np.linalg.norm(d) / np.linalg.norm(ref):.3e}  identical {np.array_equal(out[rt], ref)}')
