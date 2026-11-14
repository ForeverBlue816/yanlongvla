"""W2 (FP activations) kernel rows against the evaluated dense W2 model: actions on a few held-out observations."""
import sys, numpy as np, torch
import runner, modules
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
obs = runner.heldout40()[:n]; nz = runner.fixed_noise(0); acts = {}
for row in ('w2_dense', 'w2_fly', 'w2_hybrid', 'w2a8_resident'):
    policy, info = runner.load(row); a = []
    for i, o in obs:
        modules.decode_chunk(policy)
        a.append(np.asarray(policy.infer(o, noise=nz)['actions'], dtype=np.float64))
    acts[row] = np.stack(a); print(row, info.get('modes'), info.get('fused_mlps'), dict(modules.STATS), round(torch.cuda.memory_allocated() / 2**30, 2), 'GB', flush=True)
    del policy; torch.cuda.empty_cache()
ref = acts['w2_dense']
for row, a in acts.items():
    d = a - ref; per = np.linalg.norm(d.reshape(len(d), -1), axis=1) / np.linalg.norm(ref.reshape(len(ref), -1), axis=1)
    print(f'{row:14s} vs w2_dense: max|dA| {np.abs(d).max():.3e} rel err max {per.max():.3e} median {np.median(per):.3e}', flush=True)
