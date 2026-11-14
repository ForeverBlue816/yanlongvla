"""Lockstep vision: exactness against the per-image call structure (reference and kernel rows) and stage timing."""
import sys, numpy as np, torch
import runner, modules
obs = runner.heldout40(); nz = runner.fixed_noise(0); noise = torch.from_numpy(nz).cuda()[None]
ref = None
for row in ('w2a8_ref', 'w2a8_resident', 'w2a8_fly', 'fp'):
    policy, info = runner.load(row)
    st = runner.Stages(policy, 10, vision='per_image'); out = {}
    for mode in ('per_image', 'lockstep', 'batched'):
        st.vision = mode; a = []
        for i, o in obs[:(4 if row in ('w2a8_ref', 'fp') else 40)]:
            a.append(st.chunk(st.observation(o), noise).float().cpu().numpy())
        out[mode] = np.stack(a)
    if row == 'w2a8_ref': ref = out['per_image']
    msg = f"{row}: lockstep == per_image {np.array_equal(out['lockstep'], out['per_image'])} (n={len(out['lockstep'])}) | batched vs per_image rel {np.linalg.norm(out['batched'] - out['per_image']) / np.linalg.norm(out['per_image']):.2e}"
    if row.startswith('w2a8'): msg += f" | per_image == reference per_image (first 4): {np.array_equal(out['per_image'][:4], ref)}"
    print(msg, flush=True)
    if row != 'w2a8_ref':
        observation = st.observation(obs[0][1])
        with runner.host_memo():
            for mode in ('per_image', 'lockstep', 'batched'):
                st.vision = mode
                g, _ = runner.capture(lambda: st.embed(observation)); r = runner.timed(g, 20, 100); kc = runner.kernel_classes(g)
                print(f"   {row} vision {mode:10s} median {r['median_ms']:.3f} ms launches {kc['launches']} {kc['by_class_ms']}", flush=True)
                del g
    del policy, st; torch.cuda.empty_cache()
