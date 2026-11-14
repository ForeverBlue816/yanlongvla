import sys, numpy as np, torch, json
import runner, modules, kernels
obs = runner.heldout40()[:3]; nz = runner.fixed_noise(0); acts = {}; res = {}
for split in (False, True):
    modules.FUSE['split'] = split
    policy, info = runner.load('w2a8_resident')
    acts[split] = np.stack([np.asarray(policy.infer(o, noise=nz)['actions'], dtype=np.float64) for i, o in obs])
    st = runner.Stages(policy, 10, vision='per_image'); observation = st.observation(obs[0][1]); noise = torch.from_numpy(nz).cuda()[None]
    with runner.host_memo():
        state, padding, kv = st.prefill(observation)
        g, _ = runner.capture(lambda: st.step(0, state, padding, kv, noise)); r = runner.timed(g, 20, 200); kc = runner.kernel_classes(g)
    print('split', split, 'expert step median %.3f ms p95 %.3f' % (r['median_ms'], r['p95_ms']), kc['launches'], kc['by_class_ms'], flush=True)
    del policy, st, g; torch.cuda.empty_cache()
print('split == unsplit actions identical:', np.array_equal(acts[False], acts[True]))
