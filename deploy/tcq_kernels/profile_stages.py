"""Stage 0 / attribution: where the time goes, by stage (vision encoder + embedding, LLM prefill, one action-expert step) and by
kernel class (CUPTI trace of CUDA-graph replays). For the PyTorch-reference W2A8 path the per-operation split (decode, Hadamard,
activation quantize, matrix product, scale) is timed eagerly with synchronisation around each operation.

  python profile_stages.py --rows fp w2_dense w2a8_ref --out <json>
"""
import argparse
import json
import time

import numpy as np
import torch

import runner
import tcq_format as F
import cuda_rt


@torch.no_grad()
def profile(row, obs, noise_np, steps=10, reference_resident=True):
    kw = dict(reference_resident=reference_resident) if row == 'w2a8_ref' else {}
    policy, info = runner.load(row, steps, **kw)
    st = runner.Stages(policy, steps, vision=runner.VISION[row])
    observation = st.observation(obs)
    noise = torch.from_numpy(noise_np).cuda()[None]
    out = dict(row=row, info=info, vision=st.vision, stages={})
    with runner.host_memo():
        embs, padding, attention, state = st.embed(observation)
        kv = st.llm(embs, padding, attention)
        if st.hybrid:
            runner.modules.decode_chunk(policy)
        fns = dict(vision=lambda: st.embed(observation), llm_prefill=lambda: st.llm(embs, padding, attention),
                   expert_step=lambda: st.step(0, state, padding, kv, noise))
        if row == 'w2a8_ref':                                  # eager per-operation split
            for name, fn in fns.items():
                fn(); torch.cuda.synchronize()
                F.TIMER, F.STAGE = {}, name
                t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); wall = time.perf_counter() - t0
                ops = {k[1]: round(v * 1e3, 3) for k, v in F.TIMER.items()}
                F.TIMER = None
                t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); plain = time.perf_counter() - t0
                out['stages'].setdefault(name, {})['eager_ops_ms'] = dict(ops, everything_else=round(wall * 1e3 - sum(ops.values()), 3), wall_timed=round(wall * 1e3, 3), wall_untimed=round(plain * 1e3, 3))
        for name, fn in fns.items():
            graph, _ = runner.capture(fn)
            rec = out['stages'].setdefault(name, {})
            rec['graph'] = runner.timed(graph, 20, 100)
            rec['kernels'] = runner.kernel_classes(graph)
            del graph
    out['memory'] = dict(allocated=torch.cuda.memory_allocated(), peak=torch.cuda.max_memory_allocated(), reserved=torch.cuda.memory_reserved())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', nargs='+', default=['fp', 'w2_dense', 'w2a8_ref'])
    p.add_argument('--out', required=True)
    p.add_argument('--fly-reference', action='store_true', help='reference decodes on every call (PyTorch ops) instead of holding INT8 weights')
    a = p.parse_args()
    obs = runner.heldout40()[0][1]; nz = runner.fixed_noise(0)
    results = dict(device=cuda_rt.versions(), rows={})
    for row in a.rows:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        r = profile(row, obs, nz, reference_resident=not a.fly_reference)
        results['rows'][row] = r
        for name, s in r['stages'].items():
            k = s['kernels']
            print(f"{row:14s} {name:12s} graph median {s['graph']['median_ms']:8.3f} ms | kernels {k['kernel_ms']:8.3f} gaps {k['gaps_ms']:6.3f} launches {k['launches']:5d} | {k['by_class_ms']}", flush=True)
            if 'eager_ops_ms' in s:
                print(f"{'':14s} {'':12s} eager per-op ms: {s['eager_ops_ms']}", flush=True)
        json.dump(results, open(a.out, 'w'), indent=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
