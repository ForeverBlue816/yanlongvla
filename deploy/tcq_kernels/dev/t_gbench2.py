"""Graph-based kernel micro-benchmark (per-launch time inside a CUDA graph, 64 launches per replay) for tuning (threads, radix)."""
import sys, time, itertools, torch
import numpy as np
import kernels, tcq_format as F
names = F.layer_names()
def layer(sub):
    n = next(x for x in names if sub in x); return F.load(names=[n])[n]
def gbench(fn, reps=64):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); v = []
    for _ in range(20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b))
    return float(np.median(v)) / reps * 1e3
P = 'paligemma_with_expert.paligemma.model.language_model.layers.3.'
V = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.3.'
E = 'paligemma_with_expert.gemma_expert.model.layers.3.'
kernels.base()
def case(cls, M, width, kind, L=None, L2=None):
    if kind == 'pre':
        x = torch.randn(M, width, device='cuda').bfloat16(); su = torch.ones(width, dtype=torch.int8, device='cuda')
        q = torch.empty(M, width, dtype=torch.int8, device='cuda'); sc = torch.empty(M, device='cuda')
        return lambda: kernels.pre(x, su, q=q, scale=sc)
    if kind == 'post':
        acc = torch.randint(-2**20, 2**20, (M, width), dtype=torch.int32, device='cuda'); s = torch.rand(M, device='cuda') + .5
        rs = torch.rand(width, device='cuda'); sv = torch.ones(width, dtype=torch.int8, device='cuda'); y = torch.empty(M, width, dtype=torch.bfloat16, device='cuda')
        return lambda: kernels.post(acc, s, rs, sv, None, y)
    if kind == 'mid2':
        acc = torch.randint(-2**20, 2**20, (M, 2 * width), dtype=torch.int32, device='cuda'); s = torch.rand(M, device='cuda') + .5
        rs = torch.rand(2 * width, device='cuda') * 1e-6; sv = torch.ones(2 * width, dtype=torch.int8, device='cuda'); su = torch.ones(width, dtype=torch.int8, device='cuda')
        q = torch.empty(M, width, dtype=torch.int8, device='cuda'); sc = torch.empty(M, device='cuda')
        return lambda: kernels.mid2(acc, s, rs, sv, su, q=q, scale=sc)
    if kind == 'mid1':
        acc = torch.randint(-2**20, 2**20, (M, width), dtype=torch.int32, device='cuda'); s = torch.rand(M, device='cuda') + .5
        rs = torch.rand(width, device='cuda') * 1e-6; sv = torch.ones(width, dtype=torch.int8, device='cuda'); su = torch.ones(width, dtype=torch.int8, device='cuda')
        q = torch.empty(M, width, dtype=torch.int8, device='cuda'); sc = torch.empty(M, device='cuda')
        return lambda: kernels.mid1(acc, s, rs, sv, None, su, q=q, scale=sc)
CASES = [('large', 768, 1152, 'pre'), ('large', 768, 1152, 'post'), ('large', 768, 4304, 'mid1'), ('large', 968, 256, 'post'), ('large', 968, 2048, 'post')]
for cls, M, width, kind in CASES:
    fn = case(cls, M, width, kind); res = []
    for T in (64, 128, 256, 512, 1024):
        for rad in (2, 4):
            kernels.PLAN[(cls, width)] = (T, rad)
            res.append((gbench(fn), T, rad))
    kernels.PLAN.pop((cls, width))
    res.sort(); d = kernels.plan(M, width)
    cur = next(r[0] for r in res if (r[1], r[2]) == d)
    print(f'{cls:6s} M={M:4d} {kind:5s} w={width:6d}: best ' + '  '.join(f'{r[0]:.1f}us(T{r[1]},r{r[2]})' for r in res[:4]) + f' | default T{d[0]},r{d[1]}: {cur:.1f}us', flush=True)
