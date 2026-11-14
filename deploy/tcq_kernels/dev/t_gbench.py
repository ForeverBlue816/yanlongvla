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
CASES = [('large', 968, 2048, 'pre'), ('large', 968, 2048, 'post'), ('large', 968, 256, 'post'), ('large', 968, 16384, 'mid2'),
         ('mid', 256, 1152, 'pre'), ('mid', 256, 1152, 'post'), ('mid', 256, 4304, 'mid1'), ('mid', 256, 4304, 'pre'),
         ('small', 10, 1024, 'pre'), ('small', 10, 2048, 'pre'), ('small', 10, 2048, 'post'), ('small', 10, 1024, 'post'), ('small', 10, 256, 'post'), ('small', 10, 4096, 'mid2')]
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
# other kernels in a graph
for tag, M, pre in (('expert', 10, E),):
    g, u = layer(pre + 'mlp.gate_proj'), layer(pre + 'mlp.up_proj')
    W = torch.empty(2 * g.m, g.n, dtype=torch.int8, device='cuda'); kernels.decode(g.bits, g.t8, W[:g.m]); kernels.decode(u.bits, u.t8, W[g.m:])
    q4 = torch.randint(-2**31, 2**31 - 1, (g.n // 4, M), dtype=torch.int32, device='cuda'); s = torch.rand(M, device='cuda'); rs = torch.cat([g.rs, u.rs]).contiguous()
    z = torch.empty(M, 2 * g.m, device='cuda'); bits = torch.cat([g.bits, u.bits]).contiguous(); t8 = torch.cat([g.t8.reshape(-1), u.t8.reshape(-1)]).contiguous()
    print('expert gemm_w8 8192x1024: %.1f us | gemm_tcq: %.1f us | decode 4096x1024: %.1f us' % (gbench(lambda: kernels.gemm_w8(W, q4, s, rs, z)), gbench(lambda: kernels.gemm_tcq(bits, t8, q4, s, rs, z, g.n, (g.m // 16, 1 << 30))), gbench(lambda: kernels.decode(g.bits, g.t8, W[:g.m]))))
