import sys, torch, numpy as np
import kernels, tcq_format as F
exec(open('dev/t_gbench.py').read().split("P = 'paligemma")[0].split("names = F.layer_names()")[1])
kernels.base()
for M, N in ((968, 16384), (10, 4096), (256, 4304)):
    acc = torch.randint(-2**20, 2**20, (M, 2 * N), dtype=torch.int32, device='cuda'); s = torch.rand(M, device='cuda') + .5
    rs = torch.rand(2 * N, device='cuda') * 1e-6; sv = torch.ones(2 * N, dtype=torch.int8, device='cuda'); su = torch.ones(N, dtype=torch.int8, device='cuda')
    q = torch.empty(M, N, dtype=torch.int8, device='cuda'); sc = torch.empty(M, device='cuda'); q2 = torch.empty_like(q); sc2 = torch.empty_like(sc)
    g = torch.empty(M, N, dtype=torch.bfloat16, device='cuda'); u = torch.empty_like(g)
    if kernels.mid2_available(N):
        kernels.mid2(acc, s, rs, sv, su, q=q, scale=sc)
        a = gbench(lambda: kernels.mid2(acc, s, rs, sv, su, q=q, scale=sc))
    else: a = float('nan')
    kernels.postg(acc, s, rs, sv, None, [g, u], gelu0=True); kernels.pre2(g, u, su, q=q2, scale=sc2)
    b1 = gbench(lambda: kernels.postg(acc, s, rs, sv, None, [g, u], gelu0=True)); b2 = gbench(lambda: kernels.pre2(g, u, su, q=q2, scale=sc2))
    print(f'M={M} N={N}: mid2 {a:.1f} us | postg {b1:.1f} + pre2 {b2:.1f} = {b1 + b2:.1f} us | identical q {torch.equal(q, q2)} scale {torch.equal(sc, sc2)}', flush=True)
print(torch.cuda.get_device_name())
