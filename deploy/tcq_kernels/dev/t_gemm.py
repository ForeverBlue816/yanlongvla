import sys, torch, numpy as np
import kernels, tcq_format as F
exec(open('dev/t_gbench.py').read().split("P = 'paligemma")[0].split("names = F.layer_names()")[1])
M = 10
for (N, K) in ((8192, 1024), (1024, 4096), (2560, 1024), (1024, 2048)):
    W = torch.randint(-127, 128, (N, K), dtype=torch.int8, device='cuda'); q4 = torch.randint(-2**31, 2**31 - 1, (K // 4, M), dtype=torch.int32, device='cuda')
    s = torch.rand(M, device='cuda'); rs = torch.rand(N, device='cuda'); z = torch.empty(M, N, device='cuda'); res = []
    Wb = W.to(torch.bfloat16); xb = torch.randn(M, K, device='cuda').bfloat16()
    for direct in (False, True):
        for R in (1, 2, 4, 8, 16, 32):
            kernels.GEMM_PLAN = dict(rows=R, direct=direct); res.append((gbench(lambda: kernels.gemm_w8(W, q4, s, rs, z)), R, 'direct' if direct else 'smem'))
    res.sort()
    print(f'N={N} K={K}: ' + '  '.join(f'{r[0]:.1f}us(R{r[1]},{r[2]})' for r in res[:5]) + f' | worst {res[-1][0]:.1f} | BF16 torch linear {gbench(lambda: torch.nn.functional.linear(xb, Wb)):.1f}us | {N*K/1e6:.1f} MB', flush=True)
