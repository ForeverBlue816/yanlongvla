"""Kernel micro-benchmark at the real shapes + bit-exactness of the fused kernels against the unfused chain."""
import sys, time, torch
import kernels, tcq_format as F
names = F.layer_names()
def layer(sub):
    n = next(x for x in names if sub in x); return F.load(names=[n])[n]
def bench(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
P = 'paligemma_with_expert.paligemma.model.language_model.layers.3.'
V = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.3.'
E = 'paligemma_with_expert.gemma_expert.model.layers.3.'
for tag, M, pre in (('llm', 968, P), ('expert', 10, E)):
    gate, up, down, q, k = [layer(pre + x) for x in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'self_attn.q_proj', 'self_attn.k_proj')]
    x = torch.randn(M, gate.n, device='cuda').bfloat16()
    qb = torch.empty(M, gate.n, dtype=torch.int8, device='cuda'); s = torch.empty(M, device='cuda')
    print(tag, 'pre K=%d: %.1f us' % (gate.n, bench(lambda: kernels.pre(x, gate.su, q=qb, scale=s))))
    Wg = torch.empty(gate.m, gate.n, dtype=torch.int8, device='cuda'); kernels.decode(gate.bits, gate.t8, Wg)
    Wu = torch.empty(up.m, up.n, dtype=torch.int8, device='cuda'); kernels.decode(up.bits, up.t8, Wu)
    print(tag, 'decode %dx%d: %.1f us' % (gate.m, gate.n, bench(lambda: kernels.decode(gate.bits, gate.t8, Wg))))
    W = torch.cat([Wg, Wu]).contiguous(); rs = torch.cat([gate.rs, up.rs]).contiguous(); sv = torch.cat([gate.sv, up.sv]).contiguous()
    acc = F.int_matmul(qb, W)
    if M > 32: print(tag, 'int8 gemm %dx%dx%d: %.1f us' % (M, 2 * gate.m, gate.n, bench(lambda: torch._int_mm(qb, W.t()))))
    yg = torch.empty(M, gate.m, dtype=torch.bfloat16, device='cuda'); yu = torch.empty_like(yg)
    print(tag, 'post N=%d: %.1f us' % (gate.m, bench(lambda: kernels.post(acc, s, gate.rs, gate.sv, None, yg, col0=0))))
    kernels.post(acc, s, up.rs, up.sv, None, yu, col0=gate.m)
    prod = torch.nn.functional.gelu(yg, approximate='tanh') * yu
    q2 = torch.empty(M, down.n, dtype=torch.int8, device='cuda'); s2 = torch.empty(M, device='cuda')
    kernels.pre(prod, down.su, q=q2, scale=s2)
    print(tag, 'pre K=%d: %.1f us' % (down.n, bench(lambda: kernels.pre(prod, down.su, q=q2, scale=s2))))
    print(tag, 'torch gelu*mul: %.1f us' % bench(lambda: torch.nn.functional.gelu(yg, approximate='tanh') * yu))
    if kernels.mid2_available(gate.m):
        q3 = torch.empty_like(q2); s3 = torch.empty_like(s2)
        kernels.mid2(acc, s, rs, sv, down.su, q=q3, scale=s3)
        print(tag, 'mid2 N=%d: %.1f us | fused == chain: q %s scale %s' % (gate.m, bench(lambda: kernels.mid2(acc, s, rs, sv, down.su, q=q3, scale=s3)), torch.equal(q3, q2), torch.equal(s3, s2)))
    else: print(tag, 'mid2 not available at N =', gate.m)
    # reference check of post / pre against PyTorch
    yr = F.reference_linear(x, gate, w8=Wg); print(tag, 'post == reference:', torch.equal(yg, yr))
    # q/k/v post3
    v = layer(pre + 'self_attn.v_proj'); xa = torch.randn(M, q.n, device='cuda').bfloat16()
    qa = torch.empty(M, q.n, dtype=torch.int8, device='cuda'); sa = torch.empty(M, device='cuda'); kernels.pre(xa, q.su, q=qa, scale=sa)
    Ws = []
    for L in (q, k, v):
        w = torch.empty(L.m, L.n, dtype=torch.int8, device='cuda'); kernels.decode(L.bits, L.t8, w); Ws.append(w)
    acc3 = F.int_matmul(qa, torch.cat(Ws).contiguous())
    ys = [torch.empty(M, L.m, dtype=torch.bfloat16, device='cuda') for L in (q, k, v)]
    rs3 = torch.cat([L.rs for L in (q, k, v)]).contiguous(); sv3 = torch.cat([L.sv for L in (q, k, v)]).contiguous()
    t3 = bench(lambda: kernels.post3(acc3, sa, rs3, sv3, None, ys))
    ok = all(torch.equal(y, F.reference_linear(xa, L, w8=w)) for y, L, w in zip(ys, (q, k, v), Ws))
    print(tag, 'post3 (%s): %.1f us | == reference: %s' % ('+'.join(str(L.m) for L in (q, k, v)), t3, ok))
    print(tag, 'pre K=%d: %.1f us' % (q.n, bench(lambda: kernels.pre(xa, q.su, q=qa, scale=sa))))
fc1, fc2 = layer(V + 'mlp.fc1'), layer(V + 'mlp.fc2'); M = 256
x = torch.randn(M, fc1.n, device='cuda').bfloat16(); qb = torch.empty(M, fc1.n, dtype=torch.int8, device='cuda'); s = torch.empty(M, device='cuda')
print('vision pre K=%d: %.1f us' % (fc1.n, bench(lambda: kernels.pre(x, fc1.su, q=qb, scale=s))))
W1 = torch.empty(fc1.m, fc1.n, dtype=torch.int8, device='cuda'); kernels.decode(fc1.bits, fc1.t8, W1)
acc = F.int_matmul(qb, W1); y1 = torch.empty(M, fc1.m, dtype=torch.bfloat16, device='cuda')
print('vision int8 gemm: %.1f us' % bench(lambda: torch._int_mm(qb, W1.t())))
print('vision post N=%d: %.1f us | == reference: %s' % (fc1.m, bench(lambda: kernels.post(acc, s, fc1.rs, fc1.sv, fc1.bias, y1)), torch.equal(y1, F.reference_linear(x, fc1, w8=W1))))
g = torch.nn.functional.gelu(y1, approximate='tanh'); q2 = torch.empty(M, fc2.n, dtype=torch.int8, device='cuda'); s2 = torch.empty(M, device='cuda'); kernels.pre(g, fc2.su, q=q2, scale=s2)
q3 = torch.empty_like(q2); s3 = torch.empty_like(s2); kernels.mid1(acc, s, fc1.rs, fc1.sv, fc1.bias, fc2.su, q=q3, scale=s3)
print('vision mid1 N=%d: %.1f us | fused == chain: q %s scale %s' % (fc1.m, bench(lambda: kernels.mid1(acc, s, fc1.rs, fc1.sv, fc1.bias, fc2.su, q=q3, scale=s3)), torch.equal(q3, q2), torch.equal(s3, s2)))
print('vision pre K=%d: %.1f us' % (fc2.n, bench(lambda: kernels.pre(g, fc2.su, q=q2, scale=s2))))
