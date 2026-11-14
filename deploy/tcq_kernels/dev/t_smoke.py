import torch, json, time
import kernels, cuda_rt, tcq_format as F
print(cuda_rt.versions())
d = F.fit_dir()
for name in ['paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj', 'paligemma_with_expert.paligemma.model.language_model.layers.3.self_attn.k_proj']:
    rec = torch.load(d / f'{name}.pt', map_location='cpu'); t = rec['tensors']
    L = F.Layer(name, dict(bits=t['trellis'], tlut=t['tlut'], row_scale=t['row_scale'], su=t['signs'], sv=t['signs_out']))
    m, n = L.m, L.n
    st = L.states(); x = (st + 1) * st; k = ((x >> 6) & 511) * 2; sg = 1 - ((x >> 15) & 1) * 2
    index = torch.stack((sg * (k + 1), k + 2), -1).reshape(m, n).to(torch.int16)
    print(name.split('.')[-3:], (m, n), 'states == stored index:', torch.equal(index.cpu(), rec['index']))
    w = L.dense('fp16'); ref = rec['w_hat'].cuda()
    print(' dense bf16 equal:', torch.equal(w.bfloat16(), ref), 'rel', float((w - ref.float()).norm() / ref.float().norm()), 'int8 table rel mse', float(((L.dense('int8') - w).norm() / w.norm()) ** 2))
    w8 = L.w8_reference()
    out = torch.zeros(m, n, dtype=torch.int8, device='cuda'); kernels.decode(L.bits, L.t8, out)
    print(' k_decode == reference:', torch.equal(out, w8))
    for M in (10, 300):
        for dt in (torch.bfloat16, torch.float32):
            xx = (torch.randn(M, n, device='cuda') * 3).to(dt)
            q = torch.zeros(M, n, dtype=torch.int8, device='cuda'); s = torch.zeros(M, device='cuda'); q4 = torch.zeros(n // 4, M, dtype=torch.int32, device='cuda')
            kernels.pre(xx, L.su, q=q, q4=q4, scale=s)
            h = F.fwht(xx.float() * L.su.float()); qr, sr = F.quantize_tokens(h)
            q4r = qr.contiguous().view(M, n // 4, 4).view(torch.int32).reshape(M, n // 4).t()
            acc = F.int_matmul(q, w8)
            y = torch.zeros(M, m, dtype=torch.bfloat16, device='cuda'); kernels.post(acc, s, L.rs, L.sv, None, y)
            yr = F.reference_linear(xx, L, w8=w8)
            print(f'  M={M} {str(dt)[6:]}: pre q {torch.equal(q, qr)} scale {torch.equal(s, sr[:, 0])} q4 {torch.equal(q4, q4r)} | post == reference {torch.equal(y, yr)}', end='')
            if M <= 16:
                z = torch.zeros(M, m, device='cuda'); kernels.gemm_w8(w8, q4, s, L.rs, z)
                zr = (acc.float() * s[:, None]) * L.rs[None]
                z2 = torch.zeros(M, m, device='cuda'); kernels.gemm_tcq(L.bits, L.t8, q4, s, L.rs, z2, n)
                y2 = torch.zeros(M, m, dtype=torch.bfloat16, device='cuda'); kernels.post(z2, None, None, L.sv, None, y2)
                print(f' | gemm_w8 {torch.equal(z, zr)} gemm_tcq {torch.equal(z2, zr)} post_f32 {torch.equal(y2, yr)}', end='')
            print()
torch.cuda.synchronize(); print('ok')
