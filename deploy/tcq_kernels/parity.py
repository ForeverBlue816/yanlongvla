"""Stage 1: correctness of the kernels against the reference W2A8 model (PyTorch ops, tcq_format.reference_linear).

  layers   every quantized layer, on the inputs it receives inside the reference model (prefill and denoising step 0 of one held-out
           observation): INT8 activations, scales, decoded INT8 weights, INT32 accumulators and BF16 outputs of the kernel paths
           (resident, on the fly) compared bit for bit with the reference. Mismatching layers are listed.
  actions  final actions on the 40 held-out observations (noise seed 0): every runtime against the reference W2A8 model
           (relative error = ||A - A_ref|| / ||A_ref|| per observation), plus the evaluated W2 model, FP and the FP16-table W2A8
           model for scale.
"""
import argparse
import json

import numpy as np
import torch

import kernels
import modules
import runner
import tcq_format as F


@torch.no_grad()
def stage_layers(out):
    policy, info = runner.load('w2a8_ref')
    model = policy._model
    report, calls = [], {}

    def hook(mod, inp, y_ref):
        L = mod.layer[0]; x = inp[0]
        n = calls.get(L.name, 0); calls[L.name] = n + 1
        if n > 0:
            return
        x2 = x.reshape(-1, L.n).contiguous()
        M = x2.shape[0]
        h = F.fwht(x2.float() * L.su.float()); q_ref, s_ref = F.quantize_tokens(h)
        acc_ref = F.int_matmul(q_ref, mod.w8)
        y_ref = y_ref.reshape(M, L.m)
        rec = dict(name=L.name, shape=[L.m, L.n], tokens=M, input_dtype=str(x.dtype).replace('torch.', ''))
        w8 = torch.empty(L.m, L.n, dtype=torch.int8, device='cuda'); kernels.decode(L.bits, L.t8, w8)
        rec['decode'] = bool(torch.equal(w8, mod.w8))
        s = torch.empty(M, device='cuda')
        small = modules.part_of(L.name) == 'expert'
        if not small:
            q = torch.empty(M, L.n, dtype=torch.int8, device='cuda'); kernels.pre(x2, L.su, q=q, scale=s)
            rec['activations'] = bool(torch.equal(q, q_ref)); rec['scales'] = bool(torch.equal(s, s_ref[:, 0]))
            acc = torch._int_mm(q, w8.t())
            rec['accumulator'] = bool(torch.equal(acc, acc_ref))
            y = torch.empty(M, L.m, dtype=torch.bfloat16, device='cuda'); kernels.post(acc, s, L.rs, L.sv, L.bias, y)
            rec['output'] = bool(torch.equal(y, y_ref))
        else:
            q4 = torch.empty(L.n // 4, M, dtype=torch.int32, device='cuda'); kernels.pre(x2, L.su, q4=q4, scale=s)
            q4_ref = q_ref.contiguous().view(M, L.n // 4, 4).view(torch.int32).reshape(M, L.n // 4).t()
            rec['activations'] = bool(torch.equal(q4, q4_ref)); rec['scales'] = bool(torch.equal(s, s_ref[:, 0]))
            z_ref = (acc_ref.float() * s_ref) * L.rs[None, :]
            z1 = torch.empty(M, L.m, device='cuda'); kernels.gemm_w8(w8, q4, s, L.rs, z1)
            z2 = torch.empty(M, L.m, device='cuda'); kernels.gemm_tcq(L.bits, L.t8, q4, s, L.rs, z2, L.n)
            rec['accumulator'] = bool(torch.equal(z1, z_ref)); rec['accumulator_fly'] = bool(torch.equal(z2, z_ref))
            y = torch.empty(M, L.m, dtype=torch.bfloat16, device='cuda'); kernels.post(z2, None, None, L.sv, L.bias, y)
            rec['output'] = bool(torch.equal(y, y_ref))
        report.append(rec)

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, modules.RefLinear)]
    obs = runner.heldout40()[0][1]
    policy.infer(obs, noise=runner.fixed_noise(0))
    for h in handles:
        h.remove()
    keys = ('decode', 'activations', 'scales', 'accumulator', 'accumulator_fly', 'output')
    bad = [r for r in report if not all(r.get(k, True) for k in keys)]
    summary = dict(layers=len(report), exact={k: sum(1 for r in report if r.get(k, True)) for k in keys}, mismatching=[r['name'] for r in bad],
                   tokens={p: sorted({r['tokens'] for r in report if modules.part_of(r['name']) == p}) for p in ('vision', 'llm', 'expert')},
                   input_dtypes=sorted({r['input_dtype'] for r in report}))
    json.dump(dict(summary=summary, layers=report), open(out, 'w'), indent=1)
    print(json.dumps(summary), flush=True)


@torch.no_grad()
def stage_actions(out, rows):
    obs = runner.heldout40(); nz = runner.fixed_noise(0)
    acts, infos = {}, {}
    for row in rows:
        kw = {}
        name = row
        if row == 'w2a8_ref_fp16table':
            name, kw = 'w2a8_ref', dict(table='fp16')
        policy, info = runner.load(name, **kw)
        a = []
        for i, o in obs:
            modules.decode_chunk(policy)                          # hybrid rows: expert weights decoded once per chunk (no-op otherwise)
            a.append(np.asarray(policy.infer(o, noise=nz)['actions'], dtype=np.float64))
        acts[row] = np.stack(a); infos[row] = info
        print(row, acts[row].shape, dict(modules.STATS), flush=True)
        del policy; torch.cuda.empty_cache()
    ref = acts['w2a8_ref']
    table = {}
    for row, a in acts.items():
        d = a - ref
        per = np.linalg.norm(d.reshape(len(d), -1), axis=1) / np.linalg.norm(ref.reshape(len(ref), -1), axis=1)
        table[row] = dict(identical_observations=int(sum(np.array_equal(x, y) for x, y in zip(a, ref))), max_abs=float(np.abs(d).max()),
                          rel_err_max=float(per.max()), rel_err_mean=float(per.mean()), rel_err_pooled=float(np.linalg.norm(d) / np.linalg.norm(ref)))
        print(f"{row:22s} identical {table[row]['identical_observations']:2d}/40  max|dA| {table[row]['max_abs']:.3e}  rel err max {table[row]['rel_err_max']:.3e}  mean {table[row]['rel_err_mean']:.3e}", flush=True)
    np.savez(out.replace('.json', '.npz'), **acts)
    json.dump(dict(reference='w2a8_ref', observations=[i for i, _ in obs], noise_seed=0, rows=table, info=infos), open(out, 'w'), indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['layers', 'actions'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rows', nargs='+', default=['w2a8_ref', 'w2a8_resident', 'w2a8_fly', 'w2a8_hybrid', 'w2_dense', 'w2a8_ref_fp16table', 'fp'])
    a = p.parse_args()
    stage_layers(a.out) if a.stage == 'layers' else stage_actions(a.out, a.rows)
