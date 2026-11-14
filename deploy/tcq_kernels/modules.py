"""Model integration: replaces the 414 quantized nn.Linear modules of the pi0.5 policy by TCQ W2A8 modules.

Runtimes (install(policy, row, runtime)):
  dense        the evaluated W2 model: BF16 dense weights in the original basis, FP activations (what LIBERO-Plus ran on)
  reference    W2A8 in plain PyTorch (tcq_format.reference_linear): the definition the kernels are checked against
  resident     kernels, operating point (i):   every layer decoded once at load, weights held as INT8
  fly          kernels, operating point (ii):  every layer decoded from the 2-bit stream on each call
  hybrid       kernels, operating point (iii): backbone on the fly; action expert decoded once per chunk (decode_chunk) into INT8

Layers that read the same tensor with identical input signs share one rotation + quantization and one product (q/k/v; gate/up):
the group runs on the first member's call and the other members return their slice.
"""
import torch

import kernels
import tcq_format as F

ROOT = F.ROOT
RUNTIMES = ('dense', 'reference', 'resident', 'fly', 'hybrid')
GROUPS = (('q_proj', 'k_proj', 'v_proj'), ('gate_proj', 'up_proj'))
STATS = dict(group_runs=0, regroup_misses=0, fused_mlps=0)
FUSE = dict(post3=True, mlp=True, split=True)           # split: for few tokens (action expert) the gated-MLP middle and the q/k/v rotations run as (token, layer) blocks on different SMs (bit-identical, 3 % of a step)            # kernel fusions (bit-identical arithmetic); switch off to measure them


def part_of(name):
    return 'expert' if '.gemma_expert.' in name else 'vision' if 'vision_tower' in name else 'llm'


class Group:
    def __init__(self, layers, part, mode, a8=True):
        self.layers, self.part, self.mode, self.a8 = layers, part, mode, a8        # mode: resident | fly | chunk; a8=False: FP (BF16) activations in the rotated space
        self.K = layers[0].n
        assert all(L.n == self.K and torch.equal(L.su, layers[0].su) for L in layers), [L.name for L in layers]
        self.su = layers[0].su
        self.N = [L.m for L in layers]
        self.col0 = [sum(self.N[:i]) for i in range(len(layers))]
        self.Ntot = sum(self.N)
        self.small = part == 'expert'
        self.W = None
        self.rs = torch.cat([L.rs for L in layers]).contiguous()
        self.sv = torch.cat([L.sv for L in layers]).contiguous()
        self.bias = torch.cat([L.bias for L in layers]).contiguous() if layers[0].bias is not None else None
        if self.small:
            self.t8 = torch.cat([L.t8.reshape(-1) for L in layers]).contiguous()
            self.bits = torch.cat([L.bits for L in layers]).contiguous() if mode == 'fly' else None
            self.bounds = [c // 16 for c in self.col0[1:]] + [1 << 30] * (3 - len(layers))
        if mode in ('resident', 'chunk'):
            self.W = torch.empty(self.Ntot, self.K, dtype=torch.int8, device='cuda')
            self.decode()                    # chunk mode: valid from load on; decode_chunk() re-materialises it per chunk (the timed cost)
        if not a8 and not self.small:
            self.tb = [L.t8.to(torch.bfloat16).contiguous() for L in layers]            # BF16 bit patterns of the INT8 table values
        self.x = self.outs = None
        self.served = 0

    def decode(self, W=None):
        W = self.W if W is None else W
        for L, c in zip(self.layers, self.col0):
            kernels.decode(L.bits, L.t8, W[c:c + L.m])
        return W

    def prep(self, x):
        x2 = x.reshape(-1, self.K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if x2.dtype not in (torch.bfloat16, torch.float32):
            x2 = x2.float()
        return x2

    def buffers(self, M, dev):
        """Quantized-activation buffer in the layout of this group's product, and the per-token scales."""
        s = torch.empty(M, dtype=torch.float32, device=dev)
        if self.small:
            return dict(q4=torch.empty(self.K // 4, M, dtype=torch.int32, device=dev), scale=s)
        return dict(q=torch.empty(M, self.K, dtype=torch.int8, device=dev), scale=s)

    def quantize(self, x2):
        b = self.buffers(x2.shape[0], x2.device)
        kernels.pre(x2, self.su, **b)
        return b

    def product(self, b):
        """INT8 activations -> prefill: INT32 accumulators (M, Ntot) (cuBLASLt) | expert: FP32 z = acc * s * rs (M, Ntot)."""
        if not self.small:
            dev = b['q'].device
            W = self.W if self.W is not None else self.decode(torch.empty(self.Ntot, self.K, dtype=torch.int8, device=dev))
            return torch._int_mm(b['q'], W.t())
        M = b['q4'].shape[1]
        z = torch.empty(M, self.Ntot, dtype=torch.float32, device=b['q4'].device)
        if self.W is not None:
            kernels.gemm_w8(self.W, b['q4'], b['scale'], self.rs, z)
        else:
            kernels.gemm_tcq(self.bits, self.t8, b['q4'], b['scale'], self.rs, z, self.K, self.bounds)
        return z

    def posts(self, acc, b, shape):
        M, dev = acc.shape[0], acc.device
        ys = [torch.empty(M, n, dtype=torch.bfloat16, device=dev) for n in self.N]
        s, rs = (None, None) if self.small else (b['scale'], self.rs)
        if len(ys) == 3 and self.small and FUSE['split']:
            kernels.postg(acc, s, rs, self.sv, self.bias, ys)                 # few tokens: the three layers on different SMs at once
        elif len(ys) == 3 and FUSE['post3']:
            kernels.post3(acc, s, rs, self.sv, self.bias, ys)
        else:
            for L, c, y in zip(self.layers, self.col0, ys):
                kernels.post(acc, s, None if rs is None else L.rs, L.sv, L.bias, y, col0=c)
        return [y.reshape(*shape[:-1], n) for y, n in zip(ys, self.N)]

    def run(self, x):
        x2 = self.prep(x)
        if not self.a8:
            return self.run_float(x2, x.shape)
        b = self.quantize(x2)
        outs = self.posts(self.product(b), b, x.shape)
        STATS['group_runs'] += 1
        return outs

    def run_float(self, x2, shape):
        """W2: rotated activations rounded to BF16, INT8-table weights, BF16 product (prefill: cuBLAS on decoded BF16 weights; expert: k_gemmf_*)."""
        M, dev, outs = x2.shape[0], x2.device, []
        if not self.small:
            h = torch.empty(M, self.K, dtype=torch.bfloat16, device=dev)
            kernels.pre_h(x2, self.su, h=h)
            W = torch.empty(self.Ntot, self.K, dtype=torch.bfloat16, device=dev)
            for L, c, tb in zip(self.layers, self.col0, self.tb):
                kernels.decode_bf16(L.bits, tb, W[c:c + L.m])
            acc = h @ W.t()
            for L, c in zip(self.layers, self.col0):
                y = torch.empty(M, L.m, dtype=torch.bfloat16, device=dev)
                kernels.post(acc, None, L.rs, L.sv, L.bias, y, col0=c)
                outs.append(y.reshape(*shape[:-1], L.m))
        else:
            hT = torch.empty(self.K, M, dtype=torch.bfloat16, device=dev)
            kernels.pre_h(x2, self.su, hT=hT)
            z = torch.empty(M, self.Ntot, dtype=torch.float32, device=dev)
            if self.W is not None:
                kernels.gemmf_w8(self.W, hT, self.rs, z)
            else:
                kernels.gemmf_tcq(self.bits, self.t8, hT, self.rs, z, self.K, self.bounds)
            for L, c in zip(self.layers, self.col0):
                y = torch.empty(M, L.m, dtype=torch.bfloat16, device=dev)
                kernels.post(z, None, None, L.sv, L.bias, y, col0=c)
                outs.append(y.reshape(*shape[:-1], L.m))
        STATS['group_runs'] += 1
        return outs

    def get(self, x, index):
        if self.x is not x:
            if self.x is not None:
                STATS['regroup_misses'] += 1
            self.outs = self.run(x); self.x = x; self.served = 0
        out = self.outs[index]
        self.served += 1
        if self.served == len(self.layers):
            self.x = self.outs = None
        return out


class KLinear(torch.nn.Module):
    def __init__(self, group, index):
        super().__init__()
        self.group, self.index = [group], index                    # list: keep the group out of the module tree
        L = group.layers[index]
        self.in_features, self.out_features, self.layer_name = L.n, L.m, L.name
        self.register_buffer('weight', torch.empty(0, dtype=torch.bfloat16, device='cuda'), persistent=False)   # read by the model for its dtype only

    def forward(self, x):
        return self.group[0].get(x, self.index)


class KGatedMLP(torch.nn.Module):
    """Gemma MLP down(gelu(gate(x)) * up(x)) with the middle fused: 5 launches (rotate+quantize, product, middle, product, scale+rotate)."""
    def __init__(self, gate_up, down):
        super().__init__()
        self.groups = [gate_up, down]
        self.gate_proj, self.up_proj, self.down_proj = KLinear(gate_up, 0), KLinear(gate_up, 1), KLinear(down, 0)

    def forward(self, x):
        gu, down = self.groups
        b = gu.quantize(gu.prep(x))
        acc = gu.product(b)
        b2 = down.buffers(acc.shape[0], acc.device)
        if gu.small and FUSE['split']:                                         # few tokens: gate and up in parallel blocks, then product + rotate + quantize
            g = torch.empty(acc.shape[0], gu.N[0], dtype=torch.bfloat16, device=acc.device); u = torch.empty_like(g)
            kernels.postg(acc, None, None, gu.sv, None, [g, u], gelu0=True)
            kernels.pre2(g, u, down.su, **b2)
        else:
            kernels.mid2(acc, None if gu.small else b['scale'], None if gu.small else gu.rs, gu.sv, down.su, **b2)
        STATS['fused_mlps'] += 1
        return down.posts(down.product(b2), b2, x.shape)[0]


class KPlainMLP(torch.nn.Module):
    """SigLIP MLP fc2(gelu(fc1(x))) with the middle fused."""
    def __init__(self, fc1, fc2):
        super().__init__()
        self.groups = [fc1, fc2]
        self.fc1, self.fc2 = KLinear(fc1, 0), KLinear(fc2, 0)

    def forward(self, x):
        g1, g2 = self.groups
        b = g1.quantize(g1.prep(x))
        acc = g1.product(b)
        b2 = g2.buffers(acc.shape[0], acc.device)
        kernels.mid1(acc, None if g1.small else b['scale'], None if g1.small else g1.rs, g1.sv, g1.bias, g2.su, **b2)
        STATS['fused_mlps'] += 1
        return g2.posts(g2.product(b2), b2, x.shape)[0]


class RefLinear(torch.nn.Module):
    """W2A8 by definition (PyTorch ops only). a8=False: same INT8-table weights, FP32 activations."""
    def __init__(self, layer, a8=True, table='int8', resident=True):
        super().__init__()
        self.layer, self.a8, self.table = [layer], a8, table
        self.w8 = layer.w8_reference() if (table == 'int8' and resident) else None        # resident=False: decoded on every call (PyTorch ops)
        self.in_features, self.out_features, self.layer_name = layer.n, layer.m, layer.name
        self.register_buffer('weight', torch.empty(0, dtype=torch.bfloat16, device='cuda'), persistent=False)

    def forward(self, x):
        return F.reference_linear(x, self.layer[0], a8=self.a8, table=self.table, w8=self.w8)


def set_submodule(model, name, module):
    parent, _, leaf = name.rpartition('.')
    setattr(model.get_submodule(parent), leaf, module)


def apply_row_state(model, row):
    """Everything of the row outside the packed Linear payload: the row-INT8 token embedding of the 2-bit rows and the fine-tuned norm weights."""
    from safetensors import safe_open
    from compact_policy import install_int8_embedding, RowInt8Embedding
    embed = model.paligemma_with_expert.paligemma.language_model.embed_tokens
    if not isinstance(embed, RowInt8Embedding):
        install_int8_embedding(model)
    norms = 0
    with safe_open(str(ROOT / f'models/pi05_tcq/{row}/dense.safetensors'), framework='pt', device='cuda') as f:
        for k in sorted(f.keys()):
            if k.startswith('norm::'):
                _, name, attr = k.split('::')
                p = getattr(model.get_submodule(name), attr)
                v = f.get_tensor(k)
                assert v.shape == p.shape and v.dtype == p.dtype, k
                p.data.copy_(v); norms += 1
    return norms


def install(policy, row=F.ROW, runtime='resident', a8=True, table='int8', reference_resident=True):
    assert runtime in RUNTIMES
    model = policy._model
    names = F.layer_names()
    norms = apply_row_state(model, row)
    info = dict(row=row, runtime=runtime, layers=len(names), norm_tensors=norms)
    if runtime == 'dense':
        from safetensors import safe_open
        with safe_open(str(ROOT / f'models/pi05_tcq/{row}/dense.safetensors'), framework='pt', device='cuda') as f:
            for name in names:
                m = model.get_submodule(name)
                w = f.get_tensor(name)
                assert isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16 and m.weight.shape == w.shape, name
                m.weight.data.copy_(w)
        return info
    biases = {}
    for name in names:
        m = model.get_submodule(name)
        assert isinstance(m, torch.nn.Linear), (name, type(m))
        if m.bias is not None:
            biases[name] = m.bias.detach().float().clone()
    layers = F.load(row, names=names, biases=biases)
    if runtime == 'reference':
        for name in names:
            set_submodule(model, name, RefLinear(layers[name], a8=a8, table=table, resident=reference_resident))
        torch.cuda.empty_cache()
        return info
    groups, done = [], set()
    for name in names:
        if name in done:
            continue
        parent, _, leaf = name.rpartition('.')
        members = [name]
        for g in GROUPS:
            if leaf in g:
                cand = [f'{parent}.{x}' for x in g]
                if all(c in layers for c in cand) and all(layers[c].n == layers[cand[0]].n and torch.equal(layers[c].su, layers[cand[0]].su) for c in cand):
                    members = cand
        part = part_of(name)
        mode = 'resident' if runtime == 'resident' else 'fly' if (runtime == 'fly' or part != 'expert') else 'chunk'
        group = Group([layers[n] for n in members], part, mode, a8=a8)
        for i, n in enumerate(members):
            set_submodule(model, n, KLinear(group, i)); done.add(n)
        if mode != 'fly':
            for L in group.layers:
                L.bits_resident = mode == 'chunk'                 # resident: the bitstream is not needed after load
                if mode == 'resident':
                    L.bits = None
        groups.append(group)
    fused = 0
    if a8 and FUSE['mlp']:
        by_name = {g.layers[0].name: g for g in groups}
        for g in list(groups):
            parent, _, leaf = g.layers[0].name.rpartition('.')
            if leaf == 'gate_proj' and len(g.layers) == 2 and f'{parent}.down_proj' in by_name and kernels.mid2_available(g.layers[0].m):
                assert type(model.get_submodule(parent)).__name__ == 'GemmaMLP', type(model.get_submodule(parent))
                set_submodule(model, parent, KGatedMLP(g, by_name[f'{parent}.down_proj'])); fused += 1
            elif leaf == 'fc1' and f'{parent}.fc2' in by_name:
                assert type(model.get_submodule(parent)).__name__ == 'SiglipMLP', type(model.get_submodule(parent))
                set_submodule(model, parent, KPlainMLP(g, by_name[f'{parent}.fc2'])); fused += 1
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    policy._tcq_groups = groups
    info['fused_mlps'] = fused
    info.update(groups=len(groups), shared_rotations=sum(len(g.layers) - 1 for g in groups),
                modes={m: sum(1 for g in groups if g.mode == m) for m in ('resident', 'fly', 'chunk')})
    return info


def decode_chunk(policy):
    """Operating point (iii): decode the action expert's layers once per action chunk (called before the first denoising step)."""
    n = 0
    for g in getattr(policy, '_tcq_groups', ()):
        if g.mode == 'chunk':
            g.decode(); n += len(g.layers)
    return n
