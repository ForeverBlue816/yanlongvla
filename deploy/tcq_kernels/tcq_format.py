"""Packed deployment file of a TCQ row and the PyTorch reference arithmetic of W2A8 in this format.

packed.safetensors (models/pi05_tcq/<row>/): per quantized Linear `name`
  <name>::bits       int16 view of the uint16 words (m/16 * n/16, 32): the stored trellis bitstream, 512 bits per 16 x 16 tile, unchanged
  <name>::tlut       FP16 (512, 2)              the stored lookup table, unchanged
  <name>::row_scale  FP16 (m,)                  the stored row scale, unchanged
  <name>::su, ::sv   int8 (n,), (m,)            input / output signs of the two-sided randomized block-Walsh rotation, unchanged
Nothing is refitted. The INT8 runtime derives, at load:
  t8  = round(tlut * g), g = 127 / max|tlut|    INT8 table (512, 2)
  rs  = FP32(row_scale) / g                      row scale applied after the integer accumulation
so decoded weight = sign * t8[hash(state)] * rs[row] (rotated basis). The table rounding is the only difference from the stored FP16
model; `reference_linear(table='fp16')` gives the unrounded model for comparison.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get('VLA_ROOT', '/projects/yanlongvla'))            # project root: models/, runs/, research/
ROW = 'tcq_raqft_d1'


def block_widths(width):
    out = []
    while width:
        b = 1 << (width.bit_length() - 1); out.append(b); width -= b
    return out


_SCALES = {}


def _scale(w, device):
    """FP32 constant 1 / sqrt(w) on the device, created once (a host-to-device copy is not allowed inside a CUDA-graph capture)."""
    key = (w, str(device))
    if key not in _SCALES:
        _SCALES[key] = torch.tensor(np.float32(1.0 / math.sqrt(w)), device=device)
    return _SCALES[key]


def fwht(x):
    """Block Walsh on the last dim, FP32, butterflies in the order h = 1, 2, 4, ... (the kernels reproduce these sums bit for bit)."""
    outs, off = [], 0
    for w in block_widths(x.shape[-1]):
        y, h = x[..., off:off + w], 1
        while h < w:
            y = y.reshape(*y.shape[:-1], w // (2 * h), 2, h)
            y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), dim=-2).reshape(*y.shape[:-3], w)
            h *= 2
        outs.append(y * _scale(w, x.device))
        off += w
    return torch.cat(outs, -1) if len(outs) > 1 else outs[0]


def states_from_bits(bits, m, n):
    """Windowed read: state t of a tile = bits [4 t, 4 t + 16) of its 512-bit stream (MSB first, tail-biting) -> (m, n / 2) int64."""
    b = bits.to(torch.int32).to(torch.int64) & 0xffff
    t = torch.arange(128, device=bits.device); w, r = t // 4, t % 4
    st = (((b[:, w] << 16) | b[:, (w + 1) % 32]) >> (16 - 4 * r)) & 0xffff
    return st.reshape(m // 16, n // 16, 16, 8).transpose(1, 2).reshape(m, n // 2)


def lookup(states, table):
    """(m, n/2) states + (512, 2) table -> (m, n) values: hash = ((s + 1) s >> 6) & 511, first coordinate negated when bit 15 of (s + 1) s is set."""
    x = (states + 1) * states
    v = table[(x >> 6) & 511]
    v0 = torch.where(((x >> 15) & 1).bool(), -v[..., 0], v[..., 0])
    return torch.stack((v0, v[..., 1]), -1).reshape(states.shape[0], -1)


def snap(tlut, row_scale):
    g = (127.0 / tlut.float().abs().max()).to(torch.float32)
    t8 = torch.round(tlut.float() * g).clamp(-127, 127).to(torch.int8)
    return t8, row_scale.float() / g, g


class Layer:
    """One quantized Linear on a device: packed tensors + derived INT8 table / row scale; `w8` (m, n) INT8 decoded on demand."""
    def __init__(self, name, tensors, bias=None, device='cuda'):
        self.name = name
        self.bits = tensors['bits'].to(device)
        self.tlut = tensors['tlut'].to(device)
        self.row_scale = tensors['row_scale'].to(device)
        self.su, self.sv = tensors['su'].to(device), tensors['sv'].to(device)
        self.m, self.n = self.sv.numel(), self.su.numel()
        assert self.bits.shape == ((self.m // 16) * (self.n // 16), 32) and self.m % 16 == 0 and self.n % 16 == 0, name
        self.t8, self.rs, self.gain = snap(self.tlut, self.row_scale)
        self.t8 = self.t8.contiguous()
        self.bias = None if bias is None else bias.float().to(device)

    def states(self):
        return states_from_bits(self.bits, self.m, self.n)

    def w8_reference(self):
        return lookup(self.states(), self.t8.to(torch.int16)).to(torch.int8)

    def dense(self, table='fp16'):
        """Original-basis FP32 weight: S_v H_m diag(r) Q H_n S_u with the FP16 table (the evaluated model) or the INT8 table."""
        if table == 'fp16':
            q = lookup(self.states(), self.tlut.float()) * self.row_scale.float()[:, None]
        else:
            q = self.w8_reference().float() * self.rs[:, None]
        w = fwht(q) * self.su.float()[None, :]                       # (Q H_n) S_u  (H symmetric)
        return (fwht(w.T) * self.sv.float()[None, :]).T               # S_v H_m (...)


def int_matmul(q, w8):
    """Exact (M, K) int8 x (N, K) int8 -> (M, N) int32."""
    if q.is_cuda:
        m = q.shape[0]
        qq = torch.nn.functional.pad(q, (0, 0, 0, max(0, 32 - m)))
        return torch._int_mm(qq, w8.t())[:m]
    return (q.double() @ w8.double().t()).to(torch.int32)


def quantize_tokens(h):
    amax = h.abs().amax(dim=1, keepdim=True)
    s = torch.where(amax > 0, amax / torch.full_like(amax, 127.0), torch.ones_like(amax))      # tensor divisor: a true FP32 division (a Python scalar divisor becomes a reciprocal multiply on CUDA)
    return torch.clamp(torch.round(h / s), -127, 127).to(torch.int8), s


def reference_linear(x, layer, a8=True, table='int8', w8=None, out_dtype=torch.bfloat16):
    """The W2A8 definition of one layer in plain PyTorch. a8=False: FP32 activations in the rotated space (same weights)."""
    shape = x.shape
    if TIMER is not None and table == 'int8' and a8:
        return _timed_reference(x, layer, w8, out_dtype)
    h = fwht(x.float().reshape(-1, shape[-1]) * layer.su.float())
    if table == 'int8':
        w8 = layer.w8_reference() if w8 is None else w8
        if a8:
            q, s = quantize_tokens(h)
            z = (int_matmul(q, w8).float() * s) * layer.rs[None, :]
        else:
            z = (h.double() @ w8.double().t()).float() * layer.rs[None, :]
    else:
        wq = lookup(layer.states(), layer.tlut.float())
        if a8:
            q, s = quantize_tokens(h)
            z = ((q.double() @ wq.double().t()).float() * s) * layer.row_scale.float()[None, :]
        else:
            z = (h.double() @ wq.double().t()).float() * layer.row_scale.float()[None, :]
    y = fwht(z) * layer.sv.float()
    if layer.bias is not None:
        y = y + layer.bias
    return y.to(out_dtype).reshape(*shape[:-1], -1)


TIMER = None            # set to a dict to time the reference path per operation (eager, synchronised): keys (STAGE, operation) -> seconds
STAGE = 'all'


def _timed_reference(x, layer, w8, out_dtype):
    import time
    def tick(key, t0):
        torch.cuda.synchronize(); TIMER[(STAGE, key)] = TIMER.get((STAGE, key), 0.0) + time.perf_counter() - t0
        return time.perf_counter()
    shape = x.shape
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if w8 is None:
        w8 = layer.w8_reference(); t0 = tick('decode', t0)
    h = fwht(x.float().reshape(-1, shape[-1]) * layer.su.float()); t0 = tick('hadamard', t0)
    q, s = quantize_tokens(h); t0 = tick('activation quantize', t0)
    acc = int_matmul(q, w8); t0 = tick('matrix product', t0)
    z = (acc.float() * s) * layer.rs[None, :]; t0 = tick('scale', t0)
    y = fwht(z) * layer.sv.float()
    if layer.bias is not None:
        y = y + layer.bias
    y = y.to(out_dtype).reshape(*shape[:-1], -1); tick('hadamard', t0)
    return y


# ------------------------------------------------------------------------------------------------------------------- packing
def fit_dir(row=ROW):
    info = json.loads((ROOT / f'models/pi05_tcq/{row}/row.json').read_text())
    return ROOT / 'runs/plustrain_20261008/pi05_tcq/fit' / info['fit_key']


def packed_path(row=ROW):
    return ROOT / f'models/pi05_tcq/{row}/packed.safetensors'


def layer_names():
    rows = [r for r in json.loads((ROOT / 'research/results/model_audit.json').read_text())['parameters'] if r['quantizable']]
    assert len(rows) == 414
    return [r['name'].removesuffix('.weight') for r in rows]


def build(row=ROW, device='cuda'):
    """Fit records -> packed.safetensors. Every layer is checked: the windowed read reproduces the stored state indices (QTIP unpack), and the
    decoded FP16-table weight in the original basis equals the evaluated dense weight after BF16 rounding (reported, not assumed)."""
    from safetensors.torch import save_file
    out, report = {}, []
    d = fit_dir(row)
    for i, name in enumerate(layer_names()):
        rec = torch.load(d / f'{name}.pt', map_location='cpu')
        t = rec['tensors']
        assert rec['tcq']['rotation'] == 'two' and rec['tcq']['L'] == 16 and rec['tcq']['K'] == 2 and rec['tcq']['V'] == 2 and rec['tcq']['td'] == 16, name
        m, n = rec['w_hat'].shape
        tensors = dict(bits=t['trellis'].view(torch.int16), tlut=t['tlut'], row_scale=t['row_scale'], su=t['signs'], sv=t['signs_out'])   # the uint16 words, stored as int16 (safetensors has no uint16 here)
        L = Layer(name, tensors, device=device)
        st = L.states()
        x = (st + 1) * st; k = ((x >> 6) & 511) * 2; sg = 1 - ((x >> 15) & 1) * 2
        index = torch.stack((sg * (k + 1), k + 2), -1).reshape(m, n).to(torch.int16)
        same_index = bool(torch.equal(index.cpu(), rec['index']))
        w = L.dense('fp16')
        ref = rec['w_hat'].to(device)
        same_dense = bool(torch.equal(w.bfloat16(), ref))
        rel = float(((w - ref.float()).norm() / ref.float().norm()))
        w8err = float(((L.dense('int8') - w).norm() / w.norm()) ** 2)
        report.append(dict(name=name, shape=[m, n], states_equal=same_index, dense_bf16_equal=same_dense, dense_rel_err=rel, int8_table_rel_mse=w8err,
                           gain=float(L.gain), max_abs_t8=int(L.t8.abs().max())))
        for k2, v in tensors.items():
            out[f'{name}::{k2}'] = v.contiguous()
        if i % 40 == 0:
            print(json.dumps(report[-1]), flush=True)
        del rec, L, w, ref
    p = packed_path(row)
    save_file(out, str(p.with_suffix('.partial'))); p.with_suffix('.partial').replace(p)
    summary = dict(row=row, layers=len(report), states_equal=sum(r['states_equal'] for r in report), dense_bf16_equal=sum(r['dense_bf16_equal'] for r in report),
                   dense_rel_err_max=max(r['dense_rel_err'] for r in report), int8_table_rel_mse_mean=float(np.mean([r['int8_table_rel_mse'] for r in report])),
                   int8_table_rel_mse_max=max(r['int8_table_rel_mse'] for r in report), packed_bytes=p.stat().st_size)
    (p.parent / 'packed.json').write_text(json.dumps(dict(summary=summary, layers=report), indent=1))
    print(json.dumps(summary), flush=True)
    return summary


def load(row=ROW, device='cuda', names=None, biases=None):
    from safetensors import safe_open
    out = {}
    with safe_open(str(packed_path(row)), framework='pt', device='cpu') as f:
        for name in (names or layer_names()):
            out[name] = Layer(name, {k: f.get_tensor(f'{name}::{k}') for k in ('bits', 'tlut', 'row_scale', 'su', 'sv')},
                              bias=(biases or {}).get(name), device=device)
    return out


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ROW)
