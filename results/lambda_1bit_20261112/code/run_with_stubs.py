"""Run a baseline-harness script (e.g. groot/qtip/quantize_qtip.py) with QTIP's three native imports replaced by pure-PyTorch modules,
so that no CUDA extension has to be built (Lambda, 2026-11-12):
  glog                     logging only -> no-op
  qtip_kernels             decode kernels for inference with packed weights; never called while quantizing -> empty module
  fast_hadamard_transform  hadamard_transform(x, scale) = scale * (x @ H_n) with H_n the Sylvester Hadamard matrix of size n = x.shape[-1]
                           (n a power of 2), the operation of Dao-AILab's CUDA kernel; computed here by the in-place butterfly in FP32.
Usage: python run_with_stubs.py <script.py> [args...]"""
import runpy
import sys
import types

import torch


def hadamard_transform(x, scale=1.0):
    shape, dtype = x.shape, x.dtype
    n = shape[-1]
    assert n & (n - 1) == 0, f'hadamard_transform: size {n} is not a power of 2'
    y = x.reshape(-1, n).float()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return (y.reshape(shape) * scale).to(dtype)


glog = types.ModuleType('glog')
for name in ('info', 'warning', 'warn', 'error', 'debug', 'fatal'):
    setattr(glog, name, lambda *a, **k: None)
fht = types.ModuleType('fast_hadamard_transform')
fht.hadamard_transform = hadamard_transform
sys.modules.update(glog=glog, qtip_kernels=types.ModuleType('qtip_kernels'), fast_hadamard_transform=fht)

if __name__ == '__main__':
    n = 64
    x = torch.randn(3, n, dtype=torch.float64)
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
    assert torch.allclose(hadamard_transform(x.float(), 0.5).double(), 0.5 * x @ H, atol=1e-4), 'hadamard self-check'
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(__import__('pathlib').Path(script).resolve().parent))
    runpy.run_path(script, run_name='__main__')
