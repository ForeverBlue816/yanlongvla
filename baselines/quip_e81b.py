"""QuIP# 1-bit codebook NON-OFFICIAL: QuIP# has no
1-bit recipe; its official codebooks are E8P12 (2 bits), E8P12RVQ3B (3 bits), E8P12RVQ4B (4 bits).

What is official here: the grid is QuIP#'s own 256-point E8 grid `_E81B_CACHED` (lib/codebook/latticee8_padded12_rvq3bit.py:
get_e81bgrid, the residual stage of the official 3-bit codebook: 0, the 240 E8 roots and 15 of the 16 points +-2 e_i), used here
alone: 8 bits per 8 weights = 1 bit per weight. Rounding is the official nearest-point rule (argmax 2 x.g - |g|^2, as
E8P12RVQ3B_codebook.round). Everything else (RHT incoherence, block LDL, buffered LDLQ, 10 tune sweeps, sigma_reg) is the
unchanged official lib/algo/quip.py.

What is ours: (1) the class itself; (2) opt_scale, computed with the official procedure written in the comment block of
E8P12RVQ3B_codebook.__init__ (minimize E||Q(s x)/s - x||^2 over s for x ~ N(0, I_8)), because no official value exists for
this grid alone; (3) the packing (8 one-byte indices per int64, as the official pack_e81b).
"""
import torch
from torch import nn

from lib.codebook.latticee8_padded12_rvq3bit import _E81B_CACHED, _E81B_NORM_CACHED

_CODESZ = 8


def nearest(X, grid, grid_norm):
    idx = (2 * X @ grid.T - grid_norm).argmax(-1)
    return grid[idx], idx


def optimal_scale(samples=1_000_000, seed=0, device='cpu'):
    """The official opt_scale procedure (scalar search of the Gaussian MSE), for the E81B grid alone."""
    from scipy.optimize import minimize_scalar
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, _CODESZ, generator=g).to(device)
    grid, gn = _E81B_CACHED.float().to(device), _E81B_NORM_CACHED.float().to(device)

    def err(s):
        return float((nearest(x * s, grid, gn)[0] / s - x).square().sum(-1).mean())

    res = minimize_scalar(err, bounds=(0.1, 100), method='bounded')
    return float(res.x), float(res.fun)


class E81B_codebook(nn.Module):
    OPT_SCALE = None   # set by main() of quantize_quip.py from optimal_scale() and recorded in meta

    def __init__(self, inference=False):
        super().__init__()
        assert E81B_codebook.OPT_SCALE is not None, 'compute OPT_SCALE first (quip_e81b.optimal_scale)'
        self.opt_scale = E81B_codebook.OPT_SCALE
        self.codesz = _CODESZ
        self.idx_dtype = torch.int64
        self.packsz = 8
        self.pack_out = False
        self.version = 0
        self.register_buffer('grid', _E81B_CACHED.float())
        self.register_buffer('grid_norm', _E81B_NORM_CACHED.float())

    def quantize(self, X, return_idx=True, **kwargs):
        vals, idx = nearest(X, self.grid.to(X.dtype), self.grid_norm.to(X.dtype))
        return (vals, idx) if return_idx else vals

    def maybe_pack_idxs(self, idxs):   # (m, n/8) indices in [0, 256) -> (m, n/64) int64, 8 per word (as the official pack_e81b)
        idxs = idxs.to(torch.int64)
        assert idxs.shape[-1] % 8 == 0, idxs.shape
        accum = idxs[:, 0::8].clone()
        for i in range(1, 8):
            accum += idxs[:, i::8] << (8 * i)
        return accum

    def unpack_idxs(self, packed):
        m, k = packed.shape
        out = torch.empty(m, k * 8, dtype=torch.int64, device=packed.device)
        for i in range(8):
            out[:, i::8] = (packed >> (8 * i)) & 255
        return out

    def by_idxs(self, idxs, **kwargs):   # decode (unpacked indices) -> (m, n) values on the grid
        m, k = idxs.shape
        return self.grid[idxs.long()].reshape(m, k * _CODESZ)
