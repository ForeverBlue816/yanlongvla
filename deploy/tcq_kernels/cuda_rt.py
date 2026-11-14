"""Minimal CUDA kernel runtime on top of PyTorch's CUDA context: NVRTC (CUDA C++ source -> PTX) and the CUDA driver API (module load,
kernel launch) through ctypes. No cupy, no nvcc, no build step: the NVRTC library and the CUDA headers are the pip wheels PyTorch itself
depends on (nvidia-cuda-nvrtc, nvidia-cuda-runtime); libcuda.so.1 is the driver.

Kernels are launched on torch.cuda.current_stream(), so they are recorded by torch.cuda.graph() captures like PyTorch's own kernels
(arguments are baked at capture: pass only tensors that stay alive and scalars that do not change between replays).
"""
import ctypes
import glob
import os

import torch

_STATE = {}


def _wheel(sub, pattern):
    import nvidia
    for base in nvidia.__path__:
        hits = sorted(glob.glob(os.path.join(base, sub, pattern)))
        if hits:
            return hits[-1]
    return None


def _libs():
    if _STATE:
        return _STATE
    torch.cuda.init(); torch.zeros(1, device='cuda')                      # creates PyTorch's (primary) context and makes it current on this thread
    nvrtc = ctypes.CDLL(_wheel('cuda_nvrtc', 'lib/libnvrtc.so.*') or 'libnvrtc.so')
    cuda = ctypes.CDLL('libcuda.so.1')
    cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    include = _wheel('cuda_runtime', 'include')
    major, minor = torch.cuda.get_device_capability()
    v = (ctypes.c_int(), ctypes.c_int()); nvrtc.nvrtcVersion(ctypes.byref(v[0]), ctypes.byref(v[1]))
    _STATE.update(nvrtc=nvrtc, cuda=cuda, include=include, arch=f'compute_{major}{minor}', nvrtc_version=f'{v[0].value}.{v[1].value}', modules=[])
    return _STATE


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f'{what} failed with CUDA / NVRTC error {rc}')


def compile_module(source, defines=None, fmad=False):
    """CUDA C++ source -> dict name -> Kernel for every `extern "C" __global__` function. fmad=False keeps a*b+c un-contracted (bit-exact
    agreement with the PyTorch reference arithmetic)."""
    import re
    st = _libs(); nvrtc, cuda = st['nvrtc'], st['cuda']
    opts = [f'--gpu-architecture={st["arch"]}', '--std=c++17', f'--fmad={"true" if fmad else "false"}']
    if st['include']:
        opts.append(f'--include-path={st["include"]}')
    opts += [f'-D{k}={v}' for k, v in (defines or {}).items()]
    prog = ctypes.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b'tcq_kernels.cu', 0, None, None), 'nvrtcCreateProgram')
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        n = ctypes.c_size_t(); nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value); nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError('NVRTC compile error:\n' + log.value.decode(errors='replace'))
    n = ctypes.c_size_t(); _check(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n)), 'nvrtcGetPTXSize')
    ptx = ctypes.create_string_buffer(n.value); _check(nvrtc.nvrtcGetPTX(prog, ptx), 'nvrtcGetPTX')
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    module = ctypes.c_void_p()
    _check(cuda.cuModuleLoadData(ctypes.byref(module), ptx.value), 'cuModuleLoadData')
    st['modules'].append(module)
    out = {}
    for name in re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)', source):
        f = ctypes.c_void_p()
        _check(cuda.cuModuleGetFunction(ctypes.byref(f), module, name.encode()), f'cuModuleGetFunction({name})')
        out[name] = Kernel(f, name)
    return out


class Kernel:
    def __init__(self, func, name):
        self.func, self.name, self.launches = func, name, 0

    def max_dynamic_shared(self, nbytes):
        """Opt in to more than 48 KB of dynamic shared memory per block (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8)."""
        _check(_libs()['cuda'].cuFuncSetAttribute(self.func, 8, int(nbytes)), f'cuFuncSetAttribute({self.name}, {nbytes})')

    def __call__(self, grid, block, args, shared=0):
        vals = []
        for a in args:
            if a is None:
                vals.append(ctypes.c_void_p(0))
            elif isinstance(a, torch.Tensor):
                vals.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, bool):
                vals.append(ctypes.c_int(int(a)))
            elif isinstance(a, int):
                vals.append(ctypes.c_int(a))
            elif isinstance(a, float):
                vals.append(ctypes.c_float(a))
            else:
                raise TypeError(f'{self.name}: unsupported kernel argument {type(a)}')
        params = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
        g = tuple(grid) + (1,) * (3 - len(grid)) if not isinstance(grid, int) else (grid, 1, 1)
        b = tuple(block) + (1,) * (3 - len(block)) if not isinstance(block, int) else (block, 1, 1)
        rc = _libs()['cuda'].cuLaunchKernel(self.func, g[0], g[1], g[2], b[0], b[1], b[2], int(shared),
                                           ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), params, None)
        _check(rc, f'cuLaunchKernel({self.name})')
        self.launches += 1


def versions():
    st = _libs()
    return dict(nvrtc=st['nvrtc_version'], arch=st['arch'], torch=torch.__version__, cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name())
