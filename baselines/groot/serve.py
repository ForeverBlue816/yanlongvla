"""Policy server (runs in the GR00T environment): serves one suite checkpoint (prepared official or fake-quant) over the
websocket wire protocol of the report's clients. The flow-matching noise is seeded per episode from `_vla_episode_seed`,
so every method sees identical noise on identical units. Replies {'actions': (16, 7) float32, 'n_action_steps': 8}.

Usage:  python serve.py --checkpoint ./ckpt_gptq_w2g128/libero_spatial --port 8205
"""
import argparse
import asyncio
import json
import time
import traceback

import numpy as np
import torch

from groot_common import load_policy, infer, N_ACTION_STEPS


def client_codec():
    """openpi_client's msgpack-numpy codec (the clients' wire format; the pip package msgpack_numpy encodes arrays differently).
    Install openpi-client into this environment, or point OPENPI_CLIENT_MSGPACK at openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py."""
    import os
    try:
        from openpi_client import msgpack_numpy
        return msgpack_numpy
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location('openpi_msgpack_numpy', os.environ['OPENPI_CLIENT_MSGPACK'])
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module


def set_sdpa_backend():
    """GOAL 1 Nov 2.6: SDPA_BACKEND=math|efficient|flash restricts torch scaled_dot_product_attention to one kernel (default: torch chooses)."""
    import os
    b = os.environ.get('SDPA_BACKEND')
    if b:
        torch.backends.cuda.enable_math_sdp(b == 'math'); torch.backends.cuda.enable_mem_efficient_sdp(b == 'efficient'); torch.backends.cuda.enable_flash_sdp(b == 'flash')
        print(json.dumps(dict(sdpa_backend=b)), flush=True)
    log_sdpa_kernels()


def log_sdpa_kernels(n_calls=3):
    """2 Nov: record which CUDA kernels torch's scaled_dot_product_attention really launches (the 'default' choice was never logged).
    Wraps F.scaled_dot_product_attention, profiles its first n_calls calls, prints {"sdpa_kernels": {...}} once per call; no effect on outputs."""
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity
    orig = F.scaled_dot_product_attention
    state = dict(n=0)

    def wrapped(*args, **kwargs):
        if state['n'] >= n_calls or not torch.cuda.is_available():
            return orig(*args, **kwargs)
        state['n'] += 1
        try:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                out = orig(*args, **kwargs)
                torch.cuda.synchronize()
        except Exception as e:
            print(json.dumps(dict(sdpa_kernels_error=repr(e))), flush=True)
            state['n'] = n_calls
            return orig(*args, **kwargs)
        names = sorted({e.key for e in prof.key_averages() if e.device_type.name == 'CUDA'})
        q = args[0] if args else kwargs.get('query')
        print(json.dumps(dict(sdpa_kernels=names, q_shape=list(q.shape), dtype=str(q.dtype), is_causal=kwargs.get('is_causal'),
                              has_mask=(len(args) > 3 and args[3] is not None) or kwargs.get('attn_mask') is not None)), flush=True)
        return out
    F.scaled_dot_product_attention = wrapped


def main():
    set_sdpa_backend()
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, default=8205)
    p.add_argument('--host', default='127.0.0.1')
    a = p.parse_args()
    import websockets
    import websockets.asyncio.server as _server
    import websockets.frames
    msgpack_numpy = client_codec()
    torch.set_num_threads(4)
    policy = load_policy(a.checkpoint)
    infer(policy, {'observation/image': np.zeros((256, 256, 3), np.uint8), 'observation/wrist_image': np.zeros((256, 256, 3), np.uint8),
                   'observation/state': np.zeros(8, np.float32), 'prompt': 'warm up', '_vla_episode_seed': 0})
    print(json.dumps(dict(ready=True, checkpoint=a.checkpoint, port=a.port)), flush=True)

    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(dict(model='nvidia/GR00T-N1.7-LIBERO', checkpoint=a.checkpoint, n_action_steps=N_ACTION_STEPS)))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                t0 = time.monotonic()
                reply = {'actions': infer(policy, obs), 'n_action_steps': N_ACTION_STEPS, 'server_timing': {'infer_ms': (time.monotonic() - t0) * 1000}}
                await ws.send(packer.pack(reply))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason='Internal server error')
                raise

    async def health(connection, request):
        return connection.respond(200, 'OK\n') if request.path == '/healthz' else None

    async def run():
        async with _server.serve(handler, a.host, a.port, compression=None, max_size=None, process_request=health) as server:
            await server.serve_forever()
    asyncio.run(run())


if __name__ == '__main__':
    main()
