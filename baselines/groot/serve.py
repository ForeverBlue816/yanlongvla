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


def main():
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
