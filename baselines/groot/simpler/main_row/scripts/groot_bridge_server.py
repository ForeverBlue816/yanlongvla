"""GR00T N1.7 LIBERO policy bridge (runs in envs/groot): serves one per-suite nvidia/GR00T-N1.7-LIBERO checkpoint over the
same websocket wire protocol as our openpi policy servers, so the paired LIBERO / LIBERO-Plus clients drive it unchanged.

Request (msgpack-numpy dict from groot_libero_client.py):
  observation/image, observation/wrist_image : uint8 (256, 256, 3), already flipped [::-1, ::-1] exactly as GR00T's own
                                               LIBERO wrapper does; observation/state: float32 (8,) = eef xyz, axis-angle, gripper qpos (2);
  prompt: str; _vla_episode_seed: int (first call of an episode) -> torch.manual_seed for the flow-matching noise.
Reply: {'actions': float32 (16, 7)} in the LIBERO env convention (gripper: [0,1] -> [-1,1], binarized, inverted, as in
gr00t/eval/sim/LIBERO/libero_env.py), plus server_timing.

Backbone config source: the Gr00tN1d7 loader constructs the Qwen3-VL backbone from config.model_name
(nvidia/Cosmos-Reason2-2B, a gated repo) before loading the checkpoint's own backbone weights (all 16 kept LLM layers, the
24 vision blocks, embeddings and lm_head are stored in the checkpoint). Without gated access we substitute the public
Qwen/Qwen3-VL-2B-Instruct repo, which the N1.7 code accepts explicitly (get_backbone_cls) and which shares the architecture
and tokenizer; the substitution is recorded in the served manifest. With HF_TOKEN set and access granted the official
name is used.
"""
import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

import importlib.util
# Use openpi_client's msgpack-numpy codec (the clients' wire format), not the pip msgpack_numpy package, whose ndarray
# encoding differs; load the single module file directly so envs/groot needs no openpi_client install.
_ROOT = os.environ.get('VLA_ROOT', '/projects/yanlongvla')          # hand-over copy: paths from VLA_ROOT / OPENPI_CLIENT_MSGPACK
_spec = importlib.util.spec_from_file_location('openpi_msgpack_numpy', os.environ.get('OPENPI_CLIENT_MSGPACK', _ROOT + '/src/openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py'))
msgpack_numpy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msgpack_numpy)
import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server
import websockets.frames

HF_CACHE = Path(os.environ.get('HF_HUB_CACHE', _ROOT + '/cache/huggingface/hub'))
MODELS = Path(_ROOT + '/models/groot_n17_libero')
OFFICIAL_BACKBONE = 'nvidia/Cosmos-Reason2-2B'
PUBLIC_BACKBONE = 'Qwen/Qwen3-VL-2B-Instruct'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
N_ACTION_STEPS = 8          # official GR00T LIBERO rollout executes 8 of the 16-step window per policy call


def snapshot_dir():
    snaps = sorted((HF_CACHE / 'models--nvidia--GR00T-N1.7-LIBERO' / 'snapshots').glob('*'))
    assert snaps, 'checkpoint not downloaded'
    return snaps[-1]


def gated_access_available():
    token = os.environ.get('HF_TOKEN') or (Path.home() / '.cache/huggingface/token').read_text().strip() if (Path.home() / '.cache/huggingface/token').exists() else os.environ.get('HF_TOKEN')
    if not token:
        return False
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(OFFICIAL_BACKBONE, 'config.json', token=token)
        return True
    except Exception:
        return False


def verify_blobs(src):
    """Every weight shard must hash to its HF blob name (the blob file name is the file's SHA-256). A corrupted download was
    found on 2026-10-16 (all four first shards; Xet transfer), so the bridge refuses to serve unverified weights. Results are
    cached next to the blob as <blob>.verified."""
    import hashlib
    for shard in sorted(src.glob('model-*.safetensors')):
        blob = shard.resolve()
        stamp = blob.with_name(blob.name + '.verified')
        if stamp.exists() and stamp.read_text().strip() == str(blob.stat().st_size):
            continue
        h = hashlib.sha256()
        with open(blob, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        assert h.hexdigest() == blob.name, f'corrupted checkpoint shard {shard} (sha {h.hexdigest()[:12]} != {blob.name[:12]})'
        stamp.write_text(str(blob.stat().st_size))


def prepare_model_dir(suite, backbone_name):
    """Local model directory: symlinks to the snapshot files, config.json with the chosen backbone name."""
    src = snapshot_dir() / suite
    assert (src / 'model.safetensors.index.json').exists(), src
    verify_blobs(src)
    local = MODELS / suite
    local.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == 'config.json':
            continue
        dst = local / f.name
        if not dst.exists():
            dst.symlink_to(f)
    cfg = json.loads((src / 'config.json').read_text())
    cfg['model_name'] = backbone_name
    (local / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (local / 'bridge_manifest.json').write_text(json.dumps(dict(suite=suite, snapshot=str(src), backbone_config_source=backbone_name,
                                                                 official_backbone=OFFICIAL_BACKBONE, substituted=backbone_name != OFFICIAL_BACKBONE), indent=2) + '\n')
    return local


def env_action(chunk):
    """GR00T action window (T, 7) in the dataset convention -> LIBERO env actions (gripper [0,1] -> [-1,1], sign, inverted)."""
    a = np.array(chunk, dtype=np.float32)
    a[..., -1] = 2 * a[..., -1] - 1
    a[..., -1] = np.sign(a[..., -1])
    a[..., -1] = -a[..., -1]
    return a


class Bridge:
    def __init__(self, suite, device='cuda', model_root=None):
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        self.backbone = OFFICIAL_BACKBONE if gated_access_available() else PUBLIC_BACKBONE
        # model_root: an assembled (e.g. 2-bit PTQ fake-quant) checkpoint directory per suite, written by groot_ptq.py assemble
        self.model_dir = Path(model_root) / suite if model_root else prepare_model_dir(suite, self.backbone)
        self.policy = Gr00tPolicy(embodiment_tag='LIBERO_PANDA', model_path=str(self.model_dir), device=device, strict=True)
        self.suite = suite
        self.calls = 0

    def observation(self, obs):
        img = np.asarray(obs['observation/image'], dtype=np.uint8)
        wrist = np.asarray(obs['observation/wrist_image'], dtype=np.uint8)
        state = np.asarray(obs['observation/state'], dtype=np.float32)
        assert img.shape == (256, 256, 3) and wrist.shape == (256, 256, 3) and state.shape == (8,), (img.shape, wrist.shape, state.shape)
        s = lambda sl: state[sl].reshape(1, 1, -1)
        return {'video': {'image': img[None, None], 'wrist_image': wrist[None, None]},
                'state': {'x': s(slice(0, 1)), 'y': s(slice(1, 2)), 'z': s(slice(2, 3)), 'roll': s(slice(3, 4)), 'pitch': s(slice(4, 5)), 'yaw': s(slice(5, 6)), 'gripper': s(slice(6, 8))},
                'language': {'annotation.human.action.task_description': [[str(obs['prompt'])]]}}

    def infer(self, obs):
        obs = dict(obs)
        seed = obs.pop('_vla_episode_seed', None)
        if seed is not None:
            torch.manual_seed(int(seed))
            self.policy.reset()
        action, _ = self.policy.get_action(self.observation(obs))
        keys = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')
        get = lambda k: action[f'action.{k}'] if f'action.{k}' in action else action[k]      # Gr00tPolicy returns bare keys; the sim wrapper prefixes them
        chunk = np.concatenate([np.asarray(get(k)[0], dtype=np.float32) for k in keys], axis=-1)   # (16, 7)
        assert chunk.ndim == 2 and chunk.shape[1] == 7 and np.isfinite(chunk).all(), chunk.shape
        self.calls += 1
        return {'actions': env_action(chunk), 'n_action_steps': N_ACTION_STEPS}


async def handler(bridge, websocket):
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack(dict(model='nvidia/GR00T-N1.7-LIBERO', suite=bridge.suite, backbone_config_source=bridge.backbone, n_action_steps=N_ACTION_STEPS)))
    while True:
        try:
            obs = msgpack_numpy.unpackb(await websocket.recv())
            t0 = time.monotonic()
            reply = bridge.infer(obs)
            reply['server_timing'] = {'infer_ms': (time.monotonic() - t0) * 1000}
            await websocket.send(packer.pack(reply))
        except websockets.ConnectionClosed:
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason='Internal server error')
            raise


async def health(connection, request):
    if request.path == '/healthz':
        return connection.respond(200, 'OK\n')
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--model-root', default=None)
    a = p.parse_args()
    torch.set_num_threads(4)
    bridge = Bridge(a.suite, model_root=a.model_root)
    # warm-up on a synthetic observation so the first episode is not timed against compilation
    bridge.infer({'observation/image': np.zeros((256, 256, 3), np.uint8), 'observation/wrist_image': np.zeros((256, 256, 3), np.uint8),
                  'observation/state': np.zeros(8, np.float32), 'prompt': 'warm up', '_vla_episode_seed': 0})
    print(json.dumps(dict(ready=True, suite=a.suite, backbone_config_source=bridge.backbone, model_dir=str(bridge.model_dir))), flush=True)

    async def run():
        async with _server.serve(lambda ws: handler(bridge, ws), a.host, a.port, compression=None, max_size=None, process_request=health) as server:
            await server.serve_forever()
    asyncio.run(run())


if __name__ == '__main__':
    main()
