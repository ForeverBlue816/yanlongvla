"""GR00T N1.7 SimplerEnv-Bridge (WidowX) in our 2-bit pipeline (user 2026-11-06): calibration on datasets/bridge-calibration, TCQ two-sided
(tcq_w2_rht) fits of the 432 Linear layers on this checkpoint, assembly into a flat checkpoint directory that the official
run_gr00t_server.py serves unchanged, and held-out action MSE on datasets/bridge-heldout.

Checkpoint: nvidia/GR00T-N1.7-SimplerEnv-Bridge @ 940134b3 (flat layout, no per-suite sub-directories; embodiment SIMPLER_ENV_WIDOWX, projector 1).
Its backbone (216 quantizable Linear) is byte-identical to the LIBERO checkpoints'; the action head (216 Linear) differs. The rows below
calibrate both parts on Bridge observations; tcq_w2_rht_bbl reuses the LIBERO-calibrated backbone fit (ablation: calibration transfer).
Inputs: image_0 256x256 (the dataset / env frame, no flip), state (8) = x, y, z, roll, pitch, yaw, pad, gripper, instruction; output: 8-step
action chunk (x, y, z, roll, pitch, yaw, gripper in [0, 1]); the env executes 4 steps per call (official README protocol).
Stages (envs/groot, PYTHONPATH=src/Isaac-GR00T:research/scripts):
  prepare                  models/groot_n17_simpler_bridge/fp: symlinks to the verified snapshot + config.json with the backbone config source
  smoke                    FP policy on 2 calibration observations; counts the quantizable layers (432)
  calib --parts ...        per-layer H = E[x x^T] over the 256 calibration observations -> OUT/hessians/{backbone,head}_simpler_bridge
  fit --part P --worker w --workers n   TCQ two-sided fits -> OUT/fit/{backbone,head}_tcq_rht_simpler_bridge
  assemble --row R         models/groot_ptq/<row>/simpler_bridge/{model.safetensors, config.json, ...} + accounting.json
  offline --rows R1:R2     held-out chunk MSE vs FP (256 observations, noise SeedSequence([7, i, 0])) -> runs/.../groot/simpler_bridge/offline/
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT, ROOT, MODELS, atomic_save, dump, part_of, target_layers   # noqa: E402

SETTING = 'simpler_bridge'
EMBODIMENT = 'SIMPLER_ENV_WIDOWX'
REPO = 'nvidia/GR00T-N1.7-SimplerEnv-Bridge'
REVISION = '940134b3c2948ccfdf8e7393f2d2ca869dc42833'
HF_CACHE = Path(os.environ.get('HF_HUB_CACHE', str(ROOT / 'cache/huggingface/hub')))          # hand-over copy: default under VLA_ROOT
SNAPSHOT = HF_CACHE / 'models--nvidia--GR00T-N1.7-SimplerEnv-Bridge' / 'snapshots' / REVISION
FP_DIR = ROOT / 'models/groot_n17_simpler_bridge/fp'
CAL = ROOT / 'datasets/bridge-calibration'
HELDOUT = ROOT / 'datasets/bridge-heldout'
SB = OUT.parent / SETTING                                  # runs/plustrain_20261008/groot/simpler_bridge
STATE_KEYS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'pad', 'gripper')
ACTION_KEYS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')
ACTION_HORIZON = 8
N_ACTION_STEPS = 4
HKEY = dict(backbone=f'backbone_{SETTING}', head=f'head_{SETTING}')
FIT_KEY = dict(backbone=f'backbone_tcq_rht_{SETTING}', head=f'head_tcq_rht_{SETTING}')
ROWS = dict(tcq_w2_rht=(FIT_KEY['backbone'], FIT_KEY['head']),
            tcq_w2_rht_bbl=('backbone_tcq_rht', FIT_KEY['head']))   # LIBERO-calibrated backbone fit (identical weights), Bridge-calibrated head
BACKBONE_LAYERS, HEAD_LAYERS = 216, 216


# ------------------------------------------------------------------------------------------------------------------ checkpoint
def weights_bridge(names):
    from safetensors import safe_open
    from groot_bridge_server import verify_blobs
    verify_blobs(SNAPSHOT)
    index = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())['weight_map']
    by_shard, out = {}, {}
    for n in names:
        by_shard.setdefault(index[n + '.weight'], []).append(n)
    for shard, ns in by_shard.items():
        with safe_open(str(SNAPSHOT / shard), 'pt') as f:
            for n in ns:
                out[n] = f.get_tensor(n + '.weight')
    return out


def prepare_fp(backbone_name):
    from groot_bridge_server import verify_blobs, OFFICIAL_BACKBONE
    assert (SNAPSHOT / 'model.safetensors.index.json').exists(), f'download {REPO}@{REVISION} first'
    verify_blobs(SNAPSHOT)
    FP_DIR.mkdir(parents=True, exist_ok=True)
    for f in SNAPSHOT.iterdir():
        if f.name == 'config.json' or f.name.startswith('global_step') or f.name.startswith('rng_state') or f.name in ('latest', 'scheduler.pt', 'training_args.bin', 'zero_to_fp32.py'):
            continue
        dst = FP_DIR / f.name
        if not dst.exists():
            dst.symlink_to(f)
    cfg = json.loads((SNAPSHOT / 'config.json').read_text())
    cfg['model_name'] = backbone_name
    (FP_DIR / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (FP_DIR / 'bridge_manifest.json').write_text(json.dumps(dict(repo=REPO, revision=REVISION, embodiment=EMBODIMENT, snapshot=str(SNAPSHOT), backbone_config_source=backbone_name,
                                                                official_backbone=OFFICIAL_BACKBONE, substituted=backbone_name != OFFICIAL_BACKBONE), indent=2) + '\n')
    return FP_DIR


# ---------------------------------------------------------------------------------------------------------------------- policy
class Policy:
    """Gr00tPolicy on the Bridge checkpoint (FP, or an assembled 2-bit row via model_dir) with our observation convention."""

    def __init__(self, device='cuda', model_dir=None):
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from groot_bridge_server import gated_access_available, OFFICIAL_BACKBONE, PUBLIC_BACKBONE
        self.backbone = OFFICIAL_BACKBONE if gated_access_available() else PUBLIC_BACKBONE
        self.model_dir = Path(model_dir) if model_dir else prepare_fp(self.backbone)
        self.policy = Gr00tPolicy(embodiment_tag=EMBODIMENT, model_path=str(self.model_dir), device=device, strict=True)
        self.calls = 0

    @staticmethod
    def observation(obs):
        img = np.asarray(obs['observation/image'], dtype=np.uint8)
        state = np.asarray(obs['observation/state'], dtype=np.float32)
        assert img.shape == (256, 256, 3) and state.shape == (8,), (img.shape, state.shape)
        return {'video': {'image_0': img[None, None]},
                'state': {k: state[i].reshape(1, 1, 1) for i, k in enumerate(STATE_KEYS)},
                'language': {'annotation.human.action.task_description': [[str(obs['prompt'])]]}}

    def infer(self, obs):
        obs = dict(obs)
        seed = obs.pop('_vla_episode_seed', None)
        if seed is not None:
            torch.manual_seed(int(seed))
            self.policy.reset()
        action, _ = self.policy.get_action(self.observation(obs))
        get = lambda k: action[f'action.{k}'] if f'action.{k}' in action else action[k]
        chunk = np.concatenate([np.asarray(get(k)[0], dtype=np.float32) for k in ACTION_KEYS], axis=-1)
        assert chunk.shape == (ACTION_HORIZON, 7) and np.isfinite(chunk).all(), chunk.shape
        self.calls += 1
        return chunk


def forward_chunks(policy, observations, seed_tail=(0,)):
    out = []
    with torch.inference_mode():
        for i, (_, obs) in enumerate(observations):
            o = dict(obs)
            o['_vla_episode_seed'] = int(np.random.SeedSequence([7, i, *seed_tail]).generate_state(1)[0])
            out.append(policy.infer(o))
    return np.stack(out)


# ---------------------------------------------------------------------------------------------------------------------- stages
def stage_prepare():
    from groot_bridge_server import gated_access_available, OFFICIAL_BACKBONE, PUBLIC_BACKBONE
    d = prepare_fp(OFFICIAL_BACKBONE if gated_access_available() else PUBLIC_BACKBONE)
    print(json.dumps(dict(prepared=str(d), manifest=json.loads((d / 'bridge_manifest.json').read_text()))))


def stage_smoke(device='cuda'):
    from bridge_calibration import load_observations
    policy = Policy(device=device)
    layers = target_layers(policy.policy.model)
    parts = {p: sum(part_of(n) == p for n in layers) for p in ('backbone', 'head')}
    params = sum(m.weight.numel() for m in layers.values())
    assert parts['backbone'] == BACKBONE_LAYERS and parts['head'] == HEAD_LAYERS, parts
    _, observations = load_observations(limit=2, data_dir=CAL)
    chunks = forward_chunks(policy, observations)
    res = dict(layers=parts, quantizable_parameters=params, backbone_config_source=policy.backbone, observations=[o['prompt'] for _, o in observations],
               chunk_shape=list(chunks.shape), chunk0=chunks[0].round(4).tolist(), gpu=torch.cuda.get_device_name())
    dump(SB / 'smoke.json', res)
    print(json.dumps(res), flush=True)


def stage_calib(parts, device='cuda'):
    from bridge_calibration import load_observations
    policy = Policy(device=device)
    layers = {n: m for n, m in target_layers(policy.policy.model).items() if part_of(n) in parts}
    acc = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device=device) for n, m in layers.items()}
    tokens = {n: 0 for n in layers}
    handles = []
    for n, m in layers.items():
        def hook(mod, inp, n=n):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            acc[n].addmm_(x.t(), x)
            tokens[n] += x.shape[0]
        handles.append(m.register_forward_pre_hook(hook))
    _, observations = load_observations(data_dir=CAL)
    assert len(observations) == 256
    t0 = time.time()
    with torch.inference_mode():
        for i, (_, obs) in enumerate(observations):
            o = dict(obs)
            o['_vla_episode_seed'] = int(np.random.SeedSequence([7, i]).generate_state(1)[0])      # as groot_ptq._calibrate
            policy.infer(o)
    for h in handles:
        h.remove()
    for part in parts:
        directory = OUT / 'hessians' / HKEY[part]
        names = [n for n in layers if part_of(n) == part]
        for n in names:
            atomic_save(dict(hessian=(acc[n] / max(tokens[n], 1)).cpu(), tokens=tokens[n], suite=SETTING), directory / f'{n}.pt')
        dump(directory / 'complete.json', dict(status='complete', setting=SETTING, checkpoint=f'{REPO}@{REVISION}', part=part, layers=len(names), observations=256, calibration=str(CAL),
                                              seconds=time.time() - t0, device=torch.cuda.get_device_name(), tokens_example={n: tokens[n] for n in names[:3]}))
    print(json.dumps(dict(calib=SETTING, parts=parts, layers=len(layers), seconds=round(time.time() - t0, 1))), flush=True)


def stage_fit(part, worker, workers, rounds, device='cuda'):
    """TCQ two-sided randomized Hadamard base (groot_tcq.fit_layer, rotation='two'), the format of tcq_w2_rht; records in groot_ptq's layout."""
    from groot_tcq import fit_layer, tlut_init, make_codebook, try_compile, engine, SETTINGS
    key, out_key = HKEY[part], FIT_KEY[part]
    hdir = OUT / 'hessians' / key
    assert (hdir / 'complete.json').exists(), f'calibrate {key} first'
    names = sorted(p.name[:-3] for p in hdir.glob('*.pt'))
    mine = [n for i, n in enumerate(names) if i % workers == worker]
    directory = OUT / 'fit' / out_key
    directory.mkdir(parents=True, exist_ok=True)
    todo = [n for n in mine if not (directory / f'{n}.json').exists()]
    weights = weights_bridge(todo) if todo else {}
    tlut0 = tlut_init().to(device)
    cb = make_codebook(tlut0, device)
    compiled = try_compile(cb)
    t_all = time.time()
    for n in todo:
        w = weights[n].to(device).float()
        h = torch.load(hdir / f'{n}.pt', map_location=device)['hessian'].float()
        record, info = fit_layer(n, w, h, cb, tlut0, rounds=rounds, rotation='two')
        atomic_save(record, directory / f'{n}.pt')
        dump(directory / f'{n}.json', info)
        print(json.dumps(dict(layer=n, shape=info['shape'], rel_mse=round(info['relative_weight_mse'], 5), h_err=round(info['hessian_relative_error'], 6), damping=info['damping'],
                              round=info['selected_round'], seconds=round(info['seconds'], 1))), flush=True)
        del w, h, record
        torch.cuda.empty_cache()
    done = [n for n in names if (directory / f'{n}.json').exists()]
    if len(done) == len(names) and not (directory / 'complete.json').exists():
        metas = [json.loads((directory / f'{n}.json').read_text()) for n in names]
        dump(directory / 'complete.json', dict(status='complete', key=out_key, source_key=key, setting=SETTING, checkpoint=f'{REPO}@{REVISION}', layers=len(names),
                                              payload_bytes=sum(x['payload_bytes'] for x in metas), parameters=sum(x['shape'][0] * x['shape'][1] for x in metas),
                                              relative_weight_mse_mean=float(np.mean([x['relative_weight_mse'] for x in metas])), hessian_relative_error_mean=float(np.mean([x['hessian_relative_error'] for x in metas])),
                                              dampings={str(d): sum(x['damping'] == d for x in metas) for d in sorted({x['damping'] for x in metas})}, seconds=sum(x['seconds'] for x in metas),
                                              format=dict(SETTINGS, rotation='two'), engine=dict(qtip_commit=engine()['commit'], compiled=compiled)))
    print(json.dumps(dict(fit=out_key, worker=worker, layers=len(todo), seconds=round(time.time() - t_all, 1), compiled=compiled)), flush=True)


def stage_assemble(row):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from groot_bridge_server import verify_blobs
    parts = ROWS[row]
    verify_blobs(SNAPSHOT)
    assert (FP_DIR / 'config.json').exists(), 'run prepare first'
    out = MODELS / row / SETTING
    if (out / 'accounting.json').exists():
        return json.loads((out / 'accounting.json').read_text())
    fits = {}
    for key in parts:
        d = OUT / 'fit' / key
        assert (d / 'complete.json').exists(), f'fit {key} incomplete'
        fits.update({p.name[:-3]: p for p in d.glob('*.pt')})
    assert len(fits) == BACKBONE_LAYERS + HEAD_LAYERS, len(fits)
    index = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(SNAPSHOT / shard), 'pt') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    payload = 0
    for n, p in fits.items():
        rec = torch.load(p, map_location='cpu')
        k = n + '.weight'
        assert tensors[k].shape == rec['w_hat'].shape, k
        tensors[k] = rec['w_hat'].to(tensors[k].dtype)
        payload += json.loads(p.with_suffix('.json').read_text())['payload_bytes']
    norm_replaced = 0
    for key in parts:                                    # fine-tuned rows (groot_simpler_main.py): RMSNorm / LayerNorm weights, as groot_ptq.stage_assemble
        nf = OUT / 'fit' / key / 'norm_weights.safetensors'
        if nf.exists():
            with safe_open(str(nf), 'pt') as f:
                for k in f.keys():
                    assert k in tensors and tensors[k].shape == f.get_tensor(k).shape, k
                    tensors[k] = f.get_tensor(k).to(tensors[k].dtype); norm_replaced += 1
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'model.safetensors.partial'
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp), metadata={'format': 'pt'})
    tmp.replace(out / 'model.safetensors')
    for f in FP_DIR.iterdir():
        if f.name.startswith('model') or f.name in ('config.json', 'bridge_manifest.json'):
            continue
        dst = out / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
    shutil.copy(FP_DIR / 'config.json', out / 'config.json')
    quantized_keys = {n + '.weight' for n in fits}
    retained = sum(v.numel() * v.element_size() for k, v in tensors.items() if k not in quantized_keys and 'lm_head' not in k)
    active = sum(v.numel() for k, v in tensors.items() if 'lm_head' not in k)
    quant_params = sum(tensors[k].numel() for k in quantized_keys)
    acc = dict(row=row, suite=SETTING, checkpoint=f'{REPO}@{REVISION}', parts=list(parts), quantized_layers=len(fits), quantized_parameters=quant_params, linear_payload_bytes=payload,
               quantized_linear_bpw=payload * 8 / quant_params, retained_bytes=retained, active_parameters=active, whole_model_bpw=(payload + retained) * 8 / active, norm_weights_replaced=norm_replaced,
               note='retained = all other inference tensors at stored BF16 (lm_head excluded) - conservative, as the LIBERO rows')
    dump(out / 'accounting.json', acc)
    print(json.dumps(acc), flush=True)
    return acc


def stage_offline(rows, device='cuda'):
    from bridge_calibration import load_observations
    _, observations = load_observations(data_dir=HELDOUT)
    assert len(observations) == 256
    d = SB / 'offline'
    chunks = {}
    for row in rows:
        f = d / f'{row}.npy'
        if f.exists():
            chunks[row] = np.load(f)
            continue
        policy = Policy(device=device) if row == 'fp' else Policy(device=device, model_dir=str(MODELS / row / SETTING))
        chunks[row] = forward_chunks(policy, observations)
        d.mkdir(parents=True, exist_ok=True)
        np.save(f, chunks[row])
        del policy
        torch.cuda.empty_cache()
    fp = chunks['fp']
    gt = np.stack([o['action'] for _, o in observations])                       # the dataset action at the observation's frame (first step of the chunk)
    var = fp.reshape(-1, 7).var(axis=0)
    res = {}
    for row, c in chunks.items():
        res[row] = dict(mse=float(((c - fp) ** 2).mean()), nmse=float((((c - fp) ** 2).mean(axis=(0, 1)) / var).mean()), per_dim_mse=((c - fp) ** 2).mean(axis=(0, 1)).round(6).tolist(),
                        first_step_mse_vs_dataset=float(((c[:, 0] - gt) ** 2).mean()), gripper_agreement=float(((c[..., 6] > 0.5) == (fp[..., 6] > 0.5)).mean()))
    dump(d / f'offline_{torch.cuda.get_device_name().replace(" ", "_")}.json', dict(setting=SETTING, observations=256, rows=res, fp_action_variance=var.round(6).tolist()))
    print(json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('prepare', 'smoke', 'calib', 'fit', 'assemble', 'offline'), required=True)
    p.add_argument('--parts', nargs='+', choices=('backbone', 'head'), default=['backbone', 'head'])
    p.add_argument('--part', choices=('backbone', 'head'), default='head')
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--row', choices=tuple(ROWS), default='tcq_w2_rht')
    p.add_argument('--rows', default='fp:tcq_w2_rht')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.stage == 'prepare':
        stage_prepare()
    elif a.stage == 'smoke':
        stage_smoke(a.device)
    elif a.stage == 'calib':
        stage_calib(a.parts, a.device)
    elif a.stage == 'fit':
        stage_fit(a.part, a.worker, a.workers, a.rounds, a.device)
    elif a.stage == 'assemble':
        stage_assemble(a.row)
    elif a.stage == 'offline':
        stage_offline(a.rows.split(':'), a.device)


if __name__ == '__main__':
    main()
