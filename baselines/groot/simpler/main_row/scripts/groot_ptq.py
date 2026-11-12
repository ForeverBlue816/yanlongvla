"""GR00T N1.7 2-bit PTQ (evaluation-GPU plan, Day 2-4), runs in envs/groot.

Recipe = the π0.5 plain-VQ row (`story_plain`, 702/1000 on Plus vs 704 for the Fisher-weighted base): per Linear, deterministic
RHT (layer_signs(name) + block Walsh, backbone_rht), layer-Hessian group-8 GPTQ-encoded 2-plane RVQ (2 x 256 x 8 codebooks,
FP16 per-row scale; vq_fit.fit_uniform). Calibration: the same 256 LIBERO training observations (60 episodes) as every π0.5 row;
DiT Hessians accumulate over all 4 flow steps; noise seed SeedSequence([7, index]).

Quantized (432 Linear): backbone vision tower 104 + LLM layers 0-15 112 (identical in all four suite checkpoints: quantized
once, Hessian from the libero_spatial model); action head vl_self_attention 24 + DiT attention/FFN 192 (per suite).
Kept: DiT AdaLN linears (norm1.linear, 32) and the AdaLN-out proj_out_1 depend only on the flow timestep; with the fixed
4-step schedule their outputs are per-step constants (precomputed FP32 tables in deployment, as for π0.5's AdaRMS); proj_out_2,
per-embodiment encoders/decoders, embeddings, norms stay BF16; lm_head is unused and dropped from accounting.

Stages:
  calib  --suite S [--part backbone|head]   Hessians E[x x^T] per Linear (FP32) -> OUT/groot/ptq/hessians/<part or suite>/<layer>.pt
  fit    --suite S --part P --worker w --workers n    per-layer RVQ fit -> OUT/groot/ptq/fit/<key>/<layer>.pt (+ .json meta)
  assemble --suite S --row R                          fake-quant checkpoint models/groot_ptq/<row>/<suite> (dequantized BF16 weights
                                                      in the original basis) + accounting.json; rows: ptq_w2 (both), ptq_bb (backbone
                                                      only), ptq_head (action head only)
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

from tcq_bits import TAG as _TAG

ROOT = Path(os.environ.get('VLA_ROOT', '/projects/yanlongvla'))          # hand-over copy: the root is taken from VLA_ROOT
OUT = ROOT / f'runs/plustrain_20261008/groot{_TAG}/ptq'          # TCQ_BITS != 2: its own tree (groot_w1/...)
MODELS = ROOT / f'models/groot_ptq{_TAG}'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
BACKBONE_PREFIXES = ('backbone.model.model.visual.', 'backbone.model.model.language_model.layers.')
HEAD_PREFIXES = ('action_head.vl_self_attention.', 'action_head.model.transformer_blocks.')
EXCLUDE = ('norm1.linear', 'proj_out', 'lm_head', 'embed')
ROWS = dict(ptq_w2=('backbone', 'head'), ptq_bb=('backbone',), ptq_head=('head',),
            raqft_bb=('backbone_raqft', 'head'), distill_bb=('backbone_distill', 'head'),   # groot_raqft.py: fine-tuned backbone codebooks + 2-bit heads
            raqft_am_bb=('backbone_am', 'head'), raqft_am0_bb=('backbone_am0', 'head'),
            raqft_rowb=('backbone_rowb', 'head_rowb'), raqft_rowb2=('backbone_rowb2', 'head_rowb2'),   # row B2 (user 2026-10-27): student-head readouts + β·L_vel
            raqft_b_am=('backbone_bam', 'head_bam'),
            ptq_w2_v2=('backbone_v2', 'head_v2'),
            tcq_w2=('backbone_tcq', 'head_tcq'), tcq_raqft=('backbone_tcq_raqft', 'head_tcq_raqft'), tcq_w2_g16=('backbone_tcq_g16', 'head_tcq_g16'), tcq_raqft_g16=('backbone_tcq_raqft_g16', 'head_tcq_raqft_g16'), tcq_w2_rht=('backbone_tcq_rht', 'head_tcq_rht'), tcq_w2_rht_g16=('backbone_tcq_rht_g16', 'head_tcq_rht_g16'), tcq_raqft_rht=('backbone_tcq_raqft_rht', 'head_tcq_raqft_rht'), tcq_raqft_rht_g16=('backbone_tcq_raqft_rht_g16', 'head_tcq_raqft_rht_g16'), tcq_raqft_rht_s=('backbone_tcq_raqft_rht_s', 'head_tcq_raqft_rht_s'), tcq_w2_rht_iso=('backbone_tcq_rht_iso', 'head_tcq_rht_iso'), tcq_w2_rht_shift=('backbone_tcq_rht_shift', 'head_tcq_rht_shift'), tcq_w2_rht_d1=('backbone_tcq_rht_d1', 'head_tcq_rht_d1'), tcq_w2_rht_d2=('backbone_tcq_rht_d2', 'head_tcq_rht_d2'), tcq_raqft_c_s=('backbone_tcq_raqft_c_s', 'head_tcq_raqft_c_s'), tcq_raqft_d1_s=('backbone_tcq_raqft_d1_s', 'head_tcq_raqft_d1_s'), tcq_raqft_d2_s=('backbone_tcq_raqft_d2_s', 'head_tcq_raqft_d2_s'),   # groot_tcq.py: trellis-coded base + B2 recipe on it (user 2026-11-01)         # 2026-11-01 base audit: base fits with the flagged layers refit at 10x / 100x damping (stage v2)
            amvq_w2=('backbone_amvq', 'head_amvq'),      # groot_amvq.py: action-metric codebook fit of the base (user 2026-10-26)
            raqft_rowb_amvq=('backbone_rowb_amvq', 'head_rowb_amvq'), raqft_b_am_amvq=('backbone_bam_amvq', 'head_bam_amvq'))   # conditional fine-tune from the AM-VQ base      # groot_amraq.py --mode b_am: row B's trainable set with the action metric (user amendment 2026-10-25)   # groot_rowb.py: fine-tuned backbone + per-suite fine-tuned DiT (user 2026-10-26)   # groot_amraq.py: action-metric readout consistency (β=0.1 / β=0)


def part_of(name):
    if name.startswith(BACKBONE_PREFIXES):
        return 'backbone'
    if name.startswith(HEAD_PREFIXES):
        return 'head'
    return None


def target_layers(model):
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and part_of(name) and not any(e in name for e in EXCLUDE):
            out[name] = m
    return out


def key_of(suite, part):
    return 'backbone' if part == 'backbone' else f'head_{suite}'


def atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.partial{os.getpid()}')
    torch.save(obj, tmp)
    tmp.replace(path)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.partial{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=1) + '\n')
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------------------------- calib
def stage_calib(suite, parts):
    """Locked and idempotent: parallel jobs call it; the first one calibrates, the others wait and skip."""
    import fcntl
    lockfile = OUT / 'hessians' / f'calib_{suite}.lock'
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    with lockfile.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        todo = tuple(p for p in parts if not (OUT / 'hessians' / key_of(suite, p) / 'complete.json').exists())
        if not todo:
            print(json.dumps(dict(calib=suite, parts=parts, skipped='already complete')), flush=True)
            return
        _calibrate(suite, todo)


def _calibrate(suite, parts):
    from groot_bridge_server import Bridge
    from calibration_data import load_observations
    bridge = Bridge(suite)
    model = bridge.policy.model
    layers = {n: m for n, m in target_layers(model).items() if part_of(n) in parts}
    acc = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device='cuda') for n, m in layers.items()}
    tokens = {n: 0 for n in layers}
    handles = []
    for n, m in layers.items():
        def hook(mod, inp, n=n):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            acc[n].addmm_(x.t(), x)
            tokens[n] += x.shape[0]
        handles.append(m.register_forward_pre_hook(hook))
    _, observations = load_observations()
    assert len(observations) == 256
    t0 = time.time()
    with torch.inference_mode():
        for i, (_, obs) in enumerate(observations):
            o = dict(obs)
            o['_vla_episode_seed'] = int(np.random.SeedSequence([7, i]).generate_state(1)[0])
            bridge.infer(o)
    for h in handles:
        h.remove()
    for part in parts:
        directory = OUT / 'hessians' / key_of(suite, part)
        for n in [n for n in layers if part_of(n) == part]:
            atomic_save(dict(hessian=(acc[n] / max(tokens[n], 1)).cpu(), tokens=tokens[n], suite=suite), directory / f'{n}.pt')
        dump(directory / 'complete.json', dict(status='complete', suite=suite, part=part, layers=sum(part_of(n) == part for n in layers), observations=256, seconds=time.time() - t0,
                                              device=torch.cuda.get_device_name(), tokens_example={n: tokens[n] for n in list(layers)[:3]}))
    print(json.dumps(dict(calib=suite, parts=parts, layers=len(layers), seconds=round(time.time() - t0, 1))), flush=True)


# ---------------------------------------------------------------------------------------------------------------------- fit
def weights_for(suite, names):
    from safetensors import safe_open
    from groot_bridge_server import snapshot_dir, verify_blobs
    src = snapshot_dir() / suite
    verify_blobs(src)
    index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
    out = {}
    by_shard = {}
    for n in names:
        by_shard.setdefault(index[n + '.weight'], []).append(n)
    for shard, ns in by_shard.items():
        with safe_open(src / shard, 'pt') as f:
            for n in ns:
                out[n] = f.get_tensor(n + '.weight')
    return out


def inverse_rht(wp, signs):
    from rht import hadamard
    from backbone_rht import block_widths
    parts = wp.split(block_widths(wp.shape[-1]), dim=-1)
    return torch.cat([hadamard(p) for p in parts], dim=-1) * signs.to(wp.dtype)


def stage_fit(suite, part, worker, workers):
    from vq_fit import fit_uniform
    from vq_core import subset_reconstruction
    from rht import layer_signs
    from backbone_rht import transform, transform_covariance
    key = key_of(suite, part)
    hdir = OUT / 'hessians' / key
    for _ in range(720):                       # a parallel job may still be calibrating: wait up to 6 h
        if (hdir / 'complete.json').exists():
            break
        time.sleep(30)
    assert (hdir / 'complete.json').exists(), f'calibrate {key} first'
    names = sorted(p.name[:-3] for p in hdir.glob('*.pt'))
    mine = [n for i, n in enumerate(names) if i % workers == worker]
    weights = weights_for('libero_spatial' if part == 'backbone' else suite, mine)
    directory = OUT / 'fit' / key
    t_all = time.time()
    for n in mine:
        if (directory / f'{n}.json').exists():
            continue
        t0 = time.time()
        w = weights[n].cuda().float()
        h = torch.load(hdir / f'{n}.pt', map_location='cuda')['hessian'].float()
        signs = layer_signs(n, w.shape[1]).cuda()
        wp = transform(w, signs)
        hp = transform_covariance(h, signs)
        tensors, meta = fit_uniform(wp, hp, planes=2, name=n)
        dec = subset_reconstruction(tensors['codebooks'].float(), tensors['codes'].permute(1, 2, 0), [True, True]).reshape(w.shape) * tensors['row_scale'].float()[:, None]
        w_hat = inverse_rht(dec, signs)
        rel = float((w_hat - w).square().sum() / w.square().sum())
        herr = float(torch.sum(((w_hat - w) @ h) * (w_hat - w)) / torch.sum((w @ h) * w).clamp_min(1e-30))
        payload = sum(t.numel() * t.element_size() for t in tensors.values()) + signs.numel()   # codes, codebooks, row scale, mask + int8 signs
        atomic_save(dict(w_hat=w_hat.to(torch.bfloat16).cpu(), tensors={k: v.cpu() for k, v in tensors.items()}, signs=signs.to(torch.int8).cpu()), directory / f'{n}.pt')
        dump(directory / f'{n}.json', dict(name=n, shape=list(w.shape), payload_bytes=payload, relative_weight_mse=rel, hessian_relative_error=herr,
                                          fit=dict((k, meta[k]) for k in ('selected_round', 'hessian_relative_error') if k in meta), seconds=time.time() - t0))
        print(json.dumps(dict(layer=n, shape=list(w.shape), rel_mse=round(rel, 5), h_err=round(herr, 5), seconds=round(time.time() - t0, 1))), flush=True)
        del w, h, wp, hp, tensors, dec, w_hat
        torch.cuda.empty_cache()
    done = len(list(directory.glob('*.json'))) - (1 if (directory / 'complete.json').exists() else 0)
    if done == len(names):
        metas = [json.loads((directory / f'{n}.json').read_text()) for n in names]
        dump(directory / 'complete.json', dict(status='complete', key=key, layers=len(names), payload_bytes=sum(m['payload_bytes'] for m in metas),
                                              parameters=sum(m['shape'][0] * m['shape'][1] for m in metas),
                                              relative_weight_mse_mean=float(np.mean([m['relative_weight_mse'] for m in metas])), hessian_relative_error_mean=float(np.mean([m['hessian_relative_error'] for m in metas]))))
    print(json.dumps(dict(fit=key, worker=worker, layers=len(mine), seconds=round(time.time() - t_all, 1))), flush=True)


# ------------------------------------------------------------------------------------------------------------------ assemble
def _fit_one(n, w, h, damping):
    from vq_fit import fit_uniform
    from vq_core import subset_reconstruction
    from rht import layer_signs
    from backbone_rht import transform, transform_covariance
    signs = layer_signs(n, w.shape[1]).cuda()
    wp = transform(w, signs)
    hp = transform_covariance(h, signs)
    tensors, meta = fit_uniform(wp, hp, planes=2, name=n, damping=damping)
    dec = subset_reconstruction(tensors['codebooks'].float(), tensors['codes'].permute(1, 2, 0), [True, True]).reshape(w.shape) * tensors['row_scale'].float()[:, None]
    w_hat = inverse_rht(dec, signs)
    rel = float((w_hat - w).square().sum() / w.square().sum())
    herr = float(torch.sum(((w_hat - w) @ h) * (w_hat - w)) / torch.sum((w @ h) * w).clamp_min(1e-30))
    d = h.diagonal(); k = int(d.argmax()); mask = torch.ones_like(d); mask[k] = 0
    h2 = h * mask[:, None] * mask[None, :]
    herr_ex = float(torch.sum(((w_hat - w) @ h2) * (w_hat - w)) / torch.sum((w @ h2) * w).clamp_min(1e-30))
    payload = sum(t.numel() * t.element_size() for t in tensors.values()) + signs.numel()
    record = dict(w_hat=w_hat.to(torch.bfloat16).cpu(), tensors={k: v.cpu() for k, v in tensors.items()}, signs=signs.to(torch.int8).cpu())
    info = dict(name=n, shape=list(w.shape), payload_bytes=payload, relative_weight_mse=rel, hessian_relative_error=herr, hessian_relative_error_excluding_max_diag_channel=herr_ex,
                max_diag_channel=k, diag_max_over_mean=float(d.max() / d.mean()), damping=damping,
                fit=dict((k, meta[k]) for k in ('selected_round', 'hessian_relative_error') if k in meta))
    return record, info


def stage_refit(layers_file, dampings):
    """2026-11-01 base audit: refit the listed layers (json list of names, or {key: [names]}) with other relative dampings
    (GPTQ feedback; the base used 0.01 x mean diagonal). Output OUT/fit/refit_d<damping>/<key>/<layer>.pt + .json."""
    spec = json.loads(Path(layers_file).read_text())
    names = spec if isinstance(spec, list) else sorted({n for v in spec.values() for n in v})
    by_key = {}
    for n in names:
        if part_of(n) == 'backbone':
            by_key.setdefault('backbone', []).append(n)
        else:
            for suite in SUITES if isinstance(spec, list) else [k.removeprefix('head_') for k, v in spec.items() if n in v]:
                by_key.setdefault(f'head_{suite}', []).append(n)
    t_all = time.time()
    for key, mine in by_key.items():
        suite = 'libero_spatial' if key == 'backbone' else key.removeprefix('head_')
        weights = weights_for(suite, mine)
        for damping in dampings:
            directory = OUT / 'fit' / f'refit_d{damping:g}' / key
            for n in mine:
                if (directory / f'{n}.json').exists():
                    continue
                t0 = time.time()
                w = weights[n].cuda().float()
                h = torch.load(OUT / 'hessians' / key / f'{n}.pt', map_location='cuda')['hessian'].float()
                record, info = _fit_one(n, w, h, damping)
                info['seconds'] = time.time() - t0
                atomic_save(record, directory / f'{n}.pt')
                dump(directory / f'{n}.json', info)
                print(json.dumps(dict(refit=n, key=key, damping=damping, h_err=round(info['hessian_relative_error'], 6), h_err_ex=round(info['hessian_relative_error_excluding_max_diag_channel'], 6), rel_mse=round(info['relative_weight_mse'], 5), seconds=round(info['seconds'], 1))), flush=True)
                del w, h, record
                torch.cuda.empty_cache()
    print(json.dumps(dict(refit='done', layers=len(names), dampings=list(dampings), seconds=round(time.time() - t_all, 1))), flush=True)


def stage_v2(selection_file):
    """Compose fit/backbone_v2 and fit/head_v2_<suite>: the base fit for every layer, except the layers in the selection file
    ({layer name: damping} or {key: {name: damping}}) which come from refit_d<damping>. Symlinks; complete.json + v2_selection.json."""
    spec = json.loads(Path(selection_file).read_text())
    for key in ['backbone'] + [f'head_{s}' for s in SUITES]:
        chosen = spec.get(key, {}) if any(k in spec for k in ('backbone', 'head_libero_spatial')) else {n: d for n, d in spec.items() if (part_of(n) == 'backbone') == (key == 'backbone')}
        base = OUT / 'fit' / key
        out = OUT / 'fit' / ('backbone_v2' if key == 'backbone' else f'head_v2_{key.removeprefix("head_")}')
        out.mkdir(parents=True, exist_ok=True)
        names = sorted(p.name[:-3] for p in base.glob('*.pt'))
        replaced = {}
        for n in names:
            src_dir = base
            if n in chosen:
                src_dir = OUT / 'fit' / f'refit_d{chosen[n]:g}' / key
                assert (src_dir / f'{n}.json').exists(), (key, n, chosen[n])
                replaced[n] = chosen[n]
            for ext in ('.pt', '.json'):
                dst = out / f'{n}{ext}'
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src_dir / f'{n}{ext}')
        metas = [json.loads((out / f'{n}.json').read_text()) for n in names]
        dump(out / 'v2_selection.json', dict(key=key, replaced=replaced, base=str(base)))
        dump(out / 'complete.json', dict(status='complete', key=key, layers=len(names), payload_bytes=sum(m['payload_bytes'] for m in metas),
                                         parameters=sum(m['shape'][0] * m['shape'][1] for m in metas), replaced=len(replaced),
                                         relative_weight_mse_mean=float(np.mean([m['relative_weight_mse'] for m in metas])), hessian_relative_error_mean=float(np.mean([m['hessian_relative_error'] for m in metas]))))
        print(json.dumps(dict(v2=key, layers=len(names), replaced=replaced)), flush=True)


def stage_assemble(suite, row):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from groot_bridge_server import snapshot_dir, verify_blobs
    parts = ROWS[row]
    src = snapshot_dir() / suite
    verify_blobs(src)
    out = MODELS / row / suite
    if (out / 'accounting.json').exists():
        return json.loads((out / 'accounting.json').read_text())
    fits = {}
    for part in parts:
        d = OUT / 'fit' / (part if part.startswith('backbone_') else (f'{part}_{suite}' if part.startswith('head_') else key_of(suite, part)))
        assert (d / 'complete.json').exists(), f'fit {d} incomplete'
        fits.update({p.name[:-3]: p for p in d.glob('*.pt')})
    index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}
    for shard in sorted(set(index.values())):
        with safe_open(src / shard, 'pt') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    payload = 0
    replaced = 0
    for n, p in fits.items():
        rec = torch.load(p, map_location='cpu')
        k = n + '.weight'
        assert tensors[k].shape == rec['w_hat'].shape
        tensors[k] = rec['w_hat'].to(tensors[k].dtype)
        payload += json.loads(p.with_suffix('.json').read_text())['payload_bytes']
        replaced += 1
    norm_replaced = 0
    for part in parts:                                   # tcq_raqft (2026-11-01): fine-tuned RMSNorm / LayerNorm weights, stored at their BF16 size (retained bytes unchanged)
        d = OUT / 'fit' / (part if part.startswith('backbone_') else (f'{part}_{suite}' if part.startswith('head_') else key_of(suite, part)))
        if (d / 'norm_weights.safetensors').exists():
            with safe_open(str(d / 'norm_weights.safetensors'), 'pt') as f:
                for k in f.keys():
                    assert k in tensors and tensors[k].shape == f.get_tensor(k).shape, k
                    tensors[k] = f.get_tensor(k).to(tensors[k].dtype); norm_replaced += 1
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'model.safetensors.partial'
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp), metadata={'format': 'pt'})
    tmp.replace(out / 'model.safetensors')
    for f in src.iterdir():
        if f.name.startswith('model') or f.name in ('config.json',):
            continue
        dst = out / f.name
        if not dst.exists():
            dst.symlink_to(f)
    shutil.copy(src / 'config.json', out / 'config.json')
    # accounting (deployable): quantized payload + every other tensor used by action inference at its stored dtype,
    # except the unused lm_head and the 31 unused embodiment categories are kept (conservative); AdaLN linears counted at
    # their BF16 weight size (tables would be smaller)
    quantized_keys = {n + '.weight' for n in fits}
    retained = sum(v.numel() * v.element_size() for k, v in tensors.items() if k not in quantized_keys and 'lm_head' not in k)
    active = sum(v.numel() for k, v in tensors.items() if 'lm_head' not in k)
    quant_params = sum(tensors[k].numel() for k in quantized_keys)
    acc = dict(row=row, suite=suite, parts=list(parts), quantized_layers=replaced, norm_weights_replaced=norm_replaced, quantized_parameters=quant_params, linear_payload_bytes=payload,
               quantized_linear_bpw=payload * 8 / quant_params, retained_bytes=retained, active_parameters=active,
               whole_model_bpw=(payload + retained) * 8 / active,
               note='retained = all other inference tensors at stored BF16 (embedding not int8 here; AdaLN linears not replaced by tables) - conservative')
    dump(out / 'accounting.json', acc)
    print(json.dumps(acc), flush=True)
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('calib', 'fit', 'assemble', 'refit', 'v2'), required=True)
    p.add_argument('--layers-file', help='refit: json list of layer names (or {key: [names]})')
    p.add_argument('--damping', type=float, nargs='+', default=[0.1, 1.0], help='refit: relative dampings (base 0.01)')
    p.add_argument('--selection-file', help='v2: json {layer: damping}')
    p.add_argument('--suite', choices=SUITES, default='libero_spatial')
    p.add_argument('--part', choices=('backbone', 'head', 'both'), default='both')
    p.add_argument('--row', choices=tuple(ROWS), default='ptq_w2')
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.stage == 'calib':
        stage_calib(a.suite, ('backbone', 'head') if a.part == 'both' else (a.part,))
    elif a.stage == 'fit':
        stage_fit(a.suite, a.part, a.worker, a.workers)
    elif a.stage == 'refit':
        stage_refit(a.layers_file, a.damping)
    elif a.stage == 'v2':
        stage_v2(a.selection_file)
    else:
        stage_assemble(a.suite, a.row)


if __name__ == '__main__':
    main()
