"""GR00T N1.7 SimplerEnv-Bridge, 2-bit MAIN ROW (user 2026-11-11): the LIBERO main-row recipe tcq_raqft_d1_s ported to the Bridge
checkpoint. Nothing in the recipe changes; only the LIBERO-specific plumbing (four per-suite checkpoints, two cameras, LIBERO
simulator) is replaced by the Bridge one (one flat checkpoint, one camera, SimplerEnv).

  H_D      = H_clean + sum_s w_s H_s, w_s = tr(H_clean) / (3 tr(H_s)), s in {photometric, geometric, on-policy}  (groot_shift.stage_hsrc / stage_hd)
             H_clean      = groot_simpler.py calib (256 Bridge calibration observations)
             photometric  = copies j = 1, 2 of each calibration observation (512), geometric = copy j = 3 (256): shift_perturb stream 0
             on-policy    = 256 states of the clean-H 2-bit row tcq_w2_rht in SimplerEnv (simpler_onpolicy_client.py; user 2026-11-11:
                            captured in SimplerEnv, i.e. on evaluation layouts — recorded)
             per-observation normalisation of every source (as groot_shift.stage_hsrc); FP model activations; noise seeds as on LIBERO
  base     tcq_w2_rht_d1 = TCQ two-sided (groot_tcq.fit_layer, rotation 'two', 1 round) on H_D, damping pinned to tcq_w2_rht's per layer
  main row tcq_raqft_d1_s = V0' stabilised readout fine-tune (groot_tcq_raqft_s.run, unchanged code) on that base

Stages (envs/groot, PYTHONPATH=src/Isaac-GR00T:research/scripts, TCQ_BITS unset = 2):
  hsrc                 per-source second moments -> ptq/hessians/{backbone,head}_src1_<source>_simpler_bridge
  hd                   H_D -> ptq/hessians/{backbone,head}_hd1_simpler_bridge
  fit --part P [--worker w --workers n]   -> ptq/fit/{backbone,head}_tcq_rht_d1_simpler_bridge
  assemble --row R     models/groot_ptq/<R>/simpler_bridge (R = tcq_w2_rht_d1 | tcq_raqft_d1_s; fine-tuned norm weights applied)
  finetune [--smoke]   groot_tcq_raqft_s.run with the Bridge adapters below -> ptq/fit/{backbone,head}_tcq_raqft_d1_s_simpler_bridge
  offline --rows ...   held-out chunk MSE vs FP (groot_simpler.stage_offline)

Fine-tune port (the only deviations, all forced by the checkpoint):
  - one "suite" (simpler_bridge): one teacher / student pair, no head routing; hold-out = 40 of the 256 observations drawn with the same
    rng (SeedSequence([20261104, 0])) without the per-suite stratification (there is one stratum);
  - L_vel window: executed = 4 of the 8-step Bridge chunk (LIBERO: 8 of 16), valid dims 7;
  - observations: one camera (image_0), 8-d state x, y, z, roll, pitch, yaw, pad, gripper.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import groot_simpler as GS                                                           # noqa: E402
from groot_ptq import OUT, MODELS, atomic_save, dump, part_of, target_layers          # noqa: E402
from tcq_bits import BITS                                                            # noqa: E402

S = GS.SETTING
ONPOLICY = GS.SB / 'onpolicy_rinit'
SOURCES = ('photometric', 'geometric', 'onpolicy')
SRC_KEY = lambda part, src: f'{part}_src1_{src}_{S}'
HD_KEY = dict(backbone=f'backbone_hd1_{S}', head=f'head_hd1_{S}')
D1_KEY = dict(backbone=f'backbone_tcq_rht_d1_{S}', head=f'head_tcq_rht_d1_{S}')
FT_KEY = dict(backbone=f'backbone_tcq_raqft_d1_s_{S}', head=f'head_tcq_raqft_d1_s_{S}')
FT_HEAD_PREFIX = 'head_tcq_raqft_d1_s'                                                # groot_tcq_raqft.export_fit appends _<suite>
RUNS = GS.SB / 'raqft'
GS.ROWS.update(tcq_w2_rht_d1=(D1_KEY['backbone'], D1_KEY['head']), tcq_raqft_d1_s=(FT_KEY['backbone'], FT_KEY['head']))


# ---------------------------------------------------------------------------------------------------------------------- sources
def onpolicy_observations():
    assert (ONPOLICY / 'complete.json').exists(), f'on-policy capture incomplete ({ONPOLICY})'
    out = []
    for i in range(256):
        meta = json.loads((ONPOLICY / f'obs_{i:03d}.json').read_text()); z = np.load(ONPOLICY / f'obs_{i:03d}.npz')
        out.append({'observation/image': z['image'], 'observation/state': z['state'], 'prompt': meta['prompt']})
    return out


def perturbed(obs, i, j):
    """shift_perturb.perturbed_copy on a one-camera observation: the same rng stream and parameters as on LIBERO (the parameters are
    drawn before the images are touched and applied to every camera)."""
    from shift_perturb import perturbed_copy
    out, _ = perturbed_copy(dict(obs, **{'observation/wrist_image': obs['observation/image']}), 0, i, j)
    out.pop('observation/wrist_image')
    return out


def stage_hsrc(sources=SOURCES, device='cuda'):
    from bridge_calibration import load_observations
    _, cal = load_observations(data_dir=GS.CAL); assert len(cal) == 256
    seed_of = lambda i: int(np.random.SeedSequence([7, i]).generate_state(1)[0])
    sets = dict(photometric=[(lambda i=i, j=j, obs=obs: perturbed(obs, i, j), seed_of(i)) for i, (_, obs) in enumerate(cal) for j in (1, 2)],
                geometric=[(lambda i=i, obs=obs: perturbed(obs, i, 3), seed_of(i)) for i, (_, obs) in enumerate(cal)])
    todo = [src for src in sources if not all((OUT / 'hessians' / SRC_KEY(p, src) / 'complete.json').exists() for p in ('backbone', 'head'))]
    if 'onpolicy' in todo:
        sets['onpolicy'] = [(lambda obs=obs: obs, int(np.random.SeedSequence([7, i, 1]).generate_state(1)[0])) for i, obs in enumerate(onpolicy_observations())]
    if not todo:
        print(json.dumps(dict(hsrc='complete'))); return
    policy = GS.Policy(device=device)
    layers = target_layers(policy.policy.model)
    assert len(layers) == 432
    for src in todo:
        assert len(sets[src]) == dict(photometric=512, geometric=256, onpolicy=256)[src]
        acc = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device=device) for n, m in layers.items()}
        tmp = {n: torch.zeros_like(a) for n, a in acc.items()}
        tokens = {n: 0 for n in layers}; total_tokens = {n: 0 for n in layers}
        handles = []
        for n, m in layers.items():
            def hook(mod, inp, n=n):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                tmp[n].addmm_(x.t(), x); tokens[n] += x.shape[0]
            handles.append(m.register_forward_pre_hook(hook))
        t0 = time.time()
        with torch.inference_mode():
            for make, seed in sets[src]:
                o = dict(make()); o['_vla_episode_seed'] = seed
                for n in layers: tmp[n].zero_(); tokens[n] = 0
                policy.infer(o)
                for n in layers:
                    acc[n].add_(tmp[n], alpha=1.0 / max(tokens[n], 1)); total_tokens[n] += tokens[n]
        for h in handles: h.remove()
        n_obs = len(sets[src])
        for part in ('backbone', 'head'):
            key = SRC_KEY(part, src)
            names = [n for n in layers if part_of(n) == part]
            for n in names:
                atomic_save(dict(hessian=(acc[n] / n_obs).cpu(), tokens=total_tokens[n], suite=S, source=src, round=1, observations=n_obs, weighting='equal per observation'), OUT / 'hessians' / key / f'{n}.pt')
            dump(OUT / 'hessians' / key / 'complete.json', dict(status='complete', key=key, setting=S, part=part, source=src, round=1, observations=n_obs, layers=len(names),
                                                                onpolicy_row='tcq_w2_rht (SimplerEnv rollouts)' if src == 'onpolicy' else None, seconds=time.time() - t0, device=torch.cuda.get_device_name()))
        print(json.dumps(dict(hsrc=S, source=src, layers=len(layers), seconds=round(time.time() - t0, 1))), flush=True)
        del acc, tmp; torch.cuda.empty_cache()


def stage_hd():
    summary = {}
    for part in ('backbone', 'head'):
        directory = OUT / 'hessians' / HD_KEY[part]
        if (directory / 'complete.json').exists():
            print(json.dumps(dict(hd=HD_KEY[part], skipped='complete')), flush=True); continue
        base = OUT / 'hessians' / GS.HKEY[part]
        srcs = {src: OUT / 'hessians' / SRC_KEY(part, src) for src in SOURCES}
        for d in [base] + list(srcs.values()):
            assert (d / 'complete.json').exists(), f'H incomplete: {d}'
        names = sorted(p.name[:-3] for p in base.glob('*.pt'))
        t0, recs = time.time(), []
        for n in names:
            b = torch.load(base / f'{n}.pt', map_location='cpu')
            hc = b['hessian'].double(); trc = float(hc.diagonal().sum())
            hd = hc.clone(); w, trs = {}, {}
            for src, d in srcs.items():
                hs = torch.load(d / f'{n}.pt', map_location='cpu')['hessian'].double()
                trs[src] = float(hs.diagonal().sum()); w[src] = trc / (3.0 * max(trs[src], 1e-300))
                hd.add_(hs, alpha=w[src])
            atomic_save(dict(hessian=hd.float(), tokens=b['tokens'], suite=S, recipe='H_clean + sum_s w_s H_s, w_s = tr(H_clean)/(3 tr(H_s))', round=1, weights=w, traces=dict(clean=trc, **trs)), directory / f'{n}.pt')
            recs.append(dict(name=n, tr_clean=trc, weights=w, traces=trs))
        dump(directory / 'hd_layers.json', recs)
        summary[HD_KEY[part]] = {src: float(np.mean([r['weights'][src] for r in recs])) for src in SOURCES}
        dump(directory / 'complete.json', dict(status='complete', key=HD_KEY[part], setting=S, part=part, round=1, layers=len(names), sources=list(SOURCES), mean_weights=summary[HD_KEY[part]],
                                              recipe='H_D = H_clean + sum_s w_s H_s, w_s = tr(H_clean) / (3 tr(H_s)); added trace = tr(H_clean)', seconds=time.time() - t0))
        print(json.dumps(dict(hd=HD_KEY[part], layers=len(names), mean_weights=summary[HD_KEY[part]], seconds=round(time.time() - t0, 1))), flush=True)


# -------------------------------------------------------------------------------------------------------------------------- fit
def stage_fit(part, worker, workers, device='cuda'):
    """groot_tcq.stage_fit(metric='mix', dround=1, rounds=1, rotation='two') for the Bridge checkpoint."""
    from groot_tcq import fit_layer, tlut_init, make_codebook, try_compile, engine, SETTINGS, DAMPINGS
    assert BITS == 2, 'the Bridge main row is the 2-bit row'
    hdir = OUT / 'hessians' / HD_KEY[part]
    assert (hdir / 'complete.json').exists(), f'{HD_KEY[part]} first'
    names = sorted(p.name[:-3] for p in hdir.glob('*.pt'))
    directory = OUT / 'fit' / D1_KEY[part]; directory.mkdir(parents=True, exist_ok=True)
    todo = [n for i, n in enumerate(names) if i % workers == worker and not (directory / f'{n}.json').exists()]
    weights = GS.weights_bridge(todo) if todo else {}
    tlut0 = tlut_init().to(device)
    cb = make_codebook(tlut0, device)
    compiled = try_compile(cb)
    t_all = time.time()
    for n in todo:
        if (directory / f'{n}.json').exists():
            continue
        w = weights[n].to(device).float()
        h = torch.load(hdir / f'{n}.pt', map_location=device)['hessian'].float()
        extra = dict(metric='mix', round=1, hessian_key=HD_KEY[part])
        a_damp = json.loads((OUT / 'fit' / GS.FIT_KEY[part] / f'{n}.json').read_text())['damping']       # D differs from the clean-H row only in H: damping pinned to its value
        dampings = (a_damp,) + tuple(d for d in DAMPINGS if d > a_damp); extra['damping_pinned_to'] = f'{GS.FIT_KEY[part]}:{a_damp}'
        record, info = fit_layer(n, w, h, cb, tlut0, rounds=1, rotation='two', dampings=dampings)
        info.update(extra); record['tcq']['metric'] = 'mix'
        atomic_save(record, directory / f'{n}.pt')
        dump(directory / f'{n}.json', info)
        print(json.dumps(dict(layer=n, shape=info['shape'], rel_mse=round(info['relative_weight_mse'], 5), h_err=round(info['hessian_relative_error'], 6), damping=info['damping'], seconds=round(info['seconds'], 1))), flush=True)
        del w, h, record
        torch.cuda.empty_cache()
    done = [n for n in names if (directory / f'{n}.json').exists()]
    if len(done) == len(names) and not (directory / 'complete.json').exists():
        metas = [json.loads((directory / f'{n}.json').read_text()) for n in names]
        dump(directory / 'complete.json', dict(status='complete', key=D1_KEY[part], source_key=HD_KEY[part], setting=S, checkpoint=f'{GS.REPO}@{GS.REVISION}', layers=len(names),
                                              payload_bytes=sum(x['payload_bytes'] for x in metas), parameters=sum(x['shape'][0] * x['shape'][1] for x in metas),
                                              relative_weight_mse_mean=float(np.mean([x['relative_weight_mse'] for x in metas])), hessian_relative_error_mean=float(np.mean([x['hessian_relative_error'] for x in metas])),
                                              dampings={str(d): sum(x['damping'] == d for x in metas) for d in sorted({x['damping'] for x in metas})}, seconds=sum(x['seconds'] for x in metas),
                                              format=dict(SETTINGS, rotation='two', rounds=1), engine=dict(qtip_commit=engine()['commit'], compiled=compiled)))
    print(json.dumps(dict(fit=D1_KEY[part], worker=worker, layers=len(todo), seconds=round(time.time() - t_all, 1), compiled=compiled)), flush=True)


# --------------------------------------------------------------------------------------------------------------------- fine-tune
class BridgeAdapter:
    """groot_bridge_server.Bridge's interface (policy, observation, infer -> {'actions'}) on the SimplerEnv-Bridge checkpoint."""
    def __init__(self, suite, device='cuda', model_root=None):
        assert suite == S, suite
        self._p = GS.Policy(device=device, model_dir=str(Path(model_root) / S) if model_root else None)
        self.policy, self.backbone, self.model_dir, self.suite = self._p.policy, self._p.backbone, self._p.model_dir, suite

    def observation(self, obs):
        return GS.Policy.observation(obs)

    def infer(self, obs):
        return {'actions': self._p.infer(obs)}


def stage_finetune(smoke=False):
    """groot_tcq_raqft_s.run (V0', variant v0s_d1) with the LIBERO plumbing replaced: see the module docstring."""
    import calibration_data
    import groot_bridge_server
    import groot_tcq_raqft as R
    import groot_tcq_raqft_s as TS
    from bridge_calibration import load_observations
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / 'calibration_suites.json').write_text(json.dumps(dict(observation_suite=[S] * 256, note='one checkpoint: every Bridge calibration observation is routed to it')) + '\n')
    groot_bridge_server.Bridge = BridgeAdapter
    calibration_data.load_observations = lambda *args, **kw: load_observations(data_dir=GS.CAL)
    for mod in (R, TS):
        mod.SUITES = (S,); mod.RUNS = RUNS
    TS.HOLD = {S: 40}
    TS.CFG = dict(TS.CFG, executed=GS.N_ACTION_STEPS)                                  # L_vel window = the executed part of the chunk (4 of 8; LIBERO 8 of 16)
    R.VARIANTS['v0s_d1_bridge'] = dict(base_row='tcq_w2_rht_d1', key=FT_KEY['backbone'], head_prefix=FT_HEAD_PREFIX, row='tcq_raqft_d1_s',
                                       source={'backbone': D1_KEY['backbone'], f'head_{S}': D1_KEY['head'], 'head_libero_spatial': D1_KEY['head']})
    assert (MODELS / 'tcq_w2_rht_d1' / S / 'model.safetensors').exists(), 'assemble tcq_w2_rht_d1 first'
    TS.run(smoke=smoke, variant='v0s_d1_bridge')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('hsrc', 'hd', 'fit', 'assemble', 'finetune', 'offline'), required=True)
    p.add_argument('--part', choices=('backbone', 'head'), default='head')
    p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--row', choices=('tcq_w2_rht_d1', 'tcq_raqft_d1_s'), default='tcq_w2_rht_d1')
    p.add_argument('--rows', default='fp:tcq_w2_rht:tcq_w2_rht_d1:tcq_raqft_d1_s')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--sources', default=':'.join(SOURCES), help='hsrc: subset of sources (the perturbed copies do not need the on-policy capture)')
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    if a.stage == 'hsrc': stage_hsrc(tuple(a.sources.split(':')))
    elif a.stage == 'hd': stage_hd()
    elif a.stage == 'fit': stage_fit(a.part, a.worker, a.workers)
    elif a.stage == 'assemble': GS.stage_assemble(a.row)
    elif a.stage == 'finetune': stage_finetune(a.smoke)
    elif a.stage == 'offline': GS.stage_offline(a.rows.split(':'))


if __name__ == '__main__':
    main()
