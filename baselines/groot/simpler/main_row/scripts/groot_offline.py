"""GR00T held-out action MSE (AM-RAQ §3): actions of each assembled row on the 256 held-out observations (noise seed 0 per
observation), vs the FP model; total and per DoF group (translation / rotation / gripper) over the 16 valid steps x 7 dims.
Rows: base (ptq_w2), raqft_bb (A), raqft_am_bb (A-AM), raqft_am0_bb (A-AM0), distill_bb; FP is the reference. Also pairwise A-AM vs A.
Writes OUT/groot/offline/<row>/actions.npy and OUT/groot/offline/summary.json (the A-AM0 rollout gate reads it)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT as PTQ_OUT, ROOT, MODELS, dump, key_of
OUT = PTQ_OUT.parent.parent          # runs/plustrain_20261008 (the controller reads groot/offline/summary.json there)

HELDOUT = ROOT / 'datasets/libero-heldout'
GROUPS = dict(translation=[0, 1, 2], rotation=[3, 4, 5], gripper=[6])
ROWS = ('fp', 'ptq_w2', 'raqft_bb', 'raqft_am_bb', 'raqft_am0_bb', 'distill_bb', 'raqft_rowb', 'raqft_rowb2', 'raqft_b_am', 'amvq_w2', 'raqft_rowb_amvq', 'raqft_b_am_amvq', 'ptq_w2_v2', 'tcq_w2', 'tcq_raqft', 'tcq_w2_g16', 'tcq_raqft_g16', 'tcq_w2_rht', 'tcq_w2_rht_g16', 'tcq_raqft_rht', 'tcq_raqft_rht_g16', 'tcq_w2_rht_iso', 'tcq_w2_rht_shift', 'tcq_raqft_rht_s', 'tcq_w2_rht_d1', 'tcq_w2_rht_d2', 'tcq_raqft_c_s', 'tcq_raqft_d1_s', 'tcq_raqft_d2_s')


DEVICE = __import__('os').environ.get('GROOT_OFFLINE_DEVICE', 'cuda')
if DEVICE == 'cpu':
    sys.modules['flash_attn'] = None            # qwen3_backbone falls back to SDPA (identical BF16 arithmetic up to rounding; recorded per row)
ROWS = tuple(__import__('os').environ.get('GROOT_OFFLINE_ROWS', ':'.join(ROWS)).split(':'))   # ':'-separated (Slurm --export splits on commas)


def device_tag():
    if DEVICE == 'cpu':
        return 'cpu'
    return torch.cuda.get_device_name().replace('NVIDIA ', '').replace(' ', '_')


def actions_for(row, suite='libero_spatial'):
    from groot_bridge_server import Bridge
    from calibration_data import load_observations
    out = OUT / 'groot/offline' / device_tag() / row         # per device: rows are compared only within one device (2026-10-25)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'actions.npy').exists():
        return np.load(out / 'actions.npy')
    if row != 'fp':
        from groot_ptq import ROWS, stage_assemble
        fits = PTQ_OUT / 'fit'
        if not all(((fits / part) if part.startswith('backbone_') else (fits / f'{part}_{suite}') if part.startswith('head_') else (fits / key_of(suite, part))).joinpath('complete.json').exists() for part in ROWS[row]):
            return None
        stage_assemble(suite, row)
    bridge = Bridge(suite, device=DEVICE) if row == 'fp' else Bridge(suite, device=DEVICE, model_root=str(MODELS / row))
    _, observations = load_observations(data_dir=HELDOUT)
    assert len(observations) == 256
    keys = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')
    acts = []
    with torch.no_grad():
        for i, (_, obs) in enumerate(observations):
            torch.manual_seed(i); bridge.policy.reset()        # same noise seed per observation for every row (paired)
            action, _ = bridge.policy.get_action(bridge.observation(obs))
            get = lambda k: action[f'action.{k}'] if f'action.{k}' in action else action[k]
            chunk = np.concatenate([np.asarray(get(k)[0], dtype=np.float32) for k in keys], axis=-1)   # dataset convention, before the env mapping
            acts.append(chunk[:16, :7])
    a = np.stack(acts).astype(np.float32)
    np.save(out / 'actions.npy', a)
    (out / 'device.json').write_text(json.dumps(dict(device=DEVICE, gpu=torch.cuda.get_device_name() if DEVICE == 'cuda' else 'cpu')))
    del bridge
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return a


def mse(ref, cand):
    d = (ref - cand) ** 2
    res = dict(total=float(d.mean()), relative=float(d.sum() / max((ref ** 2).sum(), 1e-30)))
    for g, dims in GROUPS.items():
        res[g] = float(d[..., dims].mean())
    return res


def main():
    acts = {r: actions_for(r) for r in ROWS}
    if 'fp' not in acts or acts['fp'] is None:
        acts['fp'] = actions_for('fp')
    acts = {r: a for r, a in acts.items() if a is not None}
    summary = dict(observations=256, noise='torch.manual_seed(observation index) before each policy call, identical for every row', units='dataset action convention (gripper in [0,1]), 16 steps x 7 dims', rows=list(acts),
                   vs_fp={r: mse(acts['fp'], a) for r, a in acts.items() if r != 'fp'})
    if 'raqft_rowb' in acts and 'raqft_b_am' in acts:
        summary['b_am_vs_b'] = mse(acts['raqft_rowb'], acts['raqft_b_am'])
    if 'raqft_bb' in acts and 'raqft_am_bb' in acts:
        summary['am_vs_a'] = mse(acts['raqft_bb'], acts['raqft_am_bb'])
    if 'raqft_am_bb' in acts and 'raqft_am0_bb' in acts:
        a, b = summary['vs_fp']['raqft_am_bb']['total'], summary['vs_fp']['raqft_am0_bb']['total']
        summary['am0_rollout_gate'] = dict(am_total=a, am0_total=b, within_5_percent=bool(b <= 1.05 * a), rule='roll out A-AM0 only if its held-out MSE is within 5% of A-AM')
    summary['device'] = device_tag()
    dump(OUT / 'groot/offline' / device_tag() / 'summary.json', summary)
    # offline/summary.json = the per-device summary with the most rows (ties: the most recent); the controller and table read it
    import glob
    best = max((json.loads(Path(f).read_text()) for f in glob.glob(str(OUT / 'groot/offline/*/summary.json'))), key=lambda d: len(d['rows']))
    dump(OUT / 'groot/offline/summary.json', best if len(best['rows']) > len(summary['rows']) else summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
