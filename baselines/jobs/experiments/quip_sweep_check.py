"""Are QuIP#'s tuning sweeps (quip_tune_iters) broken here, or do they overfit by design? (mentor, 2026-10-26)

For a few layers, quantize with the official quip.quantize at several numbers of tuning sweeps and record, per run:
  loss_reg     tr(E Hreg E^T) / tr(W Hreg W^T) with Hreg = the damped, normalized Hessian the quantizer used
               (regularize_H: H / mean diagonal + sigma_reg on the diagonal). A sweep that minimizes this loss exactly
               never raises it, so a rise from one sweep to the next means the sweep does not do what it claims.
  weight       ||E|| / ||W||
  out_cal      tr(E H E^T) / tr(W H W^T) on the calibration Hessian (no damping)
  out_heldout  the same on the held-out Hessian
and, per layer, the eigenvalue spectrum of Hreg (how many eigenvalues lie above the damping floor, condition number).

Checks the mentor listed: (1) damping reaches the sweeps (they work on Hr = RHT(Hreg), which contains it), (2) the Hessian
normalization (H / mean diagonal, so sigma_reg is a fraction of the mean diagonal), (3) scale_override 0.9 against the
codebook's opt_scale 1.03 (also run at 1.03), and a larger sigma_reg as a candidate remedy (reported, never used).

Usage: python quip_sweep_check.py --layers L --calibration C --heldout H --quip-repo Q --out out.json
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import quantize_quip as qq   # noqa: E402

PICK = ['gemma_expert.model.layers.0.mlp.down_proj', 'gemma_expert.model.layers.0.mlp.gate_proj',
        'gemma_expert.model.layers.11.mlp.down_proj', 'language_model.layers.0.self_attn.k_proj']
SWEEPS = [0, 1, 2, 3, 5, 10]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--heldout', required=True)
    p.add_argument('--quip-repo', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from lib import codebook, utils
    from lib.algo import quip
    torch.set_grad_enabled(False)
    dev = torch.device('cuda:0')
    layers = Path(a.layers)
    listing = json.loads((layers / 'layers.json').read_text())['layers']
    held = json.loads((Path(a.heldout) / 'index.json').read_text())['layers']
    cb = codebook.get_codebook('E8P12')
    result = dict(codebook_opt_scale=float(cb.opt_scale), layers={})
    for key in PICK:
        item = next(i for i in listing if key in i['name'])
        src = load_file(str(layers / item['file']))
        w, h = src['weight'].float(), src['hessian'].double()
        hh = torch.load(Path(a.heldout) / held[item['name']]['file'], map_location='cpu', weights_only=True)['covariance'].double()
        m, n = w.shape
        assert qq.PAD_TO.get(m, m) == m and qq.PAD_TO.get(n, n) == n, 'padded layers are not part of this check'
        scale = w.square().mean().sqrt()
        runs = []

        def run(sigma_reg, sweeps, scale_override):
            hreg = utils.regularize_H(h.clone(), n, sigma_reg)
            torch.manual_seed(0)
            settings = dict(qq.SETTINGS, sigma_reg=sigma_reg, quip_tune_iters=sweeps, scale_override=scale_override)
            hat, _ = quip.quantize(hreg.float().to(dev), (w / scale).to(dev), 0, cb, Namespace(save_pfx='/tmp', **settings), dev)
            e = hat.float().cpu() * scale - w
            r = lambda hm: float(((e @ hm.float()) * e).sum() / ((w @ hm.float()) * w).sum())
            return dict(sigma_reg=sigma_reg, sweeps=sweeps, scale_override=scale_override, loss_reg=r(hreg), weight=float(e.norm() / w.norm()),
                        out_cal=r(h), out_heldout=r(hh))

        for s in SWEEPS:
            runs.append(run(0.01, s, 0.9))
        runs.append(run(0.01, 10, 1.03))
        for sr in (0.1, 1.0):
            runs.append(run(sr, 10, 0.9))
        hreg = utils.regularize_H(h.clone(), n, 0.01)
        ev = torch.linalg.eigvalsh(hreg.float()).clamp(min=0)
        spectrum = dict(above_damping_floor=int((ev > 0.01 * 1.5).sum()), n=n, largest=float(ev[-1]), smallest=float(ev[0]),
                        condition_number=float(ev[-1] / max(ev[0], 1e-12)))
        result['layers'][item['name']] = dict(shape=[m, n], spectrum=spectrum, runs=runs)
        mono = [r['loss_reg'] for r in runs[:len(SWEEPS)]]
        print(json.dumps(dict(layer=item['name'], spectrum=spectrum, loss_reg_by_sweeps=mono,
                              weight_by_sweeps=[r['weight'] for r in runs[:len(SWEEPS)]])), flush=True)
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
