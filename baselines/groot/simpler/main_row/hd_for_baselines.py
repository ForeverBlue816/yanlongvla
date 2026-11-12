"""Our source-balanced Hessians H_D (the calibration of the main row) in the calibration format of the baseline harness, so that
the 2-bit baselines (QTIP, QuIP#, VPTQ, AQLM, GPTQ, AWQ) can be quantized on exactly the Hessians our row is quantized on.

  python hd_for_baselines.py --hd $VLA_ROOT/runs/plustrain_20261008/groot/ptq/hessians --like <calibration> --out <calibration_hd>

  --hd     the directory that holds backbone_hd1_simpler_bridge/ and head_hd1_simpler_bridge/ (written by build_hd.sh)
  --like   the clean Bridge calibration of the baseline harness (GROOT_PROFILE=bridge python ../../calibrate.py ...): its index.json
           gives the layer list, the shapes and the checkpoint hash that the external quantizers check
  --out    <out>/backbone and <out>/head_bridge, same layout as --like (hessians/<layer>.pt with `covariance`, index.json)

H_D = H_clean + sum_s w_s H_s, w_s = tr(H_clean) / (3 tr(H_s)), s in {photometric, geometric, on-policy}: tr(H_D) = 2 tr(H_clean).
The script checks every layer (name, shape, symmetry, finite) and writes tr(H_D) / tr(H_clean of --like) per layer to <out>/summary.json;
the ratio should be close to 2 (it is not exactly 2: --like comes from your GPU and FP64 accumulators, our H_clean from ours)."""
import argparse
import json
from pathlib import Path

import torch

KEYS = {'backbone': 'backbone_hd1_simpler_bridge', 'head_bridge': 'head_hd1_simpler_bridge'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hd', required=True)
    p.add_argument('--like', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    summary = {}
    for key, ours in KEYS.items():
        like, src, out = Path(a.like) / key, Path(a.hd) / ours, Path(a.out) / key
        assert json.loads((src / 'complete.json').read_text())['status'] == 'complete', src
        index = json.loads((like / 'index.json').read_text())
        assert len(index['layers']) == 216 and len(list(src.glob('*.pt'))) == 216, (key, len(index['layers']))
        (out / 'hessians').mkdir(parents=True, exist_ok=True)
        layers, ratios = {}, {}
        for name, entry in sorted(index['layers'].items()):
            d = torch.load(src / f'{name}.pt', map_location='cpu')
            h = d['hessian'].float()
            n = entry['in_features']
            assert h.shape == (n, n) and torch.isfinite(h).all() and torch.allclose(h, h.T, rtol=1e-4, atol=1e-6 * float(h.diagonal().mean())), name
            clean = torch.load(like / entry['file'], map_location='cpu', weights_only=True)['covariance']
            ratios[name] = float(h.diagonal().double().sum() / clean.diagonal().double().sum())
            path = out / 'hessians' / (name.replace('.', '_') + '.pt')
            torch.save(dict(name=name, covariance=h.contiguous(), samples=int(d['tokens']), source_weights=d['weights'], source_traces=d['traces'],
                            note='H_D = H_clean + sum_s w_s H_s, w_s = tr(H_clean) / (3 tr(H_s)), s in photometric, geometric, on-policy (main_row/README.md); FP model activations'), path)
            layers[name] = dict(entry, file=str(path.relative_to(out)), samples=int(d['tokens']))
        r = sorted(ratios.values())
        summary[key] = dict(layers=len(layers), trace_ratio_to_clean=dict(min=r[0], median=r[len(r) // 2], max=r[-1]), per_layer=ratios)
        (out / 'index.json').write_text(json.dumps(dict(index, layers=layers, hessian='H_D (source-balanced, round 1)', clean_calibration=str(like),
                                                        note='checkpoint_sha256, suite, part and the layer list are those of the clean calibration; only the Hessians differ'), indent=1) + '\n')
        print(json.dumps(dict(converted=key, layers=len(layers), trace_ratio_to_clean=summary[key]['trace_ratio_to_clean'])), flush=True)
        assert 1.6 < r[0] and r[-1] < 2.4, f'{key}: tr(H_D) / tr(H_clean) outside [1.6, 2.4]: the two calibrations do not describe the same model inputs'
    (Path(a.out) / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
