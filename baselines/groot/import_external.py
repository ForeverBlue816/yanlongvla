"""Step 3 (external VQ toolkits): bring the five quantized layer sets back as four per-suite fake-quant checkpoints.

Input: the toolkit's output directories for the keys of export_layers.py, each with <layer>.safetensors (tensor `weight`,
dequantized, any float dtype) and accounting.json in the π0.5 harness format:
  {"method": "aqlm_2x8", "bits_nominal": 2, "layers": {"<layer name>": {"payload_bytes": N}, ...}, ...}
payload_bytes must count every stored tensor of the layer (codes/indices, codebooks, scales, zero points, signs, permutations).

Usage:  python import_external.py --checkpoints ./checkpoints --quantized ./aqlm_out --out ./ckpt_aqlm_w2
        (./aqlm_out/backbone, ./aqlm_out/head_libero_spatial, ... ; or pass --dirs key=path ... for other layouts)
"""
import argparse
import json
import re
from pathlib import Path

from safetensors.torch import load_file

from groot_common import SUITES, write_fakequant_checkpoint, accounting


def load_key(directory):
    acc = json.loads((Path(directory) / 'accounting.json').read_text())
    layers, payload = {}, 0
    for name, rec in acc['layers'].items():
        layers[name] = load_file(str(Path(directory) / (name.replace('.', '_') + '.safetensors')))['weight']
        payload += int(rec['payload_bytes'])
    assert len(layers) == 216, (directory, len(layers))
    return acc, layers, payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', required=True)
    p.add_argument('--quantized', help='directory holding backbone/ and head_<suite>/ toolkit outputs')
    p.add_argument('--dirs', nargs='*', default=[], help='key=path overrides, e.g. backbone=/x/bb head_libero_10=/x/h10')
    p.add_argument('--only', default=None, help='regex: take only the matching layers from the toolkit output, the others stay FP (partial checkpoint)')
    p.add_argument('--replace', nargs='*', default=[], help='LAYER=FILE: use the weight of FILE (safetensors, tensor `weight`) for that layer instead (intervention tests only)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dirs = {k: Path(a.quantized) / k for k in ['backbone'] + [f'head_{s}' for s in SUITES]} if a.quantized else {}
    dirs.update(dict(d.split('=', 1) for d in a.dirs))
    bb_acc, bb, bb_payload = load_key(dirs['backbone'])
    for suite in SUITES:
        acc, head, payload = load_key(dirs[f'head_{suite}'])
        assert acc['method'] == bb_acc['method'] and acc['bits_nominal'] == bb_acc['bits_nominal'] == 2, 'backbone and head must use the same method at 2 bits'
        meta = {k: v for k, v in acc.items() if k not in ('layers', 'linear_payload_bytes')}
        meta['damping_by_layer'] = {n: r['sigma_reg_used'] for src in (bb_acc, acc) for n, r in src['layers'].items() if 'sigma_reg_used' in r}   # QuIP#: sigma_reg actually used per layer (0.01 default; 0.1 / 1.0 after a Cholesky retry, G18)
        info = accounting(bb_payload + payload, acc['method'], acc['bits_nominal'], dict(meta, suite=suite, backbone_payload_bytes=bb_payload, head_payload_bytes=payload,
                                                                                         external_accounting={k: str(dirs[k]) for k in ('backbone', f'head_{suite}')}))
        both = dict(bb, **head)
        for r in a.replace:
            lname, fname = r.split('=', 1)
            both[lname] = load_file(fname)['weight']
        if a.replace:
            info['replaced_by_file'] = a.replace
        if a.only:
            both = {n: w for n, w in both.items() if re.search(a.only, n)}
            info['partial_only'] = a.only
        write_fakequant_checkpoint(Path(a.checkpoints) / suite, Path(a.out) / suite, both, info)
        print(json.dumps({k: info[k] for k in ('method', 'suite', 'quantized_linear_bpw', 'whole_model_bpw')}), flush=True)


if __name__ == '__main__':
    main()
