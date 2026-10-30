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
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dirs = {k: Path(a.quantized) / k for k in ['backbone'] + [f'head_{s}' for s in SUITES]} if a.quantized else {}
    dirs.update(dict(d.split('=', 1) for d in a.dirs))
    bb_acc, bb, bb_payload = load_key(dirs['backbone'])
    for suite in SUITES:
        acc, head, payload = load_key(dirs[f'head_{suite}'])
        assert acc['method'] == bb_acc['method'] and acc['bits_nominal'] == bb_acc['bits_nominal'] == 2, 'backbone and head must use the same method at 2 bits'
        meta = {k: v for k, v in acc.items() if k not in ('layers', 'linear_payload_bytes')}
        info = accounting(bb_payload + payload, acc['method'], acc['bits_nominal'], dict(meta, suite=suite, backbone_payload_bytes=bb_payload, head_payload_bytes=payload,
                                                                                         external_accounting={k: str(dirs[k]) for k in ('backbone', f'head_{suite}')}))
        write_fakequant_checkpoint(Path(a.checkpoints) / suite, Path(a.out) / suite, dict(bb, **head), info)
        print(json.dumps({k: info[k] for k in ('method', 'suite', 'quantized_linear_bpw', 'whole_model_bpw')}), flush=True)


if __name__ == '__main__':
    main()
