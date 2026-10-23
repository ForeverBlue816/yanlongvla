"""Fail the job if any baseline module would load from outside the frozen source snapshot (older copies of this harness
exist on the cluster), and log where openpi, the patched transformers models and the simulator come from."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('src')
p.add_argument('--simulator', help='expected root of the `libero` package (evaluation jobs only)')
a = p.parse_args()
src = Path(a.src).resolve()
stale = [e for e in sys.path if 'pi05-handover/baselines' in e or e.rstrip('/').endswith('openpi/diagnostics')]
assert not stale, f'stale code directories on sys.path: {stale}'
found = {}
for name in ('common', 'calibrate', 'quantize', 'quantize_vptq', 'export_layers', 'import_external', 'serve', 'eval_libero', 'eval_libero_plus', 'paired_stats'):
    origin = Path(importlib.util.find_spec(name).origin).resolve()
    assert origin.parent == src, f'{name} resolves to {origin}, not the snapshot {src}'
    found[name] = str(origin)
for name in ('openpi', 'transformers.models.gemma.modeling_gemma', 'openpi_client'):
    spec = importlib.util.find_spec(name)
    found[name] = spec.origin if spec else None
assert found['openpi'].startswith('/home/d250009/projects/openpi/src/'), found['openpi']
if a.simulator:
    origin = Path(importlib.util.find_spec('libero').submodule_search_locations[0]).resolve()
    assert Path(a.simulator).resolve() in origin.parents, f'libero resolves to {origin}, not {a.simulator}'
    found['libero'] = str(origin)
print(json.dumps(dict(provenance='ok', modules=found)), flush=True)
