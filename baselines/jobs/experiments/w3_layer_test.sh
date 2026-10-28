#!/bin/bash
# Experiment w3_layer_test (2026-10-28): do the 3-bit QuIP# (E8P12RVQ3B) and QTIP (K=3) ports run, store about 3 bits per weight
# and keep the output error at or below their 2-bit values? Four layers each, written under $RUN/experiments/w3_layer_test.
# Runs inside a bundle job (jobs/evaluate.pbs, kind exp); done.json marks completion.
set -euo pipefail
X="$RUN/experiments/w3_layer_test"; mkdir -p "$X"
EXT_SITE="$HANDOVER/runs/envs/external-toolkit/site-packages"
ONLY=(gemma_expert.model.layers.0.mlp.down_proj gemma_expert.model.layers.11.mlp.gate_proj language_model.layers.0.self_attn.k_proj vision_tower.vision_model.encoder.layers.0.mlp.fc1)
for m in quip qtip; do
  repo=$QUIP_REPO; [[ $m == qtip ]] && repo=$QTIP_REPO
  for b in 3; do
    echo "=== $m W$b on four layers $(date -Is)"
    LD_LIBRARY_PATH="/usr/local/gcc-12.3/lib64:${LD_LIBRARY_PATH:-}" PYTHONPATH="$SRC:$repo:$EXT_SITE:$OPENPI_ROOT" \
      "$PY" "quantize_$m.py" --layers "$LAYERS" --calibration "$CALIBRATION" --checkpoint "$BASE_CKPT" "--$m-repo" "$repo" \
        --bits "$b" --only "${ONLY[@]}" --out "$X/${m}_w$b"
  done
done
"$PY" - "$X" <<'PY'
import json, sys, glob, os
from safetensors import safe_open
x = sys.argv[1]
for f in sorted(glob.glob(x + '/*/*.json')):
    r = json.load(open(f))
    if 'payload_bytes' not in r: continue
    st = f[:-5] + '.safetensors'
    with safe_open(st, 'pt') as h:
        n = h.get_tensor('weight').numel() if 'weight' in h.keys() else 0
    print(f.split('/')[-2], os.path.basename(f)[-48:-5], 'bpw(payload/weights)=%.3f' % (8 * r['payload_bytes'] / n), 'rel_out_err=%.5f' % r['relative_output_error'], 'rel_weight_err=%.3f' % r['relative_weight_error'])
json.dump(dict(finished=__import__('datetime').datetime.now().isoformat()), open(x + '/done.json', 'w'))
PY
