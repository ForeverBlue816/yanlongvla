#!/bin/bash
# Experiment quip_sweep_check (mentor, 2026-10-26): are QuIP#'s tuning sweeps broken, or do they overfit by design?
# Runs inside a bundle job (jobs/evaluate.pbs, kind exp) with RUN, SRC, PY, LAYERS, CALIBRATION, QUIP_REPO, OPENPI_ROOT,
# HANDOVER set. Needs the held-out Hessians that experiment null_space writes. done.json marks completion.
set -euo pipefail
X="$RUN/experiments/quip_sweep_check"; mkdir -p "$X"
HELD="$RUN/experiments/heldout/calibration"
[[ -s "$HELD/index.json" ]] || { echo "held-out Hessians missing: $HELD"; exit 1; }
EXT_SITE="$HANDOVER/runs/envs/external-toolkit/site-packages"
LD_LIBRARY_PATH="/usr/local/gcc-12.3/lib64:${LD_LIBRARY_PATH:-}" PYTHONPATH="$SRC:$QUIP_REPO:$EXT_SITE:$OPENPI_ROOT" \
  "$PY" jobs/experiments/quip_sweep_check.py --layers "$LAYERS" --calibration "$CALIBRATION" --heldout "$HELD" \
    --quip-repo "$QUIP_REPO" --out "$X/results.json"
"$PY" -c "import json,sys,datetime; json.dump(dict(finished=datetime.datetime.now().isoformat()), open(sys.argv[1], 'w'))" "$X/done.json"
