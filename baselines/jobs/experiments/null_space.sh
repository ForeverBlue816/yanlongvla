#!/bin/bash
# Experiment null_space (DECISIONS D6/D7): does error placed in directions the calibration never exercised hurt on other
# inputs? Runs inside a bundle job (jobs/evaluate.pbs, kind exp) with RUN, SRC, PY, BASE_CKPT, CALIBRATION, LAYERS,
# HANDOVER, OPENPI_ROOT, AQLM_REPO, QUIP_REPO set. Each step resumes; done.json marks completion.
set -euo pipefail
X="$RUN/experiments/null_space"; mkdir -p "$X"
HELD="$RUN/experiments/heldout/calibration"
echo "=== held-out Hessians $(date -Is)"
[[ -s "$HELD/index.json" ]] || "$PY" calibrate.py --checkpoint "$BASE_CKPT" --out "$HELD" --hf-cache "$HANDOVER/runs/cache/libero-calibration" \
  --manifest "$SRC/data/heldout_manifest.json" --noise-seed 1
EXT_SITE="$HANDOVER/runs/envs/external-toolkit/site-packages"
echo "=== QuIP# without tuning sweeps on the 126 expert layers $(date -Is)"
LD_LIBRARY_PATH="/usr/local/gcc-12.3/lib64:${LD_LIBRARY_PATH:-}" PYTHONPATH="$SRC:$QUIP_REPO:$EXT_SITE:$OPENPI_ROOT" \
  "$PY" quantize_quip.py --layers "$LAYERS" --calibration "$CALIBRATION" --checkpoint "$BASE_CKPT" --quip-repo "$QUIP_REPO" \
    --out "$X/quip_t0_expert" --only gemma_expert --tune-iters 0
for d in 0.01 0; do
  echo "=== AQLM 2x8g8 percdamp $d on the 126 expert layers $(date -Is)"
  PYTHONPATH="$SRC:$AQLM_REPO:$OPENPI_ROOT" "$PY" quantize_aqlm.py --bits 2 --layers "$LAYERS" --calibration "$CALIBRATION" \
    --checkpoint "$BASE_CKPT" --aqlm-repo "$AQLM_REPO" --out "$X/aqlm_damp${d}_expert" --only gemma_expert --percdamp "$d"
done
echo "=== errors on calibration vs held-out Hessians $(date -Is)"
C="$RUN/checkpoints"
"$PY" jobs/experiments/heldout_errors.py --layers "$LAYERS" --calibration "$CALIBRATION" --heldout "$HELD" \
  --source gptq=$C/gptq_w2 --source awq=$C/awq_w2 --source vptq=$C/vptq_w2 --source quip=$C/quip_w2 --source qtip=$C/qtip_w2 \
  --source quip_t0=$X/quip_t0_expert --source aqlm_damp=$X/aqlm_damp0.01_expert --source aqlm_nodamp=$X/aqlm_damp0_expert \
  --out "$X/errors.json"
"$PY" -c "import json,sys; json.dump(dict(finished=__import__('datetime').datetime.now().isoformat()), open(sys.argv[1], 'w'))" "$X/done.json"
