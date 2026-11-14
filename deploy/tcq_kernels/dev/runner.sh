#!/usr/bin/env bash
# Dev runner (inside a Slurm GPU job): executes queue/*.sh in name order, one at a time; output -> <name>.out, completion marker <name>.done.
# Ends when queue/STOP exists or after 30 idle minutes.
Q=/projects/yanlongvla/runs/deploy_tcq/dev/queue; idle=0
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > $Q/../gpu.txt 2>&1; hostname >> $Q/../gpu.txt; echo $SLURM_JOB_ID >> $Q/../gpu.txt
while [ ! -f $Q/STOP ] && [ $idle -lt 360 ]; do
  f=$(ls $Q/*.sh 2>/dev/null | while read s; do [ -f ${s%.sh}.done ] || { echo $s; break; }; done)
  if [ -n "$f" ]; then idle=0; timeout ${DEV_TIMEOUT:-3000} bash $f > ${f%.sh}.out 2>&1; echo $? > ${f%.sh}.done; else idle=$((idle+1)); sleep 5; fi
done
