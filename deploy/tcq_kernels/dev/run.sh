#!/usr/bin/env bash
# run.sh <name> <command...>: queue a command for the dev runner and wait for it (up to WAIT seconds, default 540); prints the output tail.
Q=/projects/yanlongvla/runs/deploy_tcq/dev/queue; n=$(date +%s)_$1; shift
printf 'cd /projects/yanlongvla/research/deploy/tcq_kernels\nexport PYTHONPATH=/projects/yanlongvla/research/scripts:/projects/yanlongvla/research/deploy/tcq_kernels\nPY=/projects/yanlongvla/envs/openpi/bin/python\n%s\n' "$*" > $Q/$n.sh
for i in $(seq 1 $(( ${WAIT:-540} / 3 ))); do [ -f $Q/$n.done ] && break; sleep 3; done
[ -f $Q/$n.done ] && { echo "rc=$(cat $Q/$n.done)"; tail -n ${TAIL:-60} $Q/$n.out | cut -c1-${COLS:-400}; } || echo "still running: $Q/$n.out"
