#!/usr/bin/env bash
# wait.sh <name-suffix> [seconds]: wait for the newest queued command with that name to finish, then print its output tail.
Q=/projects/yanlongvla/runs/deploy_tcq/dev/queue; f=$(ls $Q/*_$1.sh | tail -1); f=${f%.sh}
for i in $(seq 1 $(( ${2:-100} / 4 ))); do [ -f $f.done ] && break; sleep 4; done
[ -f $f.done ] && echo "rc=$(cat $f.done)" || echo "(still running)"; tail -n ${TAIL:-40} $f.out 2>/dev/null | cut -c1-${COLS:-420}
