#!/bin/bash
# Run from cron every 10 minutes on the login node: starts `jobs/scheduler.py loop` unless one is running (it holds
# $RUN/scheduler.lock) or $RUN/STOP exists (touch it to pause submissions; remove it to resume).
#   */10 * * * * /home/d250009/yanlongvla-clean/baselines/jobs/watchdog.sh
RUN=${RUN:-/home/d250009/projects/pi05-handover/runs/readme_baselines}
PY=/home/d250009/projects/openpi/.venv/bin/python
export USER=${USER:-$(id -un)} PATH=/opt/pbs/bin:/usr/local/bin:/usr/bin:/bin:$PATH
[[ -s "$RUN/plan.json" && ! -e "$RUN/STOP" ]] || exit 0
flock -n "$RUN/scheduler.lock" true || exit 0   # held by the running loop
setsid nohup "$PY" -B "$(dirname "$(readlink -f "$0")")/scheduler.py" loop >> "$RUN/scheduler.out" 2>&1 < /dev/null &
