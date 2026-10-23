# Standing instructions for the baseline loop

Read this file at the start of every loop iteration and follow it. Deadline: 6 Nov 2026.

## Goal

Produce the mentor's 2-bit baseline rows for π₀.₅-LIBERO: RTN, GPTQ, VPTQ, AQLM, AWQ, QuIP#, QTIP (in that order;
QuIP# and QTIP are optional extras, last). Each row: whole-model and linear bpw, LIBERO-Plus 1000 with paired difference
and 95% CI vs `original_fp` and `original_2bit`, Plus 2000 (and the added 1000) vs `original_2bit_expanded_2000`,
LIBERO 2000 and the 1900 column, and a note that the row is pure PTQ on the shared Hessians. Keep `quant.json` /
`accounting.json`, the toolkit commit and the exact command for every row.

Sources, in priority order: (1) `baselines/README.md`, (2) the mentor's instructions recorded in `DECISIONS.md`,
(3) the official repository of each method. VPTQ is the one exception to (3): v4-k256-0 at 2 bits, never v8-k65536.

## Every iteration

1. Read `~/projects/pi05-handover/runs/readme_baselines/STATUS.md`, the tail of `scheduler.log`, `qstat -u $USER`,
   and the `.progress` files of running bundles. Confirm one scheduler loop is running (`scheduler.lock`); if not,
   run `jobs/watchdog.sh`.
2. Look at results while jobs run, not only at the end: per row, successes vs the reference on the same episodes.
   If something is definitely wrong (collapse, crash loop, a part failing repeatedly, nonsense bpw), stop the row or
   delete the job at once (`python jobs/scheduler.py stop <row> <reason>`, `qdel`), then find the cause from the job
   logs (`logs/<jobid>.OU`), fix it in `baselines/`, and requeue. Use one sequential debug job (`jobs/debug.pbs`)
   when the cause needs a GPU.
3. Keep 5 jobs waiting in the queue. The scheduler does this; if it can't (everything gated or parked), find out why
   and unblock it: finish a port, fix a failure, or `enable` a ready row.
4. Prefer fast feedback: new or changed quantizers first get the 100-episode sanity sample (the scheduler's sanity
   parts) before any full evaluation.
5. Work on the next unported method in order (AQLM, then QuIP#, then QTIP): write `quantize_<method>.py` in
   `baselines/` following README step 2b and the official repository, test it on a few layers in a short GPU job,
   then add its case to `jobs/evaluate.pbs`, `python jobs/scheduler.py refresh`, and `python jobs/scheduler.py enable <row>`.
6. Record every implementation decision, and every open question with the decision you took, in `DECISIONS.md`.
   Decide open questions yourself using the source priority above; ask the user only for things that are theirs alone
   (pushing code, spending beyond the project's GPU allowance, changing the mentor's protocol).
7. Commit code changes to the local branch `baselines-readme` at each milestone; never push.

## Rules

- Clean slate: edit files in `baselines/` in place, delete superseded scripts and unused run artifacts, never run old
  code from `~/projects/pi05-handover/baselines` or `~/projects/openpi/diagnostics`.
- Jobs: queue `gpu_bwp6000`, project `gs_ccds_taisp`, 12 h walltime packed with ~11 h of work, parts in plan order,
  no PBS dependency holds. Never reuse old evaluation outputs.
- Each iteration ends with a short note in `scheduler.log` style in this session: what changed, what was found, what
  is next. Pace the next iteration by what is pending (a few minutes while debugging, 20–30 minutes when quiet).
