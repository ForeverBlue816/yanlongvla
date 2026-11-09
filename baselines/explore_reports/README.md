# Evidence reports and analysis scripts (GR00T N1.7 2-bit baselines, 6 to 9 Nov 2026)

These folders are small copies of analysis reports, scripts and result summaries. The originals sit on the cluster scratch disk
(`/scratch/d250009/pi05-handover/runs/groot_baselines/explore/<folder>/`). Big raw files (per-episode JSON of 300 KB or more,
videos, logs, the 16 MB per-observation part of `w5.json`) are not copied.

| Folder | What it answers | Main file |
|---|---|---|
| `g1/` | What exactly is NVIDIA's open_drawer 65%, and how do our Fractal stack and NVIDIA's differ? | `NVIDIA_OPEN_DRAWER.md`, `STACK_DIFF.md` |
| `g3/` | Bridge stack_cube: ladder of protocol differences between our seeded protocol and NVIDIA's command | `LADDER.md` |
| `x22/` | Layout draws of NVIDIA's 5-env command (fixed seed 2022 chain), first-frame layout classifier | `LAYOUTS.md`, `classify.py`, `analyze.py` |
| `x23/` | Fractal open_drawer configuration chain (108 configurations), reweighting | `FRACTAL_CONFIGS.md`, `chain.py`, `decode.py` |
| `w5/` | Activation statistics at weight scale 1.0 and 0.95 (summary, no per-observation values) | `w5_summary.json`, `w5_job_log_clean.txt` |
| `x25/` | Weight-space and activation-space shrink of the 2-bit error (Hessians) | `X25.md`, `x25.py` |
| `s6/` | Shrink factor c per tensor for AQLM, QTIP and QuIP# | `S6.md`, `s6.py` |
| `x10/` | CPU shrink data for QuIP# and QTIP | `*_shrink.json` |
| `w8/` | Literature check: is the shrink an MMSE effect? | `LIT_MMSE.md` |
| `l1/` | LIBERO-Plus selection (S7), Fractal seeding (A11), Bridge determinism (A12) | `S7_A11_A12.md` |
| `l2/` | Audit items A14, A15, A5, S5, X3, S8 | `PART_I.md`, `a14_a5.py`, `x3_diverge.py` |
| `c0/`, `a4a8/` | Earlier audit of suspicious results; Holm checks | `AUDIT.md`, `a4a8.py` |

Run the scripts with the GR00T Python environment. They read checkpoints and episode files from the cluster paths named in each script.
