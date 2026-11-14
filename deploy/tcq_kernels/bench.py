"""Latency and memory of pi0.5 in the trellis-coded 2-bit format, one GPU, batch 1, LIBERO observation shapes.

  python bench.py --all --out <dir>              every row x {10, 2} denoising steps, one process per measurement, then results.csv + table.md
  python bench.py --row R --steps S --out <dir>  one measurement -> <dir>/R_s<S>.json

Protocol: CUDA graphs (as the existing deployment table), 20 warm-up replays, 200 timed replays, each bracketed by CUDA events and
synchronised; median and 95th percentile. Timed quantities: prefill (image preprocessing + SigLIP + Gemma prefix -> KV cache), full
chunk (prefill + all denoising steps, one graph, timed directly), one denoising step, and the vision / LLM split of the prefill.
Memory: allocated and reserved bytes of the process after load with only the full-chunk graph captured (peak over capture and
replays), measured before the other graphs are built. The loader starts from the FP skeleton and replaces the 414 Linear layers, so
load-time peaks are not deployment numbers and are reported separately.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROW_ORDER = ['fp', 'fp_ls', 'w2_dense', 'w2_fly', 'w2_hybrid', 'w2a8_resident', 'w2a8_fly', 'w2a8_hybrid']
LABEL = dict(fp='Full precision (BF16), official path', fp_ls='Full precision (BF16), lockstep vision calls', fp_bv='Full precision (BF16), cameras batched', w2_dense='2-bit weights, (i) decoded at load', w2_fly='2-bit weights, (ii) on the fly', w2_hybrid='2-bit weights, (iii) hybrid',
             w2a8_resident='W2A8, (i) decoded at load', w2a8_fly='W2A8, (ii) on the fly', w2a8_hybrid='W2A8, (iii) hybrid')


@torch.no_grad()
def measure(row, steps, warmup, timed):
    import runner, cuda_rt, modules
    obs = runner.heldout40()[0][1]; noise_np = runner.fixed_noise(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    policy, info = runner.load(row, steps)
    load = dict(seconds=time.time() - t0, peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    mode = os.environ.get('TCQ_VISION') or runner.VISION[row]
    st = runner.Stages(policy, steps, vision=mode)
    observation = st.observation(obs)
    noise = torch.from_numpy(noise_np).cuda()[None]
    eager = st.chunk(observation, noise).clone()
    st.vision = 'per_image'
    unbatched = st.chunk(observation, noise).clone()               # the official call structure (one vision-encoder call per image)
    st.vision = mode
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    result = dict(row=row, steps=steps, info=info, device=cuda_rt.versions(), load=load, vision=mode,
                  vs_per_image_max_abs=float((eager - unbatched).abs().max()), vs_per_image_rel=float((eager - unbatched).norm() / unbatched.norm()))
    with runner.host_memo():
        g_all, out = runner.capture(lambda: st.chunk(observation, noise))
        result['chunk'] = runner.timed(g_all, warmup, timed)
        torch.cuda.synchronize()
        result['memory'] = dict(resident_allocated=resident, peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                                steady_allocated=torch.cuda.memory_allocated(), steady_reserved=torch.cuda.memory_reserved())
        result['graph_vs_eager_max_abs'] = float((out - eager).abs().max())
        result['actions_finite'] = bool(torch.isfinite(out).all())
        g_pre, (s_state, s_padding, s_kv) = runner.capture(lambda: st.prefill(observation))
        result['prefill'] = runner.timed(g_pre, warmup, timed)
        embs, padding, attention, state = st.embed(observation)
        g_vis, _ = runner.capture(lambda: st.embed(observation))
        result['vision'] = runner.timed(g_vis, warmup, timed)
        g_llm, _ = runner.capture(lambda: st.llm(embs, padding, attention))
        result['llm_prefill'] = runner.timed(g_llm, warmup, timed)
        if st.hybrid:
            modules.decode_chunk(policy)
        x0 = noise.clone()
        g_step, _ = runner.capture(lambda: st.step(0, s_state, s_padding, s_kv, x0))
        result['expert_step'] = runner.timed(g_step, warmup, timed)
        g_exp, _ = runner.capture(lambda: st.expert(s_state, s_padding, s_kv, x0))
        result['expert_all_steps'] = runner.timed(g_exp, warmup, timed)
    result['stats'] = dict(modules.STATS)
    return result


def disk_bytes(row):
    import tcq_format as F
    size = json.loads((F.ROOT / f'models/pi05_tcq/{F.ROW}/deploy_size.json').read_text())
    return size['fp_bytes'] if row.startswith('fp') else size['packed_bytes'] + size['rest_bytes']


def aggregate(out):
    rows = {}
    for p in sorted(out.glob('*_s*.json')):
        r = json.loads(p.read_text())
        if 'chunk' in r:
            rows[(r['row'], r['steps'])] = r
    gb = lambda b: b / 2 ** 30
    names = [r for r in ROW_ORDER if (r, 10) in rows]
    fp = {s: rows[('fp', s)]['chunk']['median_ms'] for s in (10, 2) if ('fp', s) in rows}
    fields = ['row', 'label', 'weights_on_disk_gb', 'peak_allocated_gb', 'peak_reserved_gb', 'resident_allocated_gb',
              'prefill_ms_median', 'prefill_ms_p95', 'vision_ms_median', 'llm_prefill_ms_median',
              'chunk10_ms_median', 'chunk10_ms_p95', 'chunk2_ms_median', 'chunk2_ms_p95', 'expert_step_ms_median', 'expert_step_ms_p95',
              'speed_vs_fp_10', 'speed_vs_fp_2', 'graph_vs_eager_max_abs', 'vision', 'vs_per_image_max_abs', 'gpu']
    table = []
    for n in names:
        a, b = rows[(n, 10)], rows.get((n, 2))
        rec = dict(row=n, label=LABEL[n], weights_on_disk_gb=round(gb(disk_bytes(n)), 3),
                   peak_allocated_gb=round(gb(a['memory']['peak_allocated']), 3), peak_reserved_gb=round(gb(a['memory']['peak_reserved']), 3),
                   resident_allocated_gb=round(gb(a['memory']['resident_allocated']), 3),
                   prefill_ms_median=round(a['prefill']['median_ms'], 2), prefill_ms_p95=round(a['prefill']['p95_ms'], 2),
                   vision_ms_median=round(a['vision']['median_ms'], 2), llm_prefill_ms_median=round(a['llm_prefill']['median_ms'], 2),
                   chunk10_ms_median=round(a['chunk']['median_ms'], 2), chunk10_ms_p95=round(a['chunk']['p95_ms'], 2),
                   chunk2_ms_median=round(b['chunk']['median_ms'], 2) if b else '', chunk2_ms_p95=round(b['chunk']['p95_ms'], 2) if b else '',
                   expert_step_ms_median=round(a['expert_step']['median_ms'], 3), expert_step_ms_p95=round(a['expert_step']['p95_ms'], 3),
                   speed_vs_fp_10=round(fp[10] / a['chunk']['median_ms'], 3) if 10 in fp else '',
                   speed_vs_fp_2=round(fp[2] / b['chunk']['median_ms'], 3) if (b and 2 in fp) else '',
                   graph_vs_eager_max_abs=a['graph_vs_eager_max_abs'], vision=a.get('vision'), vs_per_image_max_abs=a.get('vs_per_image_max_abs'), gpu=a['device']['gpu'])
        table.append(rec)
    with open(out.parent / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(table)
    lines = ['| Row | Weights on disk (GB) | Peak GPU memory, allocated / reserved (GB) | Prefill (ms) | Chunk, 10 steps (ms) | Chunk, 2 steps (ms) | Speed vs FP, 10 / 2 steps |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['label']} | {r['weights_on_disk_gb']:.2f} | {r['peak_allocated_gb']:.2f} / {r['peak_reserved_gb']:.2f} | {r['prefill_ms_median']:.1f} ({r['prefill_ms_p95']:.1f}) | "
                     f"{r['chunk10_ms_median']:.1f} ({r['chunk10_ms_p95']:.1f}) | {r['chunk2_ms_median']} ({r['chunk2_ms_p95']}) | {r['speed_vs_fp_10']}x / {r['speed_vs_fp_2']}x |")
    dev = rows[(names[0], 10)]['device'] if names else {}
    lines += ['', f"Median (95th percentile) of 200 timed CUDA-graph replays after 20 warm-up replays, batch 1. Device: {json.dumps(dev)}"]
    (out.parent / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--row'); p.add_argument('--steps', type=int, default=10)
    p.add_argument('--all', action='store_true'); p.add_argument('--aggregate', action='store_true')
    p.add_argument('--rows', nargs='+', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--warmup', type=int, default=20); p.add_argument('--timed', type=int, default=200)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.all:
        import runner
        for steps in (10, 2):
            for row in (a.rows or [r for r in ROW_ORDER if r in runner.ROWS]):
                log = out / f'{row}_s{steps}.log'
                if (out / f'{row}_s{steps}.json').exists():            # resume after a preemption
                    print(row, steps, 'done already', flush=True); continue
                rc = subprocess.run([sys.executable, str(HERE / 'bench.py'), '--row', row, '--steps', str(steps), '--out', str(out), '--warmup', str(a.warmup), '--timed', str(a.timed)],
                                    stdout=open(log, 'w'), stderr=subprocess.STDOUT).returncode
                print(row, steps, 'rc', rc, open(log).read().strip().splitlines()[-1][:300] if log.stat().st_size else '', flush=True)
        aggregate(out)
    elif a.aggregate:
        aggregate(out)
    else:
        r = measure(a.row, a.steps, a.warmup, a.timed)
        (out / f'{a.row}_s{a.steps}.json').write_text(json.dumps(r, indent=1))
        print(json.dumps({k: (round(v['median_ms'], 3) if isinstance(v, dict) and 'median_ms' in v else v) for k, v in r.items() if k in ('row', 'steps', 'chunk', 'prefill', 'vision', 'llm_prefill', 'expert_step', 'expert_all_steps', 'graph_vs_eager_max_abs')}), flush=True)


if __name__ == '__main__':
    main()
