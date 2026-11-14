"""Markdown tables for the report from the per-row benchmark JSON files: python report_tables.py <rows dir>"""
import json
import sys
from pathlib import Path

import bench

d = Path(sys.argv[1])
rows = {}
for p in sorted(d.glob('*_s*.json')):
    r = json.loads(p.read_text()); rows[(r['row'], r['steps'])] = r
names = [n for n in bench.ROW_ORDER + ['fp_bv'] if (n, 10) in rows]
gb = lambda b: b / 2 ** 30
fp = {s: rows[('fp', s)]['chunk']['median_ms'] for s in (10, 2)}
print('| Row | Weights on disk (GB) | Peak GPU memory, allocated / reserved (GB) | Prefill, ms | Full chunk, 10 steps, ms | Full chunk, 2 steps, ms | vs FP, 10 / 2 steps |')
print('|---|---:|---:|---:|---:|---:|---:|')
for n in names:
    a, b = rows[(n, 10)], rows[(n, 2)]
    m = a['memory']
    print(f"| {bench.LABEL[n]} | {gb(bench.disk_bytes(n)):.2f} | {gb(m['peak_allocated']):.2f} / {gb(m['peak_reserved']):.2f} | {a['prefill']['median_ms']:.1f} ({a['prefill']['p95_ms']:.1f}) | "
          f"{a['chunk']['median_ms']:.1f} ({a['chunk']['p95_ms']:.1f}) | {b['chunk']['median_ms']:.1f} ({b['chunk']['p95_ms']:.1f}) | {fp[10] / a['chunk']['median_ms']:.2f}× / {fp[2] / b['chunk']['median_ms']:.2f}× |")
print()
print('| Row | Vision encoder, ms | LLM prefill, ms | Expert step (10-step schedule), ms | All 10 steps, ms | Expert step (2-step schedule), ms | Graph = eager | Same as per-image calls |')
print('|---|---:|---:|---:|---:|---:|---|---|')
for n in names:
    a, b = rows[(n, 10)], rows[(n, 2)]
    same = a.get('vs_per_image_max_abs', a.get('batched_vs_per_image_max_abs'))
    print(f"| {bench.LABEL[n]} | {a['vision']['median_ms']:.2f} | {a['llm_prefill']['median_ms']:.2f} | {a['expert_step']['median_ms']:.2f} ({a['expert_step']['p95_ms']:.2f}) | {a['expert_all_steps']['median_ms']:.1f} | "
          f"{b['expert_step']['median_ms']:.2f} | {'yes' if a['graph_vs_eager_max_abs'] == 0 else a['graph_vs_eager_max_abs']} | {'yes' if same == 0 else f'max abs diff {same:.1e}'} |")
dev = rows[(names[0], 10)]['device']
print(f"\nMedian (95th percentile) of 200 timed CUDA-graph replays after 20 warm-up replays, batch 1; GB = 2^30 bytes. {dev['gpu']}, PyTorch {dev['torch']}, CUDA {dev['cuda_runtime']}, cuDNN {dev['cudnn']}, NVRTC {dev['nvrtc']}.")
