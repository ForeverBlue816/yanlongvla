"""Builds data/calibration_manifest_1024.json: the 256 calibration observations plus 768 more (4 random frames from each of 192 episodes) taken from
episodes that are in neither the calibration set nor the held-out set (GOAL 1 Nov 1.3: does a larger calibration set close the gap between the
calibration error and the held-out error of large-codebook VPTQ?). Episodes are drawn round-robin over the task descriptions, seed 20261101.
  python make_calibration_1024.py /path/to/libero-calibration   (the LeRobot dataset root with meta/episodes.jsonl)"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

root, here = Path(sys.argv[1]), Path(__file__).resolve().parent
cal, held = (json.loads((here / f).read_text()) for f in ('calibration_manifest.json', 'heldout_manifest.json'))
used = {o['episode_index'] for o in cal['observations']} | {o['episode_index'] for o in held['observations']}
episodes = [json.loads(l) for l in (root / 'meta/episodes.jsonl').read_text().splitlines()]
rng = random.Random(20261101)
by_task = defaultdict(list)
for e in episodes:
    if e['episode_index'] not in used:
        by_task[e['tasks'][0]].append(e)
for v in by_task.values():
    rng.shuffle(v)
chosen, tasks = [], sorted(by_task)
while len(chosen) < 192:
    for t in tasks:
        if by_task[t] and len(chosen) < 192:
            chosen.append(by_task[t].pop())
obs = list(cal['observations'])
for e in sorted(chosen, key=lambda e: e['episode_index']):
    for f in sorted(rng.sample(range(e['length']), 4)):
        obs.append({'episode_index': e['episode_index'], 'frame_index': f})
out = dict(cal, selection='calibration_manifest.json (256 observations) + 4 random frames from each of 192 further episodes outside calibration and held-out, round-robin over tasks, seed 20261101',
           episodes=[e for e in episodes if e['episode_index'] in {o['episode_index'] for o in obs}], observation_count=len(obs), observations=obs)
(here / 'calibration_manifest_1024.json').write_text(json.dumps(out))
print(len(obs), len(out['episodes']), 'episodes; overlap with held-out:', len({o['episode_index'] for o in obs} & {o['episode_index'] for o in held['observations']}))
