"""Bridge (WidowX) calibration / held-out observation sets for nvidia/GR00T-N1.7-SimplerEnv-Bridge (user 2026-11-06).

Source: the checkpoint's own fine-tuning data, IPEC-COMMUNITY/bridge_orig_lerobot (LeRobot v2.0, 53,192 episodes, 5 Hz,
image_0 256x256 AV1 video, 8-d state x,y,z,roll,pitch,yaw,pad,gripper, 7-d action), pinned revision below. Never evaluation
rollouts. Mirrors datasets/libero-calibration: 256 observations (64 episodes x 4 frames), a second disjoint 256 held-out set,
written as self-contained parquet rows (PNG bytes in `image`, no video decoding downstream) with observation_manifest.json.

Selection (seeded, SeedSequence([20261106, split])): 6 SimplerEnv-Bridge task families (spoon on towel, carrot on plate,
stack, eggplant, open drawer, close drawer) matched by keywords in the instruction, 5 episodes each = 30, plus 34 uniformly
random episodes with a non-empty instruction; episodes shorter than 8 frames excluded; 4 frames per episode at the
10 / 37 / 63 / 90 % positions. Stages: select -> download (parquet + image_0 videos of the chosen episodes only) -> build
(decode with PyAV / dav1d, envs/openpi) -> datasets/bridge-calibration, datasets/bridge-heldout.
"""
import argparse
import io
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('VLA_ROOT', '/projects/yanlongvla'))
DATASET = 'IPEC-COMMUNITY/bridge_orig_lerobot'
REVISION = '0e9d76d07e9df3ea3eba257b2520d4913833fad2'
META = ROOT / 'datasets/bridge_orig_lerobot_meta'
SUBSET = ROOT / 'datasets/bridge_orig_lerobot_subset'
OUTS = {0: ROOT / 'datasets/bridge-calibration', 1: ROOT / 'datasets/bridge-heldout'}
FAMILIES = {'spoon_on_towel': (r'\bspoon\b', r'towel|cloth'), 'carrot_on_plate': (r'\bcarrot\b', r'\bplate\b'), 'stack': (r'\bstack', r'.'),
            'eggplant': (r'\beggplant\b', r'.'), 'open_drawer': (r'\bopen\b', r'\bdrawer\b'), 'close_drawer': (r'\bclose\b', r'\bdrawer\b')}
PER_FAMILY = 5
EPISODES = 64
FRAMES = 4
POSITIONS = (0.10, 0.37, 0.63, 0.90)
MIN_LENGTH = 8
SEED = 20261106


def episodes_meta():
    eps = [json.loads(l) for l in (META / 'meta/episodes.jsonl').read_text().splitlines()]
    tasks = {r['task_index']: r['task'] for r in map(json.loads, (META / 'meta/tasks.jsonl').read_text().splitlines())}
    return eps, tasks


def family_of(text):
    t = text.lower()
    return [f for f, (a, b) in FAMILIES.items() if re.search(a, t) and re.search(b, t)]


def stage_select():
    eps, _ = episodes_meta()
    usable = [e for e in eps if ' '.join(e['tasks']).strip() and e['length'] >= MIN_LENGTH]
    by_family = {f: [e for e in usable if f in family_of(' '.join(e['tasks']))] for f in FAMILIES}
    print(json.dumps({f: len(v) for f, v in by_family.items()}))
    taken = set()
    for split, out in OUTS.items():
        rng = np.random.default_rng(np.random.SeedSequence([SEED, split]))
        chosen = []
        for f, pool in by_family.items():
            pool = [e for e in pool if e['episode_index'] not in taken]
            for e in (pool[i] for i in rng.choice(len(pool), PER_FAMILY, replace=False)):
                chosen.append(dict(e, family=f)); taken.add(e['episode_index'])
        pool = [e for e in usable if e['episode_index'] not in taken]
        for e in (pool[i] for i in rng.choice(len(pool), EPISODES - len(chosen), replace=False)):
            chosen.append(dict(e, family='random')); taken.add(e['episode_index'])
        observations = [dict(episode_index=e['episode_index'], frame_index=int(round(p * (e['length'] - 1)))) for e in chosen for p in POSITIONS]
        assert len(observations) == EPISODES * FRAMES and len({(o['episode_index'], o['frame_index']) for o in observations}) == len(observations)
        manifest = dict(dataset=DATASET, revision=REVISION, split='calibration' if split == 0 else 'heldout', selection_seed=f'SeedSequence([{SEED}, {split}])',
                        selection=f'{PER_FAMILY} episodes per SimplerEnv-Bridge task family {list(FAMILIES)} (keyword match on the instruction) + {EPISODES - PER_FAMILY * len(FAMILIES)} uniformly random '
                                  f'episodes with a non-empty instruction; length >= {MIN_LENGTH}; {FRAMES} frames per episode at {POSITIONS} of the episode; held-out episodes disjoint from calibration',
                        families={f: a for f, a in FAMILIES.items()}, episodes=[dict(episode_index=e['episode_index'], tasks=e['tasks'], length=e['length'], family=e['family']) for e in chosen],
                        observation_count=len(observations), flow_steps=4, action_horizon=8, noise_seeds='SeedSequence([7, observation_index]) (calibration) as for LIBERO',
                        status='selected', observations=observations)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'observation_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        print(json.dumps(dict(split=manifest['split'], episodes=len(chosen), families={f: sum(e['family'] == f for e in chosen) for f in list(FAMILIES) + ['random']}, observations=len(observations))))


def stage_download():
    from huggingface_hub import snapshot_download
    patterns = ['meta/*']
    for out in OUTS.values():
        for e in json.loads((out / 'observation_manifest.json').read_text())['episodes']:
            i = e['episode_index']
            patterns += [f'data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet', f'videos/chunk-{i // 1000:03d}/observation.images.image_0/episode_{i:06d}.mp4']
    p = snapshot_download(DATASET, repo_type='dataset', revision=REVISION, allow_patterns=patterns, local_dir=str(SUBSET))
    print('downloaded', p, len(patterns) - 1, 'files')


def decode_video(path):
    import av
    frames = []
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format='rgb24'))
    return frames


def stage_build():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    _, tasks = episodes_meta()
    info = json.loads((META / 'meta/info.json').read_text())
    for out in OUTS.values():
        manifest = json.loads((out / 'observation_manifest.json').read_text())
        used_tasks = {}
        for e in manifest['episodes']:
            i = e['episode_index']
            table = pq.read_table(SUBSET / f'data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet').to_pylist()
            frames = decode_video(SUBSET / f'videos/chunk-{i // 1000:03d}/observation.images.image_0/episode_{i:06d}.mp4')
            assert len(frames) == len(table) == e['length'], (i, len(frames), len(table), e['length'])
            rows = []
            for r in table:
                fi = int(r['frame_index'])
                img = frames[fi]
                assert img.shape == (256, 256, 3) and abs(float(r['timestamp']) - fi / info['fps']) < 1e-3, (img.shape, r['timestamp'], fi)
                buf = io.BytesIO(); Image.fromarray(img).save(buf, format='PNG')
                rows.append(dict(episode_index=i, frame_index=fi, timestamp=float(r['timestamp']), task_index=int(r['task_index']), image=dict(bytes=buf.getvalue(), path=None),
                                 state=[float(x) for x in r['observation.state']], action=[float(x) for x in r['action']]))
                used_tasks[int(r['task_index'])] = tasks[int(r['task_index'])]
            assert all(len(r['state']) == 8 and len(r['action']) == 7 for r in rows)
            d = out / f'data/chunk-{i // 1000:03d}'; d.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), d / f'episode_{i:06d}.parquet')
        (out / 'meta').mkdir(exist_ok=True)
        (out / 'meta/tasks.jsonl').write_text(''.join(json.dumps(dict(task_index=k, task=v)) + '\n' for k, v in sorted(used_tasks.items())))
        (out / 'meta/episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=e['episode_index'], tasks=e['tasks'], length=e['length'])) + '\n' for e in manifest['episodes']))
        (out / 'meta/info.json').write_text(json.dumps(dict(source=dict(dataset=DATASET, revision=REVISION, codebase_version=info['codebase_version'], fps=info['fps'], robot_type=info['robot_type']),
                                                            derived='self-contained frames: image = PNG bytes of observation.images.image_0 at frame_index; state = observation.state (8); action (7)',
                                                            episodes=len(manifest['episodes']), frames=sum(e['length'] for e in manifest['episodes'])), indent=1) + '\n')
        manifest['status'] = 'complete'
        (out / 'observation_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        print(json.dumps(dict(split=manifest['split'], episodes=len(manifest['episodes']), tasks=len(used_tasks), bytes=sum(p.stat().st_size for p in out.rglob('*.parquet')))))


def load_observations(limit=None, data_dir=OUTS[0]):
    """Same contract as calibration_data.load_observations: [(sample, observation)], observation = image (256,256,3) uint8, state (8,), prompt."""
    import pyarrow.parquet as pq
    from PIL import Image
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'observation_manifest.json').read_text())
    assert manifest['status'] == 'complete', data_dir
    tasks = {r['task_index']: r['task'] for r in map(json.loads, (data_dir / 'meta/tasks.jsonl').read_text().splitlines())}
    compact = (data_dir / 'observations.parquet').exists()
    cached = {(r['episode_index'], r['frame_index']): r for r in pq.read_table(data_dir / 'observations.parquet').to_pylist()} if compact else {}
    result = []
    for index, sample in enumerate(manifest['observations'][:limit]):
        e = sample['episode_index']
        if not compact and e not in cached:
            cached.update({(e, int(r['frame_index'])): r for r in pq.read_table(data_dir / f'data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet').to_pylist()})
        row = cached[(e, sample['frame_index'])]
        obs = {'observation/image': np.array(Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')), 'observation/state': np.asarray(row['state'], dtype=np.float32), 'prompt': tasks[int(row['task_index'])],
               'action': np.asarray(row['action'], dtype=np.float32)}
        assert obs['observation/image'].shape == (256, 256, 3) and obs['observation/state'].shape == (8,)
        result.append((dict(sample, observation_index=index, task_index=int(row['task_index'])), obs))
    return manifest, result


def stage_compact(target):
    """Distribution copy: only the manifest frames (256 per split) as self-contained parquet + meta, for the public report repo."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    target = Path(target)
    for out in OUTS.values():
        manifest = json.loads((out / 'observation_manifest.json').read_text())
        dst = target / out.name
        (dst / 'meta').mkdir(parents=True, exist_ok=True)
        rows, cached = [], {}
        for o in manifest['observations']:
            e = o['episode_index']
            if e not in cached:
                cached[e] = {int(r['frame_index']): r for r in pq.read_table(out / f'data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet').to_pylist()}
            rows.append(cached[e][o['frame_index']])
        pq.write_table(pa.Table.from_pylist(rows), dst / 'observations.parquet')
        for f in ('tasks.jsonl', 'episodes.jsonl', 'info.json'):
            (dst / 'meta' / f).write_text((out / 'meta' / f).read_text())
        (dst / 'observation_manifest.json').write_text(json.dumps(dict(manifest, layout='compact: data in observations.parquet, one row per manifest observation in manifest order'), indent=1) + '\n')
        print(json.dumps(dict(split=manifest['split'], rows=len(rows), bytes=(dst / 'observations.parquet').stat().st_size)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('select', 'download', 'build', 'check', 'compact'), required=True)
    p.add_argument('--compact-dir', default=str(ROOT / 'public-reports/baselines/groot/simpler/data'))
    a = p.parse_args()
    if a.stage == 'compact':
        stage_compact(a.compact_dir)
    elif a.stage == 'check':
        for d in OUTS.values():
            m, obs = load_observations(data_dir=d)
            print(d.name, len(obs), obs[0][1]['prompt'], obs[0][1]['observation/state'].round(3).tolist())
    else:
        globals()[f'stage_{a.stage}']()
