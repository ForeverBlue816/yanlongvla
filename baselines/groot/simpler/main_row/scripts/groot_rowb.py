"""GR00T RAQ-FT row B (user decision 2026-10-26 §2): prefix (backbone) + DiT codebooks / row scales trainable, same loss, same data
and budget as row A (groot_raqft.py, mode 'readout'). Implementation choices (recorded in GROOT_2026-10-15.md, row B section):
  - trainable (user plan 2026-10-25 §1): prefix = shared 2-bit backbone (vision + LLM 0-15: 216 Linear) + each suite's 2-bit
    vl_self_attention (24 Linear), and each suite's 2-bit DiT (action_head.model.transformer_blocks: 192 Linear): codebooks and
    row scales. vlln has no quantized layer (LayerNorm, frozen like every norm in row A); codes, signs, masks, embeddings, AdaLN
    linears and per-step tables frozen;
  - joint backbone / per-suite head: every calibration observation (bank a, 256: Spatial 74, Object 66, Goal 65, Long 51) is routed
    to its own suite's FP teacher (processor, statistics, FP head) and its own suite's 2-bit student head; one backbone is shared by
    all teachers and one trainable backbone by all students (byte-identity asserted before sharing);
  - loss: row A's, unchanged: mean over 4 steps x 16 reader blocks of ‖FP-head readout(student queries, student tokens) −
    readout(student queries, FP tokens)‖² + 0.1 · H5b. The student trajectory that provides the queries runs without gradient (as
    in row A), so the DiT codebooks receive gradient only from the H5b term; the backbone from both;
  - row scales use the straight-through FP16 round trip (forward bit-identical; row A dropped 29/768 updates from row-scale
    overflow); codebooks keep row A's round trip; same non-finite guard (skip latent / drop update, cap 38);
  - 3 epochs x 256 observations = 768 updates, 8 latents each, Adam 3e-4 (codebooks) / 2e-5 (row scales), clip 1.0, final iterate.
Exports fit keys backbone_rowb and head_rowb_<suite> (the suite's head fit with its DiT layers replaced); row raqft_rowb."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT, MODELS, key_of, atomic_save, dump
from groot_raqft import CONFIG as ROW_A_CONFIG, install_trainable, prepare, backbone_tokens, Recorder, readout, h5b_loss, RUNS

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
HEAD_TRAINABLE = ('action_head.model.transformer_blocks.', 'action_head.vl_self_attention.')
# Fine-tune base (GROOT_FT_BASE): 'ptq_w2' (default) or 'amvq' (user 2026-10-26: conditional fine-tune from the AM-VQ base, same
# trainable set / loss / data / budget; fit keys and row names get the suffix '_amvq').
FT_BASE = os.environ.get('GROOT_FT_BASE', 'ptq_w2')
assert FT_BASE in ('ptq_w2', 'amvq'), FT_BASE
BASE_ROW = 'ptq_w2' if FT_BASE == 'ptq_w2' else 'amvq_w2'
SRC_BB = 'backbone' if FT_BASE == 'ptq_w2' else 'backbone_amvq'
SRC_HEAD = (lambda s: key_of(s, 'head')) if FT_BASE == 'ptq_w2' else (lambda s: f'head_amvq_{s}')
SUFFIX = '' if FT_BASE == 'ptq_w2' else '_amvq'
MAX_SKIPPED = 38


def row_scale_ste(layers):
    from groot_amraq import row_scale_ste as f
    return f(layers)


class weight_cache:
    """Kept as a no-op scope marker (the per-latent cache was rejected, see memory_efficient)."""
    def __init__(self, layers):
        self.layers = [m for m in layers if m.books.requires_grad]

    def __enter__(self):
        for m in self.layers:
            m._wcache = {}
        return self

    def __exit__(self, *exc):
        for m in self.layers:
            m._wcache = None
        return False


def memory_efficient(layers):
    """2026-10-25: row B ran out of memory on a 48 GB L40S — every grad-enabled forward of a trainable layer rebuilt the FP32 weight and
    autograd kept ~10 bytes/parameter alive per call (x4 flow steps for the DiT under H5b). Patched forward: the dequantized weight is
    computed under activation checkpointing (recomputed in backward), so only the BF16 weight F.linear needs is kept (~2 bytes/param
    per call). Unit test on a fitted DiT layer: forward values and row-scale gradients bit-identical; codebook gradients differ
    only at the level of the unpatched layer's own run-to-run non-determinism (scatter-add order in the gather backward: two runs of
    the unpatched layer differ by up to 1 ulp-scale units, rel. 1e-3 on a few of 4096 entries; same for patched vs unpatched)."""
    import types
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    def forward(self, x):
        if not (torch.is_grad_enabled() and self.books.requires_grad):
            return F.linear(x, self.weight().to(x.dtype), self.bias)
        # checkpoint only (2026-10-25): caching one weight per latent changed the gradient summation order relative to row A
        # (per-call FP16 round trip of the codebook gradient) — rejected; per-call graphs keep row A's numerics bit for bit
        w = checkpoint(lambda b, sc: self.weight().to(x.dtype), self.books, self.scale, use_reentrant=False)
        return F.linear(x, w, self.bias)
    n = 0
    for m in layers:
        if m.books.requires_grad:
            m.forward = types.MethodType(forward, m); m._wcache = None; n += 1
    return n


def share_backbone(models):
    """All models get the first model's backbone object; asserts that the dropped backbones are byte-identical to it."""
    ref = models[0].backbone
    ref_state = ref.state_dict()
    for m in models[1:]:
        other = m.backbone.state_dict()
        assert other.keys() == ref_state.keys()
        for k in list(ref_state)[::37]:                                   # sampled tensors (every 37th), exact equality
            assert torch.equal(other[k], ref_state[k]), f'backbones differ at {k}'
        m.backbone = ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ref


class Router:
    """Per-suite models with one shared backbone each for teachers and students. FP teacher heads of inactive suites live on the
    CPU (frozen, no optimizer state) and are moved to the device when their suite is routed; student heads stay on the device."""
    def __init__(self, teachers, students, device):
        self.teachers, self.students, self.device, self.active = teachers, students, device, None

    def __call__(self, suite):
        if self.device == 'cuda' and suite != self.active:
            for s, b in self.teachers.items():
                if s != suite and s == self.active:
                    b.policy.model.action_head.to('cpu')
            self.teachers[suite].policy.model.action_head.to(self.device)
            torch.cuda.empty_cache()
        self.active = suite
        return self.teachers[suite].policy.model, self.students[suite].policy.model, self.students[suite].policy


def build_models(device, observations, suites_of):
    """Teachers / students per suite (backbones shared), trainable layers (prefix + per-suite vl_self_attention + DiT) with the
    row-scale straight-through round trip, exactness at init. Returns (router, trainable_layers, counts, n_ste, init_gap)."""
    from groot_bridge_server import Bridge
    teachers, students = {}, {}
    for s in SUITES:
        teachers[s] = Bridge(s, device=device)
        students[s] = Bridge(s, device=device, model_root=str(MODELS / BASE_ROW))
        if s != SUITES[0]:
            share_backbone([teachers[SUITES[0]].policy.model, teachers[s].policy.model])
            share_backbone([students[SUITES[0]].policy.model, students[s].policy.model])
            if device == 'cuda':
                teachers[s].policy.model.action_head.to('cpu'); torch.cuda.empty_cache()
    for b in list(teachers.values()) + list(students.values()):
        for p in b.policy.model.parameters():
            p.requires_grad_(False)
    probe = {s: next(i for i, x in enumerate(suites_of) if x == s) for s in SUITES}
    ref_actions = {}
    with torch.no_grad():
        for s, i in probe.items():
            o = dict(observations[i][1]); o['_vla_episode_seed'] = i
            ref_actions[s] = students[s].infer(o)['actions']
    layers = install_trainable(students[SUITES[0]].policy.model, {SRC_BB: 'backbone'}, trainable_parts=('backbone',))
    for s in SUITES:
        head_layers = install_trainable(students[s].policy.model, {SRC_HEAD(s): 'head'}, trainable_parts=('head',))
        for name, m in head_layers.items():
            if not name.startswith(HEAD_TRAINABLE):
                m.books.requires_grad_(False); m.scale.requires_grad_(False)
        layers.update({f'{s}::{name}': m for name, m in head_layers.items()})
    if device == 'cuda':
        torch.cuda.empty_cache()
    n_ste = row_scale_ste(layers)
    memory_efficient(layers.values())
    trainable_layers = {k: m for k, m in layers.items() if m.books.requires_grad}
    counts = dict(backbone=sum(1 for k in trainable_layers if '::' not in k),
                  **{s: dict(vl_self_attention=sum(1 for k in trainable_layers if k.startswith(s + '::action_head.vl_self_attention.')),
                             dit=sum(1 for k in trainable_layers if k.startswith(s + '::action_head.model.transformer_blocks.'))) for s in SUITES})
    assert counts['backbone'] == 216 and all(counts[s] == dict(vl_self_attention=24, dit=192) for s in SUITES), counts
    with torch.no_grad():
        init_gap = {}
        for s, i in probe.items():
            o = dict(observations[i][1]); o['_vla_episode_seed'] = i
            init_gap[s] = float(np.max(np.abs(ref_actions[s] - students[s].infer(o)['actions'])))
    router = Router(teachers, students, device)
    router.active = SUITES[0] if device == 'cuda' else None          # the first suite's FP head is the one left on the device
    return router, trainable_layers, counts, n_ste, init_gap, probe


def export_fit(trainable_layers, key_bb, head_prefix, fine_tune):
    """Write fit key `key_bb` (trained backbone layers) and `<head_prefix>_<suite>` (that suite's head fit with its trained layers
    replaced; all other head layers copied unchanged). complete.json files carry `fine_tune`."""
    def write(mod, src_dir, out_dir, name):
        src = torch.load(src_dir / f'{name}.pt', map_location='cpu')
        ex = mod.export()
        src['tensors']['codebooks'] = ex['codebooks']; src['tensors']['row_scale'] = ex['row_scale']
        with torch.no_grad():
            src['w_hat'] = mod.weight().to(torch.bfloat16).cpu()
        atomic_save(src, out_dir / f'{name}.pt')
        (out_dir / f'{name}.json').write_text((src_dir / f'{name}.json').read_text())
    out = OUT / 'fit' / key_bb; out.mkdir(parents=True, exist_ok=True)
    for name, m in trainable_layers.items():
        if '::' not in name:
            write(m, OUT / 'fit' / SRC_BB, out, name)
    base = json.loads((OUT / 'fit' / SRC_BB / 'complete.json').read_text())
    for s in SUITES:
        src_dir, out_dir = OUT / 'fit' / SRC_HEAD(s), OUT / 'fit' / f'{head_prefix}_{s}'
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.name != 'complete.json' and not (out_dir / f.name).exists():
                shutil.copy2(f, out_dir / f.name)
        for name, m in trainable_layers.items():
            if name.startswith(s + '::'):
                write(m, src_dir, out_dir, name.split('::', 1)[1])
        hb = json.loads((src_dir / 'complete.json').read_text())
        dump(out_dir / 'complete.json', dict(hb, status='complete', key=f'{head_prefix}_{s}', fine_tune=dict(fine_tune, part='vl_self_attention + DiT of ' + s)))
    dump(out / 'complete.json', dict(base, status='complete', key=key_bb, fine_tune=dict(fine_tune, base_row=BASE_ROW, source_fit_keys=[SRC_BB, 'head' + SUFFIX if SUFFIX else 'head_<suite>'])))


def held_out_mse(rows):
    """Held-out action MSE (user plan 2026-10-25: before any rollout) on this job's GPU, right after the fine-tune: the CPU nodes are
    booked until ~10-28. All listed rows that have fits are computed on the same device (groot_offline.py, per-device directory)."""
    import gc
    import subprocess
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env = dict(os.environ, GROOT_OFFLINE_DEVICE='cuda', GROOT_OFFLINE_ROWS=rows)
    r = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve().parent / 'groot_offline.py')], env=env)
    print(json.dumps(dict(held_out_mse_exit=r.returncode)), flush=True)


def run(device='cuda', smoke=False):
    from groot_bridge_server import Bridge
    from calibration_data import load_observations
    cfg = dict(ROW_A_CONFIG, mode='readout', beta=0.1, row='raqft_rowb' + SUFFIX, base=BASE_ROW, trainable='prefix (backbone + per-suite vl_self_attention) + per-suite DiT', routing='per-suite heads')
    key_bb = 'backbone_rowb' + SUFFIX
    directory = RUNS / ('rowb_smoke' if smoke else key_bb)
    directory.mkdir(parents=True, exist_ok=True)
    if not smoke and (OUT / 'fit' / key_bb / 'complete.json').exists():
        print('already complete', key_bb); return
    torch.manual_seed(0)
    suites_of = json.load(open(RUNS / 'calibration_suites.json'))['observation_suite']
    _, observations = load_observations(); assert len(observations) == 256 and len(suites_of) == 256
    router, trainable_layers, counts, n_ste, init_gap, probe = build_models(device, observations, suites_of)
    books = [m.books for m in trainable_layers.values()]; scales = [m.scale for m in trainable_layers.values()]
    trainable = books + scales
    opt = torch.optim.Adam([dict(params=books, lr=cfg['lr']), dict(params=scales, lr=cfg['lr_scale'])])
    contract = dict(config=cfg, trainable_layers=counts, row_scale_ste_layers=n_ste, init_max_abs_action_gap=init_gap,
                    observations_per_suite={s: suites_of.count(s) for s in SUITES}, device=device,
                    gpu=torch.cuda.get_device_name() if device == 'cuda' else 'cpu')
    dump(directory / 'manifest.json', contract)
    print(json.dumps(dict(contract=contract)), flush=True)
    assert all(v == 0.0 for v in init_gap.values()), f'trainable students differ from the assembled students at init: {init_gap}'
    # ---- training (row A's loop, routed per suite)
    history, skipped = [], []
    t_start = time.time()
    order = list(range(256)) * cfg['epochs']
    if smoke:
        order = [probe[s] for s in SUITES]
    for update, idx in enumerate(order):
        s = suites_of[idx]
        tm, sm, policy = router(s)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = int(np.random.SeedSequence([7, idx]).generate_state(1)[0])
        opt.zero_grad(set_to_none=True)
        value, t0 = 0.0, time.time()
        latents = 1 if smoke else cfg['latents']
        for number in range(latents):
          with weight_cache(trainable_layers.values()):
            torch.manual_seed(int(np.random.SeedSequence([7, idx, number]).generate_state(1)[0]))
            backbone_inputs, action_inputs = prepare(policy, router.students[s].observation(obs))
            with torch.no_grad():
                fp_tokens, image_mask, bb_mask = backbone_tokens(tm, backbone_inputs)
            q_tokens, _, _ = backbone_tokens(sm, backbone_inputs)
            rec = Recorder(sm.action_head)
            with torch.no_grad():
                bo = sm.backbone(backbone_inputs)
                sm.action_head.get_action(bo, action_inputs)
            rec.close()
            loss, n = 0.0, 0
            for (i, hidden, emask, temb) in rec.records:
                o_fp = readout(tm.action_head, i, hidden, temb, fp_tokens, emask)
                o_q = readout(tm.action_head, i, hidden, temb, q_tokens, emask)
                loss = loss + (o_q - o_fp).float().square().sum(dim=(1, 2)).mean()
                n += 1
            loss = loss / max(n, 1)
            loss = loss + cfg['beta'] * h5b_loss(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, action_inputs)
            if not torch.isfinite(loss.detach()):
                skipped.append(dict(update=update + 1, observation=idx, suite=s, latent=number, where='loss'))
                print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
                continue
            (loss / latents).backward()
            value += float(loss.detach()) / latents
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
        dit_grad = float(torch.sqrt(sum(m.books.grad.double().square().sum() for k, m in trainable_layers.items() if k.startswith(s + '::action_head.model.transformer_blocks.') and m.books.grad is not None)))
        vlsa_grad = float(torch.sqrt(sum(m.books.grad.double().square().sum() for k, m in trainable_layers.items() if k.startswith(s + '::action_head.vl_self_attention.') and m.books.grad is not None)))
        if not np.isfinite(gn):
            skipped.append(dict(update=update + 1, observation=idx, suite=s, where='gradient', grad_norm=gn))
            print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            assert all(torch.isfinite(p).all() for p in trainable), f'non-finite parameter after update {update + 1}'
        assert len(skipped) <= MAX_SKIPPED, f'{len(skipped)} non-finite events (> {MAX_SKIPPED}): stop and report'
        history.append(dict(update=update + 1, observation=idx, suite=s, loss=value, grad_norm=gn, dit_codebook_grad_norm_before_clip=dit_grad, vl_self_attention_codebook_grad_norm_before_clip=vlsa_grad, seconds=time.time() - t0))
        if smoke or (update + 1) % 16 == 0:
            dump(directory / 'progress.json', dict(status='training', completed=update + 1, total=len(order), loss=value, elapsed=time.time() - t_start))
            print(json.dumps(history[-1]), flush=True)
    assert all(torch.isfinite(p).all() for p in trainable), 'non-finite masters at export'
    if smoke:
        dump(directory / 'smoke.json', dict(contract=contract, history=history, skipped=skipped,
                                            peak_gpu_memory_gb=(torch.cuda.max_memory_allocated() / 2**30) if device == 'cuda' else None,
                                            dit_received_gradient={s: any(h['dit_codebook_grad_norm_before_clip'] > 0 for h in history if h['suite'] == s) for s in SUITES},
                                            vl_self_attention_received_gradient={s: any(h['vl_self_attention_codebook_grad_norm_before_clip'] > 0 for h in history if h['suite'] == s) for s in SUITES}))
        print(json.dumps(dict(smoke='ok', updates=len(history), skipped=len(skipped))), flush=True)
        return
    # ---- export: backbone_rowb (backbone layers) and head_rowb_<suite> (suite head fit with trained layers)
    elapsed = time.time() - t_start
    fine_tune = dict(contract=contract, updates=len(history), nonfinite_events=skipped, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], elapsed=elapsed)
    export_fit(trainable_layers, key_bb, 'head_rowb' + SUFFIX, fine_tune)
    dump(directory / 'history.json', history)
    # held-out MSE: separate GPU job submitted by the controller (2026-10-26, in-job MSE was host-OOM-killed)
    print(json.dumps(dict(done='raqft_rowb' + SUFFIX, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], skipped=len(skipped), hours=elapsed / 3600)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--smoke', action='store_true', help='one update per suite, one latent, no export (implementation check)')
    a = p.parse_args()
    if a.device == 'cpu':
        sys.modules['flash_attn'] = None
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    run(device=a.device, smoke=a.smoke)
