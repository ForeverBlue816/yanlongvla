"""GR00T row tcq_raqft (user spec 2026-11-01 §3, TCQ_RAQFT_2026-11-01.md): the B2 recipe (groot_rowb2.py: student-head readouts vs the
FP head on the FP trajectory, W_O metric, + β·L_vel with β by gradient-norm matching at step 0; prefix + per-suite heads trainable, one
shared backbone against the four suite heads; bank a, 8 latents, 768 updates, Adam, clip 1, final iterate) on the TCQ base tcq_w2.
Trainable per quantized layer: the 2^9 x 2 lookup table (FP32 master, FP16 round trip in the forward = B2's codebook path) and the row
scales (FP32 master, B2's straight-through FP16 round trip); frozen: trellis codes, rotation signs. Plus every RMSNorm / LayerNorm
`weight` of the shared backbone and of the four heads (third Adam group at the row-scale learning rate 2e-5; the spec fixes the set, the
rate is the agent's choice, recorded in the manifest). Gates before step 1: non-zero tlut and scale gradient in every trainable layer on
the first batch, DiT-side gradient norm > 1e-4 (B2's gate).
Exports fit keys backbone_tcq_raqft + head_tcq_raqft_<suite> (tlut / row scales / w_hat replaced; norm weights in norm_weights.safetensors,
applied by groot_ptq.stage_assemble); row tcq_raqft."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT, MODELS, SUITES, atomic_save, dump, inverse_rht
from groot_raqft import CONFIG as ROW_A_CONFIG, RUNS
from groot_rowb import share_backbone, Router, memory_efficient, weight_cache, HEAD_TRAINABLE
from groot_rowb2 import losses, grad_norm, MAX_SKIPPED, CKPT_EVERY
from groot_amraq import ReaderTaps
from groot_tcq import KEY_OF, G16_KEY_OF, RHT_KEY_OF, RHT_G16_KEY_OF, signed_index, unpack, make_codebook, tlut_init, TCQ, group_index, inverse_rotation, METRIC_KEY_OF, D_KEY_OF

VARIANTS = {'v0': dict(base_row='tcq_w2', key='backbone_tcq_raqft', head_prefix='head_tcq_raqft', source=KEY_OF, row='tcq_raqft'),
            'g16': dict(base_row='tcq_w2_g16', key='backbone_tcq_raqft_g16', head_prefix='head_tcq_raqft_g16', source=G16_KEY_OF, row='tcq_raqft_g16'),   # addendum 2026-11-01: row-group tables + column scales
            'v0_rht': dict(base_row='tcq_w2_rht', key='backbone_tcq_raqft_rht', head_prefix='head_tcq_raqft_rht', source=RHT_KEY_OF, row='tcq_raqft_rht'),             # user 2026-11-02: two-sided-rotation base
            'g16_rht': dict(base_row='tcq_w2_rht_g16', key='backbone_tcq_raqft_rht_g16', head_prefix='head_tcq_raqft_rht_g16', source=RHT_G16_KEY_OF, row='tcq_raqft_rht_g16'),
            'v0s_rht': dict(base_row='tcq_w2_rht', key='backbone_tcq_raqft_rht_s', head_prefix='head_tcq_raqft_rht_s', source=RHT_KEY_OF, row='tcq_raqft_rht_s'),
            'v0s_c': dict(base_row='tcq_w2_rht_shift', key='backbone_tcq_raqft_c_s', head_prefix='head_tcq_raqft_c_s', source=METRIC_KEY_OF['shift'], row='tcq_raqft_c_s'),      # user 2026-11-07 #4: V0' recipe on arm C
            'v0s_d1': dict(base_row='tcq_w2_rht_d1', key='backbone_tcq_raqft_d1_s', head_prefix='head_tcq_raqft_d1_s', source=D_KEY_OF[1], row='tcq_raqft_d1_s'),               # user 2026-11-07 #2: V0' recipe on the better of D1 / D2
            'v0s_d2': dict(base_row='tcq_w2_rht_d2', key='backbone_tcq_raqft_d2_s', head_prefix='head_tcq_raqft_d2_s', source=D_KEY_OF[2], row='tcq_raqft_d2_s')}   # user 2026-11-04: V0' stabilized (groot_tcq_raqft_s.py)
V = dict(VARIANTS['v0'])
LR_TABLES = None
BASE_ROW, KEY, HEAD_PREFIX = V['base_row'], V['key'], V['head_prefix']
NORM_FILE = 'norm_weights.safetensors'


def set_variant(name):
    global V, BASE_ROW, KEY, HEAD_PREFIX
    V = dict(VARIANTS[name], name=name)
    BASE_ROW, KEY, HEAD_PREFIX = V['base_row'], V['key'], V['head_prefix']


class TrainableTCQLinear(nn.Module):
    """TCQ Linear with trainable lookup table (`books`, 512 x 2) and row scale (`scale`); forward decodes the frozen trellis through the
    table (FP16 round trip), applies the straight-through FP16 row scale and undoes the RHT every call (as groot_raqft.TrainableVQLinear)."""
    def __init__(self, rec, cb, bias=None):
        super().__init__()
        t = rec['tensors']
        m, n = rec['w_hat'].shape
        states = unpack(cb, t['trellis'], m, n)
        self.register_buffer('index', signed_index(states))                     # (m, n) int16: sign * (flat tlut index + 1)
        self.register_buffer('signs', t['signs'].clone())                       # (n,) int8 rotation signs
        self.register_buffer('signs_out', t['signs_out'].clone() if 'signs_out' in t else torch.zeros(0, dtype=torch.int8))   # (m,) output-side signs (two-sided base) or empty
        self.books = nn.Parameter(t['tlut'].float().clone())                    # (512, 2) FP32 master, or (G, 512, 2) row-group tables (V1)
        self.scale = nn.Parameter(t['row_scale'].float().clone())               # (m,) FP32 master
        self.groups = self.books.shape[0] if self.books.dim() == 3 else 1
        if self.groups > 1:
            self.register_buffer('group', group_index(m, self.groups, 'cpu'))     # (m,) row -> table
        self.col_scale = nn.Parameter(t['col_scale'].float().clone()) if 'col_scale' in t else None   # (n,) FP32 master, rotated domain (V1)
        self.bias = None if bias is None else nn.Parameter(bias.clone(), requires_grad=False)
        self.out_features, self.in_features = m, n
        self.log_scale = None                                                   # V0' (2026-11-04): row scale trained as log s

    def to_log_scale(self):
        self.log_scale = nn.Parameter(self.scale.detach().clamp_min(torch.finfo(torch.float16).tiny).log().clone())
        self.scale.requires_grad_(False)

    def row_scale(self):
        return self.log_scale.exp() if self.log_scale is not None else self.scale

    def weight(self):
        books = self.books.to(torch.float16).float().flatten()
        s = self.row_scale()
        scale = s + (s.to(torch.float16).float() - s).detach()
        idx = self.index.long()
        if self.groups > 1:
            dec = books.view(self.groups, -1)[self.group[:, None], idx.abs() - 1] * idx.sign().to(books.dtype)
        else:
            dec = books[idx.abs() - 1] * idx.sign().to(books.dtype)
        q = dec * scale[:, None]
        if self.col_scale is not None:
            cs = self.col_scale + (self.col_scale.to(torch.float16).float() - self.col_scale).detach()
            q = q * cs[None, :]
        return inverse_rotation(q, self.signs, self.signs_out if self.signs_out.numel() else None)

    def scale_params(self):
        return [self.log_scale if self.log_scale is not None else self.scale] + ([self.col_scale] if self.col_scale is not None else [])

    def set_trainable(self, flag):
        self.books.requires_grad_(flag)
        for p in self.scale_params():
            p.requires_grad_(flag)

    def forward(self, x):
        return F.linear(x, self.weight().to(x.dtype), self.bias)

    def export(self):
        out = dict(tlut=self.books.detach().to(torch.float16).cpu(), row_scale=self.row_scale().detach().to(torch.float16).cpu())
        if self.col_scale is not None:
            out['col_scale'] = self.col_scale.detach().to(torch.float16).cpu()
        return out


def install_trainable(model, key, cb, trainable):
    layers = {}
    d = OUT / 'fit' / key
    for p in sorted(d.glob('*.pt')):
        name = p.name[:-3]
        rec = torch.load(p, map_location='cpu')
        src = model.get_submodule(name)
        mod = TrainableTCQLinear(rec, cb, src.bias.detach() if src.bias is not None else None).to(src.weight.device)
        if not trainable:
            mod.set_trainable(False)
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, mod)
        layers[name] = mod
        del rec
    return layers


def norm_weights(module, prefix=''):
    """{name: parameter} of every RMSNorm / LayerNorm `weight` under `module` (class name contains 'Norm', excluding the AdaLN Linear)."""
    out = {}
    for name, m in module.named_modules():
        if 'Norm' in type(m).__name__ and isinstance(getattr(m, 'weight', None), torch.Tensor) and m.weight.dim() == 1:
            out[(prefix + name).strip('.') + '.weight'] = m.weight
    return out


def build_models(device, observations, suites_of):
    from groot_bridge_server import Bridge
    cb = make_codebook(tlut_init().to(device), device)
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
    layers = install_trainable(students[SUITES[0]].policy.model, V['source']['backbone'], cb, trainable=True)
    for s in SUITES:
        head_layers = install_trainable(students[s].policy.model, V['source'][f'head_{s}'], cb, trainable=True)
        for name, m in head_layers.items():
            if not name.startswith(HEAD_TRAINABLE):
                m.set_trainable(False)
        layers.update({f'{s}::{name}': m for name, m in head_layers.items()})
    norms = norm_weights(students[SUITES[0]].policy.model.backbone, 'backbone.')
    for s in SUITES:
        norms.update({f'{s}::{k}': v for k, v in norm_weights(students[s].policy.model.action_head, 'action_head.').items()})
    for p in norms.values():
        p.requires_grad_(True)
    if device == 'cuda':
        torch.cuda.empty_cache()
    memory_efficient(layers.values())
    trainable_layers = {k: m for k, m in layers.items() if m.books.requires_grad}
    counts = dict(backbone=sum(1 for k in trainable_layers if '::' not in k),
                  **{s: dict(vl_self_attention=sum(1 for k in trainable_layers if k.startswith(s + '::action_head.vl_self_attention.')),
                             dit=sum(1 for k in trainable_layers if k.startswith(s + '::action_head.model.transformer_blocks.'))) for s in SUITES},
                  norm_weights=dict(backbone=sum(1 for k in norms if '::' not in k), **{s: sum(1 for k in norms if k.startswith(s + '::')) for s in SUITES}))
    assert counts['backbone'] == 216 and all(counts[s] == dict(vl_self_attention=24, dit=192) for s in SUITES), counts
    with torch.no_grad():
        init_gap = {}
        for s, i in probe.items():
            o = dict(observations[i][1]); o['_vla_episode_seed'] = i
            init_gap[s] = float(np.max(np.abs(ref_actions[s] - students[s].infer(o)['actions'])))
    router = Router(teachers, students, device)
    router.active = SUITES[0] if device == 'cuda' else None
    return router, trainable_layers, norms, counts, init_gap, probe


def export_fit(trainable_layers, norms, fine_tune):
    from safetensors.torch import save_file

    def write(mod, src_dir, out_dir, name):
        src = torch.load(src_dir / f'{name}.pt', map_location='cpu')
        ex = mod.export()
        src['tensors'].update(ex)
        with torch.no_grad():
            src['w_hat'] = mod.weight().to(torch.bfloat16).cpu()
        atomic_save(src, out_dir / f'{name}.pt')
        (out_dir / f'{name}.json').write_text((src_dir / f'{name}.json').read_text())
    out = OUT / 'fit' / KEY; out.mkdir(parents=True, exist_ok=True)
    for name, m in trainable_layers.items():
        if '::' not in name:
            write(m, OUT / 'fit' / V['source']['backbone'], out, name)
    save_file({k: v.detach().cpu().contiguous() for k, v in norms.items() if '::' not in k}, str(out / NORM_FILE))
    base = json.loads((OUT / 'fit' / V['source']['backbone'] / 'complete.json').read_text())
    for s in SUITES:
        src_dir, out_dir = OUT / 'fit' / V['source'][f'head_{s}'], OUT / 'fit' / f'{HEAD_PREFIX}_{s}'
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.name != 'complete.json' and not (out_dir / f.name).exists():
                shutil.copy2(f, out_dir / f.name)
        for name, m in trainable_layers.items():
            if name.startswith(s + '::'):
                write(m, src_dir, out_dir, name.split('::', 1)[1])
        save_file({k.split('::', 1)[1]: v.detach().cpu().contiguous() for k, v in norms.items() if k.startswith(s + '::')}, str(out_dir / NORM_FILE))
        hb = json.loads((src_dir / 'complete.json').read_text())
        dump(out_dir / 'complete.json', dict(hb, status='complete', key=f'{HEAD_PREFIX}_{s}', fine_tune=dict(fine_tune, part='vl_self_attention + DiT + norm weights of ' + s)))
    dump(out / 'complete.json', dict(base, status='complete', key=KEY, fine_tune=dict(fine_tune, base_row=BASE_ROW, source_fit_keys=[V['source']['backbone'], V['source']['head_libero_spatial'].replace('libero_spatial', '<suite>')])))


def run(device='cuda', smoke=False):
    cfg = dict(ROW_A_CONFIG, mode=V['row'], row=V['row'], variant=V.get('name', 'v0'), base=BASE_ROW, executed=8, valid_dims=7, lr_norm=ROW_A_CONFIG['lr_scale'], lr=LR_TABLES if LR_TABLES is not None else ROW_A_CONFIG['lr'],
               lr_tables_note='2026-11-02: B2 rate 3e-4 aborted V0 at update 416 (39 non-finite gradients, loss 2x B2); a TCQ table scalar is shared by 4x more weights than an RVQ codebook scalar (m n / 1024 vs m n / 4096), so the table rate is divided by 4' if LR_TABLES is not None else 'B2 rate',
               trainable='prefix (backbone + per-suite vl_self_attention) + per-suite DiT: tlut + row scales; RMSNorm / LayerNorm weights (backbone + heads)' + (' + per-row-group tables (G by layer size) + column scales (rotated domain, FP16 round trip, scale rate)' if V.get('name') == 'g16' else ''),
               routing='per-suite heads', loss='B2: L_readout(student-head readouts vs FP-head readouts on the FP trajectory, W_O metric) + beta * L_vel')
    directory = RUNS / (f'{KEY}_smoke' if smoke else KEY); directory.mkdir(parents=True, exist_ok=True)
    if not smoke and (OUT / 'fit' / KEY / 'complete.json').exists():
        print('already complete', KEY); return
    torch.manual_seed(0)
    suites_of = json.load(open(RUNS / 'calibration_suites.json'))['observation_suite']
    from calibration_data import load_observations
    _, observations = load_observations(); assert len(observations) == 256 and len(suites_of) == 256
    router, trainable_layers, norms, counts, init_gap, probe = build_models(device, observations, suites_of)
    assert all(v == 0.0 for v in init_gap.values()), init_gap
    books = [m.books for m in trainable_layers.values()]; scales = [p for m in trainable_layers.values() for p in m.scale_params()]; norm_params = list(norms.values())
    trainable = books + scales + norm_params
    bb_books = [m.books for k, m in trainable_layers.items() if '::' not in k]
    dit_books = [m.books for k, m in trainable_layers.items() if '::action_head.model.transformer_blocks.' in k]
    opt = torch.optim.Adam([dict(params=books, lr=cfg['lr']), dict(params=scales, lr=cfg['lr_scale']), dict(params=norm_params, lr=cfg['lr_norm'])])
    taps_t = {s: ReaderTaps(router.teachers[s].policy.model.action_head) for s in SUITES}
    taps_s = {s: ReaderTaps(router.students[s].policy.model.action_head) for s in SUITES}
    wo = {s: {i: router.teachers[s].policy.model.action_head.model.transformer_blocks[i].attn1.to_out[0].weight.detach().float().to(device) for i in taps_t[s].cross} for s in SUITES}
    latent_seed = lambda idx, number: int(np.random.SeedSequence([7, idx, number]).generate_state(1)[0])
    ckpt = directory / 'ckpt.pt'
    gpu = torch.cuda.get_device_name() if device == 'cuda' else 'cpu'
    if ckpt.exists() and not smoke:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        for p, v in zip(trainable, state['trainable']):
            p.data.copy_(v)
        opt.load_state_dict(state['opt']); history, skipped, beta, start = state['history'], state['skipped'], state['beta'], state['update']
        contract = json.loads((directory / 'manifest.json').read_text())
        print(json.dumps(dict(resumed_at=start, gpu=gpu)), flush=True)
    else:
        idx = 0; s = suites_of[idx]; tm, sm, policy = router(s)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        with weight_cache(trainable_layers.values()):
            l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, 0), cfg, wo[s])
            opt.zero_grad(set_to_none=True); l_read.backward(retain_graph=True)
            g_bb_read, g_dit_read = grad_norm(bb_books), grad_norm(dit_books)
            opt.zero_grad(set_to_none=True); l_vel.backward()
            g_bb_vel, g_dit_vel = grad_norm(bb_books), grad_norm(dit_books)
            opt.zero_grad(set_to_none=True)
        assert g_dit_vel > 0, 'L_vel gives the DiT tables no gradient'
        beta = g_bb_read / g_dit_vel
        calibration = dict(L_readout=float(l_read), L_vel=float(l_vel), grad_norm_backbone_tlut_from_L_readout=g_bb_read, grad_norm_dit_tlut_from_L_readout=g_dit_read,
                           grad_norm_backbone_tlut_from_L_vel=g_bb_vel, grad_norm_dit_tlut_from_L_vel=g_dit_vel, beta=beta, dit_from_beta_L_vel=beta * g_dit_vel,
                           rule='beta = ||g_bb(L_readout)|| / ||g_DiT(L_vel)||, set once at step 0 (B2), no further tuning')
        # gates on the first batch (8 latents): DiT-side gradient norm > 1e-4; non-zero tlut and scale gradient in every trainable layer
        opt.zero_grad(set_to_none=True)
        n_lat = 1 if smoke else cfg['latents']
        for number in range(n_lat):
            with weight_cache(trainable_layers.values()):
                l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, number), cfg, wo[s])
                ((l_read + beta * l_vel) / n_lat).backward()
        g_dit_batch, g_bb_batch, g_norm_batch = grad_norm(dit_books), grad_norm(bb_books), grad_norm(norm_params)
        zero_books = [k for k, m in trainable_layers.items() if (k.startswith(s + '::') or '::' not in k) and (m.books.grad is None or not bool((m.books.grad != 0).any()))]
        zero_scales = [k for k, m in trainable_layers.items() if (k.startswith(s + '::') or '::' not in k) and any(p.grad is None or not bool((p.grad != 0).any()) for p in m.scale_params())]
        opt.zero_grad(set_to_none=True)
        gate = dict(dit_tlut_grad_norm_first_batch=g_dit_batch, backbone_tlut_grad_norm_first_batch=g_bb_batch, norm_weight_grad_norm_first_batch=g_norm_batch, threshold=1e-4,
                    layers_checked=sum(1 for k in trainable_layers if k.startswith(s + '::') or '::' not in k), zero_tlut_gradient=zero_books, zero_scale_gradient=zero_scales,
                    passed=bool(g_dit_batch > 1e-4 and not zero_books and not zero_scales), note='layers of the routed suite + backbone (the other suites\' heads see no gradient on this batch by construction)')
        dump(directory / 'calibration.json', dict(calibration=calibration, gate=gate, gpu=gpu))
        print(json.dumps(dict(calibration=calibration, gate=dict(gate, zero_tlut_gradient=len(zero_books), zero_scale_gradient=len(zero_scales)))), flush=True)
        if not gate['passed']:
            dump(directory / 'decision.json', dict(status='aborted', reason='gate failed', gate=gate, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
            raise SystemExit(3)
        contract = dict(config=dict(cfg, beta=beta), trainable_layers=counts, init_max_abs_action_gap=init_gap, observations_per_suite={s_: suites_of.count(s_) for s_ in SUITES},
                        calibration=calibration, gate=gate, device=device, gpu=gpu, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        dump(directory / 'manifest.json', contract)
        history, skipped, start = [], [], 0
    t_start = time.time()
    order = [probe[s_] for s_ in SUITES] if smoke else list(range(256)) * cfg['epochs']
    for update in range(start, len(order)):
        idx = order[update]; s = suites_of[idx]
        tm, sm, policy = router(s)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        opt.zero_grad(set_to_none=True)
        value, vread, vvel, t0 = 0.0, 0.0, 0.0, time.time()
        n_lat = 1 if smoke else cfg['latents']
        for number in range(n_lat):
            with weight_cache(trainable_layers.values()):
                l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, number), cfg, wo[s])
                loss = l_read + beta * l_vel
                if not torch.isfinite(loss.detach()):
                    skipped.append(dict(update=update + 1, observation=idx, suite=s, latent=number, where='loss'))
                    print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
                    continue
                (loss / n_lat).backward()
                value += float(loss.detach()) / n_lat; vread += float(l_read.detach()) / n_lat; vvel += float(l_vel.detach()) / n_lat
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
        g_dit = grad_norm(dit_books); g_nrm = grad_norm(norm_params)
        if not np.isfinite(gn):
            skipped.append(dict(update=update + 1, observation=idx, suite=s, where='gradient', grad_norm=gn))
            print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            assert all(torch.isfinite(p).all() for p in trainable), f'non-finite parameter after update {update + 1}'
        assert len(skipped) <= MAX_SKIPPED, f'{len(skipped)} non-finite events (> {MAX_SKIPPED}): stop and report'
        history.append(dict(update=update + 1, observation=idx, suite=s, loss=value, L_readout=vread, L_vel=vvel, grad_norm=gn, dit_tlut_grad_norm_after_clip=g_dit, norm_weight_grad_norm_after_clip=g_nrm, seconds=time.time() - t0))
        if smoke or (update + 1) % 16 == 0:
            dump(directory / 'progress.json', dict(status='training', completed=update + 1, total=len(order), loss=value, elapsed=time.time() - t_start))
            print(json.dumps(history[-1]), flush=True)
        if not smoke and (update + 1) % CKPT_EVERY == 0 and update + 1 < len(order):
            torch.save(dict(update=update + 1, trainable=[p.detach().cpu() for p in trainable], opt=opt.state_dict(), history=history, skipped=skipped, beta=beta), ckpt.with_suffix('.partial'))
            ckpt.with_suffix('.partial').replace(ckpt)
    assert all(torch.isfinite(p).all() for p in trainable), 'non-finite masters at export'
    if smoke:
        dump(directory / 'smoke.json', dict(contract=contract, history=history, skipped=skipped, peak_gpu_memory_gb=(torch.cuda.max_memory_allocated() / 2**30) if device == 'cuda' else None,
                                            dit_received_gradient={s_: any(h['dit_tlut_grad_norm_after_clip'] > 0 for h in history if h['suite'] == s_) for s_ in SUITES}))
        print(json.dumps(dict(smoke='ok', updates=len(history), skipped=len(skipped), peak_gb=torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else None)), flush=True)
        return
    fine_tune = dict(contract=contract, updates=len(history), nonfinite_events=skipped, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], elapsed=time.time() - t_start, beta=beta)
    export_fit(trainable_layers, norms, fine_tune)
    dump(directory / 'history.json', history)
    ckpt.unlink(missing_ok=True)
    print(json.dumps(dict(done=V['row'], first_loss=history[0]['loss'], final_loss=history[-1]['loss'], skipped=len(skipped), hours=(time.time() - t_start) / 3600)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--smoke', action='store_true', help='one update per suite, one latent, no export')
    p.add_argument('--lr-tables', type=float, default=None, help='Adam rate for the tables (default: B2 codebook rate 3e-4)')
    p.add_argument('--variant', choices=tuple(VARIANTS), default='v0', help='v0: per-layer tlut + row scales + norms; g16: + row-group tables and column scales (base tcq_w2_g16)')
    a = p.parse_args()
    set_variant(a.variant)
    LR_TABLES = a.lr_tables
    if a.device == 'cpu':
        sys.modules['flash_attn'] = None
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    run(device=a.device, smoke=a.smoke)
