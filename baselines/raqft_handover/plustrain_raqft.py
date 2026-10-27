"""P2 RAQ-FT: readout-consistency fine-tune of a quantized backbone.

L = sum_l w_l * mean_{t,h} || W_{O,h} (o^q_{l,t,h} - o^fp_{l,t,h}) ||^2  +  beta * L_H5b,   beta = 0.1, w_l = 1/18
  o^fp: attention readout of the FP expert at flow step t, layer l, head h, over [FP prefix KV ; suffix KV]
  o^q : the same queries/suffix KV/mask (teacher-forced from the FP expert's own trajectory) read over the
        quantized backbone's prefix KV.  Queries: all 10 steps x 8 heads x 10 action tokens.
  L_H5b: velocity-direction loss of the frozen uniform-M2 expert (steps 0-5) with student prefix KV vs FP prefix KV.
Trainable: backbone codebooks, row scales, backbone norms (FP32 masters; stored dtype roundtrip on every forward).
Frozen: codes, signs, masks, expert, embedding, tables. Fixed budget (final iterate); validation curves are logged
on the 300 Plus-train observations but never used for selection.
"""
import argparse
import fcntl
import json
import os
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers.models.gemma import modeling_gemma
from transformers.cache_utils import DynamicCache

from plustrain_protocol import ROOT, OUT, BASE, RAQFT_MODELS, sha, dump, freeze, load_bank
from plustrain_fit import device
from causal_h5_common import observation, latent, direction_loss
from decisions_h2 import SingleEngine
from decisions_runtime import HybridEngine
from calibration_data import fixed_noise, load_observations
from structure_runtime import masks_for
from backbone_policy import BackboneVQLinear
from backbone_rht import transform
from assemble_evening import write_checkpoint
from story_registry import checkpoint

RAQFT = RAQFT_MODELS
# v2 (E089): per-group learning rates matched to parameter magnitude. v1 used lr 3e-4 for everything and the FP16 row
# scales (|0.034| on average) drifted ~19% in 660 updates, inflating every layer's KV error; codebooks 3e-4 is the H5b value.
CONFIG = dict(version='v2', epochs=3, latents=8, h5b_steps=6, readout_steps=10, beta=0.1, layer_weight=1 / 18, lr=3e-4, lr_scale=2e-5, lr_norm=3e-4, clip=1.0, eval_every=64, validation_latents=1)


def bank_observations(bank):
    if bank == 'libero_a':
        m, rows = load_observations()
        return dict(bank='libero_a', manifest_sha256=sha(ROOT / 'datasets/libero-calibration/observation_manifest.json')), rows
    m, rows = load_bank(bank)
    return dict(bank=bank, manifest_sha256=sha(ROOT / 'datasets/libero-plus-train/banks' / bank / 'observation_manifest.json')), rows


def grid_forward(self, x):
    """INT8-grid training forward (INT8_2026-10-15 item 1): STE round of the codebook masters to integers, STE clamp of the
    two-plane sum to [-127, 127], FP16 per-row scale; the BF16 weight is exactly INT8 x FP16 scale rounded to BF16."""
    books = self.train_books
    books = books + (books.round() - books).detach()
    scale = self.train_scale.to(torch.float16).float()
    weight = sum(books[p][self.codes[p].long()] for p in range(len(self.codebooks)))
    weight = weight + (weight.clamp(-127., 127.) - weight).detach()
    weight = (weight.reshape(self.out_features, self.in_features) * scale[:, None]).bfloat16()
    return F.linear(transform(x.float(), self.signs).bfloat16(), weight, self.bias)


def install_trainable(engine, grid='none'):
    """FP32 masters for codebooks/row scales (FP16 roundtrip) and backbone norms (stored dtype roundtrip).

    grid='int8' (INT8_2026-10-15 item 1): the base is already on the INT8 grid (integer codebooks, FP16 scale / gain); the
    forward rounds the codebook masters to the integer grid and clamps the two-plane sum to [-127, 127], both with a
    straight-through estimator, so every decoded weight stays exactly INT8 x FP16 row scale throughout the fine-tune."""
    model = engine.policy._model
    backbone = model.paligemma_with_expert.paligemma
    layers = {name: m for name, m in model.named_modules() if isinstance(m, BackboneVQLinear)}
    assert len(layers) == 288
    masters = {}
    for name, layer in layers.items():
        layer.register_parameter('train_books', nn.Parameter(layer.codebooks.detach().float().clone()))
        layer.register_parameter('train_scale', nn.Parameter(layer.row_scale.detach().float().clone()))
        masters[name + '.codebooks'] = (layer.train_books, torch.float16)
        masters[name + '.row_scale'] = (layer.train_scale, torch.float16)
        if grid == 'int8':
            from plustrain_int8_runtime import grid_check
            assert grid_check(layer.codebooks, layer.codes), name
            forward = grid_forward
        else:
            def forward(self, x):
                books = self.train_books.to(torch.float16).float()
                scale = self.train_scale.to(torch.float16).float()
                weight = sum(books[p][self.codes[p].long()] for p in range(len(self.codebooks)))
                weight = (weight.reshape(self.out_features, self.in_features) * scale[:, None]).bfloat16()
                return F.linear(transform(x.float(), self.signs).bfloat16(), weight, self.bias)
        layer.forward = types.MethodType(forward, layer)
    norms = {}
    for name, m in backbone.named_modules():
        full = 'paligemma_with_expert.paligemma.' + name
        if isinstance(m, modeling_gemma.GemmaRMSNorm):
            # Backbone RMSNorms are plain (no adaptive dense); the patched openpi forward returns (normed, None).
            assert getattr(m, 'dense', None) is None
            dtype = m.weight.dtype
            m.register_parameter('train_weight', nn.Parameter(m.weight.detach().float().clone()))
            masters[full + '.weight'] = (m.train_weight, dtype)
            def rms_forward(self, x, cond=None, dtype=dtype):
                out_dtype = x.dtype
                normed = self._norm(x)
                normed = normed * (1.0 + self.train_weight.to(dtype).float())
                return normed.to(out_dtype), None
            m.forward = types.MethodType(rms_forward, m)
            norms[full] = m
        elif isinstance(m, nn.LayerNorm):
            dtype = m.weight.dtype
            m.register_parameter('train_weight', nn.Parameter(m.weight.detach().float().clone()))
            m.register_parameter('train_bias', nn.Parameter(m.bias.detach().float().clone()))
            masters[full + '.weight'] = (m.train_weight, dtype)
            masters[full + '.bias'] = (m.train_bias, dtype)
            def ln_forward(self, x, dtype=dtype):
                return F.layer_norm(x, self.normalized_shape, self.train_weight.to(dtype).to(x.dtype), self.train_bias.to(dtype).to(x.dtype), self.eps)
            m.forward = types.MethodType(ln_forward, m)
            norms[full] = m
    trainable = [p for p, _ in masters.values()]
    assert all(p.requires_grad for p in trainable)
    groups = dict(codebooks=[p for n, (p, _) in masters.items() if n.endswith('.codebooks')],
                  row_scale=[p for n, (p, _) in masters.items() if n.endswith('.row_scale')],
                  norms=[p for n, (p, _) in masters.items() if not n.endswith(('.codebooks', '.row_scale'))])
    assert len(groups['codebooks']) == 288 and len(groups['row_scale']) == 288 and len(groups['norms']) == len(masters) - 576
    return layers, norms, masters, trainable, groups


class Readout:
    """Captures the expert's teacher-forced attention inputs over prefix KV for every step/layer.

    queries='fp': the FP expert along its own trajectory with FP prefix KV (default).
    queries='student': the quantized model's own expert along its own trajectory with its (detached) prefix KV;
    o^fp is then the readout of those queries over FP KV. The readout metric W_O is the FP expert's in both cases."""
    def __init__(self, teacher, student=None, layers=tuple(range(18)), queries='fp', metric='wo'):
        self.teacher = teacher
        self.student = student
        self.layers = tuple(layers)
        self.queries = queries
        self.metric = metric            # 'wo': ||W_O,h delta o||^2 (RAQ-FT); 'identity': ||delta o||^2 (no reader metric)
        self.records = []
        self.original = modeling_gemma.eager_attention_forward
        expert = teacher.donors['fp']._model.paligemma_with_expert.gemma_expert.model
        self.wo = [layer.self_attn.o_proj.weight.detach().float() for layer in expert.layers]
        assert all(w.shape == (1024, 2048) for w in self.wo)

    @staticmethod
    def attend(query, k_prefix, v_prefix, ks, vs, mask, scaling):
        """FP32 attention readout over [prefix KV ; suffix KV]: (1, 8, 10, 256). Used identically for o^fp and o^q."""
        k_full = modeling_gemma.repeat_kv(torch.cat([k_prefix, ks], dim=2), 8)
        v_full = modeling_gemma.repeat_kv(torch.cat([v_prefix, vs], dim=2), 8)
        logits = torch.matmul(query.float(), k_full.float().transpose(2, 3)) * scaling + mask.float()
        return torch.matmul(torch.softmax(logits, dim=-1), v_full.float())

    def capture(self, state, padding, kv, noise, student_kv=None):
        """Run the query-source expert along its own trajectory; return per-step list of per-layer
        (query, suffix K, suffix V, mask, scaling, o_fp recomputed in FP32 over FP prefix KV, max |hooked BF16 output - FP32 readout|).
        kv is the FP prefix KV; for queries='student' the trajectory uses the student's own (detached) prefix KV."""
        records = []
        original = self.original
        run_kv = kv if self.queries == 'fp' else student_kv
        assert run_kv is not None
        def instrument(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            out, weights = original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
            if query.shape[-2] == 10 and key.shape[-2] > 10:
                idx = module.layer_idx
                ks, vs = key[:, :, -10:].detach(), value[:, :, -10:].detach()
                mask = attention_mask[:, :, :, :key.shape[-2]].detach()
                o_fp = Readout.attend(query.detach(), kv[idx][0], kv[idx][1], ks, vs, mask, scaling)
                o_run = o_fp if self.queries == 'fp' else Readout.attend(query.detach(), run_kv[idx][0], run_kv[idx][1], ks, vs, mask, scaling)
                hooked = float((o_run - out.detach().transpose(1, 2).float()).abs().max())
                records.append((idx, query.detach(), ks, vs, mask, scaling, o_fp, hooked))
            return out, weights
        x = noise
        try:
            modeling_gemma.eager_attention_forward = instrument
            with torch.no_grad():
                if self.queries == 'fp':
                    for step in range(10):
                        x = x + self.teacher.dt * self.teacher.velocity('fp', step, state, padding, kv, x)
                else:
                    detached = DynamicCache()
                    for l, (k, v) in enumerate(run_kv):
                        detached.update(k.detach(), v.detach(), l)
                    for step in range(10):
                        x = x + self.student.dt * self.student.velocity('q', step, state, padding, detached, x)
        finally:
            modeling_gemma.eager_attention_forward = original
        assert len(records) == 180, len(records)
        steps = [records[s * 18:(s + 1) * 18] for s in range(10)]
        for s in steps:
            assert [r[0] for r in s] == list(range(18))
        return steps

    def loss(self, steps, qkv, prefix_len, check=None):
        total = 0.
        per_layer = torch.zeros(18, dtype=torch.float64, device='cuda')
        for records in steps:
            for idx, query, ks, vs, mask, scaling, o_fp, hooked in records:
                if idx not in self.layers:
                    continue
                kq, vq = qkv[idx][0], qkv[idx][1]
                assert kq.shape[-2] == prefix_len
                o = Readout.attend(query, kq, vq, ks, vs, mask, scaling)   # (1, 8, 10, 256), same FP32 arithmetic as o_fp
                if check is not None:
                    check.append(hooked)
                delta = o - o_fp
                if self.metric == 'wo':
                    wo = self.wo[idx].view(1024, 8, 256)
                    projected = torch.einsum('bhtd,ohd->bhto', delta, wo)  # (1, 8, 10, 1024)
                else:
                    projected = delta                                        # identity metric control
                value = projected.square().sum(dim=(2, 3)).mean()      # mean over heads of sum over tokens/outputs
                per_layer[idx] += value.detach().double()
                total = total + value
        n = len(steps)
        return total / n / len(self.layers), (per_layer / n).tolist()      # uniform w_l over the selected layers


def teacher_path(student, state, padding, kv, x, steps):
    result = []
    with torch.no_grad():
        for step in range(steps):
            velocity = student.velocity('q', step, state, padding, kv, x)
            result.append((x.detach(), velocity.detach()))
            x = x + student.dt * velocity
    return result


def kv_errors(fp, q, padding, slots):
    masks = masks_for(padding, slots)
    result = {}
    with torch.no_grad():
        for group, mask in masks.items():
            f = torch.cat([v[:, :, mask[0], :].flatten() for pair in fp for v in pair]).float()
            a = torch.cat([v[:, :, mask[0], :].detach().flatten() for pair in q for v in pair]).float()
            result[group] = float((a - f).double().square().sum() / f.double().square().sum())
    return result


def make_latent(seed_namespace):
    """Paired latent noise; the default namespace reproduces causal_h5_common.latent, SEED2 gives the second-seed rows."""
    def latent_fn(index, number):
        return torch.from_numpy(np.random.default_rng(np.random.SeedSequence([7, seed_namespace, index, number])).standard_normal((1, 10, 32)).astype('float32')).cuda()
    return latent_fn


def kvmse_loss(fp, q, padding, slots, layers):
    """KV-matching control: per layer, relative squared error of the student's prefix K and V (valid tokens) vs FP, averaged over
    the selected layers. Differentiable through the student's codebooks/scales/norms; same scale-free form for every layer."""
    valid = padding[0].bool()
    total = 0.
    for l in layers:
        for comp in (0, 1):
            f = fp[l][comp][:, :, valid, :].float()
            g = q[l][comp][:, :, valid, :].float()
            total = total + (g - f).square().sum() / f.square().sum().clamp_min(1e-12)
    return total / (2 * len(layers))


def main_term(cfg, readout, steps, q, prefix_len, fp, padding, slots):
    if cfg['loss'] == 'kvmse':
        return kvmse_loss(fp, q, padding, slots, cfg['layers']), [0.0] * 18
    return readout.loss(steps, q, prefix_len)


def evaluate(teacher, student, readout, observations, cfg, latent=latent):
    values = dict(readout=[], h5b=[], image=[], language=[])
    with torch.no_grad():
        for index, (_, raw) in enumerate(observations):
            state, pad, fp = teacher.prefix(observation(teacher.policy, raw))
            obs = observation(student.policy, raw)
            qs, qp, q = student.prefix(obs)
            assert torch.equal(qp, pad)
            noise = latent(100000 + index, 0)
            steps = readout.capture(state, pad, fp, noise, student_kv=q)
            main, _ = main_term(cfg, readout, steps, q, int(pad.shape[1]), fp, pad, obs.tokenized_prompt.shape[-1])
            targets = teacher_path(student, state, pad, fp, noise, cfg['h5b_steps'])
            h5b = sum(float(direction_loss(student.velocity('q', step, qs, qp, q, x), v)) for step, (x, v) in enumerate(targets)) / cfg['h5b_steps']
            err = kv_errors(fp, q, pad, obs.tokenized_prompt.shape[-1])
            values['readout'].append(float(main)); values['h5b'].append(h5b); values['image'].append(err['image']); values['language'].append(err['language'])
            del q, fp, steps, targets
    return {k: float(np.mean(v)) for k, v in values.items()} | dict(observations=len(observations))


MIGRATABLE_SOURCES = ('plustrain_raqft.py', 'plustrain_protocol.py')
DEFAULT_CONFIG_KEYS = dict(loss='readout', grid='none')     # keys added after a row started, with the value that reproduces its old computation


def migrate_contract(directory, contract):
    """Let an interrupted row resume after edits that do not change its computation: only the hashes of this script/the protocol
    may differ, and any config key absent from the old contract must carry its backward-compatible default. Every migration is
    archived (old manifest kept) and appended to contract_migration.json; anything else still fails the frozen-manifest check."""
    path = directory / 'manifest.json'
    if not path.exists():
        return
    old = json.loads(path.read_text())
    if old == contract:
        return
    def strip(c, reference_config):
        c = json.loads(json.dumps(c))
        for n in MIGRATABLE_SOURCES:
            c['sources'].pop(n, None)
        for k, v in DEFAULT_CONFIG_KEYS.items():
            if k not in reference_config and c['config'].get(k) == v:
                c['config'].pop(k)
        return c
    if strip(contract, old['config']) != strip(old, old['config']):
        return                                  # a real change: the strict freeze below reports it
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    (directory / f'manifest.pre_{stamp}.json').write_text(json.dumps(old, indent=2) + '\n')
    resume = directory / 'training_state.pt'
    completed = None
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        assert state['contract'] == old
        state['contract'] = contract
        completed = state['completed']
        tmp = resume.with_suffix('.migrate.partial')
        torch.save(state, tmp)
        tmp.replace(resume)
    path.write_text(json.dumps(contract, indent=2) + '\n')
    log = directory / 'contract_migration.json'
    entries = json.loads(log.read_text()) if log.exists() else []
    if isinstance(entries, dict):
        entries = [entries]
    entries.append(dict(utc=stamp, completed_updates=completed, old_manifest=f'manifest.pre_{stamp}.json',
                        changed_sources={n: [old['sources'].get(n), contract['sources'].get(n)] for n in MIGRATABLE_SOURCES if old['sources'].get(n) != contract['sources'].get(n)},
                        added_default_config={k: v for k, v in DEFAULT_CONFIG_KEYS.items() if k not in old['config']},
                        rule='automatic: only script/protocol hashes changed and added config keys carry their backward-compatible defaults'))
    log.write_text(json.dumps(entries, indent=2) + '\n')
    print(json.dumps(dict(contract_migrated=str(directory), completed_updates=completed)), flush=True)


def run(model):
    torch.set_num_threads(4)
    spec = RAQFT[model]
    cfg = dict(CONFIG, beta=spec.get('beta', CONFIG['beta']), layers=list(spec.get('layers', range(18))), queries=spec.get('queries', 'fp'), latent_seed=spec.get('latent_seed', 20261005), loss=spec.get('loss', 'readout'), grid=spec.get('grid', 'none'))
    cfg['layer_weight'] = 1 / len(cfg['layers'])
    latent = make_latent(cfg['latent_seed'])
    directory = OUT / 'raqft' / model
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank_meta, observations = bank_observations(spec['bank'])
        assert len(observations) == 256
        vmanifest, validation = load_bank('validation')
        assert len(validation) == 300
        contract = dict(model=model, config=cfg, base=spec['base'], bank=bank_meta, base_sha256=sha(checkpoint(dict(model=spec['base'])) / 'model.safetensors'),
                        teacher_sha256=sha(ROOT / 'models/pi05_libero_trimmed/model.safetensors'),
                        validation_manifest_sha256=sha(ROOT / 'datasets/libero-plus-train/banks/validation/observation_manifest.json'),
                        sources={n: sha(ROOT / 'research/scripts' / n) for n in ('plustrain_raqft.py', 'plustrain_protocol.py', 'causal_h5_common.py', 'decisions_h2.py', 'decisions_runtime.py', 'flowvq_cached_runtime.py', 'backbone_policy.py') + (('plustrain_int8_runtime.py',) if cfg['grid'] != 'none' else ())})
        migrate_contract(directory, contract)
        freeze(directory / 'manifest.json', contract)
        dump(directory / 'device.json', device())
        if (directory / 'complete.json').exists():
            assert json.loads((directory / 'complete.json').read_text())['checkpoint_sha256'] == sha(checkpoint(dict(model=model)) / 'model.safetensors')
            return
        teacher = HybridEngine('fp')
        student = SingleEngine(spec['base'])
        teacher.use_graph = False
        student.use_graph = False
        expected = [student.infer(raw, noise=fixed_noise(7))['actions'].copy() for _, raw in observations]
        layers, norms, masters, trainable, groups = install_trainable(student, cfg['grid'])
        for i, (_, raw) in enumerate(observations):
            assert np.array_equal(student.infer(raw, noise=fixed_noise(7))['actions'], expected[i]), ('initial exact action', i)
        del expected
        readout = Readout(teacher, student, cfg['layers'], cfg['queries'], metric='identity' if cfg['loss'] == 'readout_identity' else 'wo')
        # Readout self-check: with FP prefix KV on both sides the readout loss is exactly zero, and the hooked BF16
        # expert output matches the FP32 recomputation loosely (insertion point) on the first observation.
        with torch.no_grad():
            state, pad, fp = teacher.prefix(observation(teacher.policy, observations[0][1]))
            steps = readout.capture(state, pad, fp, latent(0, 0), student_kv=fp)
            check = []
            zero, _ = readout.loss(steps, fp, int(pad.shape[1]), check)
            # o^fp and o^q share one FP32 readout, so the FP self-loss must vanish; the hooked BF16 expert output is
            # compared loosely (BF16 softmax/matmul rounding) as an insertion-point sanity check only.
            assert float(zero) == 0. and max(check) < 0.25, (max(check), float(zero))
            readout_check = dict(max_abs_bf16_hooked_vs_fp32=max(check), fp_self_loss=float(zero))
            del steps, fp
        if cfg['grid'] == 'int8':
            # Masters live in grid units (codebooks x gain, scale / gain): per-layer learning rates keep the original dynamics.
            gains = json.loads((checkpoint(dict(model=spec['base'])) / 'vq.json').read_text())['int8_grid']['gains']
            def gain_of(name):
                key = name if name in gains else name.replace('paligemma_with_expert.paligemma.', 'paligemma_with_expert.paligemma.model.', 1)
                return gains[key]
            param_groups = [dict(params=[p], lr=cfg['lr'] * gain_of(n.removesuffix('.codebooks'))) for n, (p, _) in masters.items() if n.endswith('.codebooks')]
            param_groups += [dict(params=[p], lr=cfg['lr_scale'] / gain_of(n.removesuffix('.row_scale'))) for n, (p, _) in masters.items() if n.endswith('.row_scale')]
            param_groups.append(dict(params=groups['norms'], lr=cfg['lr_norm']))
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam([dict(params=groups['codebooks'], lr=cfg['lr']), dict(params=groups['row_scale'], lr=cfg['lr_scale']), dict(params=groups['norms'], lr=cfg['lr_norm'])])
        resume = directory / 'training_state.pt'
        history = []
        evaluations = []
        completed = 0
        elapsed_prior = 0.
        if resume.exists():
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            assert saved['contract'] == contract
            with torch.no_grad():
                for name, (p, _) in masters.items():
                    p.copy_(saved['masters'][name].cuda())
            optimizer.load_state_dict(saved['optimizer'])
            history, evaluations, completed, elapsed_prior = saved['history'], saved['evaluations'], saved['completed'], saved['elapsed_seconds']
        started = time.monotonic()
        total = cfg['epochs'] * 256
        def validate(update):
            result = dict(update=update, **evaluate(teacher, student, readout, validation, cfg, latent))
            evaluations.append(result)
            dump(directory / 'curves.json', dict(status='training' if update < total else 'complete', training_updates=history, validation=evaluations, readout_check=readout_check,
                                                 validation_scope='300 Plus-train observations; latent 0; descriptive only, no selection'))
            print(json.dumps(dict(validation=result)), flush=True)
        if not evaluations:
            validate(0)
        for update in range(completed, total):
            epoch, index = divmod(update, 256)
            raw = observations[index][1]
            with torch.no_grad():
                state, padding, fp = teacher.prefix(observation(teacher.policy, raw))
            obs = observation(student.policy, raw)
            optimizer.zero_grad(set_to_none=True)
            value = dict(readout=0., h5b=0., total=0.)
            layer_values = np.zeros(18)
            errors = None
            for number in range(cfg['latents']):
                noise = latent(index, number)
                qs, qp, q = student.prefix(obs)
                assert torch.equal(qp, padding)
                steps = readout.capture(state, padding, fp, noise, student_kv=q)
                targets = teacher_path(student, state, padding, fp, noise, cfg['h5b_steps'])
                if number == 0:
                    errors = kv_errors(fp, q, padding, obs.tokenized_prompt.shape[-1])
                main, per_layer = main_term(cfg, readout, steps, q, int(padding.shape[1]), fp, padding, obs.tokenized_prompt.shape[-1])
                h5b = sum(direction_loss(student.velocity('q', step, qs, qp, q, x), target) for step, (x, target) in enumerate(targets)) / cfg['h5b_steps']
                loss = (main + cfg['beta'] * h5b) / cfg['latents']
                assert torch.isfinite(loss)
                value['readout'] += float(main.detach()) / cfg['latents']; value['h5b'] += float(h5b.detach()) / cfg['latents']; value['total'] += float(loss.detach())
                layer_values += np.array(per_layer) / cfg['latents']
                loss.backward()
                del q, loss, main, h5b, targets, steps
            gradients = [p.grad for p in trainable if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
            assert norm > 0
            optimizer.step()
            history.append(dict(update=update + 1, epoch=epoch, observation_index=index, loss=value, readout_per_layer=layer_values.tolist(), gradient_norm=norm, pre_update_KV=errors))
            if (update + 1) % cfg['eval_every'] == 0 or update + 1 == total:
                validate(update + 1)
            elapsed = elapsed_prior + time.monotonic() - started
            tmp = resume.with_suffix('.partial')
            torch.save(dict(contract=contract, masters={name: p.detach().cpu() for name, (p, _) in masters.items()}, optimizer=optimizer.state_dict(), history=history, evaluations=evaluations, completed=update + 1, elapsed_seconds=elapsed), tmp)
            tmp.replace(resume)
            dump(directory / 'progress.json', dict(status='training', completed=update + 1, total=total, loss=value, elapsed_seconds=elapsed))
            print(json.dumps(history[-1] | dict(readout_per_layer=None)), flush=True)
            del fp
        base = checkpoint(dict(model=spec['base']))
        tensors = load_file(str(base / 'model.safetensors'))
        old = json.loads((base / 'vq.json').read_text())
        changes = {}
        for name, (p, dtype) in masters.items():
            key = name
            if key not in tensors:
                key = name.replace('paligemma_with_expert.paligemma.', 'paligemma_with_expert.paligemma.model.', 1) if 'paligemma_with_expert.paligemma.model.' not in name else name
            assert key in tensors, name
            value = p.detach()
            if cfg['grid'] == 'int8' and name.endswith('.codebooks'):
                value = value.round()
            value = value.to(dtype).cpu()
            assert tensors[key].dtype == value.dtype and tensors[key].shape == value.shape and torch.isfinite(value).all(), key
            changes[key] = value
        for k, v in changes.items():
            tensors[k] = v
        meta = {k: v for k, v in old.items() if k not in ('checkpoint_sha256', 'checkpoint_file_bytes', 'variant', 'tensor_payload_bytes', 'whole_model_bpw')}
        meta.update(raqft=dict(contract=contract, optimizer_steps=len(history), changed_fields=dict(codebooks=288, row_scale=288, norms=len(changes) - 576), curves_sha256=sha(directory / 'curves.json')))
        if cfg['grid'] == 'int8':
            from plustrain_int8_runtime import grid_check, decoded_units, INT8_MAX
            resnapped = {}
            for lname in old['backbone_layers']:
                books, codes = tensors[lname + '.codebooks'], tensors[lname + '.codes']
                if not grid_check(books, codes):
                    # The STE clamp was active at the end: rescale this layer so the stored (unclamped) decode is within INT8.
                    amax = float(decoded_units(books, codes).abs().max())
                    factor = (INT8_MAX - 1) / amax
                    tensors[lname + '.codebooks'] = (books.float() * factor).round().to(torch.float16)
                    tensors[lname + '.row_scale'] = (tensors[lname + '.row_scale'].float() / factor).to(torch.float16)
                    assert grid_check(tensors[lname + '.codebooks'], codes), lname
                    resnapped[lname] = dict(amax_before=amax, factor=factor)
            meta['int8_grid'] = dict(old['int8_grid'], trained_backbone=True, resnapped_layers=resnapped)
        result = write_checkpoint(ROOT, model, tensors, meta)
        assert result['tensor_payload_bytes'] == old['tensor_payload_bytes']
        loaded = load_file(str(checkpoint(dict(model=model)) / 'model.safetensors'))
        assert set(loaded) == set(tensors) and all(torch.equal(loaded[n], v) and loaded[n].dtype == v.dtype for n, v in tensors.items())
        dump(directory / 'complete.json', dict(status='complete', model=model, checkpoint_sha256=result['checkpoint_sha256'], whole_model_bpw=result['whole_model_bpw'], optimizer_steps=len(history), changed_tensors=len(changes),
                                               elapsed_seconds=elapsed_prior + time.monotonic() - started, initial_exact_observations=256, strict_reload_all_tensors=True, contract=contract, readout_check=readout_check, curves_sha256=sha(directory / 'curves.json'), device=device()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=tuple(RAQFT), required=True)
    run(p.parse_args().model)
