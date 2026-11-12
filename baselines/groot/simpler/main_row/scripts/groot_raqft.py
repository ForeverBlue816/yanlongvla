"""GR00T N1.7 readout-consistency fine-tune of the 2-bit backbone codebooks (evaluation-GPU plan, Day 6-9), runs in envs/groot.

Student: both-2-bit model (ptq_w2 fits), backbone codebooks and row scales trainable (FP32 masters, FP16 round trip), codes,
signs and the 2-bit action head frozen. Teacher: the FP model. Reader (as in pi0.5 RAQ-FT, student queries): at every flow step
t and every cross-attention DiT block b, the student's own normed hidden state h_{t,b} (its trajectory, its 2-bit head) is read by
the FP head's cross-attention (vlln -> vl_self_attention -> block_b.norm1/attn1, all FP) over the student's backbone tokens (o^q) and
over the FP backbone tokens (o^fp); L = mean_{t,b} ||o^q - o^fp||^2 (attn1 includes W_O, so this is the W_O-metric readout loss).
Control: output distillation, MSE of the student's velocity vs the FP velocity along the student's trajectory, same trainables.
Data: the 256 LIBERO calibration observations (same bank as every row), 2 epochs, 2 latents each; Adam lr 3e-4 (codebooks) /
2e-5 (row scales), clip 1.0; final iterate. One backbone fine-tune with the libero_spatial head (the backbone is shared by the
four suite checkpoints; the four heads stay as fitted). Output: new backbone fit key (backbone_raqft / backbone_distill) ->
groot_ptq assemble rows raqft_bb / distill_bb -> LIBERO medium + Plus through the controller.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT, ROOT, MODELS, SUITES, key_of, atomic_save, dump, inverse_rht

RUNS = OUT / 'raqft'
MAX_SKIPPED = 38          # 5% of 768 updates; more non-finite events than this stops the run (reported, not silently trained through)
CONFIG = dict(version='v2', epochs=3, latents=8, lr=3e-4, lr_scale=2e-5, clip=1.0, beta=0.1, eval_every=64, head_suite='libero_spatial')   # row A spec (AM-RAQ track, 2026-10-23): bank a, 8 latents, 768 updates, + 0.1 H5b


class TrainableVQLinear(nn.Module):
    """2-bit RVQ Linear with trainable codebooks/row scale; forward dequantizes (FP16 round trip) and undoes the RHT every call."""
    def __init__(self, rec, bias=None):
        super().__init__()
        t = rec['tensors']
        self.register_buffer('codes', t['codes'].clone())                       # (P, N, K/8) uint8
        self.register_buffer('signs', rec['signs'].clone())                     # (K,) int8
        self.books = nn.Parameter(t['codebooks'].float().clone())              # (P, 256, 8) FP32 master
        self.scale = nn.Parameter(t['row_scale'].float().clone())              # (N,) FP32 master
        self.bias = None if bias is None else nn.Parameter(bias.clone(), requires_grad=False)
        self.out_features, self.groups = self.codes.shape[1], self.codes.shape[2]
        self.in_features = self.groups * 8

    def weight(self):
        books = self.books.to(torch.float16).float()
        scale = self.scale.to(torch.float16).float()
        dec = sum(books[p][self.codes[p].long()] for p in range(books.shape[0])).reshape(self.out_features, self.in_features) * scale[:, None]
        return inverse_rht(dec, self.signs)

    def forward(self, x):
        return F.linear(x, self.weight().to(x.dtype), self.bias)

    def export(self):
        return dict(codebooks=self.books.detach().to(torch.float16).cpu(), row_scale=self.scale.detach().to(torch.float16).cpu())


def install_trainable(model, part_keys, trainable_parts):
    """Replace the quantized Linear layers of `model` by TrainableVQLinear from the fit records; returns {name: module}."""
    layers = {}
    for key, part in part_keys.items():
        d = OUT / 'fit' / key
        for p in sorted(d.glob('*.pt')):
            name = p.name[:-3]
            rec = torch.load(p, map_location='cpu')
            src = model.get_submodule(name)
            mod = TrainableVQLinear(rec, src.bias.detach() if src.bias is not None else None).to(src.weight.device)
            if part not in trainable_parts:
                mod.books.requires_grad_(False); mod.scale.requires_grad_(False)
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, mod)
            layers[name] = mod
    return layers


def prepare(policy, observation):
    """Gr00tPolicy._get_action steps 1-3 for one observation; returns the model's (backbone_inputs, action_inputs) on device."""
    from gr00t.policy.gr00t_policy import _rec_to_dtype
    from gr00t.data.types import MessageType
    unbatched = policy._unbatch_observation(observation)
    vla = policy._to_vla_step_data(unbatched[0])
    collated = policy.collate_fn([policy.processor([{'type': MessageType.EPISODE_STEP.value, 'content': vla}])])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    return policy.model.prepare_input(collated['inputs'])          # the collator returns BatchFeature({'inputs': batch})


def backbone_tokens(model, backbone_inputs):
    """FP-reader side: backbone -> vlln -> vl_self_attention (the tokens every cross-attention block reads)."""
    out = model.backbone(backbone_inputs)
    feats = model.action_head.vlln(out['backbone_features'])
    feats = model.action_head.vl_self_attention(feats)
    return feats, out['image_mask'], out['backbone_attention_mask']


class Recorder:
    """Captures, per flow step, each cross-attention block's normed hidden state (query side) and its encoder mask."""
    def __init__(self, head):
        self.blocks = list(head.model.transformer_blocks)
        self.cross = [i for i, b in enumerate(self.blocks) if getattr(b, 'cross_attention_dim', None)]
        self.records = []
        self.handles = []
        for i in self.cross:
            b = self.blocks[i]
            def pre(mod, args, kwargs, i=i):
                self.records.append((i, args[0] if args else kwargs['hidden_states'], kwargs.get('encoder_attention_mask'), kwargs.get('temb')))
            self.handles.append(b.register_forward_pre_hook(pre, with_kwargs=True))

    def close(self):
        for h in self.handles:
            h.remove()


def readout(fp_head, block_index, hidden, temb, tokens, mask):
    """FP head's cross-attention of block `block_index`: norm1(hidden, temb) -> attn1 over `tokens` with `mask` (includes W_O)."""
    b = fp_head.model.transformer_blocks[block_index]
    normed = b.norm1(hidden, temb)
    if b.pos_embed is not None:
        normed = b.pos_embed(normed)
    return b.attn1(normed, encoder_hidden_states=tokens, attention_mask=mask)


def run(mode):
    from groot_bridge_server import Bridge
    from calibration_data import load_observations
    assert mode in ('readout', 'distill')
    cfg = dict(CONFIG, mode=mode)
    key_out = 'backbone_raqft' if mode == 'readout' else 'backbone_distill'
    directory = RUNS / key_out
    directory.mkdir(parents=True, exist_ok=True)
    if (OUT / 'fit' / key_out / 'complete.json').exists():
        print('already complete', key_out)
        return
    torch.manual_seed(0)
    suite = cfg['head_suite']
    teacher = Bridge(suite)                                              # FP checkpoint (shards verified)
    student = Bridge(suite, model_root=str(MODELS / 'ptq_w2'))           # both-2-bit assembled checkpoint
    tm, sm = teacher.policy.model, student.policy.model
    for p in tm.parameters():
        p.requires_grad_(False)
    for p in sm.parameters():
        p.requires_grad_(False)
    layers = install_trainable(sm, {'backbone': 'backbone', key_of(suite, 'head'): 'head'}, trainable_parts=('backbone',))
    trainable = [p for m in layers.values() for p in (m.books, m.scale) if p.requires_grad]
    books = [m.books for m in layers.values() if m.books.requires_grad]
    scales = [m.scale for m in layers.values() if m.scale.requires_grad]
    assert len(books) == 216, len(books)
    opt = torch.optim.Adam([dict(params=books, lr=cfg['lr']), dict(params=scales, lr=cfg['lr_scale'])])
    _, observations = load_observations()
    assert len(observations) == 256
    # exactness at init: the trainable student reproduces the assembled student's actions
    obs0 = dict(observations[0][1]); obs0['_vla_episode_seed'] = 0
    with torch.no_grad():
        a_ref = Bridge(suite, model_root=str(MODELS / 'ptq_w2')).infer(dict(obs0))['actions']
        a_init = student.infer(dict(obs0))['actions']
    init_gap = float(np.max(np.abs(a_ref - a_init)))
    contract = dict(config=cfg, suite=suite, trainable=dict(codebooks=len(books), row_scales=len(scales)), init_max_abs_action_gap=init_gap)
    dump(directory / 'manifest.json', contract)
    history = []
    skipped = []
    t_start = time.time()
    total = cfg['epochs'] * 256
    fp_head = tm.action_head
    rec = Recorder(sm.action_head)
    for update in range(total):
        idx = update % 256
        obs = dict(observations[idx][1])
        obs['_vla_episode_seed'] = int(np.random.SeedSequence([7, idx]).generate_state(1)[0])
        opt.zero_grad(set_to_none=True)
        value = 0.0
        for number in range(cfg['latents']):
            torch.manual_seed(int(np.random.SeedSequence([7, idx, number]).generate_state(1)[0]))
            # inputs exactly as Gr00tPolicy._get_action builds them (identical processors for teacher and student)
            backbone_inputs, action_inputs = prepare(student.policy, student.observation(obs))
            with torch.no_grad():
                fp_tokens, image_mask, bb_mask = backbone_tokens(tm, backbone_inputs)
            q_tokens, _, _ = backbone_tokens(sm, backbone_inputs)                       # grads -> backbone codebooks
            if mode == 'readout':
                # student trajectory with its own (2-bit) head over its own tokens, hidden states captured per cross block/step
                rec.records.clear()
                with torch.no_grad():
                    bo = sm.backbone(backbone_inputs)
                    sm.action_head.get_action(bo, action_inputs)
                loss = 0.0
                n = 0
                for (i, hidden, emask, temb) in rec.records:
                    o_fp = readout(fp_head, i, hidden, temb, fp_tokens, emask)
                    o_q = readout(fp_head, i, hidden, temb, q_tokens, emask)
                    loss = loss + (o_q - o_fp).float().square().sum(dim=(1, 2)).mean()
                    n += 1
                loss = loss / max(n, 1)
                if cfg['beta'] > 0:
                    loss = loss + cfg['beta'] * h5b_loss(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, action_inputs)
            else:
                # output distillation control: velocity along the student's trajectory vs the FP head over FP tokens, same noise
                loss = distill_loss(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, action_inputs)
            if not torch.isfinite(loss.detach()):
                # 2026-10-24: job 221246 went NaN between updates 48 and 64 and never recovered (one non-finite step poisons
                # the masters). A latent with a non-finite loss contributes nothing; the event is recorded.
                skipped.append(dict(update=update + 1, observation=idx, latent=number, where='loss'))
                print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
                continue
            (loss / cfg['latents']).backward()
            value += float(loss.detach()) / cfg['latents']
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
        if not np.isfinite(gn):
            skipped.append(dict(update=update + 1, observation=idx, where='gradient', grad_norm=gn))
            print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            assert all(torch.isfinite(p).all() for p in trainable), f'non-finite parameter after update {update + 1}'
        assert len(skipped) <= MAX_SKIPPED, f'{len(skipped)} non-finite events (> {MAX_SKIPPED}): stop and report'
        history.append(dict(update=update + 1, loss=value, grad_norm=gn))
        if (update + 1) % 16 == 0:
            dump(directory / 'progress.json', dict(status='training', completed=update + 1, total=total, loss=value, elapsed=time.time() - t_start))
            print(json.dumps(history[-1]), flush=True)
    rec.close()
    assert all(torch.isfinite(p).all() for p in trainable), 'non-finite masters at export'
    out = OUT / 'fit' / key_out
    out.mkdir(parents=True, exist_ok=True)
    for name, mod in layers.items():
        if not mod.books.requires_grad:
            continue
        src = torch.load(OUT / 'fit/backbone' / f'{name}.pt', map_location='cpu')
        ex = mod.export()
        src['tensors']['codebooks'] = ex['codebooks']; src['tensors']['row_scale'] = ex['row_scale']
        with torch.no_grad():
            src['w_hat'] = mod.weight().to(torch.bfloat16).cpu()
        atomic_save(src, out / f'{name}.pt')
        meta = json.loads((OUT / 'fit/backbone' / f'{name}.json').read_text())
        (out / f'{name}.json').write_text(json.dumps(meta) + '\n')
    base = json.loads((OUT / 'fit/backbone/complete.json').read_text())
    dump(out / 'complete.json', dict(base, status='complete', key=key_out, fine_tune=dict(contract=contract, updates=len(history), nonfinite_events=skipped, final_loss=history[-1]['loss'], first_loss=history[0]['loss'], elapsed=time.time() - t_start)))
    dump(directory / 'history.json', history)
    print(json.dumps(dict(done=key_out, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], init_gap=init_gap, hours=(time.time() - t_start) / 3600)), flush=True)


def direction_loss(student, teacher):
    a = student.float().flatten(1); b = teacher.detach().float().flatten(1)
    return (1 - F.cosine_similarity(a, b, dim=1, eps=1e-12)).mean()


def velocities(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, action_inputs, seed):
    """Student (own head, own tokens) and FP (FP head, FP tokens) velocities along the student's trajectory, same noise."""
    from transformers.feature_extraction_utils import BatchFeature
    head_s, head_t = sm.action_head, tm.action_head
    bo_s = BatchFeature(data=dict(backbone_features=q_tokens, image_mask=image_mask, backbone_attention_mask=bb_mask))
    bo_t = BatchFeature(data=dict(backbone_features=fp_tokens, image_mask=image_mask, backbone_attention_mask=bb_mask))
    fs, ft = head_s._encode_features(bo_s, action_inputs), head_t._encode_features(bo_t, action_inputs)
    emb = action_inputs.embodiment_id
    g = torch.Generator(device=q_tokens.device).manual_seed(seed)
    actions = torch.randn(1, head_s.config.action_horizon, head_s.action_dim, dtype=q_tokens.dtype, device=q_tokens.device, generator=g)
    dt = 1.0 / head_s.num_inference_timesteps
    pairs = []
    for t in range(head_s.num_inference_timesteps):
        tdisc = torch.full((1,), int(t / head_s.num_inference_timesteps * head_s.num_timestep_buckets), device=q_tokens.device)
        def velocity(head, feats, tokens):
            af = head.action_encoder(actions, tdisc, emb)
            if head.config.add_pos_embed:
                af = af + head.position_embedding(torch.arange(af.shape[1], device=af.device)).unsqueeze(0)
            sa = torch.cat((feats.state_features, af), dim=1)
            out = head.model(hidden_states=sa, encoder_hidden_states=tokens, timestep=tdisc, image_mask=image_mask, backbone_attention_mask=bb_mask)
            return head.action_decoder(out, emb)[:, -head.action_horizon:]
        v_s = velocity(head_s, fs, q_tokens)
        with torch.no_grad():
            v_t = velocity(head_t, ft, fp_tokens)
        pairs.append((v_s, v_t))
        actions = (actions + dt * v_s).detach()
    return pairs


def h5b_loss(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, action_inputs, seed=0):
    """Velocity-direction loss (1 - cos) of the student's velocity vs the FP velocity along the student's trajectory, 4 steps."""
    pairs = velocities(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, action_inputs, seed)
    return sum(direction_loss(v_s[..., :7], v_t[..., :7]) for v_s, v_t in pairs) / len(pairs)


def distill_loss(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, action_inputs):
    """Velocity MSE along the student's trajectory: student head over student tokens vs FP head over FP tokens, same noise/queries."""
    from transformers.feature_extraction_utils import BatchFeature
    head_s, head_t = sm.action_head, tm.action_head
    bo_s = BatchFeature(data=dict(backbone_features=q_tokens, image_mask=image_mask, backbone_attention_mask=bb_mask))
    bo_t = BatchFeature(data=dict(backbone_features=fp_tokens, image_mask=image_mask, backbone_attention_mask=bb_mask))
    fs = head_s._encode_features(bo_s, action_inputs)
    ft = head_t._encode_features(bo_t, action_inputs)
    emb = action_inputs.embodiment_id
    actions = torch.randn(1, head_s.config.action_horizon, head_s.action_dim, dtype=q_tokens.dtype, device=q_tokens.device)
    dt = 1.0 / head_s.num_inference_timesteps
    loss = 0.0
    for t in range(head_s.num_inference_timesteps):
        tdisc = torch.full((1,), int(t / head_s.num_inference_timesteps * head_s.num_timestep_buckets), device=q_tokens.device)
        def velocity(head, feats, tokens):
            af = head.action_encoder(actions, tdisc, emb)
            if head.config.add_pos_embed:
                af = af + head.position_embedding(torch.arange(af.shape[1], device=af.device)).unsqueeze(0)
            sa = torch.cat((feats.state_features, af), dim=1)
            out = head.model(hidden_states=sa, encoder_hidden_states=tokens, timestep=tdisc, image_mask=image_mask, backbone_attention_mask=bb_mask)
            return head.action_decoder(out, emb)[:, -head.action_horizon:]
        v_s = velocity(head_s, fs, q_tokens)
        with torch.no_grad():
            v_t = velocity(head_t, ft, fp_tokens)
        loss = loss + (v_s.float() - v_t.float()).square().mean()
        actions = (actions + dt * v_s).detach()
    return loss / head_s.num_inference_timesteps


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('readout', 'distill'), required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    run(a.mode)


# ------------------------------------------------------------------------------------------------- row A readout difference
# Δo definition (pinned 2026-10-25; identical wording in groot_amraq.py, GROOT_2026-10-15.md and GROOT_AMRAQ_2026-10-23.md):
# Δo_{l,t} is the pre-W_O output of the FP head's cross-attention in reader block l, evaluated with the student's own step-t
# hidden state as query over the student's backbone tokens, minus the same over the FP backbone tokens; the difference is taken
# in FP32 from the BF16 readouts.
class PreWO:
    """Captures the input of attn1.to_out[0] (the pre-W_O readout, all heads concatenated) of one head's reader blocks."""
    def __init__(self, head):
        self.captured = {}
        self.handles = []
        for i, b in enumerate(head.model.transformer_blocks):
            if getattr(b, 'cross_attention_dim', None):
                self.handles.append(b.attn1.to_out[0].register_forward_pre_hook(lambda mod, args, i=i: self.captured.__setitem__(i, args[0])))

    def close(self):
        for h in self.handles:
            h.remove()


def row_a_deltas(sm, tm, policy, obs, seed):
    """Row A's Δo exactly as run(mode='readout') above computes its readouts: the same calls in the same order after
    torch.manual_seed(seed) (prepare -> FP tokens -> student tokens -> student trajectory through get_action -> FP-head readout
    with the student's queries over student / FP tokens), returned before W_O. List in record order of
    (step, block, Δo in FP32, ‖post-W_O difference‖²). Added 2026-10-25 for AM-RAQ check (d); the training loop is untouched."""
    from groot_bridge_server import Bridge
    torch.manual_seed(seed)
    backbone_inputs, action_inputs = prepare(policy, Bridge.observation(None, obs))
    with torch.no_grad():
        fp_tokens, image_mask, bb_mask = backbone_tokens(tm, backbone_inputs)
        q_tokens, _, _ = backbone_tokens(sm, backbone_inputs)
        rec = Recorder(sm.action_head)
        bo = sm.backbone(backbone_inputs)
        sm.action_head.get_action(bo, action_inputs)
        rec.close()
        cap = PreWO(tm.action_head)
        nb = len(rec.cross)
        out = []
        for k, (i, hidden, emask, temb) in enumerate(rec.records):
            o_q = readout(tm.action_head, i, hidden, temb, q_tokens, emask); pre_q = cap.captured[i]
            o_fp = readout(tm.action_head, i, hidden, temb, fp_tokens, emask); pre_fp = cap.captured[i]
            out.append((k // nb, i, pre_q.float() - pre_fp.float(), float((o_q - o_fp).float().square().sum())))
        cap.close()
    return out
