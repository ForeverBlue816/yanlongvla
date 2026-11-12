"""GR00T N1.7 AM-RAQ: action-metric readout consistency (GROOT_AMRAQ_2026-10-23.md). Only the readout metric of the fine-tune
loss changes relative to row A (groot_raqft.py); trainable set, data, budget, optimizer, base checkpoint and evaluation are row A's.

Readouts o_{l,t}: the cross-attention outputs before W_O (input of attn1.to_out[0]) of the 16 reader blocks, 4 flow steps, all
heads and action tokens. o^FP: FP model on its own trajectory; o^q: the quantized model on its own trajectory (same observation
and noise). Δo = o^q − o^FP.
L_AM = Σ_t ‖Ω(t)^{1/2} · JVP(v^FP_t; o^FP_t → Δo_t)‖²: one forward-mode pass of the FP expert at step t on the FP trajectory with the
tangent injected at all reader blocks (torch.autograd.functional.jvp over the injection sites; exact J_{v←o} Δo with all cross-block
interactions, no backprop through the sampling loop). Gradient to the student: dL/dΔo = 2 Jᵀ Ω u via one VJP, then o^q.backward(g).
Ω(t): γ^h = 1 on the 8 executed steps, 0.5 on steps 8–15, 0 on padded steps/dims; λ_DoF = 1 / E‖J_d‖² per group (translation,
rotation, gripper), Hutchinson estimate on the calibration set from the FP model; ω(t) uniform. L = L_AM / s + β·L_H5b with s the
calibration mean of L_AM at step 0 (fixed). Modes: am (β = 0.1), am0 (β = 0).
Checks (before training, recorded): JVP vs finite differences in FP32 (ε = 1e-3, random direction, 2 observations); wall-clock per
update A vs A-AM (fallback to the diagonal metric if > 2×); Spearman(L_AM with Ω = I and one block, L_A) over 64 samples.
Diagnostics: per-block, per-step ‖Ω^{1/2} J_{v←o_l}‖_F² (text vs image reader blocks) from the FP model; readout errors under both
metrics for base / A / A-AM (stage diag, after the fits).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT, ROOT, MODELS, key_of, atomic_save, dump
from groot_raqft import TrainableVQLinear, install_trainable, prepare, backbone_tokens, h5b_loss, readout as local_readout, Recorder

RUNS = OUT / 'raqft'
CONFIG = dict(version='v1', epochs=3, latents=8, lr=3e-4, lr_scale=2e-5, clip=1.0, eval_every=64, head_suite='libero_spatial',
              executed=8, valid_steps=16, valid_dims=7, gamma_rest=0.5, groups=dict(translation=[0, 1, 2], rotation=[3, 4, 5], gripper=[6]),
              lambda_probes=4, lambda_observations=32, s_observations=64)


# ------------------------------------------------------------------------------------------------------------- reader hooks
class ReaderTaps:
    """Capture (and optionally perturb) the pre-W_O cross-attention outputs of the reader blocks of one action head."""
    def __init__(self, head):
        self.blocks = list(head.model.transformer_blocks)
        self.cross = [i for i, b in enumerate(self.blocks) if getattr(b, 'cross_attention_dim', None)]
        self.captured = {}
        self.inject = None          # dict block_index -> tangent tensor added to the pre-W_O readout
        self.handles = []
        for i in self.cross:
            lin = self.blocks[i].attn1.to_out[0]
            def pre(mod, args, i=i):
                x = args[0]
                self.captured[i] = x
                if self.inject is not None and i in self.inject:
                    return (x + self.inject[i],) + tuple(args[1:])
                return None
            self.handles.append(lin.register_forward_pre_hook(pre))

    def close(self):
        for h in self.handles:
            h.remove()


def head_velocity(head, feats, tokens, image_mask, bb_mask, emb, actions, t):
    tdisc = torch.full((1,), int(t / head.num_inference_timesteps * head.num_timestep_buckets), device=tokens.device)
    af = head.action_encoder(actions, tdisc, emb)
    if head.config.add_pos_embed:
        af = af + head.position_embedding(torch.arange(af.shape[1], device=af.device)).unsqueeze(0)
    sa = torch.cat((feats.state_features, af), dim=1)
    out = head.model(hidden_states=sa, encoder_hidden_states=tokens, timestep=tdisc, image_mask=image_mask, backbone_attention_mask=bb_mask)
    return head.action_decoder(out, emb)[:, -head.action_horizon:]


class _State:
    def __init__(self, state_features):
        self.state_features = state_features


def omega(cfg, lambdas, horizon, dim, device):
    w = torch.zeros(horizon, dim, device=device)
    w[:cfg['executed'], :cfg['valid_dims']] = 1.0
    w[cfg['executed']:cfg['valid_steps'], :cfg['valid_dims']] = cfg['gamma_rest']
    for name, dims in cfg['groups'].items():
        w[:, dims] *= lambdas[name]
    return w


def jvp_and_vjp(model, taps, step, tokens, image_mask, bb_mask, tangents, omega_w, need_vjp=True):
    """u = J Δo (forward-mode via the double-VJP trick inside autograd.functional.jvp) at the FP step; optionally g = 2 Jᵀ(Ω u)."""
    head = model.action_head
    blocks = taps.cross
    order = [i for i in blocks if i in tangents]
    def f(*deltas):
        taps.inject = {i: d for i, d in zip(order, deltas)}
        try:
            v = head_velocity(head, step['feats'], tokens, image_mask, bb_mask, step['emb'], step['actions'], step['t'])
        finally:
            taps.inject = None
        return v if v.dtype == torch.float64 else v.float()
    zeros = tuple(torch.zeros_like(step['readouts'][i]) for i in order)
    tan = tuple(tangents[i].to(step['readouts'][i].dtype) for i in order)
    # autograd.functional.jvp = double-backward trick; the fused SDPA kernels (efficient/flash) have no double backward, so the
    # forward-mode product runs on the math SDPA backend (same arithmetic up to rounding); the VJP below uses the default kernels
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        _, u = torch.autograd.functional.jvp(f, zeros, tan)
    loss = (omega_w * u.square()).sum()
    g = None
    if need_vjp:
        _, grads = torch.autograd.functional.vjp(f, zeros, 2.0 * omega_w * u)
        g = {i: gi for i, gi in zip(order, grads)}
    return u, loss, g


# -------------------------------------------------------------------------------------------------------- calibration set-up
def calibrate_lambda(cfg, route, observations, device, exclude=()):
    """λ_d = 1 / E‖J_d‖² (Hutchinson, N(0, I) tangents at all reader blocks), E over observations and steps; FP model of each
    observation's suite (route), pooled over suites."""
    acc = {k: [] for k in cfg['groups']}
    for idx in range(cfg['lambda_observations']):
        if idx in exclude:
            continue
        suite_, tm, sm, taps_t, policy = route(idx)
        ones = torch.ones(tm.action_head.config.action_horizon, tm.action_head.action_dim, device=device)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        fp = fp_trajectory(tm, taps_t, policy, obs, idx); steps, fp_tokens, image_mask, bb_mask = fp['steps'], fp['fp_tokens'], fp['image_mask'], fp['bb_mask']
        for t, st in enumerate(steps):
            st['t'] = t
            for _ in range(cfg['lambda_probes']):
                r = {i: torch.randn_like(o) for i, o in st['readouts'].items()}     # N(0, I): E‖J_d r‖² = ‖J_d‖_F² (spec λ, 2026-10-25)
                u, _, _ = jvp_and_vjp(tm, taps_t, st, fp_tokens, image_mask, bb_mask, r, ones, need_vjp=False)
                gam = omega(cfg, {k: 1.0 for k in cfg['groups']}, u.shape[1], u.shape[2], device)
                for k, dims in cfg['groups'].items():
                    acc[k].append(float((gam[:, dims] * u[0][:, dims].square()).sum()))
    return {k: 1.0 / max(np.mean(v), 1e-30) for k, v in acc.items()}, {k: float(np.mean(v)) for k, v in acc.items()}


def calibrate_diag(cfg, route, observations, device, omega_w, probes=8):
    """Static diagonal metric D_l = E[(Jᵀ Ω^{1/2} z)²] (z Rademacher over the action window), per reader block and readout element,
    Hutchinson over `probes` probes x all calibration observations x 4 steps on the FP model; used only if the JVP loss costs >2x row A."""
    acc, count = {}, {}
    root = omega_w.sqrt()
    for idx in range(len(observations)):
        suite_, tm, sm, taps_t, policy = route(idx)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        fp = fp_trajectory(tm, taps_t, policy, obs, idx); steps, fp_tokens, image_mask, bb_mask = fp['steps'], fp['fp_tokens'], fp['image_mask'], fp['bb_mask']
        for t, st in enumerate(steps):
            st['t'] = t
            order = list(st['readouts'])
            def f(*deltas):
                taps_t.inject = {i: d for i, d in zip(order, deltas)}
                try:
                    v = head_velocity(tm.action_head, st['feats'], fp_tokens, image_mask, bb_mask, st['emb'], st['actions'], t)
                finally:
                    taps_t.inject = None
                return v.float()
            zeros = tuple(torch.zeros_like(st['readouts'][i]) for i in order)
            for _ in range(probes):
                z = torch.randint(0, 2, zeros[0].shape[:1] + root.shape, device=device).float() * 2 - 1
                _, grads = torch.autograd.functional.vjp(f, zeros, root * z)
                for i, g in zip(order, grads):
                    acc.setdefault(suite_, {})[i] = acc.get(suite_, {}).get(i, 0.0) + g.float().square()
                count[suite_] = count.get(suite_, 0) + 1
    return {s_: {i: (v / count[s_]).detach() for i, v in a_.items()} for s_, a_ in acc.items()}     # per suite (per FP head)


def amdiag_loss_for(cfg, tm, sm, taps_t, policy, obs, seed, diag, s_scale=1.0, backprop=True, extra_fn=None):
    """Fallback metric: L = Σ_t Σ_l Σ_i D_{l,i} Δo² on row A's Δo (readout_deltas); no JVP/VJP."""
    d = readout_deltas(tm, sm, taps_t, policy, obs, seed, grad=backprop)
    loss = 0.0
    for r in d['deltas']:
        loss = loss + (diag[r['block']] * (r['o_q'].float() - r['o_fp'].float()).square()).sum()
    if backprop:
        obj = loss / s_scale
        extra = extra_fn(d) if extra_fn is not None else None
        (obj + extra if extra is not None else obj).backward()       # one backward: the H5b term shares the student backbone graph
        return float(loss), float(extra) if extra is not None else 0.0
    return float(loss), d


def row_scale_ste(layers):
    """Decision 2026-10-26 (option A, A-AM only): the row scale's FP16 round trip becomes straight-through, so the forward value is
    bit-identical (scale + (fp16(scale) - scale).detach() == fp16(scale) exactly in FP32) but its gradient stays FP32 and cannot
    overflow (row A: 29/768 updates dropped, all from row-scale gradients > 65 504). The codebook path is unchanged (row A's: FP16
    round trip in both directions; codebook gradients never overflowed in row A). Row A itself is not re-run."""
    import types
    from groot_raqft import inverse_rht

    def weight(self):
        books = self.books.to(torch.float16).float()
        scale = self.scale + (self.scale.to(torch.float16).float() - self.scale).detach()
        dec = sum(books[p][self.codes[p].long()] for p in range(books.shape[0])).reshape(self.out_features, self.in_features) * scale[:, None]
        return inverse_rht(dec, self.signs)
    n = 0
    for m in layers.values():
        if m.scale.requires_grad:
            with torch.no_grad():
                w0 = m.weight()
            m.weight = types.MethodType(weight, m)
            with torch.no_grad():
                assert torch.equal(w0, m.weight()), 'row-scale STE changed the forward weight'
            n += 1
    return n


def to_model_dtype(x, model):
    """Cast the floating tensors of a prepare() output (BF16) to the model's parameter dtype; a no-op for the deployed BF16 models."""
    dtype = next(model.backbone.parameters()).dtype
    if dtype != torch.bfloat16:
        for k, v in list(x.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                x[k] = v.to(dtype)
    return x


def policy_obs(policy, obs):
    from groot_bridge_server import Bridge
    return Bridge.observation(None, obs)


# ------------------------------------------------------------------------------------------- Δo (row A) and FP trajectory
# Δo definition (pinned 2026-10-25; identical wording in groot_raqft.py, GROOT_2026-10-15.md and GROOT_AMRAQ_2026-10-23.md):
# Δo_{l,t} is the pre-W_O output of the FP head's cross-attention in reader block l, evaluated with the student's own step-t
# hidden state as query over the student's backbone tokens, minus the same over the FP backbone tokens; the difference is taken
# in FP32 from the BF16 readouts.
class StepCapture:
    """Per-step inputs of head.get_action: the noisy actions entering each step (pre-hook on action_encoder), the state features
    (hook on state_encoder) and, through `taps`, the pre-W_O readouts of each step."""
    def __init__(self, head, taps=None):
        self.taps, self.actions, self.readouts, self.state = taps, [], [], None

        def enc_pre(mod, args):
            if self.taps is not None:
                if self.actions:
                    self.readouts.append(dict(self.taps.captured))     # captured so far = previous step
                self.taps.captured = {}
            self.actions.append(args[0].detach().clone())

        def state_hook(mod, args, out):
            self.state = out.detach()
        self.handles = [head.action_encoder.register_forward_pre_hook(enc_pre), head.state_encoder.register_forward_hook(state_hook)]

    def finish(self):
        if self.taps is not None:
            self.readouts.append(dict(self.taps.captured)); self.taps.captured = {}
        for h in self.handles:
            h.remove()
        return self.actions, self.readouts, self.state


def fp_trajectory(tm, taps_t, policy, obs, seed):
    """The FP model's own trajectory from the real get_action after torch.manual_seed(seed) (the same RNG rule as row A, so its
    noise equals the student's: asserted in am_loss_for). Per step: actions_t, readouts o^FP_t, state feats, embodiment, t."""
    torch.manual_seed(seed)
    bi, ai = prepare(policy, policy_obs(policy, obs)); to_model_dtype(bi, tm); to_model_dtype(ai, tm)
    with torch.no_grad():
        fp_tokens, image_mask, bb_mask = backbone_tokens(tm, bi)
        cap = StepCapture(tm.action_head, taps_t)
        bo = tm.backbone(bi); out = tm.action_head.get_action(bo, ai)
        actions, readouts, state = cap.finish()
    assert len(actions) == tm.action_head.num_inference_timesteps and all(len(r) == len(taps_t.cross) for r in readouts), (len(actions), [len(r) for r in readouts])
    steps = [dict(actions=a, readouts=r, feats=_State(state), emb=ai.embodiment_id, t=t) for t, (a, r) in enumerate(zip(actions, readouts))]
    return dict(steps=steps, fp_tokens=fp_tokens, image_mask=image_mask, bb_mask=bb_mask, ai=ai, noise=actions[0], action_pred=out['action_pred'])


def readout_deltas(tm, sm, taps_t, policy, obs, seed, grad=True, dtype=None):
    """A-AM's Δo: the same call sequence as row A's training loop (groot_raqft.run, mode 'readout') after torch.manual_seed(seed):
    prepare -> FP tokens (no grad) -> student tokens (grad -> backbone codebooks) -> student trajectory through get_action
    (Recorder: per-step hidden state / mask / temb of every reader block) -> FP head's norm1 + attn1 with those queries over the
    student's tokens and over the FP tokens; the pre-W_O outputs are read at attn1.to_out[0] (taps_t). Check (d) asserts
    bit-identity with groot_raqft.row_a_deltas. `dtype` casts the FP-head side (checks (c)); None = deployed BF16."""
    torch.manual_seed(seed)
    bi, ai = prepare(policy, policy_obs(policy, obs)); to_model_dtype(bi, sm); to_model_dtype(ai, sm)
    if dtype is not None:
        tm.action_head.to(torch.bfloat16)          # tokens (vlln / vl_self_attention) as deployed; only the readout runs in `dtype`
    with torch.no_grad():
        fp_tokens, image_mask, bb_mask = backbone_tokens(tm, bi)
    if dtype is not None:
        tm.action_head.to(dtype)
    with torch.set_grad_enabled(grad):
        q_tokens, _, _ = backbone_tokens(sm, bi)
    rec = Recorder(sm.action_head); cap = StepCapture(sm.action_head)
    with torch.no_grad():
        bo = sm.backbone(bi); sm.action_head.get_action(bo, ai)
    rec.close(); noise = cap.finish()[0][0]
    nb = len(taps_t.cross)
    assert len(rec.records) == nb * tm.action_head.num_inference_timesteps, (len(rec.records), nb)
    cast = (lambda x: x) if dtype is None else (lambda x: None if x is None else (x.to(dtype) if torch.is_tensor(x) and x.is_floating_point() else x))
    qt, ft = cast(q_tokens), cast(fp_tokens)
    deltas, posts = [], []
    for k, (i, hidden, emask, temb) in enumerate(rec.records):
        h, te, em = cast(hidden), cast(temb), cast(emask)
        with torch.set_grad_enabled(grad):
            post_q = local_readout(tm.action_head, i, h, te, qt, em); o_q = taps_t.captured[i]
        with torch.no_grad():
            post_fp = local_readout(tm.action_head, i, h, te, ft, em); o_fp = taps_t.captured[i]
        deltas.append(dict(t=k // nb, block=i, o_q=o_q, o_fp=o_fp))
        posts.append((post_q, post_fp))
    return dict(deltas=deltas, posts=posts, q_tokens=q_tokens, fp_tokens=fp_tokens, image_mask=image_mask, bb_mask=bb_mask, ai=ai, noise=noise)


def am_loss_for(cfg, tm, sm, taps_t, policy, obs, seed, omega_w, s_scale=1.0, single_block=None, backprop=True, extra_fn=None):
    """One latent: L_AM = Σ_t ‖Ω^{1/2} J_t Δo_t‖² with row A's Δo (readout_deltas) and J_t the FP expert's Jacobian at FP step t
    (fp_trajectory, same noise); when backprop, the gradient 2 JᵀΩu is pushed into o^q through one surrogate backward."""
    d = readout_deltas(tm, sm, taps_t, policy, obs, seed, grad=backprop)
    fp = fp_trajectory(tm, taps_t, policy, obs, seed)
    assert torch.equal(fp['noise'], d['noise']), 'FP and student trajectories drew different noise'
    total = 0.0
    surrogate = 0.0
    for t, st in enumerate(fp['steps']):
        recs = [r for r in d['deltas'] if r['t'] == t and (single_block is None or r['block'] == single_block)]
        delta = {r['block']: (r['o_q'].detach().float() - r['o_fp'].float()) for r in recs}
        u, loss_t, g = jvp_and_vjp(tm, taps_t, st, fp['fp_tokens'], fp['image_mask'], fp['bb_mask'], delta, omega_w, need_vjp=backprop)
        total += float(loss_t)
        if backprop:
            # dL/do^q = g (exact, from the VJP): one surrogate whose gradient w.r.t. the student graph equals g / s_scale
            for r in recs:
                surrogate = surrogate + (r['o_q'].float() * (g[r['block']].detach() / s_scale)).sum()
    if backprop:
        extra = extra_fn(d) if extra_fn is not None else None
        (surrogate + extra if extra is not None else surrogate).backward()   # one backward (2026-10-25: job 224038 backwarded twice)
        return total, float(extra) if extra is not None else 0.0
    return total, d


def truncated_pullback(tm, d):
    """Check (c), 2026-10-25: with Ω = I and the JVP tangent truncated right after W_O of the same block (the pullback stopped at
    the first linear map) L_AM_trunc = Σ ‖JVP(W_O; o^FP → Δo)‖² must equal L_A = Σ ‖post-W_O(student) − post-W_O(FP)‖² per sample."""
    trunc = 0.0; la = 0.0
    for r, (post_q, post_fp) in zip(d['deltas'], d['posts']):
        lin = tm.action_head.model.transformer_blocks[r['block']].attn1.to_out[0]
        delta = r['o_q'] - r['o_fp']
        _, jt = torch.autograd.functional.jvp(lambda o: lin(o), (r['o_fp'],), (delta,))
        trunc += float(jt.square().sum()); la += float((post_q - post_fp).square().sum())
    return trunc, la


# ------------------------------------------------------------------------------------- row A's non-finite set (user, 2026-10-25 §1)
def nonfinite_suite(cfg, route, observations, device, directory, trainable, layers, all_models):
    """(i) λ_DoF and s with and without row A's non-finite observations (training dtype); (ii) in FP32 (both models): L_AM, JVP and
    gradient finiteness on exactly these observations, plus a 4-step dry run (4 updates x 8 latents, gradients clipped, no optimizer
    step). Writes backbone_am/nonfinite_suite.json; the training run's gate (e) requires passed = true."""
    nf = json.load(open(RUNS / 'backbone_raqft/nonfinite.json')); idxs = [int(i) for i in nf['indices']]
    latent_seed = lambda idx, number: int(np.random.SeedSequence([7, idx, number]).generate_state(1)[0])
    res = dict(indices=idxs, recurring=nf.get('recurring'), gpu=torch.cuda.get_device_name() if device == 'cuda' else 'cpu')
    t0 = time.perf_counter()
    prev = json.loads((directory / 'nonfinite_suite.json').read_text()) if (directory / 'nonfinite_suite.json').exists() else {}
    if prev.get('status') == 'part_i_done':                              # resume after part (ii) failed: part (i) is deterministic
        res.update({k: prev[k] for k in ('lambda', 's', 'changes_over_5_percent', 'seconds_i')})
        lam_all = res['lambda']['all']
        _, tm0, _, _, _ = route(0)
        horizon, adim = tm0.action_head.config.action_horizon, tm0.action_head.action_dim
        print(json.dumps(dict(part_i='reused from the previous run')), flush=True)
    else:
        lam_all, _ = calibrate_lambda(cfg, route, observations, device)
        lam_ex, _ = calibrate_lambda(cfg, route, observations, device, exclude=set(idxs))
        _, tm0, _, _, _ = route(0)
        horizon, adim = tm0.action_head.config.action_horizon, tm0.action_head.action_dim
        omega_w = omega(cfg, lam_all, horizon, adim, device)
        s_all, s_ex = [], []
        for idx in range(cfg['s_observations']):
            suite_, tm, sm, taps_t, policy = route(idx)
            obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
            with torch.no_grad():
                total, _ = am_loss_for(cfg, tm, sm, taps_t, policy, obs, idx, omega_w, backprop=False)
            s_all.append(total)
            if idx not in idxs:
                s_ex.append(total)
        res['lambda'] = dict(all=lam_all, excluding=lam_ex, rel_change={k: abs(lam_ex[k] - lam_all[k]) / lam_all[k] for k in lam_all},
                             excluded_observations=[i for i in idxs if i < cfg['lambda_observations']])
        res['s'] = dict(all=float(np.mean(s_all)), excluding=float(np.mean(s_ex)), rel_change=abs(float(np.mean(s_ex)) - float(np.mean(s_all))) / float(np.mean(s_all)),
                        excluded_observations=[i for i in idxs if i < cfg['s_observations']])
        res['changes_over_5_percent'] = bool(max(res['lambda']['rel_change'].values()) > 0.05 or res['s']['rel_change'] > 0.05)
        res['seconds_i'] = time.perf_counter() - t0
        dump(directory / 'nonfinite_suite.json', dict(res, status='part_i_done'))
        print(json.dumps(dict(part_i=dict(lambda_rel_change=res['lambda']['rel_change'], s_rel_change=res['s']['rel_change']))), flush=True)
    # ---- (ii) FP32
    t0 = time.perf_counter()
    # FP32 on the device, suite by suite (2026-10-25: casting every model, including the CPU-offloaded FP heads, exceeded the job's
    # host memory): shared backbones and student models once; each FP teacher head only while its suite is being processed.
    teacher_models = all_models[:len(all_models) // 2] if len(all_models) > 2 else all_models[:1]
    student_models = all_models[len(all_models) // 2:] if len(all_models) > 2 else all_models[1:]
    for m_ in student_models:
        m_.float()                                                       # trainable masters are FP32 already; the rest is BF16 -> FP32
    for m_ in {id(x.backbone): x.backbone for x in teacher_models}.values():
        m_.float()
    if device == 'cuda':
        for m in {id(x.backbone.model): x.backbone.model for x in all_models}.values():
            for c in (m.config, getattr(m.config, 'text_config', None), getattr(m.config, 'vision_config', None)):
                if c is not None:
                    c._attn_implementation = 'sdpa'          # flash-attn has no FP32 kernels
    omega32 = omega(cfg, lam_all, horizon, adim, device)
    per = []
    order_suite = lambda xs: sorted(xs, key=lambda i: route.suite_of(i) if hasattr(route, 'suite_of') else 0)
    current = [None]

    def route32(idx):
        out = route(idx)
        head = out[1].action_head
        if current[0] is not None and current[0] is not head:
            current[0].to(torch.bfloat16)                                # lossless back to BF16 before it is offloaded
        if next(head.parameters()).dtype != torch.float32:
            head.float()
        current[0] = head
        return out
    for idx in order_suite(idxs):
        suite_, tm, sm, taps_t, policy = route32(idx)
        extra_fn = (lambda d, sm=sm, tm=tm: cfg['beta'] * h5b_loss(sm, tm, d['q_tokens'], d['fp_tokens'], d['image_mask'], d['bb_mask'], d['ai']) / cfg['latents']) if cfg['beta'] > 0 else None
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        for p in trainable:
            p.grad = None
        try:
            total, extra = am_loss_for(cfg, tm, sm, taps_t, policy, obs, latent_seed(idx, 0), omega32, s_scale=1.0, extra_fn=extra_fn)
            bad = [dict(layer=n, tensor=k) for n, m in layers.items() for k, p in (('codebooks', m.books), ('row_scale', m.scale)) if p.grad is not None and not torch.isfinite(p.grad).all()]
            gn = float(torch.sqrt(sum(p.grad.double().square().sum() for p in trainable if p.grad is not None and torch.isfinite(p.grad).all())))
            r = dict(observation=idx, suite=suite_, L_AM=total, h5b_term=extra, loss_finite=bool(np.isfinite(total) and np.isfinite(extra)), nonfinite_gradients=bad, finite_grad_norm=gn)
        except Exception as e:
            r = dict(observation=idx, error=repr(e)[:300], loss_finite=False, nonfinite_gradients=['exception'])
        r['finite'] = bool(r['loss_finite'] and not r['nonfinite_gradients'])
        per.append(r); print(json.dumps(r), flush=True)
    dry = []
    for idx in order_suite((nf.get('recurring') or idxs)[:4]):
        suite_, tm, sm, taps_t, policy = route32(idx)
        extra_fn = (lambda d, sm=sm, tm=tm: cfg['beta'] * h5b_loss(sm, tm, d['q_tokens'], d['fp_tokens'], d['image_mask'], d['bb_mask'], d['ai']) / cfg['latents']) if cfg['beta'] > 0 else None
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        for p in trainable:
            p.grad = None
        value = 0.0; ok = True
        for number in range(cfg['latents']):
            total, extra = am_loss_for(cfg, tm, sm, taps_t, policy, obs, latent_seed(idx, number), omega32, s_scale=res['s']['all'] * cfg['latents'], extra_fn=extra_fn)
            value += total / res['s']['all'] / cfg['latents'] + extra
            ok &= bool(np.isfinite(total) and np.isfinite(extra))
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
        for p in trainable:
            p.grad = None                                        # dry run: no optimizer step
        dry.append(dict(observation=idx, suite=suite_, loss=value, grad_norm=gn, finite=bool(ok and np.isfinite(gn) and np.isfinite(value))))
        print(json.dumps(dry[-1]), flush=True)
    res['fp32'] = dict(per_observation=per, dry_run=dry, all_finite=bool(all(r['finite'] for r in per) and all(d['finite'] for d in dry)), seconds=time.perf_counter() - t0,
                       summary=dict(observations=len(per), finite=sum(r['finite'] for r in per), dry_run_finite=sum(d['finite'] for d in dry)))
    res['passed'] = res['fp32']['all_finite']
    dump(directory / 'nonfinite_suite.json', dict(res, status='complete'))
    print(json.dumps(dict(nonfinite_suite_passed=res['passed'], summary=res['fp32']['summary'], changes_over_5_percent=res['changes_over_5_percent'])), flush=True)
    return res


# --------------------------------------------------------------------------------------------------------------------- run
def gate(directory, checks, name, value, ok):
    """Hard gate (user, 2026-10-25): record, and on failure write checks_failed.json, print the failing value and exit(3)
    before any manifest / progress / history / checkpoint is written."""
    checks.setdefault('gates', {})[name] = dict(value=value, passed=bool(ok))
    dump(directory / 'checks.json', dict(checks=checks))
    print(json.dumps(dict(check=name, value=value, passed=bool(ok))), flush=True)
    if not ok:
        dump(directory / 'checks_failed.json', dict(failed=name, value=value, checks=checks))
        print(json.dumps(dict(check_failed=name, value=value)), flush=True)
        raise SystemExit(3)


def run(mode, device='cuda', checks_only=False, inject_fail=None, nonfinite=False):
    from groot_bridge_server import Bridge
    from calibration_data import load_observations
    from groot_raqft import row_a_deltas
    assert mode in ('am', 'am0', 'b_am')
    cfg = dict(CONFIG, mode=mode, beta=0.0 if mode == 'am0' else 0.1)
    from groot_rowb import SUFFIX as FT_SUFFIX, BASE_ROW as FT_BASE_ROW
    key_out = dict(am='backbone_am', am0='backbone_am0', b_am='backbone_bam' + FT_SUFFIX)[mode]
    directory = RUNS / key_out
    if checks_only:
        directory = directory / ('checks_only_' + device + ('_inject_' + inject_fail if inject_fail else ''))
    directory.mkdir(parents=True, exist_ok=True)
    if (OUT / 'fit' / key_out / 'complete.json').exists() and not checks_only and not nonfinite:
        print('already complete', key_out); return
    torch.manual_seed(0)
    _, observations = load_observations(); assert len(observations) == 256
    if mode == 'b_am':
        # B-AM (user amendment 2026-10-25): row B's trainable set, routing and Δo (groot_rowb.build_models) with the action metric
        from groot_rowb import build_models, SUITES
        suites_of = json.load(open(RUNS / 'calibration_suites.json'))['observation_suite']
        router, layers, counts, n_ste, init_gap, _ = build_models(device, observations, suites_of)
        assert all(v == 0.0 for v in init_gap.values()), init_gap
        taps = {s_: ReaderTaps(router.teachers[s_].policy.model.action_head) for s_ in SUITES}

        def route(idx):
            s_ = suites_of[idx]
            tm_, sm_, policy_ = router(s_)
            return s_, tm_, sm_, taps[s_], policy_
        route.suite_of = lambda idx: suites_of[idx]
        all_models = [b.policy.model for b in list(router.teachers.values()) + list(router.students.values())]
        teacher_heads = [b.policy.model.action_head for b in router.teachers.values()]
        cfg.update(row='raqft_groot_b_am' + FT_SUFFIX, base=FT_BASE_ROW, trainable='prefix (backbone + per-suite vl_self_attention) + per-suite DiT (row B)', routing='per-suite heads', trainable_layers=counts)
    else:
        suite = cfg['head_suite']
        teacher = Bridge(suite, device=device); student = Bridge(suite, device=device, model_root=str(MODELS / 'ptq_w2'))
        tm0, sm0 = teacher.policy.model, student.policy.model
        for p in list(tm0.parameters()) + list(sm0.parameters()):
            p.requires_grad_(False)
        layers = install_trainable(sm0, {'backbone': 'backbone', key_of(suite, 'head'): 'head'}, trainable_parts=('backbone',))
        n_ste = row_scale_ste(layers)          # option A (2026-10-26): forward bit-identical (asserted per layer)
        taps0 = ReaderTaps(tm0.action_head)

        def route(idx):
            return suite, tm0, sm0, taps0, student.policy
        all_models = [tm0, sm0]
        teacher_heads = [tm0.action_head]
    books = [m.books for m in layers.values() if m.books.requires_grad]; scales = [m.scale for m in layers.values() if m.scale.requires_grad]
    trainable = books + scales
    opt = torch.optim.Adam([dict(params=books, lr=cfg['lr']), dict(params=scales, lr=cfg['lr_scale'])])

    def reset_heads():
        for h in teacher_heads:
            h.to(torch.bfloat16)
    gpu = torch.cuda.get_device_name() if device == 'cuda' else 'cpu'
    checks = dict(gpu=gpu, inject_fail=inject_fail, row_scale_ste_layers=n_ste, row_scale_ste='option A, 2026-10-26: straight-through FP16 round trip of the row scale (forward bit-identical, FP32 gradient); codebook path as row A')
    latent_seed = lambda idx, number: int(np.random.SeedSequence([7, idx, number]).generate_state(1)[0])
    if nonfinite:
        nonfinite_suite(cfg, route, observations, device, RUNS / key_out, trainable, layers, all_models); return
    # ---- (d) Δo bit-identity between row A's code path (groot_raqft.row_a_deltas) and A-AM's (readout_deltas), 8 fixed observations
    t0 = time.perf_counter()
    d_max, d_shapes, d_order, noise_ok, replay = 0.0, True, True, True, 0.0
    for idx in range(8):
        suite_, tm, sm, taps_t, policy = route(idx)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        seed = latent_seed(idx, 0)
        a = row_a_deltas(sm, tm, policy, obs, seed)
        b = readout_deltas(tm, sm, taps_t, policy, obs, seed, grad=False)
        d_shapes &= len(a) == len(b['deltas'])
        for (t, i, delta_a, la_a), r, (post_q, post_fp) in zip(a, b['deltas'], b['posts']):
            delta_b = r['o_q'].float() - r['o_fp'].float()
            d_order &= (t == r['t'] and i == r['block']); d_shapes &= tuple(delta_a.shape) == tuple(delta_b.shape)
            d_max = max(d_max, float((delta_a - delta_b).abs().max()) if delta_a.shape == delta_b.shape else float('inf'))
            d_max = max(d_max, abs(la_a - float((post_q - post_fp).float().square().sum())))
        fp = fp_trajectory(tm, taps_t, policy, obs, seed)
        noise_ok &= bool(torch.equal(fp['noise'], b['noise']))
        st = fp['steps'][1]
        with torch.no_grad():                                      # replay of the FP step used by the JVP reproduces the captured readouts
            taps_t.captured = {}
            head_velocity(tm.action_head, st['feats'], fp['fp_tokens'], fp['image_mask'], fp['bb_mask'], st['emb'], st['actions'], 1)
            replay = max(replay, max(float((taps_t.captured[i].float() - st['readouts'][i].float()).abs().max()) for i in st['readouts']))
    checks['d'] = dict(observations=8, max_abs_diff=d_max, same_shapes=d_shapes, same_order=d_order, same_noise_fp_vs_student=noise_ok, replay_readout_max_abs_diff=replay, seconds=time.perf_counter() - t0)
    gate(directory, checks, 'd_delta_bit_identical', checks['d'], d_max == 0.0 and d_shapes and d_order and noise_ok and replay <= 1e-2)
    # ---- (a) JVP vs finite differences and the adjoint identity, FP64 (gate) and FP32 (record), 2 observations, step 1, eps 1e-3
    t0 = time.perf_counter()
    fd, fd32, adj = [], [], []
    for idx in range(2):
        suite_, tm, sm, taps_t, policy = route(idx)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        fp = fp_trajectory(tm, taps_t, policy, obs, idx)
        st0 = fp['steps'][1]
        g = torch.Generator(device=device).manual_seed(100 + idx)
        r = {i: torch.randn(o.shape, generator=g, device=device, dtype=torch.float64) for i, o in st0['readouts'].items()}
        n = torch.sqrt(sum(v.square().sum() for v in r.values())); r = {i: v / n for i, v in r.items()}
        for dtype, sink in ((torch.float64, fd), (torch.float32, fd32)):
            tm.action_head.to(dtype)
            tok = fp['fp_tokens'].to(dtype)
            st = dict(actions=st0['actions'].to(dtype), readouts={i: o.to(dtype) for i, o in st0['readouts'].items()}, feats=_State(st0['feats'].state_features.to(dtype)), emb=st0['emb'], t=1)
            rr = {i: v.to(dtype) for i, v in r.items()}
            ones = torch.ones(st['actions'].shape[1], st['actions'].shape[2], device=device)
            u, _, _ = jvp_and_vjp(tm, taps_t, st, tok, fp['image_mask'], fp['bb_mask'], {i: (2.0 if inject_fail == 'jvp' else 1.0) * v for i, v in rr.items()}, ones, need_vjp=False)
            eps = 1e-3

            def f_eps(sign):
                taps_t.inject = {i: sign * eps * v for i, v in rr.items()}
                with torch.no_grad():
                    v = head_velocity(tm.action_head, st['feats'], tok, fp['image_mask'], fp['bb_mask'], st['emb'], st['actions'], 1).to(torch.float64)
                taps_t.inject = None
                return v
            fd_u = (f_eps(1.0) - f_eps(-1.0)) / (2 * eps)
            sink.append(float((u.to(torch.float64) - fd_u).norm() / fd_u.norm()))
            if dtype == torch.float64:
                w = torch.randn(u.shape, generator=g, device=device, dtype=torch.float64)
                order = [i for i in taps_t.cross if i in rr]

                def f(*deltas):
                    taps_t.inject = {i: dd for i, dd in zip(order, deltas)}
                    try:
                        return head_velocity(tm.action_head, st['feats'], tok, fp['image_mask'], fp['bb_mask'], st['emb'], st['actions'], 1)
                    finally:
                        taps_t.inject = None
                _, jt_w = torch.autograd.functional.vjp(f, tuple(torch.zeros_like(st['readouts'][i]) for i in order), w)
                lhs = float((w * u).sum()); rhs = float(sum((a_ * rr[i]).sum() for a_, i in zip(jt_w, order)))
                adj.append(abs(lhs - rhs) / max(abs(lhs), 1e-30))
        tm.action_head.to(torch.bfloat16)
    checks['a'] = dict(jvp_vs_fd_rel_err_fp64=fd, adjoint_rel_diff_fp64=adj, jvp_vs_fd_rel_err_fp32_record_only=fd32, eps=1e-3, tangent='unit-norm N(0,1) over all readouts', seconds=time.perf_counter() - t0)
    gate(directory, checks, 'a_jvp_fd_fp64', dict(max_rel_err=max(fd), max_adjoint=max(adj)), max(fd) < 1e-3 and max(adj) < 1e-6)
    # ---- (c) truncated-pullback identity, 64 observations, FP32 (< 1e-5) and FP64 (< 1e-10)
    t0 = time.perf_counter()
    cres = {}
    for dtype, thr in ((torch.float32, 1e-5), (torch.float64, 1e-10)):
        rel = []
        for idx in range(cfg['s_observations']):
            suite_, tm, sm, taps_t, policy = route(idx)
            obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
            with torch.no_grad():
                d = readout_deltas(tm, sm, taps_t, policy, obs, idx, grad=False, dtype=dtype)      # casts the routed FP head to dtype
            trunc, la = truncated_pullback(tm, d)
            rel.append(abs(trunc - la) / max(la, 1e-300))
            tm.action_head.to(torch.bfloat16)
        cres[str(dtype).split('.')[-1]] = dict(max_rel=max(rel), mean_rel=float(np.mean(rel)), threshold=thr, passed=max(rel) < thr)
    reset_heads()
    checks['c'] = dict(cres, observations=cfg['s_observations'], seconds=time.perf_counter() - t0)
    gate(directory, checks, 'c_truncated_pullback', {k: v['max_rel'] for k, v in cres.items()}, all(v['passed'] for v in cres.values()))
    # ---- (e) row A's non-finite observations (user, 2026-10-25 §1): FP32 check suite + 4-step dry run must be finite (nonfinite_suite.json)
    nfs = RUNS / key_out / 'nonfinite_suite.json'
    e = json.load(open(nfs)) if nfs.exists() else None
    checks['e'] = dict(file=str(nfs), status=(e or {}).get('status', 'missing'), lambda_rel_change=(e or {}).get('lambda', {}).get('rel_change'), s_rel_change=(e or {}).get('s', {}).get('rel_change'),
                       changes_over_5_percent=(e or {}).get('changes_over_5_percent'), fp32=(e or {}).get('fp32', {}).get('summary'), gpu=(e or {}).get('gpu'))
    if not checks_only:
        gate(directory, checks, 'e_nonfinite_suite_fp32', checks['e'], bool(e and e.get('status') == 'complete' and e.get('passed')))
    if checks_only:
        dump(directory / 'checks.json', dict(checks=checks, status='checks_only_passed'))
        print(json.dumps(dict(checks_only=True, passed=True)), flush=True); return
    # ---- λ_DoF and Ω (recomputed; the pre-correction values are void)
    lambdas, mean_j2 = calibrate_lambda(cfg, route, observations, device)
    _, tm_0, _, _, _ = route(0)
    horizon, adim = tm_0.action_head.config.action_horizon, tm_0.action_head.action_dim
    omega_w = omega(cfg, lambdas, horizon, adim, device)
    checks['lambda_dof'] = lambdas; checks['mean_J_d_squared'] = mean_j2
    # ---- s: calibration mean of L_AM at step 0 over 64 observations with the corrected Δo
    s_vals = []
    for idx in range(cfg['s_observations']):
        suite_, tm, sm, taps_t, policy = route(idx)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        with torch.no_grad():
            total, _ = am_loss_for(cfg, tm, sm, taps_t, policy, obs, idx, omega_w, backprop=False)
        s_vals.append(total)
    s_scale = float(np.mean(s_vals))
    checks['s_calibration_mean_L_AM_step0'] = s_scale
    dump(directory / 'checks.json', dict(checks=checks))
    # ---- (b) cost probe: 2 updates with L_A (row A's loss, row A's loop) vs 2 with L_AM, same GPU
    metric = dict(variant='am_jvp', diag=None, s=s_scale)

    def one_update(kind, idx):
        suite_, tm, sm, taps_t, policy = route(idx)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        value = 0.0
        for number in range(cfg['latents']):
            seed = latent_seed(idx, number)
            if kind == 'am':
                extra_fn = (lambda d: cfg['beta'] * h5b_loss(sm, tm, d['q_tokens'], d['fp_tokens'], d['image_mask'], d['bb_mask'], d['ai']) / cfg['latents']) if cfg['beta'] > 0 else None
                if metric['diag'] is not None:
                    total, extra = amdiag_loss_for(cfg, tm, sm, taps_t, policy, obs, seed, metric['diag'][suite_], s_scale=metric['s'] * cfg['latents'], extra_fn=extra_fn)
                else:
                    total, extra = am_loss_for(cfg, tm, sm, taps_t, policy, obs, seed, omega_w, s_scale=metric['s'] * cfg['latents'], extra_fn=extra_fn)
                value += total / metric['s'] / cfg['latents'] + extra
            else:
                # row A's loop (groot_raqft.run, mode 'readout'), line for line
                torch.manual_seed(seed)
                bi, ai = prepare(policy, policy_obs(policy, obs)); to_model_dtype(bi, sm); to_model_dtype(ai, sm)
                with torch.no_grad():
                    fp_tokens, image_mask, bb_mask = backbone_tokens(tm, bi)
                q_tokens, _, _ = backbone_tokens(sm, bi)
                rec = Recorder(sm.action_head)
                with torch.no_grad():
                    bo = sm.backbone(bi); sm.action_head.get_action(bo, ai)
                rec.close()
                loss = 0.0; n = 0
                for (i, hidden, emask, temb) in rec.records:
                    o_fp = local_readout(tm.action_head, i, hidden, temb, fp_tokens, emask)
                    o_q = local_readout(tm.action_head, i, hidden, temb, q_tokens, emask)
                    loss = loss + (o_q - o_fp).float().square().sum(dim=(1, 2)).mean(); n += 1
                loss = loss / max(n, 1)
                if cfg['beta'] > 0:
                    loss = loss + cfg['beta'] * h5b_loss(sm, tm, q_tokens, fp_tokens, image_mask, bb_mask, ai)
                (loss / cfg['latents']).backward(); value += float(loss.detach()) / cfg['latents']
        if device == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter() - t0, value
    t_a = float(np.mean([one_update('a', i)[0] for i in (0, 1)])); t_am = float(np.mean([one_update('am', i)[0] for i in (2, 3)]))
    opt.zero_grad(set_to_none=True)
    checks['seconds_per_update'] = dict(row_A=t_a, A_AM=t_am, ratio=t_am / t_a, gpu=gpu)
    variant = 'am_jvp' if t_am <= 2 * t_a else 'amdiag'
    if variant == 'amdiag':
        # spec fallback: static diagonal metric diag(JᵀΩJ) (Hutchinson, 8 probes x 256 observations), row raqft_groot_prefix_amdiag;
        # s re-calibrated as the mean of the diagonal loss at step 0 over the same 64 observations
        t0 = time.perf_counter()
        metric['diag'] = calibrate_diag(cfg, route, observations, device, omega_w)
        vals = []
        for idx in range(cfg['s_observations']):
            suite_, tm, sm, taps_t, policy = route(idx)
            obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
            with torch.no_grad():
                vals.append(amdiag_loss_for(cfg, tm, sm, taps_t, policy, obs, idx, metric['diag'][suite_], backprop=False)[0])
        metric['s'] = float(np.mean(vals))
        checks['amdiag'] = dict(seconds_calibration=time.perf_counter() - t0, s_calibration_mean_L_amdiag_step0=metric['s'],
                                diag_mean_per_block={f'{s_}:{i}': float(v.mean()) for s_, dd in metric['diag'].items() for i, v in dd.items()})
        t_diag = float(np.mean([one_update('am', i)[0] for i in (4, 5)])); opt.zero_grad(set_to_none=True)
        checks['seconds_per_update']['A_AMdiag'] = t_diag
    gate(directory, checks, 'b_cost_ratio', checks['seconds_per_update'], True)      # the 2x rule selects the variant; it never aborts
    s_scale = metric['s']
    checks['variant'] = variant
    checks['row_name'] = ('raqft_groot_b_am' + ('diag' if variant == 'amdiag' else '')) if mode == 'b_am' else ({'am_jvp': 'raqft_groot_prefix_am', 'amdiag': 'raqft_groot_prefix_amdiag'}[variant] + ('_nob' if cfg['beta'] == 0 else ''))
    assert all(v['passed'] for v in checks['gates'].values()), checks['gates']
    contract = dict(config=cfg, suite=cfg.get('routing', cfg['head_suite']), checks=checks, trainable=dict(codebooks=len(books), row_scales=len(scales)))
    dump(directory / 'manifest.json', contract)      # first training artefact: written only after every gate passed
    print(json.dumps(dict(checks=checks)), flush=True)
    history = []
    skipped = []
    t_start = time.time(); total_updates = cfg['epochs'] * 256
    for update in range(total_updates):
        idx = update % 256
        dt, value = one_update('am', idx)
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
        if not (np.isfinite(gn) and np.isfinite(value)):
            # same guard as row A (groot_raqft.py, 2026-10-24): a non-finite update is dropped and recorded, never applied
            skipped.append(dict(update=update + 1, observation=idx, loss=value, grad_norm=gn))
            print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            assert all(torch.isfinite(p).all() for p in trainable), f'non-finite parameter after update {update + 1}'
        assert len(skipped) <= 38, f'{len(skipped)} non-finite updates (> 5% of 768): stop and report'
        history.append(dict(update=update + 1, observation=idx, suite=route(idx)[0] if mode == 'b_am' else cfg['head_suite'], loss=value, grad_norm=gn, seconds=dt))
        if (update + 1) % 16 == 0:
            dump(directory / 'progress.json', dict(status='training', completed=update + 1, total=total_updates, loss=value, elapsed=time.time() - t_start))
            print(json.dumps(history[-1]), flush=True)
    assert all(torch.isfinite(p).all() for p in trainable), 'non-finite masters at export'
    fine_tune = dict(variant=variant, contract=contract, updates=len(history), nonfinite_events=skipped, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], elapsed=time.time() - t_start)
    if mode == 'b_am':
        from groot_rowb import export_fit
        export_fit({k: m for k, m in layers.items() if m.books.requires_grad}, key_out, 'head_bam' + FT_SUFFIX, fine_tune)
    else:
        out = OUT / 'fit' / key_out; out.mkdir(parents=True, exist_ok=True)
        for name, mod in layers.items():
            if not mod.books.requires_grad:
                continue
            src = torch.load(OUT / 'fit/backbone' / f'{name}.pt', map_location='cpu'); ex = mod.export()
            src['tensors']['codebooks'] = ex['codebooks']; src['tensors']['row_scale'] = ex['row_scale']
            with torch.no_grad():
                src['w_hat'] = mod.weight().to(torch.bfloat16).cpu()
            atomic_save(src, out / f'{name}.pt')
            (out / f'{name}.json').write_text((OUT / 'fit/backbone' / f'{name}.json').read_text())
        base = json.loads((OUT / 'fit/backbone/complete.json').read_text())
        dump(out / 'complete.json', dict(base, status='complete', key=key_out, fine_tune=fine_tune))
    dump(directory / 'history.json', history)
    if mode == 'b_am':
        from groot_rowb import held_out_mse
        del layers, opt, trainable
        pass        # held-out MSE: separate GPU job submitted by the controller (2026-10-26)
    print(json.dumps(dict(done=key_out, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], hours=(time.time() - t_start) / 3600)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('am', 'am0', 'b_am'), required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--checks-only', action='store_true', help='run gates (d), (a), (c) and exit; no training')
    p.add_argument('--inject-fail', choices=('jvp',), default=None, help='deliberately break check (a) (tangent x2) to verify the abort path')
    p.add_argument('--nonfinite-suite', action='store_true', help="row A's non-finite observations: λ/s with and without them; FP32 L_AM/JVP/gradient suite + 4-step dry run")
    a = p.parse_args()
    if a.device == 'cpu':
        sys.modules['flash_attn'] = None        # qwen3_backbone falls back to SDPA
    torch.set_num_threads(int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 8)))
    run(a.mode, device=a.device, checks_only=a.checks_only, inject_fail=a.inject_fail, nonfinite=a.nonfinite_suite)
