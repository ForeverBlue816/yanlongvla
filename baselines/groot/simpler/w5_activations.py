"""GOAL v8 W5: where does a uniform 0.95 scale of the 432 quantized Linear weights of GR00T N1.7 (Bridge) survive after the norms,
and does the action-head velocity get smaller?  One model load, N stored Bridge observations, scales {1.0, 0.95} applied in memory
(W <- s * W_original on the 432 keys of aqlm_w2_error.json, restored from a stash), the SAME torch seed per observation for all scales.

Observations: groot/simpler/data/bridge-calibration/observations.parquet (the 256 Bridge calibration observations of the quantizers: 256x256 image,
8-d state, instruction; groot/bridge_obs.py).  N of them are taken at evenly spaced indices.  --heldout uses bridge-heldout instead.
Run in the GR00T environment, from the Isaac-GR00T directory:
  python w5_activations.py --model-path <FP bridge checkpoint> --out w5.json [--n 128] [--scales 1.0 0.95] [--init-noise-scale 1.3]
With --model-path = the sh95 checkpoint use --scales 1.0 (its weights already carry the 0.95), or compare it with the FP run.

Measured (all in float32, after the call, per observation and per scale):
 (a) backbone (Qwen3-VL language model, 16 layers): residual-stream RMS after every decoder layer, RMS after input_layernorm / post_attention_layernorm / the final
     norm, RMS of the attention and MLP branch outputs, attention entropy (nats, mean over heads and queries, recomputed from q/k of the layer: causal mask,
     batch 1, no padding).  Action head: RMS after vlln and vl_self_attention, RMS after every DiT block (per flow step), RMS after norm_out, attention entropy of
     every DiT attention (q/k recomputed with diffusers Attention.scale; boolean key masks of the alternate-VL DiT are honoured) and of vl_self_attention.
     Relative change of the 0.95 run against the first scale: ||x_s - x_ref|| / ||x_ref|| for backbone layer outputs, the final norm and the vl embeddings.
 (b) per flow step: RMS of the predicted velocity (action_decoder output, last action_horizon positions) and of the current sample, over the padded
     (action_horizon x max_action_dim) tensor ("full") and over the real block (T_real x D_real taken from the decoded action dict: "real").
 (c) final action: RMS per dimension of the denormalised action (real dimensions), RMS of (final - start noise) in the normalised space ("real" and "full").
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                       # groot_common.py, bridge_obs.py
os.environ.setdefault('GROOT_PROFILE', 'bridge')           # groot_common: SIMPLER_ENV_WIDOWX, Bridge observation layout

DEFAULT_KEYS = '/scratch/d250009/pi05-handover/runs/groot_baselines/s5bq/perturb/aqlm_w2_error.json'
DEFAULT_OBS = HERE / 'data/bridge-calibration'
DEFAULT_MODEL = '/scratch/d250009/pi05-handover/runs/groot_baselines/s5b/checkpoints/bridge'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-path', default=DEFAULT_MODEL, help='FP Bridge checkpoint (default) or the sh95 checkpoint (then use --scales 1.0)')
    p.add_argument('--out', required=True, help='output JSON')
    p.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.95], help='weight scales; the first is the reference of the ratios')
    p.add_argument('--scale', type=float, default=None, help='shorthand: --scales 1.0 S')
    p.add_argument('--n', type=int, default=128, help='number of observations (evenly spaced over the store)')
    p.add_argument('--obs-dir', default=str(DEFAULT_OBS), help='folder with observations.parquet + meta/tasks.jsonl')
    p.add_argument('--heldout', action='store_true', help='use bridge-heldout instead of bridge-calibration')
    p.add_argument('--keys-json', default=DEFAULT_KEYS, help='JSON whose keys are the 432 weights to scale; checked against the layers found in the model')
    p.add_argument('--init-noise-scale', type=float, default=None, help='multiply the initial flow-matching noise by S (the x1.3 probe); same patch as simpler_server.py')
    p.add_argument('--seed-base', type=int, default=7, help='noise seed of observation i = SeedSequence([seed_base, i]) (7 = calibration)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-attn-entropy', action='store_true', help='skip the attention entropy hooks')
    p.add_argument('--no-per-observation', action='store_true', help='leave the per-observation values out of the JSON')
    p.add_argument('--embodiment-tag', default='SIMPLER_ENV_WIDOWX')
    a = p.parse_args(argv)
    if a.scale is not None:
        a.scales = [1.0, a.scale]
    return a


# ---------------------------------------------------------------- small tensor statistics (CPU-testable)
def rms(x):
    return float(x.detach().float().pow(2).mean().sqrt())


def rms_tok_median(x):
    """median over tokens of the per-token RMS (robust against the massive activations of a few tokens)"""
    x = x.detach().float()
    return float(x.reshape(-1, x.shape[-1]).pow(2).mean(-1).sqrt().median())


def attention_entropy(q, k, scale, causal=False, key_mask=None):
    """q (B,H,Tq,d), k (B,H,Tk,d) -> (mean entropy in nats over B,H,Tq; mean max prob; log of the number of allowed keys, mean over queries)."""
    import torch
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    allowed = torch.ones(logits.shape[-2:], dtype=torch.bool, device=logits.device)
    if causal:
        allowed = torch.tril(allowed)
    allowed = allowed[None, None].expand_as(logits)
    if key_mask is not None:   # (B, Tk) bool
        allowed = allowed & key_mask[:, None, None, :].to(logits.device).bool()
    logits = logits.masked_fill(~allowed, float('-inf'))
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * torch.where(allowed, logp, torch.zeros_like(logp))).sum(-1)
    n_allowed = allowed.sum(-1).float().clamp_min(1)
    return float(ent.mean()), float(p.max(-1).values.mean()), float(n_allowed.log().mean())


# ---------------------------------------------------------------- recorder with hooks
class Recorder:
    """Collects scalars of one policy call in self.cur (name -> value, or name -> list for per-step tensors) and, for the reference
    scale, keeps CPU copies of selected tensors in self.ref so later scales can report relative changes."""

    def __init__(self, entropy=True):
        self.cur, self.ref, self.keep, self.counts, self.handles = {}, {}, {}, {}, []
        self.entropy = entropy
        self.reset()

    def reset(self):
        self.cur, self.keep, self.counts = {}, {}, {}
        self.velocity, self.samples, self.final = [], [], None

    def put(self, name, value):
        self.cur[name] = value

    def call_key(self, name, per_call):
        if not per_call:
            return name
        k = self.counts.get(name, 0)
        self.counts[name] = k + 1
        return f'{name}@{k}'   # flow step k of a DiT module

    def stat(self, name, x, per_call=False, keep=False, tok=False):
        key = self.call_key(name, per_call)
        self.cur[f'{key}.rms'] = rms(x)
        if tok:
            self.cur[f'{key}.rms_tok_med'] = rms_tok_median(x)
        if keep:
            self.keep[key] = x.detach().float().cpu()

    def hook_out(self, module, name, per_call=False, keep=False, tok=False, pick=lambda o: o):
        def f(mod, args, out):
            self.stat(name, pick(out), per_call, keep, tok)
        self.handles.append(module.register_forward_hook(f))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def tensor_of(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def install_hooks(model, rec):
    """Hooks on the real module names (read from gr00t/model/gr00t_n1d7, modules/dit.py, transformers 4.57 Qwen3-VL, diffusers 0.35)."""
    import torch
    bb = model.backbone.model.language_model            # Qwen3VLTextModel; layers 0-15 kept (Qwen3Backbone pops the rest)
    head = model.action_head
    for i, layer in enumerate(bb.layers):
        p = f'bb.layer{i:02d}'
        rec.hook_out(layer, f'{p}.resid', keep=True, tok=True, pick=tensor_of)           # residual stream after the layer
        rec.hook_out(layer.input_layernorm, f'{p}.ln_in', tok=True)                        # after RMSNorm, before attention (carries the norm weight)
        rec.hook_out(layer.post_attention_layernorm, f'{p}.ln_post', tok=True)             # after RMSNorm, before the MLP
        rec.hook_out(layer.self_attn, f'{p}.attn_out', tok=True, pick=tensor_of)           # attention branch output (o_proj)
        rec.hook_out(layer.mlp, f'{p}.mlp_out', tok=True)                                  # MLP branch output (down_proj)
    rec.hook_out(bb.norm, 'bb.final_norm', keep=True, tok=True)                            # = backbone_features (hidden_states[-1])
    if rec.entropy:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        def bb_attn_pre(layer_idx):
            def f(mod, args, kwargs):
                h = args[0] if args else kwargs['hidden_states']
                cos, sin = kwargs['position_embeddings']
                shp = (*h.shape[:-1], -1, mod.head_dim)
                q = mod.q_norm(mod.q_proj(h).view(shp)).transpose(1, 2)
                k = mod.k_norm(mod.k_proj(h).view(shp)).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = k.repeat_interleave(mod.num_key_value_groups, dim=1)                   # GQA
                ent, pmax, logn = attention_entropy(q, k, mod.scaling, causal=True)
                rec.put(f'bb.layer{layer_idx:02d}.attn.entropy', ent)
                rec.put(f'bb.layer{layer_idx:02d}.attn.maxprob', pmax)
                rec.put(f'bb.layer{layer_idx:02d}.attn.log_nkeys', logn)
            return f
        for i, layer in enumerate(bb.layers):
            rec.handles.append(layer.self_attn.register_forward_pre_hook(bb_attn_pre(i), with_kwargs=True))
    # action head
    rec.hook_out(head.vlln, 'head.vlln', tok=True)
    rec.hook_out(head.vl_self_attention, 'head.vlsa_out', keep=True, tok=True, pick=tensor_of)
    for i, blk in enumerate(head.model.transformer_blocks):
        rec.hook_out(blk, f'dit.block{i:02d}.resid', per_call=True, tok=True, pick=tensor_of)
    rec.hook_out(head.model.norm_out, 'dit.norm_out', per_call=True)
    if rec.entropy:
        def attn_pre(name, per_call):
            def f(mod, args, kwargs):
                h = args[0] if args else kwargs['hidden_states']
                enc = kwargs.get('encoder_hidden_states', args[1] if len(args) > 1 else None)
                src = h if enc is None else enc
                mask = kwargs.get('attention_mask', None)
                key_mask = mask if (mask is not None and mask.dtype == torch.bool and mask.dim() == 2) else None   # (B, S) key mask of the alternate-VL DiT
                B = h.shape[0]
                hd = mod.to_q.out_features // mod.heads
                q = mod.to_q(h).view(B, -1, mod.heads, hd).transpose(1, 2)
                k = mod.to_k(src).view(B, -1, mod.heads, hd).transpose(1, 2)
                ent, pmax, logn = attention_entropy(q, k, mod.scale, causal=False, key_mask=key_mask)
                key = rec.call_key(name, per_call)
                rec.put(f'{key}.entropy', ent)
                rec.put(f'{key}.maxprob', pmax)
                rec.put(f'{key}.log_nkeys', logn)
            return f
        for root, prefix, per_call in ((head.model, 'dit', True), (head.vl_self_attention, 'head.vlsa', False)):
            for n, m in root.named_modules():
                if m.__class__.__name__ == 'Attention' and hasattr(m, 'to_q'):
                    rec.handles.append(m.register_forward_pre_hook(attn_pre(f'{prefix}.{n}.attn', per_call), with_kwargs=True))
    # sampler: sample entering each flow step = input of action_encoder; velocity = action_decoder output (last action_horizon positions)
    H = head.action_horizon

    def enc_pre(mod, args):
        rec.samples.append(args[0].detach().float().cpu())
    def dec_post(mod, args, out):
        rec.velocity.append(out[:, -H:].detach().float().cpu())
    rec.handles.append(head.action_encoder.register_forward_pre_hook(enc_pre))
    rec.handles.append(head.action_decoder.register_forward_hook(dec_post))
    # final sample (after the last Euler step) and the init-noise probe: wrap the sampler like simpler_server.py does
    orig = head.get_action_with_features

    def sampler(*args, **kw):
        out = orig(*args, **kw)
        rec.final = out['action_pred'].detach().float().cpu()
        return out
    head.get_action_with_features = sampler
    return orig


def patch_init_noise(head, s):
    """Same mechanism as simpler_server.py --init-noise-scale: scale only the (B, action_horizon, action_dim) torch.randn draw."""
    import torch
    inner = head.get_action_with_features
    shape_of = lambda B: (B, head.config.action_horizon, head.action_dim)
    hits = []

    def noisy(*args, **kw):
        randn = torch.randn

        def scaled_randn(*ra, **rk):
            x = randn(*ra, **rk)
            size = rk.get('size', ra[0] if ra else None)
            if isinstance(size, (tuple, list, torch.Size)) and len(size) == 3 and tuple(size) == shape_of(size[0]):
                hits.append(1)
                return x * s
            return x
        torch.randn = scaled_randn
        try:
            return inner(*args, **kw)
        finally:
            torch.randn = randn
    head.get_action_with_features = noisy
    return hits


# ---------------------------------------------------------------- per-call derived metrics
def derive(rec, action, ref_keep):
    """(b), (c) and the relative changes; returns the flat metric dict of this call."""
    import torch
    m = dict(rec.cur)
    if isinstance(action.get('action'), dict):   # Gr00tPolicy (no sim wrapper) returns {'action': {'x': (B, T, 1), ...}} or flat unprefixed keys
        action = action['action']
    keys = [k for k in action if getattr(action[k], 'ndim', 0) == 3]
    assert keys, f'no (B, T, D) action arrays in {list(action)}'
    dims = [action[k].shape[-1] for k in keys]
    T_real, D_real = action[keys[0]].shape[1], sum(dims)
    m['n_real_dims'], m['T_real'] = D_real, T_real
    full = lambda x: x
    real = lambda x: x[:, :T_real, :D_real]
    for t, (v, x) in enumerate(zip(rec.velocity, rec.samples)):
        for tag, sel in (('full', full), ('real', real)):
            m[f'flow.step{t}.velocity_rms_{tag}'] = rms(sel(v))
            m[f'flow.step{t}.sample_rms_{tag}'] = rms(sel(x))
    noise, final = rec.samples[0], rec.final
    m['flow.noise_rms_real'], m['flow.noise_rms_full'] = rms(real(noise)), rms(full(noise))
    m['final.normalised_rms_real'], m['final.normalised_rms_full'] = rms(real(final)), rms(full(final))
    m['final.dist_from_noise_rms_real'] = rms(real(final - noise))
    m['final.dist_from_noise_rms_full'] = rms(full(final - noise))
    for k in keys:   # denormalised action, per dimension (RMS over the T_real steps)
        a = torch.as_tensor(action[k]).float()
        for d in range(a.shape[-1]):
            m[f'final.action.{k.removeprefix("action.")}{"" if a.shape[-1] == 1 else d}.rms'] = rms(a[..., d])
    m['final.action.all_rms'] = rms(torch.cat([torch.as_tensor(action[k]).float() for k in keys], -1))
    if ref_keep is not None:   # relative change of selected tensors against the reference scale (same observation, same seed)
        for name, x in rec.keep.items():
            r = ref_keep.get(name)
            if r is not None and r.shape == x.shape:
                m[f'{name}.rel_change'] = float((x - r).norm() / r.norm().clamp_min(1e-12))
        for t, (v, rv) in enumerate(zip(rec.velocity, ref_keep['_velocity'])):
            m[f'flow.step{t}.velocity_rel_change'] = float((real(v) - real(rv)).norm() / real(rv).norm().clamp_min(1e-12))
        m['final.normalised_rel_change'] = float((real(final) - real(ref_keep['_final'])).norm() / real(ref_keep['_final']).norm().clamp_min(1e-12))
    return m


# ---------------------------------------------------------------- summaries
def mean_sd(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd


def summarise(per_obs):
    """per_obs: list of flat dicts -> {metric: {mean, sd, n}}"""
    out = {}
    for k in per_obs[0]:
        vals = [d[k] for d in per_obs if k in d and isinstance(d[k], (int, float)) and math.isfinite(d[k])]
        if vals:
            mu, sd = mean_sd(vals)
            out[k] = dict(mean=mu, sd=sd, n=len(vals))
    return out


def ratios(per_scale, per_obs_all, ref, other):
    """ratio of means and mean of per-observation ratios (+ s.e.) other / ref"""
    sref, soth = per_scale[ref], per_scale[other]
    out = {}
    for k in sref:
        if k not in soth or k.endswith('rel_change') or k in ('n_real_dims', 'T_real', 'obs_index') or abs(sref[k]['mean']) < 1e-12:
            continue
        r = dict(ratio_of_means=soth[k]['mean'] / sref[k]['mean'])
        pr = [o[other][k] / o[ref][k] for o in per_obs_all if k in o[other] and k in o[ref] and abs(o[ref][k]) > 1e-12]
        if len(pr) > 1:
            mu, sd = mean_sd(pr)
            r.update(mean_of_ratios=mu, sem_of_ratios=sd / math.sqrt(len(pr)))
        out[k] = r
    return out


def print_tables(per_scale, rat, ref, other, n_layers, summ_other):
    g = lambda k, f='ratio_of_means': rat.get(k, {}).get(f, float('nan'))
    print(f'\n=== ratio {other} / {ref} (ratio of means over observations) ===')
    print('backbone layer | resid  ln_in  ln_post attn_out mlp_out | resid rel.change | attn entropy ref -> new')
    for i in range(n_layers):
        p = f'bb.layer{i:02d}'
        ent = (per_scale[ref].get(f'{p}.attn.entropy', {}).get('mean', float('nan')), per_scale[other].get(f'{p}.attn.entropy', {}).get('mean', float('nan')))
        print(f'  {i:2d}           | {g(p + ".resid.rms"):.4f} {g(p + ".ln_in.rms"):.4f} {g(p + ".ln_post.rms"):.4f} {g(p + ".attn_out.rms"):.4f} {g(p + ".mlp_out.rms"):.4f} '
              f'| {summ_other.get(p + ".resid.rel_change", {}).get("mean", float("nan")):.4f}           | {ent[0]:.4f} -> {ent[1]:.4f}')
    print(f'  final norm rms ratio {g("bb.final_norm.rms"):.4f}  (rel change {summ_other.get("bb.final_norm.rel_change", {}).get("mean", float("nan")):.4f}); '
          f'vlln {g("head.vlln.rms"):.4f}; vl_self_attention out {g("head.vlsa_out.rms"):.4f}')
    print('flow step | velocity rms (real) ref  new  ratio | velocity rel.change | sample rms ratio')
    t = 0
    while f'flow.step{t}.velocity_rms_real' in per_scale[ref]:
        k = f'flow.step{t}.velocity_rms_real'
        print(f'  {t}       | {per_scale[ref][k]["mean"]:.4f}  {per_scale[other][k]["mean"]:.4f}  {g(k):.4f} | {summ_other.get(f"flow.step{t}.velocity_rel_change", {}).get("mean", float("nan")):.4f} | {g(f"flow.step{t}.sample_rms_real"):.4f}')
        t += 1
    print('final: ' + '  '.join(f'{k.split(".", 2)[2]} {g(k):.4f}' for k in sorted(per_scale[ref]) if k.startswith('final.action.') and k.endswith('.rms')))
    for k in ('final.dist_from_noise_rms_real', 'final.dist_from_noise_rms_full', 'final.normalised_rms_real', 'flow.noise_rms_real'):
        print(f'  {k}: ref {per_scale[ref][k]["mean"]:.4f}  new {per_scale[other][k]["mean"]:.4f}  ratio {g(k):.4f}')
    dit = [k for k in rat if k.startswith('dit.') and k.endswith('.entropy')]
    if dit:
        print(f'DiT attention entropy ratio: mean {sum(g(k) for k in dit) / len(dit):.4f} over {len(dit)} (module, step) pairs; '
              f'range {min(g(k) for k in dit):.4f} .. {max(g(k) for k in dit):.4f}')


def main(argv=None):
    a = parse_args(argv)
    import numpy as np
    import torch
    from groot_common import load_policy, quantizable_layers, bridge_observation, dump_json
    from bridge_obs import load_observations

    obs_dir = Path(a.obs_dir if not a.heldout else str(DEFAULT_OBS).replace('bridge-calibration', 'bridge-heldout'))
    store = load_observations(obs_dir)
    idx = np.unique(np.linspace(0, len(store) - 1, min(a.n, len(store))).round().astype(int))
    print(f'observations: {len(idx)} of {len(store)} from {obs_dir}', flush=True)

    policy = load_policy(a.model_path, a.device)
    model = policy.model
    layers = quantizable_layers(model)               # 432 Linear modules (asserted inside)
    keys = set(json.loads(Path(a.keys_json).read_text())) if Path(a.keys_json).exists() else None
    if keys is not None:
        found = {n + '.weight' for n in layers}
        assert found == keys, f'scaled set differs from {a.keys_json}: {len(found - keys)} extra, {len(keys - found)} missing'
        print('scaled set == keys of', a.keys_json, flush=True)
    else:
        print('WARNING: keys json not found; scaling the 432 quantizable_layers() weights', flush=True)
    stash = {n: m.weight.data.clone() for n, m in layers.items()}   # originals (bf16, ~4.7 GB on the GPU)

    def set_scale(s):
        with torch.no_grad():
            for n, m in layers.items():
                m.weight.data.copy_(stash[n] * s) if s != 1.0 else m.weight.data.copy_(stash[n])

    rec = Recorder(entropy=not a.no_attn_entropy)
    install_hooks(model, rec)
    head = model.action_head
    if a.init_noise_scale is not None:
        hits = patch_init_noise(head, a.init_noise_scale)
    scales = [str(s) for s in a.scales]
    ref_name = scales[0]
    results = {s: [] for s in scales}
    t0 = time.time()
    try:
        for j, i in enumerate(idx):
            seed = int(np.random.SeedSequence([a.seed_base, int(i)]).generate_state(1)[0])
            ref_keep = None
            for s_name, s in zip(scales, a.scales):
                set_scale(s)
                rec.reset()
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                policy.reset()
                with torch.inference_mode():
                    action, _ = policy.get_action(bridge_observation(store[int(i)]))
                m = derive(rec, action, ref_keep if s_name != ref_name else None)
                m.update(obs_index=int(i), prompt=store[int(i)]['prompt'])
                results[s_name].append(m)
                if s_name == ref_name:
                    ref_keep = dict(rec.keep, _velocity=list(rec.velocity), _final=rec.final)
            if (j + 1) % 8 == 0 or j == 0:
                print(f'{j + 1}/{len(idx)} observations, {time.time() - t0:.0f} s', flush=True)
    finally:
        set_scale(1.0)
        rec.remove()
    if a.init_noise_scale is not None:
        assert hits, 'the init-noise patch never matched the sampler randn'
        print(f'init noise scaled by {a.init_noise_scale}: {len(hits)} draws', flush=True)

    numeric = lambda lst: [{k: v for k, v in d.items() if isinstance(v, (int, float))} for d in lst]
    per_scale = {s: summarise(numeric(results[s])) for s in scales}
    nums = {s: numeric(results[s]) for s in scales}
    per_obs_num = [{s: nums[s][j] for s in scales} for j in range(len(idx))]
    out = dict(model_path=a.model_path, scales=a.scales, init_noise_scale=a.init_noise_scale, seed_base=a.seed_base, obs_dir=str(obs_dir),
               observation_indices=[int(i) for i in idx], n_layers_backbone=len(model.backbone.model.language_model.layers),
               notes=dict(rms='sqrt(mean(x^2)) over all elements, float32', entropy='nats, mean over heads and queries; backbone causal batch-1 recomputation; DiT with key masks',
                          real='first T_real steps x D_real dims of the (action_horizon x max_action_dim) padded tensor; D_real is the concatenated size of the decoded action keys (assumed to come first)',
                          rel_change='||x_s - x_ref|| / ||x_ref|| for the same observation and seed against the first scale',
                          sampler='velocity = action_decoder output, sample = action_encoder input; 4 Euler steps unless the checkpoint says otherwise'),
               summary=per_scale)
    if len(scales) > 1:
        out['ratios'] = {s: ratios(per_scale, per_obs_num, ref_name, s) for s in scales[1:]}
    if not a.no_per_observation:
        out['per_observation'] = [{s: results[s][j] for s in scales} for j in range(len(idx))]
    dump_json(a.out, out)
    print('wrote', a.out, flush=True)
    for s in scales[1:]:
        print_tables(per_scale, out['ratios'][s], ref_name, s, out['n_layers_backbone'], per_scale[s])


if __name__ == '__main__':
    main()
