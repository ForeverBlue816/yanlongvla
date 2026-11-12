"""GR00T N1.7 trellis-coded quantization (TCQ) base in our pipeline (user spec 2026-11-01, TCQ_RAQFT_2026-11-01.md).

Format (fixed): 2 bits/weight, bitshift trellis L = 16 / K = 2 / V = 2, per-layer 2^9-entry FP16 lookup table (QTIP HYB "quantlut_sym"
structure); our RHT (rht.layer_signs + backbone_rht block Walsh, identical to ptq_w2); per-output-row FP16 scale (WLS-refit); fit on our
layer Hessians (bank a) with LDLQ column feedback and Viterbi inside 16 x 16 tiles (QTIP's td 16), 1 % mean-diagonal damping, retries at
10x / 100x on Cholesky failure (recorded per layer). Encoding engine = QTIP (src/qtip, commit e90c668, GPL-3): lib.codebook.bitshift
(Viterbi / pack / unpack), lib.algo.ldlq.LDLQ, lib.utils.math_utils.block_LDL — imported, not copied; its kernels / fast_hadamard_transform /
glog are stubbed (unused). Everything else here is ours.

Record format = groot_ptq's (w_hat BF16 in the original basis + payload_bytes in the json), so stage_assemble, groot_offline and the Plus
stage work unchanged with the rows tcq_w2 = (backbone_tcq, head_tcq) and tcq_raqft = (backbone_tcq_raqft, head_tcq_raqft).

Stages:
  tlut                      deterministic initial lookup table -> OUT/groot/tcq/tlut_init.pt
  gate                      §1 (a) three layers (+ LLM layer-2 down_proj) vs the base fits, (b) decode round trip + 2-observation forward,
                            (c) tiling, (d) timing + extrapolation -> OUT/groot/tcq/gate.json
  fit --key K --worker w --workers n     per-layer fits -> OUT/groot/ptq/fit/<K>/<layer>.pt + .json (K = backbone_tcq | head_tcq_<suite>)
  roundtrip --suite S --row R            full-model decode check against the assembled checkpoint
  readout --rows R1:R2 ...               B2 readout error on the 256 held-out observations (vs the FP head)
  raqft                                  §3 fine-tune (groot_tcq_raqft.py)
"""
import argparse
import json
import math
import os
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('TQDM_DISABLE', '1')            # QTIP's LDLQ wraps its column loop in tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT, ROOT, MODELS, SUITES, key_of, atomic_save, dump, inverse_rht, weights_for, part_of   # noqa: E402

TCQ = OUT.parent / 'tcq'                      # runs/plustrain_20261008/groot/tcq
QTIP = ROOT / 'src/qtip'
from tcq_bits import BITS as _BITS
SETTINGS = dict(L=16, K=_BITS, V=2, tlut_bits=9, td=16, decode_mode='quantlut_sym', scale_override=0.9)   # K = bits per weight (TCQ_BITS, default 2)
DAMPINGS = (0.01, 0.1, 1.0)                   # relative to the mean diagonal; retried in this order on Cholesky failure
GATE_LAYERS = {'llm_attention': 'backbone.model.model.language_model.layers.8.self_attn.q_proj',
               'llm_mlp': 'backbone.model.model.language_model.layers.8.mlp.down_proj',
               'dit_attention': 'action_head.model.transformer_blocks.8.attn1.to_q',          # N1.7 DiT has no attn2: VLM tokens are attended by attn1 (joint attention)
               'llm_layer2_down_proj': 'backbone.model.model.language_model.layers.2.mlp.down_proj'}
KEY_OF = {'backbone': 'backbone_tcq', **{f'head_{s}': f'head_tcq_{s}' for s in SUITES}}
G16_KEY_OF = {'backbone': 'backbone_tcq_g16', **{f'head_{s}': f'head_tcq_g16_{s}' for s in SUITES}}
RHT_KEY_OF = {'backbone': 'backbone_tcq_rht', **{f'head_{s}': f'head_tcq_rht_{s}' for s in SUITES}}           # user 2026-11-02: two-sided randomized Hadamard base
RHT_G16_KEY_OF = {'backbone': 'backbone_tcq_rht_g16', **{f'head_{s}': f'head_tcq_rht_g16_{s}' for s in SUITES}}
METRIC_KEY_OF = {m: {'backbone': f'backbone_tcq_rht_{m}', **{f'head_{s}': f'head_tcq_rht_{m}_{s}' for s in SUITES}} for m in ('iso', 'shift')}   # SHIFT_2026-11-03: arms B / C
D_KEY_OF = {r: {'backbone': f'backbone_tcq_rht_d{r}', **{f'head_{s}': f'head_tcq_rht_d{r}_{s}' for s in SUITES}} for r in (1, 2)}              # arm D (user 2026-11-06 #4): base recipe on the mixed-set H
D_HKEY_OF = {r: {'backbone': f'backbone_hd{r}', **{f'head_{s}': f'head_hd{r}_{s}' for s in SUITES}} for r in (1, 2)}   # 2026-11-07 recipe: H_clean + trace-balanced per-source H (groot_shift.py --stage hd)
SHIFT_DAMPINGS = (0.005, 0.05, 0.5)          # relative to the combined metric (mean diag = 2 x H's): = 1 % / 10 % / 100 % of H's mean diagonal, as the base
BASES = {'tcq': (KEY_OF, G16_KEY_OF, 'one'), 'rht': (RHT_KEY_OF, RHT_G16_KEY_OF, 'two')}   # V1 base: per-row-group tables re-fitted by WLS, column scales = 1


# ------------------------------------------------------------------------------------------------------------------- engine
_ENGINE = {}


def engine():
    """Import QTIP's codebook / LDLQ once (kernels, Hadamard and logging stubbed). Returns dict(bitshift_codebook, LDLQ, block_LDL, commit)."""
    if _ENGINE:
        return _ENGINE
    stubbed = []
    for name in ('fast_hadamard_transform', 'glog', 'qtip_kernels'):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.info = stub.warning = stub.error = (lambda *a, **k: None)
            sys.modules[name] = stub
            stubbed.append(name)
    if 'datasets' not in sys.modules:
        try:
            import datasets  # noqa: F401
        except Exception:
            stub = types.ModuleType('datasets'); stub.load_dataset = None; sys.modules['datasets'] = stub
    if str(QTIP) not in sys.path:
        sys.path.insert(0, str(QTIP))
    from lib.codebook.bitshift import bitshift_codebook
    from lib.algo.ldlq import LDLQ
    from lib.utils.math_utils import block_LDL
    for name in stubbed:                                   # QTIP holds its references; a spec-less stub breaks transformers' find_spec probe
        del sys.modules[name]
    import subprocess
    commit = subprocess.run(['git', '-C', str(QTIP), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    _ENGINE.update(bitshift_codebook=bitshift_codebook, LDLQ=LDLQ, block_LDL=block_LDL, commit=commit, compiled=None)
    return _ENGINE


def tlut_init():
    """QTIP's Gaussian k-means table (2^9 x 2, unit variance x 0.968, FP16-rounded), built deterministically once (no /tmp cache)."""
    path = TCQ / 'tlut_init.pt'
    if path.exists():
        return torch.load(path)
    import scipy.cluster.vq
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1 << 20, 2)).astype(np.float32)
    guess = rng.standard_normal((1 << SETTINGS['tlut_bits'], 2)).astype(np.float32)
    centers, _ = scipy.cluster.vq.kmeans(data, guess)
    assert centers.shape == (1 << SETTINGS['tlut_bits'], 2), centers.shape
    t = torch.tensor(centers, dtype=torch.float32)
    t = (t / t.std(unbiased=False)) * 0.9682458365518543
    t = t.half().float()
    TCQ.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)
    return t


def make_codebook(tlut, device):
    cb = engine()['bitshift_codebook'](L=SETTINGS['L'], K=SETTINGS['K'], V=SETTINGS['V'], tlut_bits=SETTINGS['tlut_bits'],
                                       decode_mode=SETTINGS['decode_mode'], tlut=tlut.float().to(device))
    cb.to(device)
    return cb


def set_tlut(cb, tlut):
    cb.tlut = tlut.float().to(cb.state.device)
    cb.recons_lut()
    cb.recons_state = cb.recons(cb.state)


def try_compile(cb):
    """The Viterbi state update is @torch.compile'd in QTIP; try it on this device, fall back to eager (recorded)."""
    e = engine()
    if e['compiled'] is not None:
        return e['compiled']
    x = torch.randn(2, SETTINGS['td'] ** 2, device=cb.state.device)
    try:
        cb.quantize(x)
        e['compiled'] = True
    except Exception as exc:                               # noqa: BLE001
        from torch import _dynamo as dynamo                # `import torch._dynamo` would shadow the module-level torch
        dynamo.config.disable = True
        cb.quantize(x)
        e['compiled'] = False
        e['compile_error'] = str(exc)[:300]
    return e['compiled']


# ------------------------------------------------------------------------------------------------------------------- codes
def state_index(states):
    """(…) int states -> (flat tlut index hash*2+v, sign) per weight pair, exactly QTIP quantlut_sym: hash = ((s+1)s >> 6) & 511, sign
    flips the first coordinate when bit 15 of (s+1)s is set."""
    x = states.long(); x = (x + 1) * x
    h = (x >> (16 - SETTINGS['tlut_bits'] - 1)) & ((1 << SETTINGS['tlut_bits']) - 1)
    sflp = 1 - ((x >> 15) & 1) * 2
    k = torch.stack([h * 2, h * 2 + 1], -1)
    sg = torch.stack([sflp, torch.ones_like(sflp)], -1)
    return k.reshape(*states.shape[:-1], -1), sg.reshape(*states.shape[:-1], -1)


def groups_for(m, n):
    """V1 (addendum 2026-11-01): row-group table count by layer size so the model stays <= 2.05 bpw linear (16 tables everywhere = 2.066):
    16 for >= 8 M weights, 8 for >= 4 M, 4 for >= 2 M, else 2; a layer with < 256 rows keeps 1 group."""
    if m < 256:
        return 1
    mn = m * n
    return 16 if mn >= 8 << 20 else 8 if mn >= 4 << 20 else 4 if mn >= 2 << 20 else 2


def group_index(m, G, device):
    return torch.arange(m, device=device) // (m // G)


def decode_normalized(states, tlut):
    """(m, n/2) states + (512, 2) tlut, or (G, 512, 2) row-group tables (rows split evenly into G contiguous groups) -> (m, n) normalized
    weights (before row / column scales and inverse RHT)."""
    k, sg = state_index(states)
    if tlut.dim() == 3:
        G = tlut.shape[0]
        g = group_index(k.shape[0], G, k.device)
        return tlut.float().reshape(G, -1)[g[:, None], k] * sg.float()
    return tlut.float().flatten()[k] * sg.float()


def decode(states, tlut, row_scale, signs, col_scale=None, signs_out=None):
    """Deployed decode: normalized weights x FP16 row scale (x FP16 column scale in the rotated domain, V1) -> inverse rotation (original
    basis, FP32): input-side block Walsh with signs; with `signs_out` also the output-side block Walsh (two-sided randomized Hadamard)."""
    q = decode_normalized(states, tlut) * row_scale.float()[:, None]
    if col_scale is not None:
        q = q * col_scale.float()[None, :]
    return inverse_rotation(q, signs, signs_out)


def inverse_rotation(q, signs, signs_out=None):
    """W = S_V H_m Q H_n S_U (two-sided) or Q H_n S_U (one-sided); H = block Walsh of backbone_rht.block_widths."""
    if signs_out is not None:
        q = inverse_rht(q.T, signs_out).T
    return inverse_rht(q, signs)


def rotation_signs(m, n, device):
    """QTIP's per-layer incoherence signs as in Chamadol's quantize_qtip.py: torch.manual_seed(0), SU = sign(randn(n)) on the inputs, then
    SV = sign(randn(m)) on the outputs (drawn on the fit device)."""
    torch.manual_seed(0)
    su = (torch.randn(n, device=device).sign() + 1e-5).sign()
    sv = (torch.randn(m, device=device).sign() + 1e-5).sign()
    return su.to(torch.int8), sv.to(torch.int8)


def rotate(name, w, h, rotation):
    """(w (m, n), h (n, n)) -> (wp, hp, signs_in, signs_out). 'one': the base's input-side block Walsh with layer_signs; 'two': QTIP's two-sided
    randomized Hadamard Wr = H_m S_V W S_U H_n with our block Walsh on every dim (differs from QTIP's had12 x H_2^k on 1536 / 3072 / 6144)."""
    from rht import layer_signs
    from backbone_rht import transform, transform_covariance
    m, n = w.shape
    if rotation == 'one':
        su, sv = layer_signs(name, n).to(w.device), None
        wp = transform(w, su)
    else:
        su, sv = rotation_signs(m, n, w.device)
        wp = transform(transform(w, su).T, sv).T
    hp = transform_covariance(h, su); hp = (hp + hp.T) / 2
    return wp, hp, su, sv


def pack(cb, states, m, n):
    td, V = SETTINGS['td'], SETTINGS['V']
    seq = states.reshape(m // td, td, n // td, td // V).transpose(1, 2).reshape(-1, td * td // V)
    return cb.pack_trellis(seq.to(cb.state.device))


def unpack(cb, packed, m, n):
    td, V = SETTINGS['td'], SETTINGS['V']
    seq = cb.unpack_trellis(packed.to(cb.state.device), td * td)
    return seq.reshape(m // td, n // td, td, td // V).transpose(1, 2).reshape(m, n // V)


def signed_index(states):
    """int16 per weight: sign * (flat tlut index + 1) — the trainable layer's buffer (groot_tcq_raqft.py)."""
    k, sg = state_index(states)
    return (sg * (k + 1)).to(torch.int16)


# ------------------------------------------------------------------------------------------------------------------- WLS
def h_error(wp, hp, dec):
    e = wp - dec
    return float(torch.sum((e @ hp) * e))


def wls_scale(wp, hp, q):
    """Row scale s_r = (W'_r H' Q_r) / (Q_r H' Q_r) with the normalized decode Q fixed; FP16."""
    num = torch.sum((wp @ hp) * q, dim=1)
    den = torch.sum((q @ hp) * q, dim=1).clamp_min(1e-30)
    return (num / den).clamp_min(torch.finfo(torch.float16).tiny).half()


def wls_tlut(wp, hp, k, sg, scale, tlut, rows_per_chunk=8):
    """The 1024 table values enter W' linearly (w'_rc = s_r sg_rc t[k_rc]); solve the full-H' normal equations, ridge toward the current
    table on unidentified entries (as vq_core.fixed_code_wls). Returns the FP16-rounded candidate (acceptance is the caller's)."""
    m, n = wp.shape
    E = tlut.numel()
    s = scale.float()
    ss = sg.float() * s[:, None]
    dev = wp.device
    b = torch.zeros(E, device=dev).index_add_(0, k.flatten(), (ss * (wp @ hp)).flatten())
    M = torch.zeros(E, E, device=dev)
    for r0 in range(0, m, rows_per_chunk):
        kk, sk = k[r0:r0 + rows_per_chunk], ss[r0:r0 + rows_per_chunk]
        R = kk.shape[0]
        flat = (torch.arange(R, device=dev)[:, None] * E + kk).flatten()
        src = (hp[None].expand(R, -1, -1) * sk[:, :, None]).reshape(R * n, n)
        HA = torch.zeros(R * E, n, device=dev).index_add_(0, flat, src).view(R, E, n)
        src2 = (HA * sk[:, None, :]).permute(0, 2, 1).reshape(R * n, E)
        Y = torch.zeros(R * E, E, device=dev).index_add_(0, flat, src2)
        M += Y.view(R, E, E).sum(0)
        del src, HA, src2, Y
    strength = M.diagonal()
    active = strength > 0
    ridge = (strength * 1e-7).clamp_min(torch.finfo(torch.float32).tiny)
    A = M[active][:, active] + torch.diag(ridge[active])
    rhs = b[active] + ridge[active] * tlut.flatten()[active]
    new = tlut.flatten().clone()
    new[active] = torch.linalg.solve(A, rhs)
    return new.view_as(tlut).half().float(), int(active.sum())


# ------------------------------------------------------------------------------------------------------------------- fit
def fit_layer(name, w, h, cb, tlut0, rounds=2, log=None, dampings=None, rotation='one'):
    """One Linear. w (m, n) FP32 cuda, h (n, n) FP32 cuda (E[x x^T], original basis). Returns (record, info)."""
    e = engine()
    LDLQ, block_LDL = e['LDLQ'], e['block_LDL']
    td, V = SETTINGS['td'], SETTINGS['V']
    m, n = w.shape
    assert m % td == 0 and n % td == 0, (name, m, n)
    t0 = time.time()
    wp, hp, signs, signs_out = rotate(name, w, h, rotation)
    hp = hp / hp.diagonal().mean().clamp_min(torch.finfo(hp.dtype).eps)
    energy = float(torch.sum((wp @ hp) * wp))
    ldl, damping, retries = None, None, 0
    for damping in (dampings or DAMPINGS):
        hd = hp.clone(); hd.diagonal().add_(damping)
        ldl = block_LDL(hd, td)
        if ldl is not None:
            break
        retries += 1
    assert ldl is not None, f'{name}: block LDL failed at every damping {dampings or DAMPINGS}'
    L = ldl[0]
    diag = torch.arange(n, device=L.device); L[diag, diag] = 0
    scale = (wp.square().mean(1).sqrt() / SETTINGS['scale_override']).clamp_min(torch.finfo(torch.float16).tiny).half()
    tlut = tlut0.clone().to(w.device)
    args = Namespace(td_x=td, td_y=td, V=V)
    best, history = None, []
    for r in range(rounds):
        t_enc = time.time()
        set_tlut(cb, tlut)
        wn = wp / scale.float()[:, None]
        _, states = LDLQ(wn, L, cb, args, buf_cols=td, for_kernel=False)
        states = states.to(w.device)
        t_enc = time.time() - t_enc
        k, sg = state_index(states)
        q = tlut.flatten()[k] * sg.float()
        err = h_error(wp, hp, q * scale.float()[:, None])
        # our WLS: row scales, then the table, then the scales again; each accepted only if the calibration H-error decreases
        steps, active = [('encode', err)], None
        for what in ('scale', 'tlut', 'scale'):
            if what == 'scale':
                cand = wls_scale(wp, hp, q)
                e_c = h_error(wp, hp, q * cand.float()[:, None])
                if e_c < err:
                    scale, err = cand, e_c; steps.append(('scale', err))
            else:
                cand, active = wls_tlut(wp, hp, k, sg, scale, tlut)
                qc = cand.flatten()[k] * sg.float()
                e_c = h_error(wp, hp, qc * scale.float()[:, None])
                if e_c < err:
                    tlut, q, err = cand, qc, e_c; steps.append(('tlut', err))
        history.append(dict(round=r, encode_seconds=t_enc, steps=[(a, float(b_)) for a, b_ in steps], tlut_active_entries=active, relative_error=err / max(energy, 1e-30)))
        if best is None or err < best['err']:
            best = dict(err=err, round=r, states=states.clone(), tlut=tlut.clone(), scale=scale.clone())
        if log:
            print(json.dumps(dict(layer=name, round=r, encode_s=round(t_enc, 1), rel_err=round(err / max(energy, 1e-30), 6))), flush=True)
    states, tlut, scale = best['states'], best['tlut'], best['scale']
    w_hat = decode(states, tlut, scale, signs, None, signs_out)
    rel = float((w_hat - w).square().sum() / w.square().sum())
    herr = float(torch.sum(((w_hat - w) @ h) * (w_hat - w)) / torch.sum((w @ h) * w).clamp_min(1e-30))
    packed = pack(cb, states, m, n).cpu()
    assert packed.dtype == torch.uint16 and packed.numel() * 2 * 8 == m * n * SETTINGS['K'], (packed.shape, m, n)      # K bits per weight, tail-biting: no padding
    tensors = dict(trellis=packed, tlut=tlut.half().cpu(), row_scale=scale.cpu(), signs=signs.to(torch.int8).cpu())
    if signs_out is not None:
        tensors['signs_out'] = signs_out.to(torch.int8).cpu()
    payload = sum(t.numel() * t.element_size() for t in tensors.values())
    record = dict(w_hat=w_hat.to(torch.bfloat16).cpu(), tensors=tensors, tcq=dict(SETTINGS, rotation=rotation))
    info = dict(name=name, shape=[m, n], payload_bytes=payload, trellis_bytes=packed.numel() * 2, tlut_bytes=tlut.numel() * 2, row_scale_bytes=m * 2, sign_bytes=n + (m if signs_out is not None else 0), rotation=rotation,
                padding=0, relative_weight_mse=rel, hessian_relative_error=herr, damping=damping, cholesky_retries=retries, selected_round=best['round'],
                rounds=history, encode_seconds=sum(x['encode_seconds'] for x in history), seconds=time.time() - t0, engine=dict(qtip_commit=e['commit'], compiled=e['compiled']))
    return record, info


def viterbi_steps(m, n):
    """Viterbi state updates per encode (LDLQ with buf_cols = td): column tiles x batches of <= 256 sequences x 2 passes x 128 symbols."""
    td, V = SETTINGS['td'], SETTINGS['V']
    return (n // td) * math.ceil((m // td) / 256) * 2 * (td * td // V)


def stage_fit(key, worker, workers, rounds, device='cuda', rotation='one', metric='base', dround=1):
    """key = backbone | head_<suite>; output fit key KEY_OF[key] (one-sided) or RHT_KEY_OF[key] (two-sided); metric iso / shift (SHIFT_2026-11-03,
    two-sided only): H_iso = H + tr(H)/d I, H_rob = H + tr(H)/tr(Σ) Σ_shift, damping ladder SHIFT_DAMPINGS."""
    assert metric == 'base' or rotation == 'two'
    out_key = (KEY_OF if rotation == 'one' else RHT_KEY_OF)[key] if metric == 'base' else (D_KEY_OF[dround][key] if metric == 'mix' else METRIC_KEY_OF[metric][key])
    hdir = OUT / 'hessians' / (D_HKEY_OF[dround][key] if metric == 'mix' else key)
    assert (hdir / 'complete.json').exists(), f'calibrate {key} first'
    names = sorted(p.name[:-3] for p in hdir.glob('*.pt'))
    mine = [n for i, n in enumerate(names) if i % workers == worker]
    suite = 'libero_spatial' if key == 'backbone' else key.removeprefix('head_')
    directory = OUT / 'fit' / out_key
    directory.mkdir(parents=True, exist_ok=True)
    todo = [n for n in mine if not (directory / f'{n}.json').exists()]
    weights = weights_for(suite, todo) if todo else {}
    tlut0 = tlut_init().to(device)
    cb = make_codebook(tlut0, device)
    compiled = try_compile(cb)
    t_all = time.time()
    for n in todo:
        w = weights[n].to(device).float()
        h = torch.load(hdir / f'{n}.pt', map_location=device)['hessian'].float()
        extra = {} if metric != 'mix' else dict(metric='mix', round=dround, hessian_key=D_HKEY_OF[dround][key])
        if metric not in ('base', 'mix'):
            trh = float(h.double().diagonal().sum())
            if metric == 'iso':
                lam = trh / h.shape[0]; h = h + lam * torch.eye(h.shape[0], device=device)
            else:
                sg = torch.load(OUT.parent / 'shift' / 'sigma' / key / f'{n}.pt', map_location=device)['sigma'].float()
                trs = float(sg.double().diagonal().sum()); lam = trh / max(trs, 1e-300); h = h + lam * sg; del sg
            extra = dict(metric=metric, lam=lam, tr_h=trh, tr_metric=float(h.double().diagonal().sum()))
        dampings = SHIFT_DAMPINGS if metric not in ('base', 'mix') else None
        if metric == 'mix':                                                     # user 2026-11-07: D differs from A only in H -> damping pinned to A's recorded value for this layer
            a_json = OUT / 'fit' / RHT_KEY_OF[key] / f'{n}.json'
            if a_json.exists():
                a_damp = json.loads(a_json.read_text())['damping']
                dampings = (a_damp,) + tuple(d for d in DAMPINGS if d > a_damp); extra['damping_pinned_to'] = f'{RHT_KEY_OF[key]}:{a_damp}'
            else:                                                               # no clean-H row in this tree (1-bit runs): the base ladder, whose first value 0.01 is what arm A recorded on all 432 layers
                dampings = None; extra['damping_pinned_to'] = 'base ladder 0.01 / 0.1 / 1.0 (arm A recorded 0.01 on every layer)'
        record, info = fit_layer(n, w, h, cb, tlut0, rounds=rounds, rotation=rotation, dampings=dampings)
        if extra:
            info.update(extra); record['tcq']['metric'] = metric
        atomic_save(record, directory / f'{n}.pt')
        dump(directory / f'{n}.json', info)
        print(json.dumps(dict(layer=n, shape=info['shape'], rel_mse=round(info['relative_weight_mse'], 5), h_err=round(info['hessian_relative_error'], 6), damping=info['damping'],
                              round=info['selected_round'], encode_s=round(info['encode_seconds'], 1), seconds=round(info['seconds'], 1))), flush=True)
        del w, h, record
        torch.cuda.empty_cache()
    done = [n for n in names if (directory / f'{n}.json').exists()]
    if len(done) == len(names) and not (directory / 'complete.json').exists():
        metas = [json.loads((directory / f'{n}.json').read_text()) for n in names]
        dump(directory / 'complete.json', dict(status='complete', key=out_key, source_key=key, layers=len(names), payload_bytes=sum(x['payload_bytes'] for x in metas),
                                              parameters=sum(x['shape'][0] * x['shape'][1] for x in metas),
                                              relative_weight_mse_mean=float(np.mean([x['relative_weight_mse'] for x in metas])), hessian_relative_error_mean=float(np.mean([x['hessian_relative_error'] for x in metas])),
                                              dampings={str(d): sum(x['damping'] == d for x in metas) for d in sorted({x['damping'] for x in metas})}, metric=metric, encode_seconds=sum(x['encode_seconds'] for x in metas), seconds=sum(x['seconds'] for x in metas),
                                              format=dict(SETTINGS, rotation=rotation), engine=dict(qtip_commit=engine()['commit'], compiled=compiled)))
    print(json.dumps(dict(fit=out_key, worker=worker, layers=len(todo), seconds=round(time.time() - t_all, 1), compiled=compiled, rotation=rotation)), flush=True)


def stage_g16(key, worker, workers, device='cuda', base='tcq'):
    """V1 base (addendum 2026-11-01): from the tcq fit record of every layer, split the rows into G = groups_for(m, n) contiguous groups,
    give each group its own 2^9 x 2 table initialised from the layer's fitted tlut and re-fitted by our WLS on that group's rows under
    H' (codes and row scales fixed; accepted per group only if the group's H-error decreases); column scales = 1. Writes fit key
    G16_KEY_OF[key] (record: tlut (G, 512, 2) FP16, col_scale (n,) FP16, groups)."""
    src_key, out_key = BASES[base][0][key], BASES[base][1][key]
    hdir = OUT / 'hessians' / key
    src = OUT / 'fit' / src_key
    assert (src / 'complete.json').exists(), f'fit {src_key} first'
    names = sorted(p.name[:-3] for p in src.glob('*.pt'))
    mine = [n for i, n in enumerate(names) if i % workers == worker]
    suite = 'libero_spatial' if key == 'backbone' else key.removeprefix('head_')
    directory = OUT / 'fit' / out_key
    directory.mkdir(parents=True, exist_ok=True)
    todo = [n for n in mine if not (directory / f'{n}.json').exists()]
    weights = weights_for(suite, todo) if todo else {}
    cb = make_codebook(tlut_init().to(device), device)
    t_all = time.time()
    for name in todo:
        t0 = time.time()
        rec = torch.load(src / f'{name}.pt', map_location='cpu'); t = rec['tensors']
        base_info = json.loads((src / f'{name}.json').read_text())
        w = weights[name].to(device).float()
        h = torch.load(hdir / f'{name}.pt', map_location=device)['hessian'].float()
        m, n = w.shape
        rotation = rec['tcq'].get('rotation', 'one')
        wp, hp, signs, signs_out = rotate(name, w, h, rotation)
        assert torch.equal(signs, t['signs'].to(device)) and (signs_out is None or torch.equal(signs_out, t['signs_out'].to(device))), name
        hp = hp / hp.diagonal().mean().clamp_min(torch.finfo(hp.dtype).eps)
        energy = float(torch.sum((wp @ hp) * wp))
        states = unpack(cb, t['trellis'], m, n).to(device)
        k, sg = state_index(states)
        scale = t['row_scale'].to(device)
        tlut0 = t['tlut'].float().to(device)
        G = groups_for(m, n)
        rows = m // G
        tables, accepted, err_before, err_after = [], 0, 0.0, 0.0
        for g in range(G):
            sl = slice(g * rows, (g + 1) * rows)
            q0 = tlut0.flatten()[k[sl]] * sg[sl].float()
            e0 = h_error(wp[sl], hp, q0 * scale[sl].float()[:, None])
            cand, _ = wls_tlut(wp[sl], hp, k[sl], sg[sl], scale[sl], tlut0)
            q1 = cand.flatten()[k[sl]] * sg[sl].float()
            e1 = h_error(wp[sl], hp, q1 * scale[sl].float()[:, None])
            if e1 < e0:
                tables.append(cand); accepted += 1; err_after += e1
            else:
                tables.append(tlut0.clone()); err_after += e0
            err_before += e0
        tlut = torch.stack(tables).half()                                    # (G, 512, 2)
        col_scale = torch.ones(n, dtype=torch.float16, device=device)
        w_hat = decode(states, tlut, scale, signs, col_scale, signs_out)
        rel = float((w_hat - w).square().sum() / w.square().sum())
        herr = float(torch.sum(((w_hat - w) @ h) * (w_hat - w)) / torch.sum((w @ h) * w).clamp_min(1e-30))
        tensors = dict(trellis=t['trellis'], tlut=tlut.cpu(), row_scale=t['row_scale'], col_scale=col_scale.cpu(), signs=t['signs'], **({'signs_out': t['signs_out']} if signs_out is not None else {}))
        payload = sum(x.numel() * x.element_size() for x in tensors.values())
        record = dict(w_hat=w_hat.to(torch.bfloat16).cpu(), tensors=tensors, tcq=dict(SETTINGS, groups=G, rotation=rotation))
        info = dict(base_info, name=name, shape=[m, n], groups=G, rows_per_group=rows, payload_bytes=payload, tlut_bytes=tlut.numel() * 2, col_scale_bytes=n * 2,
                    relative_weight_mse=rel, hessian_relative_error=herr, source_key=src_key, source_hessian_relative_error=base_info['hessian_relative_error'],
                    group_refit=dict(accepted_groups=accepted, calibration_error_before=err_before / max(energy, 1e-30), calibration_error_after=err_after / max(energy, 1e-30)),
                    seconds=time.time() - t0)
        atomic_save(record, directory / f'{name}.pt')
        dump(directory / f'{name}.json', info)
        print(json.dumps(dict(layer=name, shape=[m, n], groups=G, accepted=accepted, h_err=round(herr, 6), base=round(base_info['hessian_relative_error'], 6), rel_mse=round(rel, 5), seconds=round(info['seconds'], 1))), flush=True)
        del w, h, rec, record
        torch.cuda.empty_cache()
    done = [n for n in names if (directory / f'{n}.json').exists()]
    if len(done) == len(names) and not (directory / 'complete.json').exists():
        metas = [json.loads((directory / f'{n}.json').read_text()) for n in names]
        dump(directory / 'complete.json', dict(status='complete', key=out_key, source_key=src_key, layers=len(names), payload_bytes=sum(x['payload_bytes'] for x in metas),
                                              parameters=sum(x['shape'][0] * x['shape'][1] for x in metas), groups={g: sum(x['groups'] == g for x in metas) for g in (1, 2, 4, 8, 16)},
                                              relative_weight_mse_mean=float(np.mean([x['relative_weight_mse'] for x in metas])), hessian_relative_error_mean=float(np.mean([x['hessian_relative_error'] for x in metas])),
                                              source_hessian_relative_error_mean=float(np.mean([x['source_hessian_relative_error'] for x in metas])), accepted_groups=sum(x['group_refit']['accepted_groups'] for x in metas),
                                              seconds=sum(x['seconds'] for x in metas), format=dict(SETTINGS, groups='groups_for(m, n)', col_scale='FP16 ones'), engine=dict(qtip_commit=engine()['commit'])))
    print(json.dumps(dict(g16=out_key, worker=worker, layers=len(todo), seconds=round(time.time() - t_all, 1))), flush=True)


def stage_refit(key, layers, dampings, rounds, device='cuda', rotation='one'):
    """§2 audit (2026-11-02): refit named layers at explicit dampings (one record per damping) under fit/<KEY_OF[key]>_audit/<damping>/."""
    hdir = OUT / 'hessians' / key
    suite = 'libero_spatial' if key == 'backbone' else key.removeprefix('head_')
    weights = weights_for(suite, layers)
    tlut0 = tlut_init().to(device)
    cb = make_codebook(tlut0, device)
    try_compile(cb)
    for name in layers:
        w = weights[name].to(device).float()
        h = torch.load(hdir / f'{name}.pt', map_location=device)['hessian'].float()
        base = json.loads((OUT / 'fit' / KEY_OF[key] / f'{name}.json').read_text())
        for d in dampings:
            out = OUT / 'fit' / f'{KEY_OF[key]}_audit{"_rht" if rotation == "two" else ""}' / str(d); out.mkdir(parents=True, exist_ok=True)
            if (out / f'{name}.json').exists():
                continue
            record, info = fit_layer(name, w, h, cb, tlut0, rounds=rounds, dampings=(d,), rotation=rotation)
            info['audit'] = dict(rotation=rotation, requested_damping=d, base_damping=base['damping'], base_relative_weight_mse=base['relative_weight_mse'], base_hessian_relative_error=base['hessian_relative_error'])
            atomic_save(record, out / f'{name}.pt'); dump(out / f'{name}.json', info)
            print(json.dumps(dict(layer=name, rotation=rotation, damping=d, rel_mse=round(info['relative_weight_mse'], 5), h_err=info['hessian_relative_error'], base_mse=round(base['relative_weight_mse'], 5), base_h_err=base['hessian_relative_error'], seconds=round(info['seconds'], 1))), flush=True)


# ------------------------------------------------------------------------------------------------------------------- gates
def forward_actions(bridge, observations, weights=None):
    """Actions of the bridge's model on the given (index, obs) pairs after loading `weights` ({layer name: BF16 weight}) in place."""
    model = bridge.policy.model
    with torch.no_grad():
        for name, w in (weights or {}).items():
            model.get_submodule(name).weight.copy_(w.to(model.get_submodule(name).weight.dtype))
    acts = []
    with torch.no_grad():
        for i, obs in observations:
            torch.manual_seed(i); bridge.policy.reset()
            action, _ = bridge.policy.get_action(bridge.observation(obs))
            acts.append({k: np.asarray(v).copy() for k, v in action.items() if np.asarray(v).dtype.kind in 'fiu'})   # every numeric action key (this checkpoint's keys carry no 'action.' prefix)
    assert acts and all(a for a in acts), 'no numeric action keys'
    return acts


def stage_gate(rounds, device='cuda'):
    from calibration_data import load_observations
    from groot_offline import HELDOUT
    base_keys = {'backbone': 'backbone', 'head_libero_spatial': 'head_libero_spatial'}
    tlut0 = tlut_init().to(device)
    cb = make_codebook(tlut0, device)
    compiled = try_compile(cb)
    result = dict(engine=dict(qtip_commit=engine()['commit'], compiled=compiled, compile_error=engine().get('compile_error')), format=dict(SETTINGS), gpu=torch.cuda.get_device_name(), layers={})
    records = {}
    for role, name in GATE_LAYERS.items():
        key = 'backbone' if part_of(name) == 'backbone' else 'head_libero_spatial'
        w = weights_for('libero_spatial', [name])[name].to(device).float()
        h = torch.load(OUT / 'hessians' / key / f'{name}.pt', map_location=device)['hessian'].float()
        record, info = fit_layer(name, w, h, cb, tlut0, rounds=rounds, log=True)
        base = json.loads((OUT / 'fit' / base_keys[key] / f'{name}.json').read_text())
        info['base_hessian_relative_error'] = base['hessian_relative_error']
        info['ratio_tcq_over_rvq'] = info['hessian_relative_error'] / max(base['hessian_relative_error'], 1e-30)
        info['base_relative_weight_mse'] = base['relative_weight_mse']
        info['base_payload_bytes'] = base['payload_bytes']
        info['viterbi_steps_per_encode'] = viterbi_steps(*info['shape'])
        # (b) decode round trip on this layer: payload -> states -> weights == w_hat bit for bit
        t = record['tensors']
        states = unpack(cb, t['trellis'], *info['shape'])
        w_dec = decode(states, t['tlut'].to(device), t['row_scale'].to(device), t['signs'].to(device)).to(torch.bfloat16).cpu()
        info['roundtrip_bitwise'] = bool(torch.equal(w_dec, record['w_hat']))
        set_tlut(cb, t['tlut'].to(device))                      # the engine's table must be the selected one before comparing its recons
        info['recons_matches_engine'] = bool(torch.equal(decode_normalized(states, t['tlut'].to(device)).half(), cb.recons(states.to(cb.state.device)).permute(1, 2, 0).reshape(*info['shape']).half().to(device)))
        records[name] = (record, w_dec)
        result['layers'][role] = info
        print(json.dumps(dict(gate=role, layer=name, ratio=round(info['ratio_tcq_over_rvq'], 4), h_err=info['hessian_relative_error'], base=base['hessian_relative_error'], damping=info['damping'], retries=info['cholesky_retries'],
                              roundtrip=info['roundtrip_bitwise'], engine_match=info['recons_matches_engine'], encode_s=round(info['encode_seconds'], 1), seconds=round(info['seconds'], 1))), flush=True)
        del w, h
        torch.cuda.empty_cache()
    three = [result['layers'][r]['ratio_tcq_over_rvq'] for r in ('llm_attention', 'llm_mlp', 'dit_attention')]
    result['gate_a'] = dict(ratios=three, passed=bool(all(x < 1 for x in three)), rule='TCQ H-weighted error below the 2x8 RVQ fit on all three layers')
    # (c) tiling: every quantized width is a multiple of 16 (no padding)
    shapes = {}
    for key in ('backbone', 'head_libero_spatial'):
        for p in (OUT / 'fit' / key).glob('*.json'):
            if p.name != 'complete.json':
                s = json.loads(p.read_text())['shape']; shapes[tuple(s)] = shapes.get(tuple(s), 0) + 1
    result['gate_c'] = dict(shapes={f'{a}x{b}': c for (a, b), c in sorted(shapes.items())}, all_tiled=bool(all(a % 16 == 0 and b % 16 == 0 for a, b in shapes)), padding_bytes=0)
    # (d) speed: seconds per Viterbi step from the measured encodes -> per-model and all-suite extrapolation
    enc = [(info['encode_seconds'] / rounds, info['viterbi_steps_per_encode']) for info in result['layers'].values()]
    sec_per_step = sum(a for a, _ in enc) / sum(b for _, b in enc)
    other = sum(info['seconds'] - info['encode_seconds'] for info in result['layers'].values()) / len(result['layers'])
    steps_model = sum(viterbi_steps(a, b) * c for (a, b), c in shapes.items())
    layers_model = sum(shapes.values())
    per_model_h = (steps_model * rounds * sec_per_step + layers_model * other) / 3600
    bb_shapes, hd_shapes = {}, {}
    for key, store in (('backbone', bb_shapes), ('head_libero_spatial', hd_shapes)):
        for p in (OUT / 'fit' / key).glob('*.json'):
            if p.name != 'complete.json':
                s = tuple(json.loads(p.read_text())['shape']); store[s] = store.get(s, 0) + 1
    hours = lambda st: (sum(viterbi_steps(a, b) * c for (a, b), c in st.items()) * rounds * sec_per_step + sum(st.values()) * other) / 3600
    result['gate_d'] = dict(seconds_per_viterbi_step=sec_per_step, non_encode_seconds_per_layer=other, rounds=rounds, per_model_gpu_hours=per_model_h, backbone_gpu_hours=hours(bb_shapes), head_gpu_hours=hours(hd_shapes),
                            all_suites_gpu_hours=hours(bb_shapes) + 4 * hours(hd_shapes), budget_gpu_hours=6, within_budget=bool(per_model_h <= 6), compiled=compiled)
    # (b) forward on 2 held-out observations: fake-quant weights (w_hat) vs decoded weights, bit for bit
    from groot_bridge_server import Bridge
    _, observations = load_observations(limit=2, data_dir=HELDOUT)
    bridge = Bridge('libero_spatial', device=device)
    obs = [(i, o) for i, (_, o) in enumerate(observations)]
    a_fake = forward_actions(bridge, obs, {n: r['w_hat'] for n, (r, _) in records.items()})
    a_dec = forward_actions(bridge, obs, {n: d for n, (_, d) in records.items()})
    a_fake2 = forward_actions(bridge, obs, {n: r['w_hat'] for n, (r, _) in records.items()})
    same = all(all(np.array_equal(x[k], y[k]) for k in x) for x, y in zip(a_fake, a_dec))
    repeat = all(all(np.array_equal(x[k], y[k]) for k in x) for x, y in zip(a_fake, a_fake2))
    result['gate_b'] = dict(observations=2, layers=list(GATE_LAYERS.values()), decoded_weights_bitwise=all(i['roundtrip_bitwise'] for i in result['layers'].values()), forward_bitwise=bool(same), forward_repeat_bitwise=bool(repeat),
                            max_abs_action_gap=float(max(np.abs(x[k] - y[k]).max() for x, y in zip(a_fake, a_dec) for k in x)))
    result['passed'] = bool(result['gate_a']['passed'] and result['gate_b']['decoded_weights_bitwise'] and result['gate_b']['forward_bitwise'] and result['gate_c']['all_tiled'])
    dump(TCQ / 'gate.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'layers'}, indent=1), flush=True)


def stage_roundtrip(suite, row, device='cuda'):
    """Full model: decode every fit record's payload and compare with the assembled checkpoint's weights (bit for bit); then 2 observations."""
    from safetensors import safe_open
    from groot_ptq import ROWS
    from groot_offline import HELDOUT
    from calibration_data import load_observations
    from groot_bridge_server import Bridge
    cb = make_codebook(tlut_init().to(device), device)
    ck = MODELS / row / suite / 'model.safetensors'
    assert ck.exists(), f'assemble {row} {suite} first'
    fits = {}
    for part in ROWS[row]:
        d = OUT / 'fit' / (part if part.startswith('backbone') else f'{part}_{suite}')
        fits.update({p.name[:-3]: p for p in d.glob('*.pt')})
    mism, n_layers, payload = [], 0, 0
    with safe_open(str(ck), 'pt') as f:
        for name, p in sorted(fits.items()):
            rec = torch.load(p, map_location='cpu')
            t = rec['tensors']
            m, n = rec['w_hat'].shape
            states = unpack(cb, t['trellis'], m, n)
            w = decode(states, t['tlut'].to(device), t['row_scale'].to(device), t['signs'].to(device), t['col_scale'].to(device) if 'col_scale' in t else None, t['signs_out'].to(device) if 'signs_out' in t else None).to(torch.bfloat16).cpu()
            if not torch.equal(w, f.get_tensor(name + '.weight')):
                mism.append(name)
            n_layers += 1; payload += json.loads(p.with_suffix('.json').read_text())['payload_bytes']
    _, observations = load_observations(limit=2, data_dir=HELDOUT)
    obs = [(i, o) for i, (_, o) in enumerate(observations)]
    bridge = Bridge(suite, device=device, model_root=str(MODELS / row))
    a1 = forward_actions(bridge, obs)
    # reload decoded weights (identical bytes if mism is empty) and run again
    dec = {}
    for name, p in sorted(fits.items()):
        rec = torch.load(p, map_location='cpu'); t = rec['tensors']; m, n = rec['w_hat'].shape
        dec[name] = decode(unpack(cb, t['trellis'], m, n), t['tlut'].to(device), t['row_scale'].to(device), t['signs'].to(device), t['col_scale'].to(device) if 'col_scale' in t else None, t['signs_out'].to(device) if 'signs_out' in t else None).to(torch.bfloat16).cpu()
    a2 = forward_actions(bridge, obs, dec)
    same = all(all(np.array_equal(x[k], y[k]) for k in x) for x, y in zip(a1, a2))
    res = dict(suite=suite, row=row, layers=n_layers, payload_bytes=payload, weight_mismatches=mism, weights_bitwise=not mism, forward_bitwise=bool(same), observations=2, gpu=torch.cuda.get_device_name())
    dump(TCQ / f'roundtrip_{row}_{suite}.json', res)
    print(json.dumps(res), flush=True)


# ------------------------------------------------------------------------------------------------------------------- readout error
def stage_readout(rows, device='cuda'):
    """B2's L_readout (student-head readouts vs the FP head on the FP trajectory, W_O metric) and L_vel, averaged over the 256 held-out
    observations (noise seed SeedSequence([7, i, 0])), per assembled row; FP = 0 by definition."""
    from groot_bridge_server import Bridge
    from calibration_data import load_observations
    from groot_offline import HELDOUT, device_tag
    from groot_amraq import ReaderTaps
    from groot_rowb2 import losses
    from groot_raqft import CONFIG
    cfg = dict(CONFIG, executed=8, valid_dims=7)
    _, observations = load_observations(data_dir=HELDOUT)
    assert len(observations) == 256
    out = OUT.parent / 'offline' / device_tag()
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for row in rows:
        path = out / row / 'readout.json'
        if path.exists():
            summary[row] = json.loads(path.read_text()); continue
        per_suite = {}
        for suite in ('libero_spatial',):                      # as the held-out MSE (groot_offline): every held-out observation through the spatial checkpoint
            idx = list(range(256))
            teacher = Bridge(suite, device=device)
            student = Bridge(suite, device=device, model_root=str(MODELS / row))
            tm, sm, policy = teacher.policy.model, student.policy.model, student.policy
            taps_t, taps_s = ReaderTaps(tm.action_head), ReaderTaps(sm.action_head)
            wo = {i: tm.action_head.model.transformer_blocks[i].attn1.to_out[0].weight.detach().float().to(device) for i in taps_t.cross}
            lr, lv = [], []
            with torch.no_grad():
                for i in idx:
                    obs = dict(observations[i][1]); obs['_vla_episode_seed'] = i
                    a, b = losses(tm, sm, taps_t, taps_s, policy, obs, int(np.random.SeedSequence([7, i, 0]).generate_state(1)[0]), cfg, wo)
                    lr.append(float(a)); lv.append(float(b))
            per_suite[suite] = dict(observations=len(idx), L_readout=float(np.mean(lr)), L_vel=float(np.mean(lv)))
            del teacher, student; torch.cuda.empty_cache()
        total = sum(v['observations'] for v in per_suite.values())
        summary[row] = dict(row=row, L_readout=sum(v['L_readout'] * v['observations'] for v in per_suite.values()) / total, L_vel=sum(v['L_vel'] * v['observations'] for v in per_suite.values()) / total, per_suite=per_suite, gpu=torch.cuda.get_device_name())
        dump(path, summary[row])
        print(json.dumps(summary[row]), flush=True)
    dump(out / 'readout_summary.json', summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('tlut', 'gate', 'fit', 'g16', 'refit', 'roundtrip', 'readout'), required=True)
    p.add_argument('--metric', choices=('base', 'iso', 'shift', 'mix'), default='base', help='fit metric (SHIFT_2026-11-03 arms B / C; mix = arm D mixed-set H, base recipe; two-sided only)')
    p.add_argument('--dround', type=int, default=1, help='metric mix: arm D round (hessians backbone_mix<r> / head_mix<r>_<suite>)')
    p.add_argument('--rotation', choices=('one', 'two'), default='one', help='fit / refit: input-side block Walsh (base) or two-sided randomized Hadamard (user 2026-11-02)')
    p.add_argument('--base', choices=tuple(BASES), default='tcq', help='g16: source base')
    p.add_argument('--layers', default='backbone.model.model.language_model.layers.2.mlp.down_proj', help="refit: ':'-separated layer names")
    p.add_argument('--dampings', default='0.1:1.0:0.001', help="refit: ':'-separated dampings")
    p.add_argument('--key', choices=tuple(KEY_OF), default='backbone')
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--rounds', type=int, default=2, help='encode rounds (each: LDLQ + scale / tlut / scale WLS); best by calibration H-error')
    p.add_argument('--suite', choices=SUITES, default='libero_spatial')
    p.add_argument('--row', default='tcq_w2')
    p.add_argument('--rows', default='ptq_w2:tcq_w2')
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    if a.stage == 'tlut':
        t = tlut_init(); print(json.dumps(dict(tlut=list(t.shape), rms=float(t.square().mean().sqrt()), path=str(TCQ / 'tlut_init.pt'))))
    elif a.stage == 'gate':
        stage_gate(a.rounds)
    elif a.stage == 'fit':
        stage_fit(a.key, a.worker, a.workers, a.rounds, rotation=a.rotation, metric=a.metric, dround=a.dround)
    elif a.stage == 'g16':
        stage_g16(a.key, a.worker, a.workers, base=a.base)
    elif a.stage == 'refit':
        stage_refit(a.key, a.layers.split(':'), [float(x) for x in a.dampings.split(':')], a.rounds, rotation=a.rotation)
    elif a.stage == 'roundtrip':
        stage_roundtrip(a.suite, a.row)
    elif a.stage == 'readout':
        stage_readout(a.rows.split(':'))


if __name__ == '__main__':
    main()
