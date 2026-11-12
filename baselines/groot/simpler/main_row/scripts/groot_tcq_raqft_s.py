"""V0' — stabilized fine-tune on tcq_w2_rht (user 2026-11-04, TCQ_RAQFT_2026-11-01.md addendum 3). Same recipe as V0
(groot_tcq_raqft.py --variant v0_rht: B2 loss, trainable tables / row scales / norm weights, 768 updates, 8 latents, β by gradient-norm
matching at step 0, clip 1) with only these changes:
  1. Adam lr: tables 2e-5, norm weights 2e-5, row scales 1e-4 as log s; 5 % linear warmup (38 updates) then cosine to 0; global clip
     1.0; EMA (0.99) of every trainable, applied after each step; the EMA weights are evaluated and exported.
  2. Gauge fix after each step: every table is rescaled to its initial RMS and the factor is folded into the row scales (log s).
  3. Anchor μ·Σ_l ‖Ŵ_l − W_l^FP‖²_F/‖W_l^FP‖²_F over the trainable layers of the routed model (shared backbone + the suite's head),
     μ such that the anchor is 10 % of the initial total loss (μ·A0 = (L0 + μ·A0)/10 -> μ = L0/(9·A0)), recorded; the anchor is
     back-propagated layer by layer after the latents.
  4. 40 of the 256 observations held out (stratified by suite: 12 / 10 / 10 / 8, rng SeedSequence([20261104, 0])); training cycles the
     216 others in index order to 768 updates; held-out L_readout / L_vel of the EMA weights (4 latents per observation, seeds
     SeedSequence([7, i, k])) at update 0 and every 64 updates; the best held-out L_readout snapshot is exported. Curve recorded.
Stage holdout: held-out loss of assembled rows (dense fake-quant students), for the V0 / base comparison points on the same 40 observations.
Exports fit keys backbone_tcq_raqft_rht_s / head_tcq_raqft_rht_s_<suite>; row tcq_raqft_rht_s."""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import groot_tcq_raqft as R
from groot_ptq import OUT, MODELS, SUITES, dump
from groot_raqft import CONFIG as ROW_A_CONFIG, RUNS
from groot_rowb import weight_cache
from groot_rowb2 import losses, grad_norm, MAX_SKIPPED, CKPT_EVERY
from groot_amraq import ReaderTaps

CFG = dict(updates=768, latents=8, lr_tables=2e-5, lr_norms=2e-5, lr_log_scale=1e-4, warmup_frac=0.05, clip=1.0, ema=0.99, anchor_share=0.10,
           holdout=40, eval_every=64, eval_latents=4, executed=8, valid_dims=7)
HOLD = dict(libero_spatial=12, libero_object=10, libero_goal=10, libero_10=8)


def split(suites_of):
    rng = np.random.default_rng(np.random.SeedSequence([20261104, 0]))
    hold = []
    for s, k in HOLD.items():
        idx = [i for i, x in enumerate(suites_of) if x == s]
        hold += sorted(rng.choice(idx, size=k, replace=False).tolist())
    hold = sorted(hold)
    return [i for i in range(256) if i not in set(hold)], hold


def lr_factor(update, total, warm):
    if update < warm:
        return (update + 1) / warm
    return 0.5 * (1.0 + math.cos(math.pi * (update - warm) / max(total - warm, 1)))


def heldout_loss(router, taps_t, taps_s, wo, observations, suites_of, hold, cfg, beta, latents):
    seed = lambda i, k: int(np.random.SeedSequence([7, i, k]).generate_state(1)[0])
    lr_, lv_ = [], []
    with torch.no_grad():
        for i in hold:
            s = suites_of[i]; tm, sm, policy = router(s)
            obs = dict(observations[i][1]); obs['_vla_episode_seed'] = i
            for k in range(latents):
                a, b = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, seed(i, k), cfg, wo[s])
                lr_.append(float(a)); lv_.append(float(b))
    L_r, L_v = float(np.mean(lr_)), float(np.mean(lv_))
    return dict(L_readout=L_r, L_vel=L_v, total=L_r + beta * L_v)


def run(device='cuda', smoke=False, variant='v0s_rht'):
    R.set_variant(variant)
    ROW = R.V['row']                                                                   # v0s_rht: tcq_raqft_rht_s (V0'); v0s_c / v0s_d1 / v0s_d2: the same recipe on arm C / D1 / D2 (user 2026-11-07)
    cfg = dict(ROW_A_CONFIG, **CFG, mode=ROW, row=ROW, base=R.BASE_ROW)
    directory = RUNS / (R.KEY + '_smoke' if smoke else R.KEY); directory.mkdir(parents=True, exist_ok=True)
    if not smoke and (OUT / 'fit' / R.KEY / 'complete.json').exists():
        print('already complete', R.KEY); return
    torch.manual_seed(0)
    suites_of = json.load(open(RUNS / 'calibration_suites.json'))['observation_suite']
    from calibration_data import load_observations
    _, observations = load_observations(); assert len(observations) == 256
    train, hold = split(suites_of)
    if smoke:
        hold = [hold[0], hold[-1]]
    router, layers, norms, counts, init_gap, probe = R.build_models(device, observations, suites_of)
    assert all(v == 0.0 for v in init_gap.values()), init_gap
    for m in layers.values():
        m.to_log_scale()
    books = [m.books for m in layers.values()]; logs = [m.log_scale for m in layers.values()]; norm_params = list(norms.values())
    trainable = books + logs + norm_params
    r0 = {k: float(m.books.detach().square().mean().sqrt()) for k, m in layers.items()}
    # FP weights of every trainable layer (teachers keep the original Linear modules; non-first heads live on the CPU)
    fp_w, fp_n = {}, {}
    for k in layers:
        s, name = (k.split('::', 1) if '::' in k else (SUITES[0], k))
        w = router.teachers[s].policy.model.get_submodule(name).weight.detach()
        fp_w[k] = w.to('cpu', torch.bfloat16).pin_memory() if w.device.type == 'cpu' else w.to(torch.bfloat16)
        fp_n[k] = float(w.float().square().sum())
    active = {s: [k for k in layers if '::' not in k or k.startswith(s + '::')] for s in SUITES}
    bb_books = [m.books for k, m in layers.items() if '::' not in k]
    dit_books = [m.books for k, m in layers.items() if '::action_head.model.transformer_blocks.' in k]
    opt = torch.optim.Adam([dict(params=books, lr=cfg['lr_tables']), dict(params=logs, lr=cfg['lr_log_scale']), dict(params=norm_params, lr=cfg['lr_norms'])])
    base_lr = [g['lr'] for g in opt.param_groups]
    taps_t = {s: ReaderTaps(router.teachers[s].policy.model.action_head) for s in SUITES}
    taps_s = {s: ReaderTaps(router.students[s].policy.model.action_head) for s in SUITES}
    wo = {s: {i: router.teachers[s].policy.model.action_head.model.transformer_blocks[i].attn1.to_out[0].weight.detach().float().to(device) for i in taps_t[s].cross} for s in SUITES}
    latent_seed = lambda idx, number: int(np.random.SeedSequence([7, idx, number]).generate_state(1)[0])
    total = len(SUITES) if smoke else cfg['updates']
    warm = max(1, round(cfg['warmup_frac'] * total))
    order = [next(i for i in train if suites_of[i] == s) for s in SUITES] if smoke else [train[u % len(train)] for u in range(total)]
    n_lat = 1 if smoke else cfg['latents']
    ev_lat = 1 if smoke else cfg['eval_latents']
    gpu = torch.cuda.get_device_name()

    def anchor_raw(s):
        a = 0.0
        with torch.no_grad():
            for k in active[s]:
                a += float((layers[k].weight() - fp_w[k].to(device, non_blocking=True).float()).square().sum()) / fp_n[k]
        return a

    def anchor_backward(s, mu):
        a = 0.0
        for k in active[s]:
            w = layers[k].weight()
            l = mu * (w - fp_w[k].to(device, non_blocking=True).float()).square().sum() / fp_n[k]
            l.backward(); a += float(l.detach())
        return a

    def gauge_fix(params_books=None, params_logs=None):
        with torch.no_grad():
            for (k, m), b, lg in zip(layers.items(), params_books or books, params_logs or logs):
                c = r0[k] / float(b.square().mean().sqrt().clamp_min(1e-30))
                b.mul_(c); lg.sub_(math.log(c))

    ema = [p.detach().clone() for p in trainable]

    def swap(src):
        bak = [p.detach().clone() for p in trainable]
        with torch.no_grad():
            for p, v in zip(trainable, src): p.copy_(v)
        return bak

    def evaluate(beta):
        bak = swap(ema)
        res = heldout_loss(router, taps_t, taps_s, wo, observations, suites_of, hold, cfg, beta, ev_lat)
        swap(bak)
        return res

    ckpt = directory / 'ckpt.pt'
    if ckpt.exists() and not smoke:
        st = torch.load(ckpt, map_location=device, weights_only=False)
        with torch.no_grad():
            for p, v in zip(trainable, st['trainable']): p.copy_(v)
            for e, v in zip(ema, st['ema']): e.copy_(v)
        opt.load_state_dict(st['opt'])
        history, curve, skipped, beta, mu, start, best = st['history'], st['curve'], st['skipped'], st['beta'], st['mu'], st['update'], st['best']
        contract = json.loads((directory / 'manifest.json').read_text())
        print(json.dumps(dict(resumed_at=start, gpu=gpu)), flush=True)
    else:
        idx = order[0]; s = suites_of[idx]; tm, sm, policy = router(s)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        with weight_cache(layers.values()):
            l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, 0), cfg, wo[s])
            opt.zero_grad(set_to_none=True); l_read.backward(retain_graph=True)
            g_bb_read = grad_norm(bb_books)
            opt.zero_grad(set_to_none=True); l_vel.backward()
            g_dit_vel = grad_norm(dit_books)
            opt.zero_grad(set_to_none=True)
        beta = g_bb_read / g_dit_vel
        # μ: initial total loss L0 (first batch, all latents, B2 loss) and initial anchor A0 on the same routed model
        L0 = 0.0
        with torch.no_grad():
            for number in range(n_lat):
                a, b = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, number), cfg, wo[s])
                L0 += float(a + beta * b) / n_lat
        A0 = anchor_raw(s)
        mu = L0 / (9.0 * max(A0, 1e-30))
        # gate (as V0): non-zero table / log-scale gradients in every routed layer, DiT gradient > 1e-4
        opt.zero_grad(set_to_none=True)
        with weight_cache(layers.values()):
            a, b = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, 0), cfg, wo[s])
            (a + beta * b).backward()
        zero_books = [k for k in active[s] if layers[k].books.grad is None or not bool((layers[k].books.grad != 0).any())]
        zero_logs = [k for k in active[s] if layers[k].log_scale.grad is None or not bool((layers[k].log_scale.grad != 0).any())]
        g_dit = grad_norm(dit_books); opt.zero_grad(set_to_none=True)
        gate = dict(dit_table_grad_norm=g_dit, zero_table_gradient=zero_books, zero_log_scale_gradient=zero_logs, passed=bool(g_dit > 1e-4 and not zero_books and not zero_logs))
        calibration = dict(beta=beta, L0=L0, A0=A0, mu=mu, rule='mu * A0 = 10 % of (L0 + mu * A0)', anchor_scope='trainable layers of the routed model (backbone + suite head)')
        dump(directory / 'calibration.json', dict(calibration=calibration, gate=dict(gate, zero_table_gradient=len(zero_books), zero_log_scale_gradient=len(zero_logs)), gpu=gpu))
        print(json.dumps(dict(calibration=calibration, gate=dict(gate, zero_table_gradient=len(zero_books), zero_log_scale_gradient=len(zero_logs)))), flush=True)
        if not gate['passed']:
            dump(directory / 'decision.json', dict(status='aborted', reason='gate failed', gate=gate)); raise SystemExit(3)
        contract = dict(config=cfg, trainable_layers=counts, train=len(train), holdout=hold, calibration=calibration, gpu=gpu, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        dump(directory / 'manifest.json', contract)
        history, skipped, start = [], [], 0
        ev = evaluate(beta); curve = [dict(update=0, **ev)]
        best = dict(update=0, L_readout=ev['L_readout'], ema=[e.detach().cpu().clone() for e in ema])
        print(json.dumps(dict(heldout=curve[-1])), flush=True)
    t_start = time.time()
    for update in range(start, total):
        for g, lr in zip(opt.param_groups, base_lr):
            g['lr'] = lr * lr_factor(update, total, warm)
        idx = order[update]; s = suites_of[idx]; tm, sm, policy = router(s)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        opt.zero_grad(set_to_none=True)
        value, vread, vvel, t0 = 0.0, 0.0, 0.0, time.time()
        for number in range(n_lat):
            with weight_cache(layers.values()):
                l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, number), cfg, wo[s])
                loss = l_read + beta * l_vel
                if not torch.isfinite(loss.detach()):
                    skipped.append(dict(update=update + 1, observation=idx, suite=s, latent=number, where='loss')); continue
                (loss / n_lat).backward()
                value += float(loss.detach()) / n_lat; vread += float(l_read.detach()) / n_lat; vvel += float(l_vel.detach()) / n_lat
        anc = anchor_backward(s, mu)
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
        if not np.isfinite(gn):
            skipped.append(dict(update=update + 1, observation=idx, suite=s, where='gradient', grad_norm=gn))
            print(json.dumps(dict(nonfinite=skipped[-1])), flush=True); opt.zero_grad(set_to_none=True)
        else:
            opt.step(); gauge_fix()
            with torch.no_grad():
                for e, p in zip(ema, trainable): e.mul_(cfg['ema']).add_(p.detach(), alpha=1 - cfg['ema'])
            assert all(torch.isfinite(p).all() for p in trainable), f'non-finite parameter after update {update + 1}'
        assert len(skipped) <= MAX_SKIPPED, f'{len(skipped)} non-finite events (> {MAX_SKIPPED}): stop and report'
        history.append(dict(update=update + 1, observation=idx, suite=s, loss=value + anc, L_readout=vread, L_vel=vvel, anchor=anc, grad_norm=gn, lr_factor=lr_factor(update, total, warm), seconds=time.time() - t0))
        if smoke or (update + 1) % 16 == 0:
            dump(directory / 'progress.json', dict(status='training', completed=update + 1, total=total, loss=value + anc, elapsed=time.time() - t_start))
            print(json.dumps(history[-1]), flush=True)
        if (update + 1) % cfg['eval_every'] == 0 or update + 1 == total:
            ev = evaluate(beta); curve.append(dict(update=update + 1, **ev)); print(json.dumps(dict(heldout=curve[-1])), flush=True)
            if ev['L_readout'] < best['L_readout']:
                best = dict(update=update + 1, L_readout=ev['L_readout'], ema=[e.detach().cpu().clone() for e in ema])
        if not smoke and (update + 1) % CKPT_EVERY == 0 and update + 1 < total:
            torch.save(dict(update=update + 1, trainable=[p.detach().cpu() for p in trainable], ema=[e.detach().cpu() for e in ema], opt=opt.state_dict(), history=history, curve=curve,
                            skipped=skipped, beta=beta, mu=mu, best=best), ckpt.with_suffix('.partial'))
            ckpt.with_suffix('.partial').replace(ckpt)
    dump(directory / 'curve.json', curve)
    if smoke:
        dump(directory / 'smoke.json', dict(contract=contract, history=history, curve=curve, skipped=skipped, peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 2**30))
        print(json.dumps(dict(smoke='ok', updates=len(history), curve=curve, peak_gb=torch.cuda.max_memory_allocated() / 2**30)), flush=True); return
    with torch.no_grad():                                                              # export the best EMA snapshot, gauge-fixed
        for p, v in zip(trainable, best['ema']): p.copy_(v.to(device))
    gauge_fix()
    fine_tune = dict(contract=contract, updates=len(history), nonfinite_events=skipped, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], elapsed=time.time() - t_start,
                     beta=beta, mu=mu, exported=dict(update=best['update'], heldout_L_readout=best['L_readout'], weights='EMA 0.99, gauge-fixed'), curve=curve)
    R.export_fit(layers, norms, fine_tune)
    dump(directory / 'history.json', history)
    ckpt.unlink(missing_ok=True)
    print(json.dumps(dict(done=ROW, exported_update=best['update'], heldout_L_readout=best['L_readout'], skipped=len(skipped), hours=(time.time() - t_start) / 3600)), flush=True)


def stage_holdout(rows, device='cuda', variant='v0s_rht'):
    """Held-out L_readout / L_vel (same 40 observations, 4 latents, same seeds) of assembled rows, β from the V0' calibration."""
    from groot_bridge_server import Bridge
    from groot_rowb import Router
    from calibration_data import load_observations
    cfg = dict(ROW_A_CONFIG, **CFG)
    suites_of = json.load(open(RUNS / 'calibration_suites.json'))['observation_suite']
    _, observations = load_observations()
    _, hold = split(suites_of)
    beta = json.loads((RUNS / 'backbone_tcq_raqft_rht_s' / 'calibration.json').read_text())['calibration']['beta']
    out = RUNS / R.VARIANTS[variant]['key'] / 'holdout_rows.json'        # β stays the V0' calibration's (same metric for every row)
    res = json.loads(out.read_text()) if out.exists() else {}
    for row in rows:
        if row in res:
            continue
        per = {}
        for s in SUITES:
            idx = [i for i in hold if suites_of[i] == s]
            t = Bridge(s, device=device); st = Bridge(s, device=device, model_root=str(MODELS / row))
            tm, sm, policy = t.policy.model, st.policy.model, st.policy
            taps_t, taps_s = ReaderTaps(tm.action_head), ReaderTaps(sm.action_head)
            wo = {i: tm.action_head.model.transformer_blocks[i].attn1.to_out[0].weight.detach().float().to(device) for i in taps_t.cross}
            router = lambda _s: (tm, sm, policy)
            per[s] = heldout_loss(router, {s: taps_t}, {s: taps_s}, {s: wo}, observations, suites_of, idx, cfg, beta, cfg['eval_latents'])
            per[s]['observations'] = len(idx)
            del t, st; torch.cuda.empty_cache()
        n = sum(v['observations'] for v in per.values())
        res[row] = dict(L_readout=sum(v['L_readout'] * v['observations'] for v in per.values()) / n, L_vel=sum(v['L_vel'] * v['observations'] for v in per.values()) / n, per_suite=per, gpu=torch.cuda.get_device_name())
        dump(out, res); print(json.dumps({row: res[row]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('train', 'holdout'), default='train')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--rows', default='tcq_w2_rht:tcq_raqft_rht:tcq_raqft_rht_s')
    p.add_argument('--variant', choices=('v0s_rht', 'v0s_c', 'v0s_d1', 'v0s_d2'), default='v0s_rht')
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    if a.stage == 'train':
        run(smoke=a.smoke, variant=a.variant)
    else:
        stage_holdout(a.rows.split(':'), variant=a.variant)
