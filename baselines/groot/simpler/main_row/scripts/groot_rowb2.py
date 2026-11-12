"""GR00T row B2 (user, 2026-10-27 §1; last GR00T attempt): row B's trainable set with the loss path corrected so the DiT codebooks are
actually trained.
  Δo_student: readouts taken inside the STUDENT's own 2-bit head (student queries, student cross-attention over the student's quantized
  prefix tokens, before W_O), compared with the FP head's readouts on the FP trajectory (same observation, same noise). The row-A Δo
  (FP head reader with student queries) is not used.
  Loss = L_readout(Δo_student, W_O metric of the FP head) + β · L_vel, with L_vel the velocity MSE of the student head against the FP
  head over the same quantized prefix (student trajectory), executed window (8 steps) and valid dims (7) only, averaged over the 4 flow
  steps. β is set once at step 0 so that the DiT-codebook gradient norm from β·L_vel equals the backbone-codebook gradient norm from
  L_readout (ratio 1, within the required 10×); β and both norms are recorded. No further tuning.
  Gate before the first optimizer step: DiT codebook gradient norm of the full loss on the first batch > 1e-4, else abort (no manifest).
  Trainable set = row B's (backbone + per-suite vl_self_attention + per-suite DiT codebooks and row scales, FP16 row-scale straight-through);
  256 observations routed to their suites, 8 latents, 768 updates, final iterate; checkpoint every 8 updates (resumable; was 64 until 2026-10-27).
Exports backbone_rowb2 + head_rowb2_<suite>; row raqft_rowb2."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_ptq import OUT, dump
from groot_raqft import CONFIG as ROW_A_CONFIG, prepare, backbone_tokens, RUNS
from groot_rowb import SUITES, build_models, export_fit, weight_cache
from groot_amraq import ReaderTaps, fp_trajectory, head_velocity, _State, policy_obs

MAX_SKIPPED = 38
CKPT_EVERY = 8          # 2026-10-27: was 64; job 229308 lost 48 updates to a killable preemption (save is atomic, ~64 MB)
KEY = 'backbone_rowb2'


def encode_state(head, action_inputs):
    state = action_inputs.state
    dtype = next(head.state_encoder.parameters()).dtype
    return _State(head.state_encoder(state.view(state.shape[0], 1, -1).to(dtype), action_inputs.embodiment_id))


def losses(tm, sm, taps_t, taps_s, policy, obs, seed, cfg, wo):
    """One latent. Returns (L_readout, L_vel) with graphs through the student backbone and the student head."""
    fp = fp_trajectory(tm, taps_t, policy, obs, seed)                                  # FP head on its own trajectory: o_fp per step/block
    torch.manual_seed(seed)
    bi, ai = prepare(policy, policy_obs(policy, obs))
    q_tokens, image_mask, bb_mask = backbone_tokens(sm, bi)                           # grad -> backbone + vl_self_attention codebooks
    head_s, head_t = sm.action_head, tm.action_head
    feats_s = encode_state(head_s, ai)
    with torch.no_grad():
        feats_t = encode_state(head_t, ai)
    x = fp['noise'].clone()
    l_read, l_vel, n = 0.0, 0.0, 0
    for t, st in enumerate(fp['steps']):
        taps_s.captured = {}
        v_s = head_velocity(head_s, feats_s, q_tokens, image_mask, bb_mask, ai.embodiment_id, x, t)   # grad -> DiT codebooks (and the backbone through q_tokens)
        o_s = dict(taps_s.captured)
        with torch.no_grad():
            v_fq = head_velocity(head_t, feats_t, q_tokens.detach(), image_mask, bb_mask, ai.embodiment_id, x, t)   # FP head, same quantized prefix, same x_t
        for i, o_fp in st['readouts'].items():
            delta = (o_s[i].float() - o_fp.float())                                      # (1, tokens, inner), pre-W_O, student-head readout vs FP-head readout
            l_read = l_read + (delta @ wo[i].t()).square().sum(dim=(1, 2)).mean(); n += 1
        l_vel = l_vel + (v_s[:, :cfg['executed'], :cfg['valid_dims']].float() - v_fq[:, :cfg['executed'], :cfg['valid_dims']].float()).square().mean()
        x = (x + (1.0 / head_s.num_inference_timesteps) * v_s).detach()        # Euler step as in get_action; trajectory fixed between steps
    return l_read / max(n, 1), l_vel / len(fp['steps'])


def grad_norm(params):
    return float(torch.sqrt(sum(p.grad.double().square().sum() for p in params if p.grad is not None)))


def run(device='cuda'):
    cfg = dict(ROW_A_CONFIG, mode='rowb2', row='raqft_rowb2', executed=8, valid_dims=7, trainable='prefix (backbone + per-suite vl_self_attention) + per-suite DiT', routing='per-suite heads',
               loss='L_readout(student-head readouts vs FP-head readouts, W_O metric) + beta * L_vel(student head vs FP head on the quantized prefix, executed window, valid dims)')
    directory = RUNS / KEY; directory.mkdir(parents=True, exist_ok=True)
    if (OUT / 'fit' / KEY / 'complete.json').exists():
        print('already complete', KEY); return
    torch.manual_seed(0)
    suites_of = json.load(open(RUNS / 'calibration_suites.json'))['observation_suite']
    from calibration_data import load_observations
    _, observations = load_observations(); assert len(observations) == 256 and len(suites_of) == 256
    router, trainable_layers, counts, n_ste, init_gap, probe = build_models(device, observations, suites_of)
    assert all(v == 0.0 for v in init_gap.values()), init_gap
    books = [m.books for m in trainable_layers.values()]; scales = [m.scale for m in trainable_layers.values()]
    trainable = books + scales
    bb_books = [m.books for k, m in trainable_layers.items() if '::' not in k]
    dit_books = [m.books for k, m in trainable_layers.items() if '::action_head.model.transformer_blocks.' in k]
    opt = torch.optim.Adam([dict(params=books, lr=cfg['lr']), dict(params=scales, lr=cfg['lr_scale'])])
    taps_t = {s: ReaderTaps(router.teachers[s].policy.model.action_head) for s in SUITES}
    taps_s = {s: ReaderTaps(router.students[s].policy.model.action_head) for s in SUITES}
    wo = {s: {i: router.teachers[s].policy.model.action_head.model.transformer_blocks[i].attn1.to_out[0].weight.detach().float().to(device) for i in taps_t[s].cross} for s in SUITES}   # teachers live on the host between suites (job 229180 failed here: W_O on cpu)
    latent_seed = lambda idx, number: int(np.random.SeedSequence([7, idx, number]).generate_state(1)[0])
    ckpt = directory / 'ckpt.pt'
    gpu = torch.cuda.get_device_name() if device == 'cuda' else 'cpu'
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        for p, v in zip(trainable, state['trainable']):
            p.data.copy_(v)
        opt.load_state_dict(state['opt']); history, skipped, beta, start = state['history'], state['skipped'], state['beta'], state['update']
        contract = json.loads((directory / 'manifest.json').read_text())
        print(json.dumps(dict(resumed_at=start, gpu=gpu)), flush=True)
    else:
        # ---- β at step 0 (first observation, first latent): backbone-codebook gradient norm of L_readout vs DiT-codebook gradient norm of L_vel
        idx = 0; s = suites_of[idx]; tm, sm, policy = router(s)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        with weight_cache(trainable_layers.values()):
            l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, 0), cfg, wo[s])
            opt.zero_grad(set_to_none=True); l_read.backward(retain_graph=True)
            g_bb_read, g_dit_read = grad_norm(bb_books), grad_norm(dit_books)
            opt.zero_grad(set_to_none=True); l_vel.backward()
            g_bb_vel, g_dit_vel = grad_norm(bb_books), grad_norm(dit_books)
            opt.zero_grad(set_to_none=True)
        assert g_dit_vel > 0, 'L_vel gives the DiT codebooks no gradient'
        beta = g_bb_read / g_dit_vel                                    # DiT gradient from beta*L_vel == backbone gradient from L_readout
        calibration = dict(L_readout=float(l_read), L_vel=float(l_vel), grad_norm_backbone_codebooks_from_L_readout=g_bb_read, grad_norm_dit_codebooks_from_L_readout=g_dit_read,
                           grad_norm_backbone_codebooks_from_L_vel=g_bb_vel, grad_norm_dit_codebooks_from_L_vel=g_dit_vel, beta=beta, dit_from_beta_L_vel=beta * g_dit_vel,
                           rule='beta = ||g_bb(L_readout)|| / ||g_DiT(L_vel)||, set once at step 0 (user 2026-10-27: within 10x), no further tuning')
        # ---- gate: DiT codebook gradient norm of the full loss on the first batch (8 latents) > 1e-4
        opt.zero_grad(set_to_none=True)
        for number in range(cfg['latents']):
            with weight_cache(trainable_layers.values()):
                l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, number), cfg, wo[s])
                ((l_read + beta * l_vel) / cfg['latents']).backward()
        g_dit_batch, g_bb_batch = grad_norm(dit_books), grad_norm(bb_books)
        opt.zero_grad(set_to_none=True)
        gate = dict(dit_codebook_grad_norm_first_batch=g_dit_batch, backbone_codebook_grad_norm_first_batch=g_bb_batch, threshold=1e-4, passed=bool(g_dit_batch > 1e-4))
        dump(directory / 'calibration.json', dict(calibration=calibration, gate=gate, gpu=gpu))
        print(json.dumps(dict(calibration=calibration, gate=gate)), flush=True)
        if not gate['passed']:
            dump(directory / 'decision.json', dict(status='aborted', reason='DiT codebook gradient norm on the first batch <= 1e-4', gate=gate, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
            raise SystemExit(3)
        contract = dict(config=dict(cfg, beta=beta), trainable_layers=counts, row_scale_ste_layers=n_ste, init_max_abs_action_gap=init_gap, observations_per_suite={s_: suites_of.count(s_) for s_ in SUITES},
                        calibration=calibration, gate=gate, device=device, gpu=gpu, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        dump(directory / 'manifest.json', contract)
        history, skipped, start = [], [], 0
    # ---- training (row B's loop with the B2 loss)
    t_start = time.time()
    order = list(range(256)) * cfg['epochs']
    for update in range(start, len(order)):
        idx = order[update]; s = suites_of[idx]
        tm, sm, policy = router(s)
        obs = dict(observations[idx][1]); obs['_vla_episode_seed'] = idx
        opt.zero_grad(set_to_none=True)
        value, vread, vvel, t0 = 0.0, 0.0, 0.0, time.time()
        for number in range(cfg['latents']):
            with weight_cache(trainable_layers.values()):
                l_read, l_vel = losses(tm, sm, taps_t[s], taps_s[s], policy, obs, latent_seed(idx, number), cfg, wo[s])
                loss = l_read + beta * l_vel
                if not torch.isfinite(loss.detach()):
                    skipped.append(dict(update=update + 1, observation=idx, suite=s, latent=number, where='loss'))
                    print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
                    continue
                (loss / cfg['latents']).backward()
                value += float(loss.detach()) / cfg['latents']; vread += float(l_read.detach()) / cfg['latents']; vvel += float(l_vel.detach()) / cfg['latents']
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']))
        g_dit = grad_norm(dit_books)
        if not np.isfinite(gn):
            skipped.append(dict(update=update + 1, observation=idx, suite=s, where='gradient', grad_norm=gn))
            print(json.dumps(dict(nonfinite=skipped[-1])), flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            assert all(torch.isfinite(p).all() for p in trainable), f'non-finite parameter after update {update + 1}'
        assert len(skipped) <= MAX_SKIPPED, f'{len(skipped)} non-finite events (> {MAX_SKIPPED}): stop and report'
        history.append(dict(update=update + 1, observation=idx, suite=s, loss=value, L_readout=vread, L_vel=vvel, grad_norm=gn, dit_codebook_grad_norm_after_clip=g_dit, seconds=time.time() - t0))
        if (update + 1) % 16 == 0:
            dump(directory / 'progress.json', dict(status='training', completed=update + 1, total=len(order), loss=value, elapsed=time.time() - t_start))
            print(json.dumps(history[-1]), flush=True)
        if (update + 1) % CKPT_EVERY == 0 and update + 1 < len(order):
            torch.save(dict(update=update + 1, trainable=[p.detach().cpu() for p in trainable], opt=opt.state_dict(), history=history, skipped=skipped, beta=beta), ckpt.with_suffix('.partial'))
            ckpt.with_suffix('.partial').replace(ckpt)
    assert all(torch.isfinite(p).all() for p in trainable), 'non-finite masters at export'
    fine_tune = dict(contract=contract, updates=len(history), nonfinite_events=skipped, first_loss=history[0]['loss'], final_loss=history[-1]['loss'], elapsed=time.time() - t_start, beta=beta)
    export_fit(trainable_layers, KEY, 'head_rowb2', fine_tune)
    dump(directory / 'history.json', history)
    ckpt.unlink(missing_ok=True)
    print(json.dumps(dict(done='raqft_rowb2', first_loss=history[0]['loss'], final_loss=history[-1]['loss'], skipped=len(skipped), hours=(time.time() - t_start) / 3600)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    a = p.parse_args()
    if a.device == 'cpu':
        sys.modules['flash_attn'] = None
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    run(device=a.device)
