"""GR00T policy server for SimplerEnv evaluation = the official gr00t/eval/run_gr00t_server.py (Gr00tPolicy + Gr00tSimPolicyWrapper, ZMQ
PolicyServer) plus one addition: `reset(options={'noise_seed': s})` seeds torch before the episode, so the flow-matching noise of every
episode is fixed by the client (simpler_eval.py) and identical across policies. Runs in the GR00T environment.
  python simpler_server.py --model-path <checkpoint or assembled 2-bit dir> --port 5555 [--embodiment-tag SIMPLER_ENV_WIDOWX] [--device cuda]
Optional probes (all off by default = the code path is unchanged): --action-scale S (x,y,z of the output action), --action-scale-all S2 (x,y,z,roll,pitch,yaw),
--init-noise-scale S (initial flow-matching noise), --denoise-steps N, --attn-temp T (action-head attention logits / T), --attn-temp-backbone T (Qwen3 text attention logits / T), --dup-batch N (batch of N copies, keep copy 0).
"""
import argparse

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--embodiment-tag', default='SIMPLER_ENV_WIDOWX')
    p.add_argument('--device', default='cuda')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--action-scale', type=float, default=None, help='multiply the denormalised x, y, z deltas by S before they go to the client')
    p.add_argument('--action-scale-all', type=float, default=None, help='multiply the denormalised x, y, z, roll, pitch, yaw (not gripper) by S2; with --action-scale the xyz factors multiply')
    p.add_argument('--init-noise-scale', type=float, default=None, help='multiply the initial Gaussian noise of the flow-matching sampler by S (the RNG stream is unchanged)')
    p.add_argument('--denoise-steps', type=int, default=None, help='override num_inference_timesteps of the action head')
    p.add_argument('--attn-temp', type=float, default=None, help='divide the attention logits of the action head (diffusers Attention modules of action_head.model) by T')
    p.add_argument('--attn-temp-backbone', type=float, default=None, help='divide the attention logits of the backbone language model (Qwen3 self-attention) by T')
    p.add_argument('--dup-batch', type=int, default=None, help='X20: run the model on N identical copies of each observation (server batch N, as with 5 async envs) and return copy 0')
    p.add_argument('--resid-in-scale', type=float, nargs=2, default=None, metavar=('LM', 'VIS'), help='X24: multiply the residual-stream input of the backbone language model (layer 0) by LM and of the vision tower (block 0) by VIS, weights unchanged')
    a = p.parse_args()
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    class SeededSimPolicy(Gr00tSimPolicyWrapper):
        def reset(self, options=None):
            if options and options.get('noise_seed') is not None:
                torch.manual_seed(int(options['noise_seed']))
                torch.cuda.manual_seed_all(int(options['noise_seed']))
            return super().reset(options)

        def _get_action(self, observation, options=None):
            if a.dup_batch and a.dup_batch > 1:   # X20: run the model on N copies of the observation (batch N, as with 5 envs) and keep copy 0
                observation = {k: (np.repeat(v, a.dup_batch, axis=0) if isinstance(v, np.ndarray) else type(v)(list(v) * a.dup_batch) if isinstance(v, (list, tuple)) else v)
                               for k, v in observation.items()}
                action, info = super()._get_action(observation, options)
                action = {k: (v[:1] if isinstance(v, np.ndarray) else v) for k, v in action.items()}
            else:
                action, info = super()._get_action(observation, options)
            if a.action_scale is not None or a.action_scale_all is not None:
                for k, f in (('x', 1), ('y', 1), ('z', 1), ('roll', 0), ('pitch', 0), ('yaw', 0)):   # keys of SIMPLER_ENV_WIDOWX; 'gripper' is never scaled
                    s = (a.action_scale if f and a.action_scale is not None else 1.0) * (a.action_scale_all if a.action_scale_all is not None else 1.0)
                    action[f'action.{k}'] = action[f'action.{k}'] * np.float32(s)
            return action, info

    policy = SeededSimPolicy(Gr00tPolicy(embodiment_tag=a.embodiment_tag, model_path=a.model_path, device=a.device, strict=True))
    head = policy.policy.model.action_head
    if a.denoise_steps is not None:
        head.num_inference_timesteps = a.denoise_steps
    if a.init_noise_scale is not None:   # the sampler draws its start noise with one torch.randn of shape (B, action_horizon, action_dim): scale only that draw
        hits = []
        sampler, shape_of = head.get_action_with_features, (lambda B: (B, head.config.action_horizon, head.action_dim))
        def noisy_sampler(*args, **kw):
            randn = torch.randn
            def scaled_randn(*ra, **rk):
                x = randn(*ra, **rk)
                size = rk.get('size', ra[0] if ra else None)
                if isinstance(size, (tuple, list, torch.Size)) and len(size) == 3 and tuple(size) == shape_of(size[0]):
                    if not hits:
                        print(f'init noise scaled by {a.init_noise_scale}: shape {tuple(size)}', flush=True)   # proof in the server log that the probe acts
                    hits.append(1)
                    return x * a.init_noise_scale
                return x
            torch.randn = scaled_randn
            try:
                return sampler(*args, **kw)
            finally:
                torch.randn = randn
        head.get_action_with_features = noisy_sampler
    for T, root, pick in ((a.attn_temp, head.model, lambda n, m: n.endswith('to_q')),   # the to_q linear of every diffusers Attention (self and cross) of the DiT: q/T divides q.k by T
                          (a.attn_temp_backbone, policy.policy.model.backbone, lambda n, m: n.endswith('q_norm') and 'language_model' in n)):   # Qwen3 has RMSNorm on q after q_proj: scale after it
        if T is not None:
            hooked = [m for n, m in root.named_modules() if pick(n, m)]
            assert hooked, 'no attention module found for the attention temperature'
            for m in hooked:
                m.register_forward_hook(lambda mod, inp, out, T=T: out / T)
            print(f'attention temperature {T}: {len(hooked)} modules', flush=True)
    if a.resid_in_scale is not None:   # X24: a weight shrink of the residual writers (v/o, MLP) by w is, through the norms, the same as scaling the residual input by 1/w
        bb = policy.policy.model.backbone
        named = dict(bb.named_modules())
        lm0 = [m for n, m in named.items() if n.endswith('language_model.layers.0')]
        vis0 = [m for n, m in named.items() if n.endswith('visual.blocks.0')]
        assert len(lm0) == 1 and len(vis0) == 1, (len(lm0), len(vis0))
        def pre(k):
            def hook(mod, args, kwargs):
                if 'hidden_states' in kwargs:
                    kwargs['hidden_states'] = kwargs['hidden_states'] * k
                else:
                    args = (args[0] * k,) + tuple(args[1:])
                return args, kwargs
            return hook
        lm0[0].register_forward_pre_hook(pre(a.resid_in_scale[0]), with_kwargs=True)
        vis0[0].register_forward_pre_hook(pre(a.resid_in_scale[1]), with_kwargs=True)
        print(f'residual input scaled: language model x {a.resid_in_scale[0]}, vision x {a.resid_in_scale[1]}', flush=True)
    opts = {k: v for k, v in vars(a).items() if k in ('action_scale', 'action_scale_all', 'init_noise_scale', 'denoise_steps', 'attn_temp', 'attn_temp_backbone', 'dup_batch', 'resid_in_scale') and v is not None}
    print(f'serving {a.model_path} ({a.embodiment_tag}) on {a.host}:{a.port}' + (f' options {opts}' if opts else ''), flush=True)
    with PolicyServer(policy=policy, host=a.host, port=a.port) as server:
        server.run()


if __name__ == '__main__':
    main()
