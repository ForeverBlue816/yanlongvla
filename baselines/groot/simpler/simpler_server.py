"""GR00T policy server for SimplerEnv evaluation = the official gr00t/eval/run_gr00t_server.py (Gr00tPolicy + Gr00tSimPolicyWrapper, ZMQ
PolicyServer) plus one addition: `reset(options={'noise_seed': s})` seeds torch before the episode, so the flow-matching noise of every
episode is fixed by the client (simpler_eval.py) and identical across policies. Runs in the GR00T environment.
  python simpler_server.py --model-path <checkpoint or assembled 2-bit dir> --port 5555 [--embodiment-tag SIMPLER_ENV_WIDOWX] [--device cuda]
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--embodiment-tag', default='SIMPLER_ENV_WIDOWX')
    p.add_argument('--device', default='cuda')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5555)
    a = p.parse_args()
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    class SeededSimPolicy(Gr00tSimPolicyWrapper):
        def reset(self, options=None):
            if options and options.get('noise_seed') is not None:
                torch.manual_seed(int(options['noise_seed']))
                torch.cuda.manual_seed_all(int(options['noise_seed']))
            return super().reset(options)

    policy = SeededSimPolicy(Gr00tPolicy(embodiment_tag=a.embodiment_tag, model_path=a.model_path, device=a.device, strict=True))
    print(f'serving {a.model_path} ({a.embodiment_tag}) on {a.host}:{a.port}', flush=True)
    with PolicyServer(policy=policy, host=a.host, port=a.port) as server:
        server.run()


if __name__ == '__main__':
    main()
