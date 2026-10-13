"""Policy server for the evaluation clients. Seeds the flow-matching noise per episode from the `_vla_episode_seed`
field the clients send with the first observation, so every method sees identical noise on identical episodes.

Usage:  python serve.py --checkpoint /path/ckpt --port 8105
"""
import argparse

import torch

from common import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, default=8105)
    p.add_argument('--host', default='127.0.0.1')
    a = p.parse_args()
    from openpi.serving import websocket_policy_server
    policy = load_policy(a.checkpoint)

    class Seeded:
        def infer(self, obs):
            obs = dict(obs)
            seed = obs.pop('_vla_episode_seed', None)
            if seed is not None:
                torch.manual_seed(int(seed))
            return policy.infer(obs)

        def reset(self):
            pass
    websocket_policy_server.WebsocketPolicyServer(Seeded(), host=a.host, port=a.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
