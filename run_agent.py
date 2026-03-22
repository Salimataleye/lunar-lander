#!/usr/bin/env python

import argparse
import json

from train_test import test


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='my_agent')
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render', action='store_true')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    summary = test(
        model_path=f'{args.f}.zip',
        num_episodes=args.N,
        render=args.render,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
