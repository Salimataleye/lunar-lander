#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from train_test import train


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='my_agent')
    parser.add_argument('--timesteps', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--test-episodes', type=int, default=20)
    parser.add_argument('--print-every', type=int, default=20)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--learning-rate-schedule', choices=['constant', 'linear'], default='linear')
    parser.add_argument('--n-steps', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.98)
    parser.add_argument('--ent-coef', type=float, default=0.005)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--target-kl', type=float)
    parser.add_argument('--initial-model-path', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--progress-bar', action='store_true')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    output_path = Path(f'{args.f}.zip')

    if output_path.exists() and not args.overwrite:
        raise RuntimeError(
            f'File {output_path} already exists. Restart with --overwrite to replace it.'
        )

    agent, summary = train(
        iteration=0,
        total_timesteps=args.timesteps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        test_episodes=args.test_episodes,
        print_every=args.print_every,
        progress_bar=args.progress_bar,
        best_model_path=output_path,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        initial_model_path=args.initial_model_path,
    )
    print(json.dumps(summary, indent=2))
