import argparse
import json
import shutil
import tempfile
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from agent_class import LunarLanderAgent, constant_schedule, linear_schedule
from env_setup import ENV_ID, make_env


def make_monitored_env(render=False, seed=None):
    return Monitor(make_env(render=render, seed=seed))


def make_training_env(n_envs=1, seed=42):
    return make_vec_env(ENV_ID, n_envs=n_envs, seed=seed, wrapper_class=Monitor)


def train(
    iteration=1,
    total_timesteps=500000,
    seed=42,
    eval_freq=10000,
    eval_episodes=20,
    test_episodes=20,
    print_every=20,
    progress_bar=False,
    best_model_path=None,
    n_envs=1,
    learning_rate=3e-4,
    learning_rate_schedule='constant',
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef=0.01,
    clip_range=0.2,
    target_kl=None,
    initial_model_path=None,
    robustness_penalty=0.25,
    length_penalty=0.0,
    selection_metric='robust_reward',
    evaluate_initial_model=False,
):
    train_env = make_training_env(n_envs=n_envs, seed=seed)
    eval_env = make_monitored_env(seed=seed + 1000)
    if learning_rate_schedule == 'linear':
        resolved_learning_rate = linear_schedule(learning_rate)
    else:
        resolved_learning_rate = constant_schedule(learning_rate)

    hyperparameters = {
        'learning_rate': resolved_learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'ent_coef': ent_coef,
        'clip_range': clip_range,
        'target_kl': target_kl,
    }

    if initial_model_path is None:
        agent = LunarLanderAgent(
            seed=seed,
            **hyperparameters,
        )
    else:
        agent = LunarLanderAgent.load(
            initial_model_path,
            env=train_env,
            seed=seed,
            custom_objects=hyperparameters,
        )

    with tempfile.TemporaryDirectory(prefix=f'lunar_lander_iter_{iteration}_') as tmp_dir:
        training_summary = agent.train(
            train_env=train_env,
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            best_model_dir=tmp_dir,
            print_every=print_every,
            progress_bar=progress_bar,
            robustness_penalty=robustness_penalty,
            length_penalty=length_penalty,
            selection_metric=selection_metric,
            evaluate_initial_model=evaluate_initial_model,
        )

        if best_model_path is None:
            best_model_path = Path('best') / f'iteration{iteration}.zip'
        else:
            best_model_path = Path(best_model_path)

        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        callback_best_model_path = Path(training_summary['best_model_path'])
        if callback_best_model_path.exists():
            shutil.copyfile(callback_best_model_path, best_model_path)
        else:
            agent.save(best_model_path)

    train_env.close()
    eval_env.close()

    best_agent = LunarLanderAgent.load(best_model_path, seed=seed)
    evaluation_summary = best_agent.evaluate(num_episodes=test_episodes, render=False)

    summary = {
        'iteration': iteration,
        'environment_id': ENV_ID,
        'initial_model_path': initial_model_path,
        'starting_timesteps': training_summary['starting_timesteps'],
        'timesteps_trained_this_run': training_summary['timesteps_trained_this_run'],
        'total_timesteps': training_summary['total_timesteps'],
        'episodes_seen': training_summary['episodes'],
        'final_train_stats': training_summary['final_train_stats'],
        'best_mean_reward': training_summary['best_mean_reward'],
        'best_std_reward': training_summary['best_std_reward'],
        'best_checkpoint_score': training_summary['best_checkpoint_score'],
        'selection_metric': training_summary['selection_metric'],
        'best_model_timestep': training_summary['best_model_timestep'],
        'best_model_timestep_in_run': training_summary['best_model_timestep_in_run'],
        'best_model_path': str(best_model_path),
        'configuration': {
            'n_envs': n_envs,
            'learning_rate': learning_rate,
            'learning_rate_schedule': learning_rate_schedule,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'ent_coef': ent_coef,
            'clip_range': clip_range,
            'target_kl': target_kl,
            'robustness_penalty': robustness_penalty,
            'length_penalty': length_penalty,
            'selection_metric': selection_metric,
            'evaluate_initial_model': evaluate_initial_model,
        },
        'evaluation': evaluation_summary,
    }
    return best_agent, summary


def test(agent=None, model_path=None, num_episodes=10, render=False, seed=42):
    if agent is None:
        if model_path is None:
            raise RuntimeError('Either an agent instance or a model_path must be provided.')
        agent = LunarLanderAgent.load(model_path, seed=seed)
    return agent.evaluate(num_episodes=num_episodes, render=render)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--test-episodes', type=int, default=20)
    parser.add_argument('--print-every', type=int, default=20)
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--learning-rate-schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--n-steps', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.98)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--target-kl', type=float)
    parser.add_argument('--robustness-penalty', type=float, default=0.25)
    parser.add_argument('--length-penalty', type=float, default=0.0)
    parser.add_argument('--selection-metric', choices=['efficiency', 'lcb_efficiency', 'robust_efficiency', 'robust_reward', 'robust_reward_length'], default='robust_reward')
    parser.add_argument('--initial-model-path', type=str)
    parser.add_argument('--evaluate-initial-model', action='store_true')
    parser.add_argument('--progress-bar', action='store_true')
    parser.add_argument('--render', action='store_true')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    agent, summary = train(
        iteration=args.iteration,
        total_timesteps=args.timesteps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        test_episodes=args.test_episodes,
        print_every=args.print_every,
        progress_bar=args.progress_bar,
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
        robustness_penalty=args.robustness_penalty,
        length_penalty=args.length_penalty,
        selection_metric=args.selection_metric,
        evaluate_initial_model=args.evaluate_initial_model,
    )
    print(json.dumps(summary, indent=2))

    if args.render:
        render_summary = test(
            agent=agent,
            num_episodes=1,
            render=True,
            seed=args.seed,
        )
        print(json.dumps({'render_summary': render_summary}, indent=2))
