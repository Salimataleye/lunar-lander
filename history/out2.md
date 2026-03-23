# Iteration 2 Results

## Summary

Iteration 2 improved the baseline from iteration 1 by changing checkpoint selection from a mean-only evaluation rule to a more robustness-aware rule and then fine-tuning from `best/iteration1.zip`.

Relative to iteration 1, iteration 2 improved both training-side and held-out metrics:

- Final avg reward over last 100 training episodes: `223.81 -> 251.39`
- Final avg episode length over last 100 training episodes: `217.44 -> 230.64`
- 20-episode evaluation mean reward: `262.23 -> 274.22`
- 20-episode evaluation std: `55.07 -> 21.53`
- 50-episode verification mean reward: `258.80 -> 265.28`
- 50-episode verification std: `57.13 -> 41.83`
- 50-episode mean episode length: `260.18 -> 242.50`

The saved artifact for this iteration is `best/iteration2.zip`.

## Code / Workflow Changes

- Replaced the default `EvalCallback` checkpoint selection path with a custom robustness-aware evaluation callback in `agent_class.py`.
- The new callback records:
  - evaluation mean reward,
  - evaluation standard deviation,
  - a checkpoint score defined as `mean_reward - robustness_penalty * std_reward`.
- Exposed `robustness_penalty` through `train_test.py` and `train_agent.py`.

## Fine-Tuning Start Point

The main iteration started from:

- Initial checkpoint: `best/iteration1.zip`
- Starting timestep of that checkpoint: `360000`

## Pilots Run

### Pilot 1: moderate fine-tune

Configuration:

- start from `best/iteration1.zip`
- `learning_rate=1e-4`
- `ent_coef=0.005`
- `clip_range=0.15`
- `target_kl=0.02`
- `n_epochs=10`
- `robustness_penalty=0.25`
- `eval_episodes=30`

Result:

- Best eval mean reward: `274.56`
- Best eval std: `20.08`
- Best checkpoint score: `269.54`
- 20-episode evaluation mean reward: `269.84`
- Final avg reward over last 100 training episodes: `247.96`

This was a clear upgrade over iteration 1, but not the best candidate.

### Pilot 2: more conservative fine-tune

Configuration:

- start from `best/iteration1.zip`
- `learning_rate=5e-5`
- `ent_coef=0.002`
- `clip_range=0.1`
- `target_kl=0.01`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `eval_episodes=30`

Result:

- Best eval mean reward: `281.48`
- Best eval std: `13.25`
- Best checkpoint score: `278.17`
- 20-episode evaluation mean reward: `274.22`
- Final avg reward over last 100 training episodes: `251.39`

This was the strongest pilot and became the main-run configuration.

## Main Run

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 2 \
  --timesteps 100000 \
  --eval-freq 2500 \
  --eval-episodes 30 \
  --test-episodes 20 \
  --print-every 50 \
  --initial-model-path best/iteration1.zip \
  --n-envs 1 \
  --learning-rate 5e-5 \
  --learning-rate-schedule constant \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 5 \
  --gamma 0.99 \
  --gae-lambda 0.98 \
  --ent-coef 0.002 \
  --clip-range 0.1 \
  --target-kl 0.01 \
  --robustness-penalty 0.25
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: `best/iteration1.zip`
- Starting timestep: `360000`
- Timesteps trained in iteration 2: `100352`
- Total timestep of final model state: `460352`
- Episodes seen during this run: `426`
- Final avg reward over last 100 training episodes: `251.39`
- Final avg episode length over last 100 training episodes: `230.64`
- Best eval mean reward observed during training: `281.48`
- Best eval std at selected checkpoint: `13.25`
- Best checkpoint score: `278.17`
- Best checkpoint absolute timestep: `445000`
- Best checkpoint timestep within this run: `85000`
- Saved model: `best/iteration2.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `274.22`
- Reward std: `21.53`
- Mean episode length: `233.00`

Additional verification run over 50 episodes:

- Mean reward: `265.28`
- Reward std: `41.83`
- Mean episode length: `242.50`

## Comparison To Iteration 1

What improved:

- Final avg reward over last 100 training episodes: `223.81 -> 251.39`
- 20-episode evaluation mean reward: `262.23 -> 274.22`
- 20-episode evaluation std: `55.07 -> 21.53`
- 50-episode verification mean reward: `258.80 -> 265.28`
- 50-episode verification std: `57.13 -> 41.83`
- 50-episode mean episode length: `260.18 -> 242.50`

What did not improve:

- Final avg episode length over the last 100 training episodes increased from `217.44` to `230.64`
- The saved checkpoint still exhibits a small number of bad held-out episodes

## Conclusion

Iteration 2 is a meaningful upgrade over iteration 1. The main win came from robustness-aware checkpoint selection during fine-tuning, which substantially reduced evaluation variance and improved held-out reward without giving up the strong PPO training signal.
