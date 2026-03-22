# Iteration 2 Results

## Summary

Iteration 2 kept the PPO approach from iteration 1, added explicit best-checkpoint timestep logging, and tested whether training stability could be improved enough to raise the sustained training score.

The best checkpoint found in iteration 2 did **not** exceed the iteration 1 best evaluation score, but the main run did improve the training metric that `AGENTS.md` emphasizes:

- Final avg reward over the last 100 training episodes improved from `192.93` to `223.81`
- Final avg episode length over the last 100 training episodes improved from `232.47` to `217.44`

The saved artifact for this iteration is `best/iteration2.zip`.

## Code / Workflow Changes

- Added explicit evaluation-history parsing and best-checkpoint timestep reporting in `agent_class.py`.
- Extended `train_test.py` and `train_agent.py` to support:
  - configurable PPO hyperparameters,
  - learning-rate scheduling,
  - configurable number of training environments.

## Pilots Run

### Pilot 1: 4-env PPO with linear LR decay

Configuration:

- `n_envs=4`
- `learning_rate=3e-4`
- linear learning-rate decay
- `n_steps=1024`
- `batch_size=256`
- `n_epochs=10`
- `ent_coef=0.005`

Result:

- Best eval mean reward: `-138.56`
- Final avg reward over last 100 training episodes: `14.90`
- 20-episode evaluation mean reward: `-129.85`

This was clearly worse than iteration 1 and was rejected.

### Pilot 2: 1-env PPO with linear LR decay

Configuration:

- `n_envs=1`
- `learning_rate=3e-4`
- linear learning-rate decay
- `n_steps=1024`
- `batch_size=64`
- `n_epochs=10`
- `ent_coef=0.01`

Result:

- Best eval mean reward: `253.16`
- Best checkpoint timestep: `170000`
- Final avg reward over last 100 training episodes: `175.47`
- 20-episode evaluation mean reward: `210.41`

This was better than the 4-env pilot, but still worse than iteration 1.

## Main Run

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 2 \
  --timesteps 500000 \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --test-episodes 20 \
  --print-every 100 \
  --n-envs 1 \
  --learning-rate 3e-4 \
  --learning-rate-schedule constant \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 10 \
  --gamma 0.99 \
  --gae-lambda 0.98 \
  --ent-coef 0.01
```

Main run summary:

- Environment: `LunarLander-v3`
- Total timesteps trained: `500736`
- Episodes seen during training: `1937`
- Final avg reward over last 100 training episodes: `223.81`
- Final avg episode length over last 100 training episodes: `217.44`
- Best eval mean reward observed during training: `280.59`
- Best checkpoint timestep: `220000`
- Saved model: `best/iteration2.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `275.48`
- Reward std: `21.99`
- Mean episode length: `241.05`

Additional verification run over 50 episodes:

- Mean reward: `276.64`
- Reward std: `21.47`
- Mean episode length: `238.02`

## Comparison To Iteration 1

What improved:

- Final avg reward over last 100 training episodes: `192.93 -> 223.81`
- Final avg episode length over last 100 training episodes: `232.47 -> 217.44`
- Best checkpoint timing is now recorded explicitly: `220000`

What did not improve:

- Best eval mean reward stayed at `280.59`
- 20-episode and 50-episode saved-model evaluations matched iteration 1

## Conclusion

Iteration 2 improved sustained end-of-training behavior and fixed a logging gap, but it did not discover a stronger checkpoint than iteration 1. The rejected pilots suggest that the next improvement is unlikely to come from naive vectorization or simple linear learning-rate decay alone.
