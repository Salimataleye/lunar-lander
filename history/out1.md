# Iteration 1 Results

## Summary

Iteration 1 established the first recorded baseline for this repo state and saved the best checkpoint to `best/iteration1.zip`.

The main improvement made before running the experiment was to align the default training configuration with the stronger single-environment PPO setup already supported by the current codebase, rather than the weaker multi-environment linear-learning-rate default that was still present in the CLI.

## Code / Workflow Changes

- Updated the default PPO training arguments in `train_test.py` and `train_agent.py` to the stronger single-environment setup:
  - `n_envs=1`
  - constant learning rate
  - `batch_size=64`
  - `ent_coef=0.01`
  - `total_timesteps=500000`
  - `eval_episodes=20`

## Main Run

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 1 \
  --timesteps 500000 \
  --eval-freq 5000 \
  --eval-episodes 20 \
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
  --ent-coef 0.01 \
  --clip-range 0.2
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: none
- Timesteps trained in this run: `500736`
- Total timesteps: `500736`
- Episodes seen during training: `1937`
- Final avg reward over last 100 training episodes: `223.81`
- Final avg episode length over last 100 training episodes: `217.44`
- Best eval mean reward observed during training: `276.52`
- Best checkpoint absolute timestep: `360000`
- Saved model: `best/iteration1.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `262.23`
- Reward std: `55.07`
- Mean episode length: `240.90`

Additional verification run over 50 episodes:

- Mean reward: `258.80`
- Reward std: `57.13`
- Mean episode length: `260.18`

## Notes

- The training-side metric was much stronger than the held-out evaluation metric. The saved checkpoint appears to be somewhat noisy and picked up several low-return failures during the 50-episode verification run.
- This first recorded iteration still provides a useful baseline and a saved model artifact, but future iterations should focus on improving checkpoint robustness and evaluation stability rather than only improving late-training averages.
