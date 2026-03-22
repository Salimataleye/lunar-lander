# Iteration 5 Results

## Summary

Iteration 5 reused the existing fine-tuning pipeline and searched from `best/iteration4.zip` with moderately larger PPO updates than iteration 4.

This iteration improved the short-run training dynamics during pilot selection, but the recorded main run did not beat the best held-out results from iterations 3 or 4:

- Best eval mean reward during training: `286.45`
- 20-episode evaluation mean reward: `277.38`
- 50-episode verification mean reward: `278.14`
- Final avg reward over last 100 training episodes: `236.92`
- Final avg episode length over last 100 training episodes: `227.51`

The saved artifact for this iteration is `best/iteration5.zip`.

## Code / Workflow Changes

- No functional code changes were needed for iteration 5.
- The iteration reused the checkpoint fine-tuning controls added in earlier iterations.

## Fine-Tuning Start Point

The main iteration started from:

- Initial checkpoint: `best/iteration4.zip`
- Starting timestep of that checkpoint: `305000`

## Pilots Run

### Pilot 1: moderate fine-tune from iteration 4

Configuration:

- start from `best/iteration4.zip`
- `learning_rate=5e-5`
- `ent_coef=0.002`
- `clip_range=0.1`
- `target_kl=0.02`
- `n_epochs=10`
- trained for `50176` additional timesteps

Result:

- Best eval mean reward: `286.45`
- Best checkpoint timestep: `332500`
- 20-episode evaluation mean reward: `277.38`
- 50-episode evaluation mean reward: `278.14`
- Final avg reward over last 100 training episodes: `256.16`
- Final avg episode length over last 100 training episodes: `212.77`

This was the strongest pilot for the `AGENTS.md` objective because it improved the last-100 training reward and reduced episode length substantially.

### Pilot 2: middle-ground fine-tune from iteration 4

Configuration:

- start from `best/iteration4.zip`
- `learning_rate=7.5e-5`
- `ent_coef=0.003`
- `clip_range=0.12`
- `target_kl=0.025`
- `n_epochs=8`
- trained for `50176` additional timesteps

Result:

- Best eval mean reward: `286.20`
- Best checkpoint timestep: `342500`
- 20-episode evaluation mean reward: `263.99`
- Final avg reward over last 100 training episodes: `246.90`
- Final avg episode length over last 100 training episodes: `225.53`

This was rejected because it introduced an unstable low-return failure in evaluation.

## Main Run

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 5 \
  --timesteps 100000 \
  --eval-freq 2500 \
  --eval-episodes 20 \
  --test-episodes 20 \
  --print-every 50 \
  --initial-model-path best/iteration4.zip \
  --n-envs 1 \
  --learning-rate 5e-5 \
  --learning-rate-schedule constant \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 10 \
  --gamma 0.99 \
  --gae-lambda 0.98 \
  --ent-coef 0.002 \
  --clip-range 0.1 \
  --target-kl 0.02
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: `best/iteration4.zip`
- Starting timestep: `305000`
- Timesteps trained in iteration 5: `100352`
- Total timestep of final model state: `405352`
- Episodes seen during this run: `453`
- Final avg reward over last 100 training episodes: `236.92`
- Final avg episode length over last 100 training episodes: `227.51`
- Best eval mean reward observed during training: `286.45`
- Best checkpoint absolute timestep: `332500`
- Best checkpoint timestep within this run: `27500`
- Saved model: `best/iteration5.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `277.38`
- Reward std: `22.95`
- Mean episode length: `234.30`

Additional verification run over 50 episodes:

- Mean reward: `278.14`
- Reward std: `21.98`
- Mean episode length: `232.74`

## Comparison To Earlier Iterations

Compared with iteration 4:

- Best eval mean reward during training improved: `282.79 -> 286.45`
- 20-episode evaluation mean reward regressed: `278.55 -> 277.38`
- 50-episode verification mean reward regressed: `280.23 -> 278.14`
- Final avg reward over last 100 training episodes regressed: `241.24 -> 236.92`
- 50-episode mean length regressed: `228.36 -> 232.74`

Compared with iteration 3:

- Best eval mean reward during training stayed below the prior best: `286.45 < 287.47`
- 20-episode evaluation mean reward regressed: `278.32 -> 277.38`
- 50-episode verification mean reward regressed: `280.20 -> 278.14`
- Final avg reward over last 100 training episodes regressed: `248.73 -> 236.92`

## Conclusion

Iteration 5 did not improve the best overall result. The moderate fine-tuning settings were good for short-horizon training behavior, but when extended to a longer main run they did not preserve the pilot’s stronger last-100 averages, and the selected checkpoint underperformed the best held-out models from iterations 3 and 4.
