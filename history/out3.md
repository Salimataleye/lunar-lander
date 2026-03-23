# Iteration 3 Results

## Summary

Iteration 3 tested whether checkpoint selection could be aligned more directly to the repo objective by adding an efficiency-oriented score based on:

`(avg_reward ** 2) / avg_steps`

I first tried pure efficiency-based selection in pilot runs, then promoted a robustness-adjusted efficiency rule for the main run:

`((mean_reward ** 2) / mean_length) - robustness_penalty * std_reward`

The result was negative overall. Training-side efficiency improved, but held-out evaluation got worse than iteration 2 because the selected checkpoint still allowed rare but severe failures.

The saved artifact for this iteration is `best/iteration3.zip`.

## Code / Workflow Changes

- Extended the custom evaluation callback in `agent_class.py` to record:
  - `efficiency_score = (mean_reward ** 2) / mean_length`
  - `robust_efficiency_score = efficiency_score - robustness_penalty * std_reward`
- Added `selection_metric` support for:
  - `efficiency`
  - `robust_efficiency`
  - `robust_reward`
- After seeing the held-out regression, restored the default checkpoint selector in `agent_class.py`, `train_test.py`, and `train_agent.py` to `robust_reward` so the repo does not silently default to the weaker rule.

## Fine-Tuning Start Point

The main iteration started from:

- Initial checkpoint: `best/iteration2.zip`
- Starting timestep of that checkpoint: `445000`

## Exploratory Pilots

### Pilot 1: pure efficiency, frequent evals

Configuration:

- start from `best/iteration2.zip`
- `learning_rate=5e-5`
- `ent_coef=0.002`
- `clip_range=0.1`
- `target_kl=0.01`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=efficiency`
- `eval_episodes=30`
- `eval_freq=2500`

Result:

- Final avg reward over last 100 training episodes: `256.04`
- Final avg episode length over last 100 training episodes: `227.87`
- Best eval mean reward: `276.83`
- Best eval std: `19.52`
- Best checkpoint efficiency score: `347.76`
- Best checkpoint absolute timestep: `535000`
- 20-episode evaluation mean reward: `255.02`
- 20-episode evaluation std: `67.23`
- 20-episode mean episode length: `271.20`

This improved the training proxy but failed badly on held-out behavior because one catastrophic episode dominated evaluation.

### Pilot 2: pure efficiency, larger eval batches

Configuration:

- start from `best/iteration2.zip`
- `learning_rate=2.5e-5`
- `ent_coef=0.001`
- `clip_range=0.08`
- `target_kl=0.01`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=efficiency`
- `eval_episodes=50`
- `eval_freq=5000`

Result:

- Final avg reward over last 100 training episodes: `255.12`
- Final avg episode length over last 100 training episodes: `236.53`
- Best eval mean reward: `273.57`
- Best eval std: `35.94`
- Best checkpoint efficiency score: `326.84`
- Best checkpoint absolute timestep: `460000`
- 20-episode evaluation mean reward: `252.20`
- 20-episode evaluation std: `70.84`
- 20-episode mean episode length: `316.30`

This confirmed that pure efficiency selection was too easy to fool.

## Main Run

Main run configuration:

- start from `best/iteration2.zip`
- `learning_rate=5e-5`
- `ent_coef=0.002`
- `clip_range=0.1`
- `target_kl=0.01`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=robust_efficiency`
- `eval_episodes=30`
- `eval_freq=2500`

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 3 \
  --timesteps 100000 \
  --eval-freq 2500 \
  --eval-episodes 30 \
  --test-episodes 20 \
  --print-every 1000 \
  --initial-model-path best/iteration2.zip \
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
  --robustness-penalty 0.25 \
  --selection-metric robust_efficiency
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: `best/iteration2.zip`
- Starting timestep: `445000`
- Timesteps trained in iteration 3: `100352`
- Total timestep of final model state: `545352`
- Episodes seen during this run: `428`
- Final avg reward over last 100 training episodes: `256.04`
- Final avg episode length over last 100 training episodes: `227.87`
- Final training efficiency score: `287.69`
- Best eval mean reward observed during training: `276.83`
- Best eval std at selected checkpoint: `19.52`
- Best checkpoint robust-efficiency score: `342.88`
- Best checkpoint absolute timestep: `535000`
- Best checkpoint timestep within this run: `90000`
- Saved model: `best/iteration3.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `255.02`
- Reward std: `67.23`
- Mean episode length: `271.20`
- Efficiency score: `239.80`

Additional verification run over 50 episodes:

- Mean reward: `258.47`
- Reward std: `64.01`
- Mean episode length: `241.54`
- Efficiency score: `276.59`

Notable failure mode:

- The 20-episode evaluation included a `-22.67` reward episode with a `1000`-step timeout.
- The 50-episode verification also contained major outliers, including rewards of `59.00` and `34.22`.

## Comparison To Iteration 2

What improved:

- Final avg reward over last 100 training episodes: `251.39 -> 256.04`
- Final avg episode length over last 100 training episodes: `230.64 -> 227.87`
- Final training efficiency score: `274.01 -> 287.69`

What got worse:

- 20-episode evaluation mean reward: `274.22 -> 255.02`
- 20-episode evaluation std: `21.53 -> 67.23`
- 20-episode mean episode length: `233.00 -> 271.20`
- 20-episode efficiency score: `322.73 -> 239.80`
- 50-episode verification mean reward: `265.28 -> 258.47`
- 50-episode verification std: `41.83 -> 64.01`
- 50-episode efficiency score: `290.20 -> 276.59`

## Conclusion

Iteration 3 showed that selecting checkpoints directly for reward-per-step efficiency did not improve the actual held-out objective, even after adding a robustness penalty. The metric increased training-side efficiency but still over-selected brittle checkpoints that occasionally collapse.

The useful output from this iteration is the code support for `efficiency` and `robust_efficiency` as explicit experimental options. The practical takeaway is that `robust_reward` remains the safer default checkpoint selector for future iterations.
