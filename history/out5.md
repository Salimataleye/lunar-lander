# Iteration 5 Results

## Summary

Iteration 5 focused on two workflow gaps exposed by iteration 4:

1. When fine-tuning from a strong checkpoint, the starting model was not being evaluated as a candidate.
2. The repo objective values high reward with fewer steps, but the recent best checkpoints still tended to run longer episodes than necessary.

I added start-of-run checkpoint evaluation and a step-penalized robust selector, then used those tools to polish `best/iteration4.zip`.

The new length-aware selector did not change the chosen checkpoint on the tested trajectory, but the start-of-run evaluation was useful, and a very short conservative continuation from iteration 4 found a stronger efficiency-oriented checkpoint almost immediately.

The saved artifact for this iteration is `best/iteration5.zip`.

## Code / Workflow Changes

- Extended the evaluation callback in `agent_class.py` so it can evaluate the current model at training start.
- Added a new logged score:
  - `robust_reward_length_score = mean_reward - robustness_penalty * std_reward - length_penalty * mean_length`
- Added CLI support in `train_test.py` and `train_agent.py` for:
  - `--length-penalty`
  - `--evaluate-initial-model`
  - `selection_metric=robust_reward_length`

## Fine-Tuning Start Point

The main iteration started from:

- Initial checkpoint: `best/iteration4.zip`
- Starting timestep of that checkpoint: `515000`

## Pilots Run

### Pilot 1: gentle 50k continuation with starting-checkpoint evaluation

Configuration:

- start from `best/iteration4.zip`
- `learning_rate=1.5e-5`
- `ent_coef=0.0005`
- `clip_range=0.08`
- `target_kl=0.006`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=robust_reward`
- `evaluate_initial_model=true`
- `eval_episodes=50`
- `eval_freq=5000`
- `timesteps=50000`

Result:

- Final avg reward over last 100 training episodes: `252.40`
- Final avg episode length over last 100 training episodes: `212.85`
- Best eval mean reward: `276.70`
- Best eval std: `20.09`
- Best checkpoint score: `271.67`
- Best checkpoint absolute timestep: `560000`
- 20-episode evaluation mean reward: `261.46`
- 20-episode evaluation std: `50.82`
- 20-episode mean episode length: `242.50`
- 50-episode verification mean reward: `256.93`
- 50-episode verification std: `64.27`
- 50-episode mean episode length: `246.80`

This run shortened training episodes, but the selected checkpoint was too brittle on held-out evaluation.

### Pilot 2: step-penalized robust selector

Configuration:

- same PPO settings as Pilot 1
- `selection_metric=robust_reward_length`
- `length_penalty=0.03`

Result:

- Selected the same checkpoint as Pilot 1 at timestep `560000`
- Same 20-episode evaluation result as Pilot 1:
  - mean reward `261.46`
  - std `50.82`
  - mean episode length `242.50`

This showed that the new selector support is useful, but not every trajectory provides a checkpoint ordering where it changes the winner.

### Pilot 3: short 20k continuation

Configuration:

- start from `best/iteration4.zip`
- `learning_rate=1e-5`
- `ent_coef=0.0005`
- `clip_range=0.08`
- `target_kl=0.005`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=robust_reward`
- `evaluate_initial_model=true`
- `eval_episodes=50`
- `eval_freq=2500`
- `timesteps=20000`

Result:

- Final avg reward over last 100 training episodes: `256.68`
- Final avg episode length over last 100 training episodes: `223.34`
- Best eval mean reward: `278.02`
- Best eval std: `19.37`
- Best checkpoint score: `273.18`
- Best checkpoint absolute timestep: `520000`
- Best checkpoint timestep within run: `5000`
- 20-episode evaluation mean reward: `274.24`
- 20-episode evaluation std: `19.98`
- 20-episode mean episode length: `233.65`
- 50-episode verification mean reward: `265.47`
- 50-episode verification std: `51.03`
- 50-episode mean episode length: `233.80`

This was the strongest pilot by the repo objective because it preserved reward while cutting episode length enough to produce the best efficiency score so far.

## Main Run

Main run configuration:

- start from `best/iteration4.zip`
- `learning_rate=1e-5`
- `ent_coef=0.0005`
- `clip_range=0.08`
- `target_kl=0.005`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=robust_reward`
- `evaluate_initial_model=true`
- `eval_episodes=50`
- `eval_freq=2500`
- `timesteps=20000`

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 5 \
  --timesteps 20000 \
  --eval-freq 2500 \
  --eval-episodes 50 \
  --test-episodes 20 \
  --print-every 1000 \
  --initial-model-path best/iteration4.zip \
  --evaluate-initial-model \
  --n-envs 1 \
  --learning-rate 1e-5 \
  --learning-rate-schedule constant \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 5 \
  --gamma 0.99 \
  --gae-lambda 0.98 \
  --ent-coef 0.0005 \
  --clip-range 0.08 \
  --target-kl 0.005 \
  --robustness-penalty 0.25 \
  --selection-metric robust_reward
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: `best/iteration4.zip`
- Starting timestep: `515000`
- Timesteps trained in iteration 5: `20480`
- Total timestep of final model state: `535480`
- Episodes seen during this run: `91`
- Final avg reward over last 100 training episodes: `256.68`
- Final avg episode length over last 100 training episodes: `223.34`
- Final training efficiency score: `294.99`
- Best eval mean reward observed during training: `278.02`
- Best eval std at selected checkpoint: `19.37`
- Best checkpoint score: `273.18`
- Best checkpoint absolute timestep: `520000`
- Best checkpoint timestep within this run: `5000`
- Saved model: `best/iteration5.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `274.24`
- Reward std: `19.98`
- Mean episode length: `233.65`
- Efficiency score: `321.87`

Additional verification run over 50 episodes:

- Mean reward: `265.47`
- Reward std: `51.03`
- Mean episode length: `233.80`
- Efficiency score: `301.42`

Notable behavior:

- The 20-episode evaluation is much stronger than iteration 4 and closely matches iteration 2 while keeping similar step count.
- The 50-episode verification still includes two bad outliers (`49.40` and `17.91` reward), so the policy is not strictly more robust in variance terms than iteration 4.

## Comparison To Iteration 4

What improved:

- Final avg reward over last 100 training episodes: `255.12 -> 256.68`
- Final avg episode length over last 100 training episodes: `236.53 -> 223.34`
- Final training efficiency score: `275.16 -> 294.99`
- 20-episode evaluation mean reward: `268.34 -> 274.24`
- 20-episode evaluation std: `29.17 -> 19.98`
- 20-episode mean episode length: `262.60 -> 233.65`
- 20-episode efficiency score: `274.21 -> 321.87`
- 50-episode mean episode length: `246.62 -> 233.80`
- 50-episode efficiency score: `291.54 -> 301.42`

What got worse:

- 50-episode verification mean reward: `268.14 -> 265.47`
- 50-episode verification std: `40.28 -> 51.03`

## Comparison To Iteration 2

What improved:

- Final avg reward over last 100 training episodes: `251.39 -> 256.68`
- Final avg episode length over last 100 training episodes: `230.64 -> 223.34`
- Final training efficiency score: `274.01 -> 294.99`
- 50-episode verification mean episode length: `242.50 -> 233.80`
- 50-episode verification efficiency score: `290.20 -> 301.42`
- 20-episode evaluation std: `21.53 -> 19.98`

What stayed roughly flat:

- 20-episode evaluation mean reward: `274.22 -> 274.24`
- 20-episode mean episode length: `233.00 -> 233.65`

What got worse:

- 50-episode verification std: `41.83 -> 51.03`

## Conclusion

Iteration 5 is the strongest result so far if the primary objective is reward efficiency rather than raw broad-horizon reward. The saved checkpoint nearly matches iteration 2 on 20-episode reward, improves substantially on step count in both training and 50-episode verification, and produces the best reward-squared-per-step scores seen so far.

The tradeoff is that iteration 5 is not the most stable model on 50-episode variance. Iteration 4 still looks slightly better if the only priority is broader raw-reward robustness, but iteration 5 is the best artifact if the repo objective is taken literally as maximizing reward while minimizing steps.
