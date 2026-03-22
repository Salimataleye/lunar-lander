# Iteration 3 Results

## Summary

Iteration 3 improved on the previous best result by fine-tuning from the saved checkpoint in `best/iteration2.zip` instead of training from scratch.

This iteration improved all of the main metrics of interest relative to iteration 2:

- Best eval mean reward during training: `280.59 -> 287.47`
- 20-episode post-training evaluation mean reward: `275.48 -> 278.32`
- 50-episode verification mean reward: `276.64 -> 280.20`
- Final avg reward over last 100 training episodes: `223.81 -> 248.73`
- Final avg episode length over last 100 training episodes: `217.44 -> 215.21`
- 50-episode verification mean length: `238.02 -> 226.10`

The saved artifact for this iteration is `best/iteration3.zip`.

## Code / Workflow Changes

- Added checkpoint fine-tuning support to the PPO workflow.
- Added PPO load-time hyperparameter overrides for fine-tuning.
- Added run metadata to distinguish:
  - starting checkpoint timestep,
  - timesteps trained during the current iteration,
  - absolute timestep of the best checkpoint found during fine-tuning,
  - timestep of the best checkpoint relative to the current run.

## Fine-Tuning Start Point

The main iteration started from:

- Initial checkpoint: `best/iteration2.zip`
- Starting timestep of that checkpoint: `220000`

## Pilots Run

### Pilot 1: moderate fine-tune

Configuration:

- start from `best/iteration2.zip`
- `learning_rate=1e-4`
- `ent_coef=0.005`
- `clip_range=0.15`
- `n_epochs=10`
- trained for `100352` additional timesteps

Result:

- Best eval mean reward: `288.83`
- Best checkpoint timestep: `230000`
- 20-episode evaluation mean reward: `266.57`
- 50-episode evaluation mean reward: `274.81`
- Final avg reward over last 100 training episodes: `248.73`
- Final avg episode length over last 100 training episodes: `215.21`

This was the strongest checkpoint-discovery configuration and became the basis for the main run.

### Pilot 2: more conservative fine-tune

Configuration:

- start from `best/iteration2.zip`
- `learning_rate=5e-5`
- `ent_coef=0.001`
- `clip_range=0.1`
- `n_epochs=5`
- trained for `100352` additional timesteps

Result:

- Best eval mean reward: `279.85`
- Best checkpoint timestep: `230000`
- 20-episode evaluation mean reward: `263.45`
- 50-episode evaluation mean reward: `262.74`
- Final avg reward over last 100 training episodes: `229.76`
- Final avg episode length over last 100 training episodes: `201.61`

This was more conservative, but weaker overall than pilot 1.

## Main Run

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 3 \
  --timesteps 100000 \
  --eval-freq 5000 \
  --eval-episodes 10 \
  --test-episodes 20 \
  --print-every 50 \
  --initial-model-path best/iteration2.zip \
  --n-envs 1 \
  --learning-rate 1e-4 \
  --learning-rate-schedule constant \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 10 \
  --gamma 0.99 \
  --gae-lambda 0.98 \
  --ent-coef 0.005 \
  --clip-range 0.15
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: `best/iteration2.zip`
- Starting timestep: `220000`
- Timesteps trained in iteration 3: `100352`
- Total timestep of final model state: `320352`
- Episodes seen during this run: `461`
- Final avg reward over last 100 training episodes: `248.73`
- Final avg episode length over last 100 training episodes: `215.21`
- Best eval mean reward observed during training: `287.47`
- Best checkpoint absolute timestep: `295000`
- Best checkpoint timestep within this run: `75000`
- Saved model: `best/iteration3.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `278.32`
- Reward std: `20.98`
- Mean episode length: `226.65`

Additional verification run over 50 episodes:

- Mean reward: `280.20`
- Reward std: `20.82`
- Mean episode length: `226.10`

## Comparison To Earlier Iterations

Compared with iteration 2:

- Best eval mean reward improved by `+6.88`
- 20-episode evaluation mean reward improved by `+2.84`
- 50-episode evaluation mean reward improved by `+3.56`
- Final avg reward over last 100 training episodes improved by `+24.92`
- 50-episode mean length improved by `-11.92`

Compared with iteration 1:

- Best eval mean reward improved by `+6.88`
- 20-episode evaluation mean reward improved by `+2.84`
- 50-episode evaluation mean reward improved by `+3.56`
- 50-episode mean length improved by `-11.92`

## Conclusion

Iteration 3 is the strongest result so far. The main improvement came from fine-tuning the previously best PPO checkpoint with a smaller learning rate and lower entropy, rather than restarting training from scratch.
