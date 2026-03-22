# Iteration 4 Results

## Summary

Iteration 4 continued the checkpoint fine-tuning strategy from iteration 3, now starting from `best/iteration3.zip` and using more conservative PPO updates.

This iteration produced only a marginal held-out evaluation improvement:

- 20-episode evaluation mean reward: `278.32 -> 278.55`
- 50-episode verification mean reward: `280.20 -> 280.23`

However, the training-side metrics were worse than iteration 3:

- Final avg reward over last 100 training episodes: `248.73 -> 241.24`
- Final avg episode length over last 100 training episodes: `215.21 -> 220.73`
- Best eval mean reward during training: `287.47 -> 282.79`

So iteration 4 slightly improved held-out reward, but not the sustained training metric emphasized in `AGENTS.md`.

The saved artifact for this iteration is `best/iteration4.zip`.

## Code / Workflow Changes

- Added `target_kl` support to the configurable PPO training path in `train_test.py` and `train_agent.py`.
- Used that new control to test tighter PPO updates during fine-tuning.

## Fine-Tuning Start Point

The main iteration started from:

- Initial checkpoint: `best/iteration3.zip`
- Starting timestep of that checkpoint: `295000`

## Pilots Run

### Pilot 1: moderate conservative fine-tune

Configuration:

- start from `best/iteration3.zip`
- `learning_rate=5e-5`
- `ent_coef=0.002`
- `clip_range=0.1`
- `target_kl=0.02`
- `n_epochs=10`
- trained for `50176` additional timesteps

Result:

- Best eval mean reward: `284.51`
- Best checkpoint timestep: `317500`
- 20-episode evaluation mean reward: `275.95`
- Final avg reward over last 100 training episodes: `245.77`
- Final avg episode length over last 100 training episodes: `222.81`

This stayed strong, but did not beat iteration 3.

### Pilot 2: very conservative fine-tune

Configuration:

- start from `best/iteration3.zip`
- `learning_rate=2e-5`
- `ent_coef=0.001`
- `clip_range=0.08`
- `target_kl=0.01`
- `n_epochs=5`
- trained for `50176` additional timesteps

Result:

- Best eval mean reward: `282.79`
- Best checkpoint timestep: `305000`
- 20-episode evaluation mean reward: `278.55`
- 50-episode evaluation mean reward: `280.23`
- Final avg reward over last 100 training episodes: `241.24`
- Final avg episode length over last 100 training episodes: `220.73`

This gave the best held-out evaluation signal, so it was selected for the main run.

## Main Run

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 4 \
  --timesteps 50000 \
  --eval-freq 2500 \
  --eval-episodes 20 \
  --test-episodes 20 \
  --print-every 50 \
  --initial-model-path best/iteration3.zip \
  --n-envs 1 \
  --learning-rate 2e-5 \
  --learning-rate-schedule constant \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 5 \
  --gamma 0.99 \
  --gae-lambda 0.98 \
  --ent-coef 0.001 \
  --clip-range 0.08 \
  --target-kl 0.01
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: `best/iteration3.zip`
- Starting timestep: `295000`
- Timesteps trained in iteration 4: `50176`
- Total timestep of final model state: `345176`
- Episodes seen during this run: `228`
- Final avg reward over last 100 training episodes: `241.24`
- Final avg episode length over last 100 training episodes: `220.73`
- Best eval mean reward observed during training: `282.79`
- Best checkpoint absolute timestep: `305000`
- Best checkpoint timestep within this run: `10000`
- Saved model: `best/iteration4.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `278.55`
- Reward std: `22.33`
- Mean episode length: `230.65`

Additional verification run over 50 episodes:

- Mean reward: `280.23`
- Reward std: `21.75`
- Mean episode length: `228.36`

## Comparison To Iteration 3

What improved:

- 20-episode evaluation mean reward: `278.32 -> 278.55`
- 50-episode verification mean reward: `280.20 -> 280.23`

What regressed:

- Best eval mean reward during training: `287.47 -> 282.79`
- Final avg reward over last 100 training episodes: `248.73 -> 241.24`
- Final avg episode length over last 100 training episodes: `215.21 -> 220.73`
- 50-episode mean length: `226.10 -> 228.36`

## Conclusion

Iteration 4 found a slightly better held-out checkpoint, but the gain was very small and came with weaker sustained training behavior. The main lesson is that ultra-conservative PPO polishing can preserve performance and occasionally nudge reward up, but it does not seem to improve the broader training dynamics as well as iteration 3 did.
