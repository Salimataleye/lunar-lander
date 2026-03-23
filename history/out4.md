# Iteration 4 Results

## Summary

Iteration 4 followed up on the iteration-3 failure by making the efficiency-oriented selector more risk-aware and then testing whether the main improvement lever was actually checkpoint scoring or PPO fine-tuning dynamics.

I added a lower-confidence efficiency metric:

`lcb_reward = max(mean_reward - robustness_penalty * std_reward, 0.0)`

`lcb_efficiency_score = (lcb_reward ** 2) / mean_length`

In practice, that new selector did not change the chosen checkpoint on the tested trajectory. The best result of the iteration came instead from a more conservative PPO fine-tune starting from `best/iteration2.zip`.

The saved artifact for this iteration is `best/iteration4.zip`.

## Code / Workflow Changes

- Extended the custom evaluation callback in `agent_class.py` to log:
  - `lcb_reward`
  - `lcb_efficiency_score`
- Added `selection_metric=lcb_efficiency` support to `train_test.py` and `train_agent.py`.
- Kept `robust_reward` as the default selector after iteration 3 showed that stronger efficiency pressure can over-select brittle checkpoints.
- Increased pilot and main-run evaluation batches to `50` episodes so rare failures are more likely to be seen during checkpoint selection.

## Fine-Tuning Start Point

The main iteration started from:

- Initial checkpoint: `best/iteration2.zip`
- Starting timestep of that checkpoint: `445000`

## Pilots Run

### Pilot 1: robust-reward control with larger eval batches

Configuration:

- start from `best/iteration2.zip`
- `learning_rate=5e-5`
- `ent_coef=0.002`
- `clip_range=0.1`
- `target_kl=0.01`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=robust_reward`
- `eval_episodes=50`
- `eval_freq=5000`

Result:

- Final avg reward over last 100 training episodes: `256.04`
- Final avg episode length over last 100 training episodes: `227.87`
- Best eval mean reward: `273.42`
- Best eval std: `22.20`
- Best checkpoint score: `267.87`
- Best checkpoint absolute timestep: `525000`
- 20-episode evaluation mean reward: `267.55`
- 20-episode evaluation std: `32.01`
- 20-episode mean episode length: `251.80`
- 50-episode verification mean reward: `256.94`
- 50-episode verification std: `70.62`
- 50-episode mean episode length: `235.76`

This control was clearly better than iteration 3, but it still had large held-out collapses and did not beat iteration 2 overall.

### Pilot 2: lower-confidence efficiency selector

Configuration:

- same PPO settings as Pilot 1
- `selection_metric=lcb_efficiency`

Result:

- Selected the same checkpoint as Pilot 1 at timestep `525000`
- Same 20-episode evaluation result as Pilot 1:
  - mean reward `267.55`
  - std `32.01`
  - mean episode length `251.80`

This showed that, on this trajectory, the new selector math was not the main bottleneck.

### Pilot 3: more conservative PPO fine-tune

Configuration:

- start from `best/iteration2.zip`
- `learning_rate=2.5e-5`
- `ent_coef=0.001`
- `clip_range=0.08`
- `target_kl=0.008`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=robust_reward`
- `eval_episodes=50`
- `eval_freq=5000`

Result:

- Final avg reward over last 100 training episodes: `255.12`
- Final avg episode length over last 100 training episodes: `236.53`
- Best eval mean reward: `275.26`
- Best eval std: `19.08`
- Best checkpoint score: `270.49`
- Best checkpoint absolute timestep: `515000`
- 20-episode evaluation mean reward: `268.34`
- 20-episode evaluation std: `29.17`
- 20-episode mean episode length: `262.60`
- 50-episode verification mean reward: `268.14`
- 50-episode verification std: `40.28`
- 50-episode mean episode length: `246.62`

This was the strongest pilot because its broader 50-episode verification beat iteration 2 on mean reward and slightly improved the reward-squared-per-step efficiency score.

## Main Run

Main run configuration:

- start from `best/iteration2.zip`
- `learning_rate=2.5e-5`
- `ent_coef=0.001`
- `clip_range=0.08`
- `target_kl=0.008`
- `n_epochs=5`
- `robustness_penalty=0.25`
- `selection_metric=robust_reward`
- `eval_episodes=50`
- `eval_freq=5000`

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 4 \
  --timesteps 100000 \
  --eval-freq 5000 \
  --eval-episodes 50 \
  --test-episodes 20 \
  --print-every 1000 \
  --initial-model-path best/iteration2.zip \
  --n-envs 1 \
  --learning-rate 2.5e-5 \
  --learning-rate-schedule constant \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 5 \
  --gamma 0.99 \
  --gae-lambda 0.98 \
  --ent-coef 0.001 \
  --clip-range 0.08 \
  --target-kl 0.008 \
  --robustness-penalty 0.25 \
  --selection-metric robust_reward
```

Main run summary:

- Environment: `LunarLander-v3`
- Initial model: `best/iteration2.zip`
- Starting timestep: `445000`
- Timesteps trained in iteration 4: `100352`
- Total timestep of final model state: `545352`
- Episodes seen during this run: `428`
- Final avg reward over last 100 training episodes: `255.12`
- Final avg episode length over last 100 training episodes: `236.53`
- Final training efficiency score: `275.16`
- Best eval mean reward observed during training: `275.26`
- Best eval std at selected checkpoint: `19.08`
- Best checkpoint score: `270.49`
- Best checkpoint absolute timestep: `515000`
- Best checkpoint timestep within this run: `70000`
- Saved model: `best/iteration4.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `268.34`
- Reward std: `29.17`
- Mean episode length: `262.60`
- Efficiency score: `274.21`

Additional verification run over 50 episodes:

- Mean reward: `268.14`
- Reward std: `40.28`
- Mean episode length: `246.62`
- Efficiency score: `291.54`

Notable behavior:

- The saved checkpoint no longer showed the severe multi-outlier collapse seen in the Pilot-1 control.
- There was still one clearly weak episode in the 50-episode verification (`41.86` reward), so the policy is not fully rid of occasional bad landings.

## Comparison To Iteration 2

What improved:

- Final avg reward over last 100 training episodes: `251.39 -> 255.12`
- Final training efficiency score: `274.01 -> 275.16`
- 50-episode verification mean reward: `265.28 -> 268.14`
- 50-episode verification std: `41.83 -> 40.28`
- 50-episode efficiency score: `290.20 -> 291.54`

What got worse:

- Final avg episode length over the last 100 training episodes: `230.64 -> 236.53`
- 20-episode evaluation mean reward: `274.22 -> 268.34`
- 20-episode evaluation std: `21.53 -> 29.17`
- 20-episode mean episode length: `233.00 -> 262.60`
- 20-episode efficiency score: `322.73 -> 274.21`
- 50-episode mean episode length: `242.50 -> 246.62`

## Conclusion

Iteration 4 produced a modest but real improvement on the broader 50-episode verification metric, even though the shorter 20-episode evaluation snapshot regressed relative to iteration 2. The most useful lesson was that the new `lcb_efficiency` selector is a reasonable experimental option, but the bigger gain in this iteration came from gentler PPO fine-tuning rather than a new checkpoint-scoring formula.

Because the held-out story is mixed, I would treat iteration 4 as a cautious upgrade rather than a decisive one. It is probably the better artifact if the priority is broader 50-episode robustness, but iteration 2 still looks stronger on short-horizon evaluation.
