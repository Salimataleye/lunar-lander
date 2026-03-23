# Iteration 3 Plan

## Context

Iteration 2 improved substantially over iteration 1 by switching to robustness-aware checkpoint selection during PPO fine-tuning from `best/iteration1.zip`.

Current best recorded metrics from iteration 2:

- Final avg reward over last 100 training episodes: `251.39`
- Final avg episode length over last 100 training episodes: `230.64`
- 20-episode evaluation mean reward: `274.22`
- 20-episode evaluation mean length: `233.00`
- 50-episode verification mean reward: `265.28`
- 50-episode verification mean length: `242.50`

The remaining weakness is that checkpoint selection still optimizes a proxy score based on reward mean and reward std, rather than a metric that directly matches the repo objective of high reward with fewer steps.

## Goal

Improve checkpoint quality by selecting directly for reward efficiency, using a score aligned with the objective:

`(avg_reward)^2 / avg_steps`

## Planned Changes

1. Extend evaluation logging to compute and record an efficiency-oriented score:
   - `(mean_reward ** 2) / mean_episode_length`
2. Use that score as the primary checkpoint selection criterion during training.
3. Preserve mean reward, std, and mean episode length in the logs for analysis.
4. Fine-tune from `best/iteration2.zip`.
5. Save the best checkpoint from the main run as `best/iteration3.zip`.

## Training Strategy

- Start from `best/iteration2.zip`.
- Keep PPO fine-tuning conservative enough to preserve stability.
- Compare pilots using the efficiency score plus held-out evaluation behavior.
- Prefer configurations that improve efficiency without reintroducing catastrophic failures.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- `train_agent.py`
- `history/plan3.md`
- `history/out3.md`

## Success Criteria

- Improve the efficiency-oriented score on held-out evaluation relative to iteration 2.
- Maintain or improve held-out reward while reducing average episode length.
- Produce `best/iteration3.zip` from the main run.
