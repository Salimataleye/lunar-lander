# Iteration 2 Plan

## Context

Iteration 1 established a strong PPO baseline on `LunarLander-v3`:

- Best eval mean reward during training: `280.59`
- 20-episode evaluation mean reward: `275.48`
- 50-episode verification mean reward: `276.64`
- Final avg reward over the last 100 training episodes: `192.93`
- Mean evaluation episode length over 50 episodes: `238.02`

The strongest signal from iteration 1 is that PPO works well, but late-training behavior is still somewhat unstable:

- The best checkpoint was better than the final training average.
- The exact timestep of the best checkpoint was not captured in the logs.

## Goal

Improve on iteration 1 by increasing sustained reward and, if possible, reducing episode length, while also fixing the missing metadata around the best checkpoint.

## Planned Changes

1. Keep PPO as the main algorithm; do not revert to the weaker DQN path.
2. Improve PPO training stability by testing a stronger training setup, likely including:
   - parallel training environments,
   - a decaying learning-rate schedule,
   - adjusted rollout / batch settings for PPO.
3. Extend the training summary so the best evaluation score is paired with the timestep where it was found.
4. Keep the iteration workflow intact:
   - do not modify any previous files in `history/` or `best/`,
   - add only one new model artifact in `best/iteration2.zip`,
   - record the chosen configuration and results in `history/out2.md`.

## Training Strategy

- Run one or more pilots outside `best/` to compare candidate PPO configurations.
- Choose the strongest candidate based on evaluation reward and stability.
- Use the selected configuration for the main iteration 2 run.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- possibly `train_agent.py` / `run_agent.py` if argument support needs to expand

## Success Criteria

- Best eval mean reward exceeds iteration 1 if possible.
- Post-training evaluation stays at or above the iteration 1 level.
- The main run summary records the timestep of the best checkpoint.
