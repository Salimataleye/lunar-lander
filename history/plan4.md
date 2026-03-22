# Iteration 4 Plan

## Context

Iteration 3 is currently the strongest result:

- Best eval mean reward: `287.47`
- 20-episode evaluation mean reward: `278.32`
- 50-episode verification mean reward: `280.20`
- Final avg reward over last 100 training episodes: `248.73`
- 50-episode verification mean length: `226.10`

The key lesson from the recent iterations is:

- training from scratch is no longer the strongest path,
- checkpoint fine-tuning works,
- large PPO changes can easily hurt performance,
- more careful fine-tuning is the best next bet.

## Goal

Improve on iteration 3 by polishing the existing best checkpoint rather than relearning. The main target is a higher evaluation reward while preserving or improving the shorter episode lengths achieved in iteration 3.

## Planned Changes

1. Start from `best/iteration3.zip`.
2. Add one more PPO fine-tuning control if needed to stabilize updates during polishing.
3. Run short fine-tuning pilots with:
   - lower learning rates than iteration 3,
   - smaller clip ranges,
   - lower entropy,
   - more evaluation episodes and/or more frequent evaluation.
4. Use the best pilot configuration for the main iteration 4 run.
5. Save only the best checkpoint from the main run as `best/iteration4.zip`.

## Training Strategy

- Use cautious PPO updates, not exploratory retraining.
- Prefer stable held-out evaluation performance over a noisy callback-only improvement.
- Compare pilots using both checkpoint reward and post-training evaluation.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- `train_agent.py`

## Success Criteria

- Best eval mean reward exceeds `287.47`, if possible.
- 20-episode and 50-episode evaluations should stay at or above iteration 3 if possible.
- Mean episode length should not regress materially.
