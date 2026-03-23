# Iteration 2 Plan

## Context

Iteration 1 established a solid PPO baseline and saved `best/iteration1.zip`, but it exposed a clear weakness:

- Final avg reward over last 100 training episodes: `223.81`
- Best eval mean reward during training: `276.52`
- 20-episode held-out evaluation mean reward: `262.23`
- 50-episode verification mean reward: `258.80`

The main issue is not that the agent cannot reach high reward, but that the selected checkpoint is not robust enough on held-out runs.

## Goal

Improve held-out robustness and checkpoint quality relative to iteration 1, while keeping or improving training-side performance if possible.

## Planned Changes

1. Replace the current mean-only checkpoint selection logic with a more robust evaluation callback.
2. Track richer evaluation metrics during training:
   - mean reward,
   - reward standard deviation,
   - a robustness-oriented checkpoint score.
3. Fine-tune from `best/iteration1.zip` rather than restarting from scratch.
4. Use more frequent evaluation during fine-tuning so the best robust checkpoint is less likely to be missed.
5. Save the best checkpoint from the main run as `best/iteration2.zip`.

## Training Strategy

- Start from `best/iteration1.zip`.
- Use a smaller learning rate than iteration 1.
- Reduce entropy slightly to stabilize behavior.
- Select checkpoints using a robustness-aware score rather than pure mean reward.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- `history/plan2.md`
- `history/out2.md`

## Success Criteria

- 20-episode and 50-episode evaluation improve over iteration 1.
- Evaluation variance and catastrophic failures are reduced.
- `best/iteration2.zip` is produced from the main run.
