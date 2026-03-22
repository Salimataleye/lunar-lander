# Iteration 3 Plan

## Context

Iteration 1 found the strongest checkpoint so far, and iteration 2 improved the last-100 training average without improving checkpoint quality:

- Iteration 1 best eval mean reward: `280.59`
- Iteration 2 best eval mean reward: `280.59`
- Iteration 2 final avg reward over last 100 training episodes: `223.81`
- Iteration 2 final avg episode length over last 100 training episodes: `217.44`

The pilot evidence from iteration 2 also ruled out two simple ideas:

- naive multi-environment PPO,
- simple linear learning-rate decay from scratch.

## Goal

Improve model quality beyond the current best checkpoint by fine-tuning from the best existing PPO model instead of restarting from scratch.

## Planned Changes

1. Extend the training pipeline so a run can start from an existing checkpoint.
2. Support PPO hyperparameter overrides during checkpoint loading so fine-tuning can use gentler settings than the original run.
3. Record training metadata that distinguishes:
   - starting timestep of the loaded checkpoint,
   - timesteps trained during the current iteration,
   - absolute timestep of the best checkpoint during fine-tuning.
4. Run short fine-tuning pilots from `best/iteration2.zip` with lower learning rates / lower entropy.
5. Use the strongest pilot configuration for the main run and save its best checkpoint as `best/iteration3.zip`.

## Training Strategy

- Start from `best/iteration2.zip`.
- Test small-step PPO fine-tuning rather than another full from-scratch sweep.
- Prefer configurations that improve evaluation reward without making episode lengths worse.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- `train_agent.py`

## Success Criteria

- Best eval mean reward exceeds `280.59`, if possible.
- If best eval does not improve, still aim to improve sustained reward or mean episode length.
- `history/out3.md` clearly reports whether gains came from fine-tuning and how many extra timesteps were used.
