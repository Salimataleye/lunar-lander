# Iteration 1 Plan

## Context

There are currently no recorded iteration files in `history/` and no saved model artifacts in `best/`.

The current codebase already contains a fairly advanced PPO training pipeline, including:

- a `LunarLanderAgent` PPO wrapper,
- checkpoint selection during training,
- optional checkpoint fine-tuning,
- training/evaluation metadata reporting.

However, the current CLI defaults in `train_test.py` / `train_agent.py` still point at a weaker multi-environment, linear-learning-rate PPO setup. Based on the current code and prior tuning work reflected in the codebase itself, a stronger baseline is the single-environment PPO configuration with constant learning rate and smaller batch size.

## Goal

Produce the first recorded baseline model and report in this repo state, using the strongest available from-scratch PPO configuration in the current codebase.

## Planned Changes

1. Update the default training arguments to the stronger single-environment PPO setup.
2. Run a full training/evaluation cycle with that configuration.
3. Save the best checkpoint from the main run to `best/iteration1.zip`.
4. Record the resulting metrics in `history/out1.md`.

## Training Strategy

- Use PPO on `LunarLander-v3`.
- Train with a single environment.
- Use constant learning rate rather than linear decay.
- Favor the configuration with stronger sustained training behavior:
  - `n_envs=1`
  - `learning_rate=3e-4`
  - `learning_rate_schedule=constant`
  - `n_steps=1024`
  - `batch_size=64`
  - `n_epochs=10`
  - `gamma=0.99`
  - `gae_lambda=0.98`
  - `ent_coef=0.01`
  - `clip_range=0.2`

## Files Expected To Change

- `train_test.py`
- `train_agent.py`
- `history/plan1.md`
- `history/out1.md`

## Success Criteria

- Training completes successfully with `./venv/bin/python`.
- `best/iteration1.zip` is created.
- `history/out1.md` records the final metrics and verification runs.
