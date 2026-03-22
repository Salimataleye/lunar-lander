# Iteration 1 Plan

## Context

This repository currently has no prior recorded iterations in `history/`.
The main issues discovered during initial inspection are:

- The runnable environment in the repo venv is `LunarLander-v3`; `LunarLander-v2` raises a deprecation error and cannot be used for training or evaluation.
- The system Python is missing required dependencies, but the repo-local virtual environment at `./venv/bin/python` contains the needed packages, including `gymnasium`, `torch`, and `stable_baselines3`.
- The existing training code is older custom RL code and does not align well with the simpler `train_test.py` / `env_setup.py` flow.

## Goal

Produce a strong first working baseline that improves reward reliably on `LunarLander-v3`, tracks the best checkpoint, and leaves a clean training and evaluation path for later iterations.

## Planned Changes

1. Replace the current agent implementation used by the lightweight training flow with a practical DQN agent built on top of `stable_baselines3`.
2. Standardize environment creation on `LunarLander-v3`.
3. Upgrade `train_test.py` into a reproducible train/evaluate workflow that:
   - logs average reward over the last 100 episodes,
   - tracks episode lengths,
   - saves the best checkpoint found during the main training run,
   - can evaluate the saved checkpoint after training.
4. Keep repo constraints intact:
   - do not modify `trained_agents/`,
   - do not modify `AGENTS.md`,
   - only add new files to `history/` and `best/`.

## Training Strategy

- Use DQN with a multilayer perceptron policy.
- Favor a larger replay buffer, target updates, and exploration decay suitable for Lunar Lander.
- Evaluate periodically during training and use the best-performing checkpoint from the main run as the iteration artifact.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- `env_setup.py`
- possibly `train_agent.py` / `run_agent.py` if needed for compatibility with `LunarLander-v3`

## Success Criteria

- Training runs successfully with `./venv/bin/python`.
- A best model checkpoint is produced in `best/iteration1.zip`.
- `history/out1.md` records the resulting performance, including best evaluation score and any important observations.
