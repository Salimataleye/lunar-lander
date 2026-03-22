# Iteration 1 Results

## Summary

Iteration 1 produced a strong working baseline on `LunarLander-v3` and saved the best checkpoint to `best/iteration1.zip`.

The final recorded training run used a PPO-backed agent through `stable_baselines3` instead of the originally planned DQN approach. That change was made after a pilot DQN run underperformed badly, while PPO showed a clear path to a high-scoring policy on the same environment.

## Code / Workflow Changes

- Standardized environment creation on `LunarLander-v3`.
- Added a modern `LunarLanderAgent` wrapper in `agent_class.py`.
- Reworked `train_test.py` into a reproducible train/evaluate pipeline with best-checkpoint selection.
- Reworked `train_agent.py` and `run_agent.py` into lightweight CLI wrappers around the new pipeline.
- Kept `trained_agents/` and `AGENTS.md` unchanged.

## Main Run

Equivalent command:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./venv/bin/python train_test.py \
  --iteration 1 \
  --timesteps 300000 \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --test-episodes 20 \
  --print-every 50
```

Main run summary:

- Environment: `LunarLander-v3`
- Total timesteps trained: `300032`
- Episodes seen during training: `1202`
- Final avg reward over last 100 training episodes: `192.93`
- Final avg episode length over last 100 training episodes: `232.47`
- Best eval mean reward observed during training: `280.59`
- Saved model: `best/iteration1.zip`

## Evaluation Results

Post-training evaluation of the saved checkpoint over 20 episodes:

- Mean reward: `275.48`
- Reward std: `21.99`
- Mean episode length: `241.05`

Additional verification run over 50 episodes:

- Mean reward: `276.64`
- Reward std: `21.47`
- Mean episode length: `238.02`

Short 5-episode rerun:

- Mean reward: `271.43`
- Mean episode length: `238.80`

## Pilot Findings

The original DQN pilot did not look competitive enough for the recorded iteration:

- 50k-step DQN pilot final avg reward over last 100 episodes: `-140.56`
- 5-episode DQN pilot evaluation mean reward: `-0.52`

The PPO pilot at 200k timesteps was much stronger and justified the switch:

- Final avg reward over last 100 training episodes: `229.80`
- 10-episode evaluation mean reward: `256.85`
- Best eval mean reward during that pilot: `275.84`

## Notes

- The exact timestep where the best checkpoint was found during the main run was not persisted by the current logging. The best evaluation score was persisted, and the corresponding best model was saved correctly to `best/iteration1.zip`.
- Future iterations should record evaluation timestep metadata explicitly in the training summary so the best checkpoint timing can be reported precisely.
