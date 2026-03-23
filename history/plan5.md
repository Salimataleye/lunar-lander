# Iteration 5 Plan

## Context

Iteration 4 is currently the strongest result on broader verification:

- 50-episode verification mean reward: `268.14`
- 50-episode verification std: `40.28`
- 50-episode mean episode length: `246.62`
- 50-episode efficiency score: `291.54`

But it still has two weaknesses:

1. Episode length remains relatively high.
2. When fine-tuning from a strong existing checkpoint, the current callback does not evaluate the starting model before training, so the original checkpoint cannot win unless later checkpoints beat it.

Iteration 4 also suggested that training dynamics mattered more than fancy efficiency formulas. That points toward a more conservative next step: fine-tune from `best/iteration4.zip`, preserve robustness with `robust_reward`, but add a mild direct penalty for long episodes during checkpoint selection.

## Goal

Improve reward-per-step efficiency without giving up the broader robustness gained in iteration 4.

## Planned Changes

1. Add an option to evaluate the current model at training start so fine-tuning can keep the starting checkpoint if it remains the best candidate.
2. Add a new checkpoint selector that gently penalizes episode length:
   - `robust_reward_length_score = mean_reward - robustness_penalty * std_reward - length_penalty * mean_length`
3. Log the new score during evaluation for later analysis.
4. Expose `length_penalty` and `evaluate_initial_model` through the training CLIs.
5. Fine-tune from `best/iteration4.zip` using cautious PPO settings and compare:
   - `robust_reward`
   - `robust_reward_length`
6. Promote the better pilot to the main run and save it as `best/iteration5.zip`.

## Training Strategy

- Start from `best/iteration4.zip`.
- Use short, conservative fine-tuning so the iteration-4 policy is polished rather than overwritten.
- Evaluate `50` episodes per checkpoint and treat the starting checkpoint as an eligible candidate.
- Prefer any configuration that keeps or improves 50-episode reward while reducing episode length.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- `train_agent.py`
- `history/plan5.md`
- `history/out5.md`

## Success Criteria

- Improve the 50-episode efficiency score relative to iteration 4.
- Hold 50-episode mean reward near or above iteration 4 while reducing average episode length.
- Produce `best/iteration5.zip` from the main run and document the result honestly even if the starting checkpoint remains the best candidate.
