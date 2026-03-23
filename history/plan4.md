# Iteration 4 Plan

## Context

Iteration 2 is still the strongest held-out checkpoint so far:

- 20-episode evaluation mean reward: `274.22`
- 20-episode evaluation std: `21.53`
- 20-episode mean episode length: `233.00`
- 50-episode verification mean reward: `265.28`
- 50-episode verification mean episode length: `242.50`

Iteration 3 improved the training-side objective but failed on held-out evaluation. The core problem was that the efficiency-oriented checkpoint rules still allowed brittle checkpoints with occasional catastrophic episodes.

The specific weakness in iteration 3 was the scaling of the robust-efficiency score:

`((mean_reward ** 2) / mean_length) - robustness_penalty * std_reward`

Because the variance penalty was applied only after squaring reward, it was too weak relative to the scale of the efficiency term.

## Goal

Keep pursuing the repo objective of high reward with fewer steps, but make the efficiency selector much more risk-aware so it stops preferring brittle checkpoints.

## Planned Changes

1. Add a lower-confidence efficiency metric to checkpoint selection:
   - `lcb_reward = max(mean_reward - robustness_penalty * std_reward, 0.0)`
   - `lcb_efficiency_score = (lcb_reward ** 2) / mean_length`
2. Log `lcb_reward` and `lcb_efficiency_score` during evaluation for analysis.
3. Keep `robust_reward` as the default selector, and expose `lcb_efficiency` as an explicit experiment option.
4. Fine-tune from `best/iteration2.zip`, since iteration 3 regressed on held-out metrics.
5. Use larger evaluation batches than iteration 2 when testing the new selector so rare failures are more likely to be seen during checkpoint selection.

## Training Strategy

- Start from `best/iteration2.zip`.
- Use conservative PPO fine-tuning settings similar to the successful iteration-2 recipe.
- Run a pilot with `selection_metric=lcb_efficiency`.
- Compare it to a pilot with `selection_metric=robust_reward` under the same PPO settings.
- Promote the better configuration to the main run and save it as `best/iteration4.zip`.

## Files Expected To Change

- `agent_class.py`
- `train_test.py`
- `train_agent.py`
- `history/plan4.md`
- `history/out4.md`

## Success Criteria

- Improve held-out reward or held-out episode length relative to iteration 2.
- Improve the held-out efficiency score without reintroducing large-variance catastrophic failures.
- Produce `best/iteration4.zip` from the main run and record the result honestly even if the experiment does not beat iteration 2.
