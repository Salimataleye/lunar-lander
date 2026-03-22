# Iteration 5 Plan

## Context

Iteration 3 remains the strongest overall training iteration, while iteration 4 slightly improved held-out evaluation:

- Iteration 3 best eval mean reward: `287.47`
- Iteration 3 final avg reward over last 100 training episodes: `248.73`
- Iteration 3 50-episode evaluation mean reward: `280.20`
- Iteration 4 50-episode evaluation mean reward: `280.23`
- Iteration 4 final avg reward over last 100 training episodes: `241.24`

The current evidence suggests:

- checkpoint fine-tuning is still the best search strategy,
- very conservative updates preserve or slightly improve held-out score,
- but those same updates hurt sustained training metrics,
- the next useful experiment is to fine-tune from the slightly stronger held-out checkpoint in `best/iteration4.zip` while restoring somewhat larger updates.

## Goal

Beat iteration 4 on held-out evaluation while recovering or exceeding iteration 3 on sustained training quality if possible.

## Planned Changes

1. Reuse the current fine-tuning pipeline without major structural changes unless a blocker appears.
2. Start pilots from `best/iteration4.zip`.
3. Test configurations that are less conservative than iteration 4, but still gentler than the original iteration 3 jump.
4. Use more frequent evaluation and more evaluation episodes so checkpoint selection is less noisy.
5. Save the best checkpoint from the main run as `best/iteration5.zip`.

## Training Strategy

- Start from `best/iteration4.zip`.
- Compare one moderate fine-tune and one middle-ground fine-tune.
- Choose the best candidate using both checkpoint reward and post-training evaluation, not callback score alone.

## Files Expected To Change

- likely no major code changes unless pilots expose a missing control

## Success Criteria

- 50-episode evaluation mean reward exceeds `280.23`, if possible.
- Final avg reward over last 100 training episodes should move back toward or above `248.73`.
- Mean episode length should stay near or below the iteration 3 / 4 range.
