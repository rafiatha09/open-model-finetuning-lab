# Phase 6: Advanced Post-Training and Future Extensions

## Goal

Create a clear bridge from the Goal A production path into more advanced
post-training work, without making it part of the first required build.

## What to build here

- concise learning docs for DPO, reward models, GRPO/RL, and distillation
- notes on when these methods become useful after SFT or LoRA
- a clear boundary between "good first production version" and "later upgrade"
- a short self-check to confirm the advanced concepts are clear

## Advanced bridge docs

1. `docs/06_advanced/01_dpo.md`
2. `docs/06_advanced/02_reward_models.md`
3. `docs/06_advanced/03_grpo_and_rl.md`
4. `docs/06_advanced/04_distillation.md`
5. `docs/06_advanced/05_self_check_qa.md`

## Exit criteria

You understand what comes after SFT / LoRA, when it is worth reaching for, and
why it is not required for the first production version.

## Why this matters in real LLM engineering

This phase gives you a practical map of what comes after the core pipeline, so
you can recognize advanced methods, sequence them correctly, and avoid adding
post-training complexity too early.
