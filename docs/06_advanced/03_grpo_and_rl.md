# GRPO and RL

## What this means

GRPO and related RL-style methods use a reward signal to improve generation
behavior beyond plain supervised learning.

The important idea is:

- SFT learns from target answers
- RL-style post-training learns from a reward or preference signal

## When to use it after SFT / LoRA

These methods only make sense after:

- your SFT or LoRA baseline is already solid
- your evaluation setup is trustworthy
- you understand what behavior you are trying to optimize

If the baseline is still weak, RL-style training usually adds confusion instead
of clarity.

## Why people use it

RL-style methods are used when simple supervised data is not enough and you want
to optimize toward more nuanced behavior such as reasoning quality, style, or
task success.

## Not required for first production version

For Goal A, GRPO and RL are not required for the first production version.
These are later-stage methods, not the starting point.

## Why this matters in real LLM engineering

Even if you do not implement RL early, knowing where it fits helps you make
sense of advanced model-training stacks and avoid reaching for them too soon.
