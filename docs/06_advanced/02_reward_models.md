# Reward Models

## What they are

A reward model is a model trained to score outputs.

In practice, it tries to answer:

"How good is this answer for this prompt?"

That score can later be used for ranking, filtering, or reinforcement learning.

## When to use them after SFT / LoRA

Reward models become useful after you can already produce multiple candidate
responses and you have some idea of what "better" means in your application.

They are often used when:

- pairwise preferences are available
- you want a reusable scoring model
- you are preparing for RL-style post-training

## Why people use them

They separate "generate answers" from "judge answers," which can make advanced
training and evaluation workflows more structured.

## Not required for first production version

For Goal A, reward models are not required for the first production version.
They add complexity that only makes sense once the basic assistant is already
usable.

## Why this matters in real LLM engineering

Reward models are one of the key ideas behind more advanced alignment and RL
workflows, so understanding them helps you see where post-training systems get
their optimization signal.
