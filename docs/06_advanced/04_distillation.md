# Distillation

## What it is

Distillation means training a smaller or cheaper model to imitate a stronger
teacher model.

The practical goal is usually:

- lower serving cost
- lower latency
- keep as much useful behavior as possible

## When to use it after SFT / LoRA

Distillation is useful after you already have a stronger model that behaves the
way you want and you need a lighter deployment option.

It often comes after:

- baseline tuning
- evaluation
- early serving experiments

## Why people use it

It can help turn a good but expensive system into something easier to serve in
production.

## Not required for first production version

For Goal A, distillation is not required for the first production version.
It is a later optimization step once the stronger assistant behavior is already
proven.

## Why this matters in real LLM engineering

Distillation is a practical bridge between model quality and deployment
constraints, which is why it often shows up after a system already works and now
needs to become cheaper or faster.
