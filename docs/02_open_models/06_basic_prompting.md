# Basic Prompting

## Core idea

Before fine-tuning anything, you should learn to get a strong baseline with
clear prompting.

A good prompt usually makes the task explicit, gives enough context, and sets
clear output expectations.

## A simple prompt structure

Useful building blocks:

- task: what you want the model to do
- context: the relevant information
- constraints: style, length, or format limits
- output format: bullets, JSON, short answer, and so on

That often works better than vague one-line requests.

## Example

Instead of:

`Explain LoRA`

try:

`Explain LoRA for an ML engineer new to LLM fine-tuning. Use 3 short bullets and mention memory savings.`

The second version gives the model:

- audience
- scope
- format
- emphasis

## Practical intuition

Prompting is how you discover whether the model already has enough capability
before deciding to fine-tune.

That matters because many early fine-tuning attempts are really prompt problems,
evaluation problems, or model-choice problems.

## Simple prompting habits

- ask for one task at a time
- define the audience
- specify output format when it matters
- keep instructions concrete
- compare prompts on a small benchmark set

## Before you fine-tune

Try to answer:

1. is the current model actually failing?
2. is the failure consistent?
3. does better prompting fix most of it?
4. would a different open model be enough?

If prompting already solves the issue, you may not need training yet.

## See also

- `docs/02_open_models/04_generation_parameters.md`
- `docs/02_open_models/05_chat_templates.md`
- `examples/generation_demo.py`

## Why this matters in real LLM engineering

Prompting is the cheapest way to improve open-model behavior. Strong baselines
help you decide whether fine-tuning is necessary and give you a fair reference
point for later evaluation.
