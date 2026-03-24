# Instruction Following Checks

## What to test

Instruction following asks whether the model actually did the requested task shape.

Examples:

- answer in 2 sentences
- include a specific keyword
- stay concise
- respond with bullets

These are often the first behaviors people want from a fine-tuned assistant.

## Why it matters

A model can be broadly "about the right topic" while still failing the task.

For example, it may:

- ignore requested length
- skip a required concept
- answer too vaguely

So instruction following deserves its own check.

## In the starter eval set

The JSONL format supports simple optional checks such as:

- `must_include`
- `max_sentences`

These are intentionally small because the goal is to make the evaluation loop easy to read and extend.

## How to use the results

If instruction following fails often, inspect:

- prompt formatting
- dataset formatting
- whether the tuning examples clearly teach the behavior
- whether the eval reference is too strict

This helps separate model quality from task framing quality.

## Why this matters in real LLM engineering

Many assistant regressions are not about knowledge. They are about failing to do the task in the requested format. Instruction-following checks help you catch those failures early.
