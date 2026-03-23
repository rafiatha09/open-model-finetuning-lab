# Generation Parameters

## Core idea

Generation parameters control how the model turns next-token probabilities into
actual text.

The four most important ones for this phase are:

- `temperature`
- `top_p`
- `max_new_tokens`
- greedy decoding vs sampling

## Greedy decoding

Greedy decoding means:

- always choose the highest-probability next token

This is simple and deterministic, but it can be repetitive or dull.

Use it when you want:

- repeatable demos
- stable extraction-like outputs
- baseline comparisons

## Sampling

Sampling means:

- draw the next token from a probability distribution instead of always taking the top one

This gives more variety, but also more randomness.

Use it when you want:

- more diverse outputs
- creative wording
- multiple candidate generations

## Temperature

Temperature changes how sharp or flat the next-token distribution feels before
sampling.

- lower temperature: more conservative, less random
- higher temperature: more diverse, more random

A practical mental model:

- `temperature=0.2` feels cautious
- `temperature=0.7` feels moderate
- `temperature=1.2` feels more adventurous

Temperature matters only when sampling is enabled.

## Top-p

`top_p` is nucleus sampling.

Instead of sampling from the full vocabulary, the model samples only from the
smallest set of tokens whose cumulative probability reaches `p`.

Practical intuition:

- lower `top_p`: narrower candidate set
- higher `top_p`: broader candidate set

This is often easier to reason about than top-k for beginner workflows.

## Max new tokens

`max_new_tokens` limits how many tokens the model may generate beyond the input.

This matters because it controls:

- latency
- cost
- risk of rambling or runaway output

It does not change the input length. It limits only the new generated continuation.

## Practical defaults

Good starting instincts:

- greedy for deterministic baselines
- sampling for creative comparison
- small `max_new_tokens` while debugging
- modest `temperature` and `top_p` before trying wider randomness

## See also

- `docs/02_open_models/06_basic_prompting.md`
- `docs/02_open_models/05_chat_templates.md`
- `examples/generation_demo.py`

## Why this matters in real LLM engineering

Generation quality is not just about the model checkpoint. Decoding settings can
make the same model feel rigid, unstable, concise, or overly verbose, so they
must be treated as first-class inference choices.
