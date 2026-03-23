# Transformer Basics

## Core idea

A transformer processes tokens by repeatedly mixing information across the
sequence and then refining each position's representation.

At a high level, a transformer block does two important things:

1. attention lets tokens look at other relevant tokens
2. a feed-forward network transforms each position after that information is mixed

Stacking many such blocks gives the model depth.

## Intuition

Earlier NLP systems often had trouble keeping track of long-range relationships.
Transformers work better because each token can directly attend to other tokens
instead of passing information only step by step through time.

That does not mean the model "understands" in a human sense. It means the
architecture is very good at building useful contextual representations.

## Minimal mental picture

For each layer:

- every token starts with a current representation
- attention decides which other positions matter right now
- the model blends information from those positions
- a feed-forward network reshapes the result
- residual connections keep earlier information available

The same pattern repeats across many layers, and the representation becomes more
task-aware as it moves upward.

## Where attention fits

Attention is the mechanism that says:

"For this token at this layer, which earlier tokens should matter most?"

That is why attention is so central to LLM explanations. It is not the whole
model, but it is the part that makes contextual lookup efficient and flexible.

## What to remember

- a transformer is not just attention
- each layer produces new hidden states, not final text
- the model gradually builds context-aware representations
- the final output is produced only after many layers and a prediction step

## Common engineering implications

Transformer basics help you reason about:

- why prompt order matters
- why adding useful context can help
- why irrelevant context can distract generation
- why model depth and size often affect capability

This also explains why debugging LLMs is different from debugging traditional
ML pipelines. You are often steering contextual behavior, not just feeding
static features into a classifier.

## See also

- `examples/attention_demo.py`
- `docs/01_foundation/04_causal_masking.md`
- `docs/01_foundation/05_decoder_only_models.md`

## Why this matters in real LLM engineering

Transformers are the core architecture behind modern LLMs. If you understand the
basic block structure, prompt behavior, context sensitivity, and the impact of
fine-tuning become much easier to reason about.
