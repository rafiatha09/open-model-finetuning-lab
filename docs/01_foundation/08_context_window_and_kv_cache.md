# Context Window and KV Cache

## Core idea

The context window is the maximum amount of token history the model can use at
once. The KV cache is an inference optimization that helps reuse previous
attention computations while generating new tokens.

These are related but not the same thing.

## Intuition

The context window answers:

"How much text can the model consider right now?"

The KV cache answers:

"How can we avoid recomputing everything from scratch for every new token?"

## Context window mental model

If a model has a context window of 8,000 tokens, then the prompt plus generated
history must fit within that budget.

Once you exceed the limit, something has to happen:

- truncate earlier text
- summarize old content
- retrieve only the most relevant chunks
- choose a model with a larger window

So context management is an application design problem, not just a model spec.

## KV cache mental model

In decoder-only generation, each new token attends to the tokens that came
before it.

Without caching, the system would repeatedly recompute key and value tensors for
the whole prefix at every generation step.

With a KV cache, the model stores those earlier tensors and reuses them, so new
generation focuses mainly on the newest token and the stored history.

## Why the cache matters

The KV cache is one reason autoregressive serving is practical at all.

It improves:

- latency for long generations
- throughput in serving systems
- efficiency of chat-style multi-turn continuation

But it also consumes memory, which becomes important when prompts are long or
many requests are active at once.

## Common engineering implications

This helps explain:

- why long contexts are expensive
- why prompt bloat hurts latency
- why serving systems care about sequence length and batching
- why retrieval and summarization are often needed even with large-context models

## See also

- `docs/01_foundation/01_tokenization.md`
- `docs/01_foundation/04_causal_masking.md`
- `docs/01_foundation/07_training_vs_inference.md`

## Why this matters in real LLM engineering

Context-window limits shape product design, and KV-cache behavior shapes serving
performance. If you understand both, you can reason much more clearly about
latency, truncation, memory pressure, and retrieval strategy.
