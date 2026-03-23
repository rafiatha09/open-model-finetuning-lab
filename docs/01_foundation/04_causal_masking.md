# Causal Masking

## One-line idea

Causal masking means:

**a token can look at itself and earlier tokens, but never future tokens.**

That is the whole rule.

## Start with a simple sentence

Imagine the sequence:

`I study open models`

Number the positions:

1. `I`
2. `study`
3. `open`
4. `models`

Now ask:

"When the model is processing token 3, what is it allowed to see?"

Answer:

- position 1: yes
- position 2: yes
- position 3: yes
- position 4: no

So token `open` can look backward, but not forward.

## Visual picture

This is the allowed attention pattern:

```text
token 1 -> can see: 1
token 2 -> can see: 1, 2
token 3 -> can see: 1, 2, 3
token 4 -> can see: 1, 2, 3, 4
```

Or as a matrix:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Here:

- `1` means "allowed to attend"
- `0` means "blocked"

This triangle shape is the causal mask.

## What it actually does in attention

Normally, attention gives every token a score against every other token.

Without a mask, token 3 could score token 4 and use information from it.

Causal masking stops that. It says:

"future positions must be blocked."

So the model still computes attention, but the future-token scores are forced to
be unusable. After softmax, those future positions get effectively zero weight.

## Why this exists

Decoder-only LLMs are trained to predict the next token.

For example:

`I study open ___`

The model should predict `models`.

But if the model were allowed to look at the future token `models` before making
that prediction, it would be peeking at the answer ahead of time.

Causal masking prevents that. It forces the model to predict using only the
tokens that already came before.

## The easiest intuition

Think of the model as writing text left to right.

When writing the next word, you are allowed to use:

- what has already been written

You are not allowed to use:

- the future words that have not been generated yet

That is exactly what causal masking enforces.

## Why it matters for generation

At inference time, the model generates one token at a time:

1. read the prompt
2. predict the next token
3. append it
4. repeat

This only works cleanly because the model is built to depend on past context,
not future context.

So causal masking is one of the core rules behind autoregressive generation.

## Common misunderstanding

Causal masking does **not** mean the model has no context.

It means the model has only **past context**.

That is still a lot of context:

- the whole prompt so far
- the whole generated answer so far
- anything earlier in the sequence that fits in the context window

So the model is not "blind." It is just not allowed to peek ahead.

## A useful engineering takeaway

If you remember only one thing, remember this:

**attention decides what to look at, and causal masking limits that search to the past.**

That is why:

- prompt order matters
- decoder-only models generate left to right
- KV cache is useful
- future tokens cannot help during next-token prediction

## Try the example

Run:

```bash
python examples/causal_mask_demo.py
```

That script prints the triangular mask and shows that each token gets attention
weights only over itself and earlier tokens.

## See also

- `examples/causal_mask_demo.py`
- `docs/01_foundation/05_decoder_only_models.md`
- `docs/01_foundation/06_next_token_prediction.md`

## Why this matters in real LLM engineering

Causal masking is the rule that makes decoder-only text generation valid. It
explains why chat models work left to right, why prompt order matters, and why a
model cannot use information from tokens that have not been generated yet.
