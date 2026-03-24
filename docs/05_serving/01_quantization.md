# Quantization

## Core idea

Quantization reduces the precision used to store model weights, often from
`float16` or `float32` down to lower-bit formats such as 8-bit or 4-bit.

The practical goal is simple:

- reduce memory usage
- fit larger models on smaller hardware
- sometimes improve serving cost

## What it changes

Quantization usually affects:

- model loading memory
- throughput
- latency
- numerical fidelity

So it is a serving tradeoff, not a free win.

## Why it matters for Phase 5

As soon as you serve real models locally, memory becomes a hard constraint.
Quantization is one of the first tools engineers reach for when the model is too
large or too expensive to keep in full precision.

## What to watch

- some quantization stacks are easier on GPU than CPU
- not every model family behaves the same way after quantization
- lower memory does not always mean lower latency

## Why this matters in real LLM engineering

Quantization is often the difference between a model that is theoretically
useful and a model that you can actually afford to load, serve, and iterate on.
