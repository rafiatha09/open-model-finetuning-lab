# Batching

## Core idea

Batching means serving multiple requests together instead of one at a time.

This can improve hardware utilization, but it changes the latency profile.

## The tradeoff

Batching can help:

- throughput
- cost efficiency

But it can also hurt:

- single-request latency
- implementation simplicity

## Why it matters for Phase 5

The starter local backend in this repository processes requests sequentially.
That keeps the code easy to read.

The API layer still exposes a batch endpoint because the interface is a good
place to prepare for later backends like vLLM or TGI, which handle batching much
more efficiently.

## What to learn from this

Separating:

- request schema
- backend implementation

makes it easier to add smarter batching later without changing the external API.

## Why this matters in real LLM engineering

Batching is one of the clearest examples of why serving is not just "call
generate in a loop." It is a systems problem where throughput, latency, and API
design all meet.
