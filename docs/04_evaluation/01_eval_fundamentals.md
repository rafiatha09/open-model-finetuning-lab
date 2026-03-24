# Evaluation Fundamentals

## What evaluation is

Evaluation asks a simple question:

"Did the model get better in the way I actually care about?"

For this repository, that usually means comparing:

- a base model
- a fine-tuned model
- the same prompts
- the same evaluation set

The point is not to produce one magic score. The point is to collect evidence.

## What to evaluate

At a minimum, an evaluation loop should help you inspect:

- answer quality against a reference
- instruction following
- obvious regressions
- suspicious unsupported claims
- examples that need manual review

This is why a small readable eval set is often more useful early on than a big complicated benchmark.

## What good early evaluation looks like

A beginner-friendly evaluation loop should be:

- small enough to run often
- structured enough to compare versions
- readable enough to inspect by hand

That is the design goal for Phase 4.

## In this repository

The starter evaluation flow compares two model variants on the same JSONL eval set and writes:

- row-level outputs in JSONL
- an aggregate summary in JSON
- a qualitative review markdown file

That gives you both machine-readable outputs and human-readable review material.

## Limits of simple evals

Heuristic metrics can help, but they are not truth.

For example:

- lexical overlap does not fully measure correctness
- a hallucination warning is not a factuality proof
- exact match can be too strict for good paraphrases

So the right mindset is:

"use lightweight metrics to narrow attention, then inspect examples."

## Why this matters in real LLM engineering

Evaluation is how you decide whether tuning helped, hurt, or simply changed style. Without it, you are guessing from a few prompts instead of making engineering decisions from evidence.
