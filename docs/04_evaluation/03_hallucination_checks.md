# Hallucination Checks

## What this means here

In this repository, a hallucination check means:

"Does the answer add claims that are not supported by the reference or task context?"

That is a narrower and more practical question than trying to solve factuality in full.

## Lightweight offline check

A simple starter workflow is:

1. keep a trusted reference answer
2. compare the model output against that reference
3. flag low-overlap answers with extra unsupported content
4. manually inspect the flagged rows

This is not a perfect metric. It is a triage tool.

## What the starter script does

The Phase 4 script includes a `possible_hallucination` warning.

It is a rough heuristic based on things like:

- very low overlap with the reference
- relatively long output

So treat it as:

"look here first"

not:

"this row is definitely false"

## Better manual review questions

When you inspect a flagged row, ask:

- Did the model answer the task at all?
- Did it invent specific claims?
- Did it drift into unrelated content?
- Is the reference too narrow for the answer to score fairly?

These questions usually teach you more than a single number.

## Why this matters in real LLM engineering

Hallucination risk often appears as an evaluation and product problem before it appears as a training problem. Simple checks help you catch risky outputs earlier and review them in a more focused way.
