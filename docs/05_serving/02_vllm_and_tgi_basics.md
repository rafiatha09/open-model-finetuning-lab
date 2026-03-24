# vLLM and TGI Basics

## What these are

vLLM and Text Generation Inference (TGI) are serving systems designed for
high-throughput LLM inference.

They are not training tools. They are runtime systems for getting better
serving behavior from deployed models.

## Why people use them

Compared with a simple local `transformers` loop, these systems can improve:

- batching efficiency
- memory usage patterns
- throughput under concurrency
- production-style API behavior

## Why this repository starts smaller

Phase 5 begins with a lightweight local serving layer so the basic moving parts
are easy to understand:

- prompt construction
- model loading
- request handling
- JSON API shape

That small interface can later swap from a local `transformers` backend to a
vLLM or TGI backend.

## Mental model

Think of the serving layer as two parts:

1. backend runtime
2. API surface

The backend decides how tokens are generated. The API decides how requests enter
and responses leave.

## Why this matters in real LLM engineering

If you keep backend concerns separate from API concerns, you can move from a
simple local server to a faster production runtime without rewriting the whole
application interface.
