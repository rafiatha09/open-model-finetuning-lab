# Serving API

## What the starter API does

The Phase 5 API is intentionally small. It exposes:

- `GET /health`
- `POST /generate`
- `POST /generate_batch`

This is enough to learn the serving flow without building a full platform.

## Why these endpoints

`/health` answers:

"Is the server alive and what backend is it using?"

`/generate` answers:

"Generate one response for this instruction."

`/generate_batch` answers:

"Accept a list of requests behind the same interface."

## Why the API stays simple

The goal is to keep local serving readable while leaving space for future
backends such as vLLM or TGI.

That is why the serving layer is split into:

- generation backend logic
- API wrapper logic

## Practical note

In a production system, you would likely add:

- authentication
- request validation policies
- structured logging
- timeouts and concurrency controls
- observability

Phase 5 intentionally stops earlier than that.

## Why this matters in real LLM engineering

A small clean API makes it easier to test, demo, and later replace the runtime
backend. Good serving design starts with a narrow interface and grows only when
the requirements become real.
