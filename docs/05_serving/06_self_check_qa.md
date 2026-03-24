# Serving Self Check

### 1. What problem does quantization try to solve in serving?

It reduces model memory use so larger checkpoints can fit on limited hardware and sometimes lowers serving cost.

### 2. Why are vLLM and TGI different from a simple `transformers` loop?

They are specialized serving runtimes that improve throughput, batching, and production-style inference behavior.

### 3. Why separate the backend from the API layer?

It lets you keep the request interface stable while swapping the runtime later.

### 4. What is the tradeoff behind batching?

Batching can improve throughput and cost efficiency, but it can also increase per-request latency.

### 5. What does `GET /health` tell you?

It confirms the server is alive and reports the assistant name, backend, and batching support.

### 6. What does `POST /generate` do?

It takes one instruction-style request and returns the generated text plus prompt and backend metadata.

### 7. Why keep a `generate_batch` endpoint even if the backend is sequential today?

Because the API shape can stay the same when you later add smarter batching backends.

### 8. Why is latency not the only serving metric that matters?

Because a faster system can still be too expensive, and a cheap system can still be too slow for the product.

### 9. What is the simplest local serving loop in this repo?

Load the model once, expose a small API, and send JSON requests to it locally.

### 10. What is the practical goal of Phase 5?

To move from checkpoint files to a usable local assistant interface that can later grow into a production runtime.

## Why this matters in real LLM engineering

Serving is where model quality meets systems constraints. A short self-check helps you confirm that you understand not only how to call the model, but why the serving design looks the way it does.
