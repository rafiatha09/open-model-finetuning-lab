# Latency vs Cost

## The tradeoff

Serving decisions often balance two pressures:

- make responses fast
- keep hardware cost under control

You usually cannot optimize both perfectly at the same time.

## What increases latency

Common causes of slower serving:

- larger models
- longer prompts
- longer generated outputs
- weak batching behavior
- CPU-only inference for models that want acceleration

## What increases cost

Common causes of higher serving cost:

- overprovisioned hardware
- keeping too many replicas warm
- large models with poor utilization
- low-throughput deployments

## Why this matters in local learning

Even on one machine, you are already making the same category of decisions:

- use a smaller model or not
- quantize or not
- batch or not
- serve with a simple backend or a more advanced runtime

That is why Phase 5 includes these ideas early.

## Why this matters in real LLM engineering

Latency and cost are product constraints, not just infrastructure details.
Serving work becomes much clearer once you see every deployment choice as a
tradeoff between responsiveness and resource usage.
