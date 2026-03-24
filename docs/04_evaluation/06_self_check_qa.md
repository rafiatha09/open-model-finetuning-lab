# Evaluation Self Check

### 1. What is the simplest useful evaluation loop?

Use the same eval set and generation settings to compare a base model and a tuned model, then inspect both metrics and raw outputs.

### 2. Why is before-and-after comparison important?

It helps you tell whether tuning changed the behavior you actually care about instead of relying on a few impressions.

### 3. Why is one average score not enough?

Because models can improve on some tasks while getting worse on others, and many failures only show up when you read the outputs.

### 4. What is a lightweight hallucination check?

Compare the answer against a trusted reference and flag outputs that add unsupported content or have very low overlap.

### 5. What is an instruction-following check?

It tests whether the model followed task constraints such as required keywords, brevity, or answer format.

### 6. What does the Phase 4 script save?

It saves row-level outputs, an aggregate summary, and a qualitative review file for manual inspection.

### 7. Why should row-level outputs be saved?

Because debugging usually happens on individual failures, not only on averages.

### 8. What kinds of patterns belong in error analysis?

Repetition, missing required concepts, unsupported claims, weak task following, and cases where the metric seems misleading.

### 9. What is a good sign that an eval set is useful?

It is small enough to run often, clear enough to inspect manually, and representative enough to catch real regressions.

### 10. What is the practical value of qualitative review?

It helps you see style, usefulness, and failure modes that simple metrics miss.

## Why this matters in real LLM engineering

Evaluation only becomes valuable when you can explain what a score means, inspect concrete failures, and decide what to improve next. A short self-check helps make that workflow explicit.
