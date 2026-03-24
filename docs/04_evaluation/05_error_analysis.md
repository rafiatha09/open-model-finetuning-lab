# Error Analysis

## Core idea

Error analysis means looking at failures in groups instead of treating every bad output as random noise.

The useful question is:

"What kinds of failures keep happening?"

## A practical loop

Start simple:

1. run the evaluation script
2. inspect the worst rows
3. group failures by pattern
4. decide what to fix next

Common patterns:

- instruction not followed
- answer too repetitive
- missing required terms
- unsupported claims
- answer is correct but poorly scored by a weak metric

## What to record

For each failure, try to note:

- task type
- likely cause
- whether the base or tuned model is worse
- what fix you would test next

This turns evaluation from passive scoring into active iteration.

## Good next-step questions

- Is this a data problem?
- Is this a prompt-format problem?
- Is this a decoding problem?
- Is this an eval-set problem?
- Is the metric misleading for this case?

Those questions usually lead to better fixes than just "train longer."

## Why this matters in real LLM engineering

Model improvement usually comes from understanding recurring failure modes, not from staring at one average score. Error analysis is how evaluation turns into better datasets, prompts, and training runs.
