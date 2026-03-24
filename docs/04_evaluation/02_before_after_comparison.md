# Before and After Comparison

## Core idea

A good fine-tuning comparison keeps the task fixed and changes the model variant.

That means:

- same eval set
- same prompt format
- same generation settings
- different model checkpoints

If you change several things at once, you learn much less.

## What to compare

The most common comparison is:

- base model
- tuned model

But the same pattern also works for:

- checkpoint A vs checkpoint B
- LoRA vs SFT
- prompt version A vs prompt version B

## What to look for

Ask questions like:

- Did the tuned model follow the instruction more often?
- Did the answer become more concise or more repetitive?
- Did new hallucinations appear?
- Did reference overlap improve on the examples that matter?

This keeps comparison practical.

## In Phase 4

The evaluation script writes both outputs side by side so you can inspect:

- base output
- candidate output
- per-row metrics
- a simple winner label

That is often enough for a first useful comparison loop.

## Common mistake

Do not compare outputs that were generated with very different decoding settings unless that difference is the thing you want to test.

Otherwise you may think tuning changed behavior when decoding actually did.

## Why this matters in real LLM engineering

Most model work is really comparison work. Reliable before-and-after evaluation is how you choose models, checkpoints, prompts, and tuning strategies without fooling yourself.
