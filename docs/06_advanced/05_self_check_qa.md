# Self Check Q&A

## Questions and short answers

### What is DPO?

DPO is a post-training method that learns from preference pairs such as a chosen
answer and a rejected answer.

### When should I consider DPO?

After SFT or LoRA is already working and you want to improve preference,
helpfulness, or response ranking.

### What is a reward model?

A reward model scores model outputs so later systems can rank, filter, or
optimize generations.

### When do reward models become useful?

When you already have candidate outputs, some notion of "better," and a reason
to separate generation from judging.

### What do GRPO and RL-style methods add?

They optimize behavior using a reward signal instead of only copying target
answers from supervised data.

### When are GRPO or RL methods a bad next step?

When the SFT or LoRA baseline is still weak, the eval setup is unclear, or the
team does not yet know what behavior should be optimized.

### What is distillation?

Distillation trains a smaller model to imitate a stronger teacher model.

### When is distillation useful?

After you already have a strong assistant and need cheaper or faster serving.

### Which of these is required for the first production version?

None of them. For Goal A, they are future extensions after the core pipeline is
stable.

## Why this matters in real LLM engineering

These methods help you see what comes after the first working system, which
makes it easier to prioritize well and avoid premature complexity.
