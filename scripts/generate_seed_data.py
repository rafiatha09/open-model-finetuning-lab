"""Generate a larger seeded instruction dataset for the lab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.instruction_dataset import (  # noqa: E402
    InstructionExample,
    train_validation_split,
    write_instruction_jsonl,
)


TOPICS = [
    {
        "name": "tokenization",
        "summary": "splits raw text into model-readable pieces before those pieces become token IDs.",
        "why": "It affects context length, truncation, latency, and cost.",
        "pitfall": "assuming word count is the same as token count",
        "compare": "embeddings",
        "action": "inspect token counts and truncated samples early",
    },
    {
        "name": "embeddings",
        "summary": "map token IDs to learned dense vectors that give the model a useful starting representation.",
        "why": "They help the network represent similarity and context before deeper layers transform the signal.",
        "pitfall": "thinking token IDs already contain meaning on their own",
        "compare": "tokenization",
        "action": "check whether domain vocabulary is represented cleanly",
    },
    {
        "name": "self-attention",
        "summary": "lets each token gather information from other relevant tokens in the sequence.",
        "why": "It gives the model a flexible way to use context instead of relying only on local neighbors.",
        "pitfall": "assuming attention alone is the whole transformer",
        "compare": "feed-forward layers",
        "action": "trace which tokens should matter for a prompt",
    },
    {
        "name": "transformer blocks",
        "summary": "combine self-attention, feed-forward layers, and residual connections to update token representations.",
        "why": "They are the repeating core unit behind modern decoder-only LLMs.",
        "pitfall": "treating the model as one giant black box without understanding the block structure",
        "compare": "decoder-only models",
        "action": "separate attention behavior from architecture behavior when debugging",
    },
    {
        "name": "causal masking",
        "summary": "prevents each position from attending to future tokens during left-to-right generation.",
        "why": "It keeps training aligned with autoregressive next-token prediction.",
        "pitfall": "thinking it removes context instead of restricting context to the past",
        "compare": "next-token prediction",
        "action": "remember that decoder-only models cannot peek ahead",
    },
    {
        "name": "decoder-only models",
        "summary": "generate text by repeatedly predicting one next token from the prompt and generated history.",
        "why": "That makes them a natural fit for chat, completion, and instruction following.",
        "pitfall": "forgetting that chat is still just serialized text generation under the hood",
        "compare": "encoder-style processing",
        "action": "reason about outputs as continuations of a token sequence",
    },
    {
        "name": "next-token prediction",
        "summary": "trains the model to predict the probability distribution for the next token after a prefix.",
        "why": "It is the base objective behind most open decoder-only LLMs.",
        "pitfall": "expecting the model to retrieve one exact answer instead of modeling plausible continuations",
        "compare": "causal masking",
        "action": "think in terms of probabilities before decoding",
    },
    {
        "name": "context windows",
        "summary": "set how many tokens the model can actively consider at once.",
        "why": "They shape prompt design, retrieval strategy, and truncation behavior.",
        "pitfall": "assuming a large window makes prompt quality irrelevant",
        "compare": "KV cache",
        "action": "budget tokens intentionally and trim irrelevant context",
    },
    {
        "name": "KV cache",
        "summary": "reuses past attention computations during generation instead of recomputing everything from scratch.",
        "why": "It improves autoregressive inference latency for longer generations.",
        "pitfall": "thinking it changes what the model knows instead of how fast it serves",
        "compare": "context windows",
        "action": "separate model quality questions from runtime optimization questions",
    },
    {
        "name": "Hugging Face model loading",
        "summary": "turns saved tokenizer files, config, and weights into usable Python objects.",
        "why": "Most open-model workflows start by loading the tokenizer and model correctly.",
        "pitfall": "loading a checkpoint without checking tokenizer compatibility",
        "compare": "tokenizer usage",
        "action": "verify the tokenizer, config, and prompt format before generation",
    },
    {
        "name": "tokenizer usage",
        "summary": "covers encoding raw text into model inputs and decoding generated token IDs back into text.",
        "why": "Most inference bugs appear first at the tokenizer boundary.",
        "pitfall": "ignoring special tokens, padding, or truncation behavior",
        "compare": "model loading",
        "action": "inspect input IDs and decoded outputs when behavior looks odd",
    },
    {
        "name": "generation parameters",
        "summary": "control how next-token probabilities turn into actual text during decoding.",
        "why": "The same checkpoint can feel very different under different decoding settings.",
        "pitfall": "judging a model without recording temperature, top-p, or token limits",
        "compare": "basic prompting",
        "action": "lock down decoding settings when comparing prompts or models",
    },
    {
        "name": "chat templates",
        "summary": "format role-based messages into the exact text layout a chat model expects.",
        "why": "A good template can unlock strong behavior while a bad one can make a good model look broken.",
        "pitfall": "sending plain text to a chat model that expects structured role markers",
        "compare": "basic prompting",
        "action": "inspect the serialized chat prompt, not just the message list",
    },
    {
        "name": "basic prompting",
        "summary": "uses clear task, context, constraints, and output format to guide model behavior.",
        "why": "It is the cheapest way to improve an open model before fine-tuning.",
        "pitfall": "jumping to training before checking whether prompting already solves the issue",
        "compare": "fine-tuning",
        "action": "establish a strong prompt baseline before training anything",
    },
    {
        "name": "instruction dataset formatting",
        "summary": "turns each example into a consistent prompt-response text pattern for fine-tuning.",
        "why": "The model learns the format you feed it, not just the intent you had in mind.",
        "pitfall": "mixing multiple prompt styles inside one training set without a reason",
        "compare": "supervised fine-tuning",
        "action": "review formatted samples before the first training run",
    },
    {
        "name": "train/validation splitting",
        "summary": "separates examples used for learning from examples used for checking generalization.",
        "why": "It helps you see whether improvements hold beyond the exact records seen during training.",
        "pitfall": "training without a validation split and then overtrusting training loss",
        "compare": "checkpointing",
        "action": "create the split before tuning hyperparameters",
    },
    {
        "name": "supervised fine-tuning",
        "summary": "updates a language model on labeled prompt-response examples so it better follows the desired behavior.",
        "why": "It is the clearest baseline path from a curated dataset to a tuned assistant.",
        "pitfall": "expecting SFT to fix poor data quality or missing evaluation",
        "compare": "LoRA",
        "action": "start with a tiny run and compare to the base model",
    },
    {
        "name": "PEFT",
        "summary": "adapts models by training a small subset of parameters instead of the full network.",
        "why": "It lowers cost and makes experimentation with open models more practical.",
        "pitfall": "treating PEFT as magic instead of another training design choice",
        "compare": "full fine-tuning",
        "action": "choose the lightest tuning method that solves the task",
    },
    {
        "name": "LoRA",
        "summary": "adds low-rank adapters so the base model stays mostly frozen while small trainable matrices learn the task.",
        "why": "It is often the best first practical fine-tuning method for open models.",
        "pitfall": "starting with large adapter settings before proving the pipeline works",
        "compare": "QLoRA",
        "action": "run a tiny adapter experiment before scaling up",
    },
    {
        "name": "QLoRA",
        "summary": "combines LoRA with low-bit model loading to reduce training memory requirements further.",
        "why": "It can make larger-model adaptation feasible on limited hardware.",
        "pitfall": "underestimating extra dependency and platform complexity",
        "compare": "LoRA",
        "action": "treat it as the memory-efficient extension after plain LoRA is clear",
    },
    {
        "name": "checkpointing",
        "summary": "saves training state so runs can be resumed, compared, and tracked over time.",
        "why": "Without checkpoints, training results are hard to reproduce or recover.",
        "pitfall": "saving artifacts without clear naming or config traceability",
        "compare": "train/validation splitting",
        "action": "name output directories clearly and save config files with runs",
    },
    {
        "name": "baseline evaluation",
        "summary": "compares untuned and tuned behavior on a fixed set of prompts or tasks.",
        "why": "It tells you whether training improved the model in ways that matter.",
        "pitfall": "declaring success from one good-looking sample",
        "compare": "basic prompting",
        "action": "keep a stable comparison set before and after tuning",
    },
]


def build_example(topic: dict[str, str], pattern_index: int) -> InstructionExample:
    name = topic["name"]
    compare = topic["compare"]
    summary = topic["summary"]
    why = topic["why"]
    pitfall = topic["pitfall"]
    action = topic["action"]

    if pattern_index == 0:
        return InstructionExample(
            instruction=f"Explain {name} for an ML engineer new to LLM engineering.",
            input="Keep it practical and concise.",
            response=f"{name.capitalize()} {summary} {why}",
        )
    if pattern_index == 1:
        return InstructionExample(
            instruction=f"Summarize {name} in two sentences.",
            input="Focus on real engineering intuition.",
            response=f"{name.capitalize()} {summary} In practice, {why.lower()}",
        )
    if pattern_index == 2:
        return InstructionExample(
            instruction=f"What is the difference between {name} and {compare}?",
            input="Answer for a beginner-to-intermediate ML engineer.",
            response=f"{name.capitalize()} and {compare} are related but not the same. {name.capitalize()} {summary} {compare.capitalize()} should be understood separately so you do not confuse the role each concept plays.",
        )
    if pattern_index == 3:
        return InstructionExample(
            instruction=f"What is a common mistake people make with {name}?",
            input="Give one mistake and one correction.",
            response=f"A common mistake is {pitfall}. A better habit is to {action}.",
        )
    if pattern_index == 4:
        return InstructionExample(
            instruction=f"Why does {name} matter in real LLM engineering?",
            input="Answer in 2 short bullets.",
            response=f"- {why}\n- Good intuition about {name} helps you make better model, prompt, and training decisions.",
        )
    if pattern_index == 5:
        return InstructionExample(
            instruction=f"Give me a debugging tip related to {name}.",
            input="Assume I am troubleshooting a small open-model workflow.",
            response=f"Start by checking whether {name} is the real bottleneck. A reliable first move is to {action} before blaming the whole model.",
        )
    if pattern_index == 6:
        return InstructionExample(
            instruction=f"When should I inspect {name} first?",
            input="I have a workflow that behaves unexpectedly.",
            response=f"Inspect {name} early when the symptoms match its failure mode. In particular, {why.lower()}",
        )
    if pattern_index == 7:
        return InstructionExample(
            instruction=f"Give me an analogy for {name}.",
            input="Keep it concrete, not poetic.",
            response=f"A useful analogy is to think of {name} as a control layer for the workflow. It is not the whole system, but it strongly shapes how information is used or adapted.",
        )
    if pattern_index == 8:
        return InstructionExample(
            instruction=f"How does {name} connect to fine-tuning?",
            input="Mention the practical impact on data or training.",
            response=f"{name.capitalize()} matters because training only amplifies the patterns already present in your setup. If you ignore it, you may train longer without fixing the real issue.",
        )
    if pattern_index == 9:
        return InstructionExample(
            instruction=f"How does {name} connect to inference?",
            input="Focus on runtime behavior.",
            response=f"At inference time, {name} shapes how the system behaves after the model is loaded. That is why it often shows up in latency, output quality, or prompt behavior discussions.",
        )
    if pattern_index == 10:
        return InstructionExample(
            instruction=f"Teach me {name} like I am onboarding to an LLM team.",
            input="Use 3 short bullets.",
            response=f"- {name.capitalize()} {summary}\n- {why}\n- Avoid {pitfall} and {action}.",
        )
    if pattern_index == 11:
        return InstructionExample(
            instruction=f"What should I log or inspect when learning about {name}?",
            input="Answer with a small checklist.",
            response=f"Inspect real examples, note where {name} shows up in the workflow, and record the before/after effect when you change one variable at a time.",
        )
    if pattern_index == 12:
        return InstructionExample(
            instruction=f"Give me a short interview-style answer for: why is {name} important?",
            input="Keep it to 3 sentences.",
            response=f"{name.capitalize()} is important because {summary} {why} Engineers who understand it are less likely to misdiagnose model behavior.",
        )
    if pattern_index == 13:
        return InstructionExample(
            instruction=f"How would you explain {name} to someone who keeps overcomplicating it?",
            input="Use plain language.",
            response=f"Keep the core idea simple: {name} {summary} The main engineering point is that {why.lower()}",
        )
    if pattern_index == 14:
        return InstructionExample(
            instruction=f"Give me one reason to care about {name} before scaling an experiment.",
            input="I only have time for one important point.",
            response=f"Care early because {why.lower()} That makes {name} a leverage point before you spend more compute.",
        )
    if pattern_index == 15:
        return InstructionExample(
            instruction=f"What happens if I ignore {name}?",
            input="Answer in a practical way.",
            response=f"If you ignore {name}, you risk making the wrong optimization or training decision. A common failure mode is {pitfall}.",
        )
    if pattern_index == 16:
        return InstructionExample(
            instruction=f"How can {name} affect evaluation results?",
            input="Connect it to fair comparison.",
            response=f"{name.capitalize()} can distort evaluation if it changes the actual task setup between runs. Keep it controlled so you can compare outputs fairly.",
        )
    if pattern_index == 17:
        return InstructionExample(
            instruction=f"Give me a one-paragraph note on {name}.",
            input="Make it readable for future study notes.",
            response=f"{name.capitalize()} {summary} {why} When learning it, avoid {pitfall} and {action}.",
        )
    if pattern_index == 18:
        return InstructionExample(
            instruction=f"What is the smallest useful workflow habit related to {name}?",
            input="I want one concrete habit.",
            response=f"A strong habit is to {action}. That habit turns {name} from abstract theory into something you can verify in practice.",
        )
    return InstructionExample(
        instruction=f"Compare the role of {name} in a beginner workflow versus a more production-minded workflow.",
        input="Keep the answer short and practical.",
        response=f"In a beginner workflow, {name} mainly helps you build correct intuition. In a more production-minded workflow, {why.lower()} In both cases, avoid {pitfall}.",
    )


def build_seed_examples(count: int) -> list[InstructionExample]:
    patterns_per_topic = 20
    examples: list[InstructionExample] = []
    for topic in TOPICS:
        for pattern_index in range(patterns_per_topic):
            examples.append(build_example(topic, pattern_index))
    return examples[:count]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a larger seeded instruction dataset for the lab.")
    parser.add_argument("--count", type=int, default=440, help="Number of examples to generate. Default: 440")
    parser.add_argument(
        "--output",
        default="data/sample/domain_assistant_examples.jsonl",
        help="Path for the full generated dataset. Default: data/sample/domain_assistant_examples.jsonl",
    )
    parser.add_argument(
        "--train-output",
        default="data/sample/domain_assistant_train.jsonl",
        help="Path for the generated training split. Default: data/sample/domain_assistant_train.jsonl",
    )
    parser.add_argument(
        "--validation-output",
        default="data/sample/domain_assistant_validation.jsonl",
        help="Path for the generated validation split. Default: data/sample/domain_assistant_validation.jsonl",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split ratio between 0 and 1. Default: 0.2",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split. Default: 42")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    examples = build_seed_examples(args.count)
    output_path = write_instruction_jsonl(examples, ROOT / args.output)
    train_records, validation_records = train_validation_split(
        examples,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    train_path = write_instruction_jsonl(train_records, ROOT / args.train_output)
    validation_path = write_instruction_jsonl(validation_records, ROOT / args.validation_output)

    print("Generated seeded dataset")
    print(f"Total examples:      {len(examples)}")
    print(f"Training examples:   {len(train_records)}")
    print(f"Validation examples: {len(validation_records)}")
    print(f"Full dataset path:   {output_path}")
    print(f"Train path:          {train_path}")
    print(f"Validation path:     {validation_path}")


if __name__ == "__main__":
    main()
