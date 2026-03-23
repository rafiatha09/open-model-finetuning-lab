from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    number: int
    name: str
    goal: str
    next_output: str


PHASES = [
    Phase(
        number=1,
        name="LLM foundations",
        goal="Understand the concepts behind prompting, tuning, and decoding.",
        next_output="Add a small set of explanatory examples and terminology notes.",
    ),
    Phase(
        number=2,
        name="Open-model usage",
        goal="Run and compare open models before training anything.",
        next_output="Create a baseline inference script for one chosen model.",
    ),
    Phase(
        number=3,
        name="SFT / LoRA / QLoRA",
        goal="Fine-tune efficiently on a focused instruction dataset.",
        next_output="Implement a minimal supervised fine-tuning workflow.",
    ),
    Phase(
        number=4,
        name="Evaluation",
        goal="Compare base and tuned behavior with simple, readable metrics.",
        next_output="Add a baseline eval script and a small report template.",
    ),
    Phase(
        number=5,
        name="Inference and serving",
        goal="Package the tuned model as a usable domain assistant.",
        next_output="Expose a local CLI or API interface for inference.",
    ),
    Phase(
        number=6,
        name="Deployment basics",
        goal="Make model artifacts and runtime configuration easy to manage.",
        next_output="Write a deployment checklist and local serving notes.",
    ),
    Phase(
        number=7,
        name="DPO and advanced post-training",
        goal="Extend the lab into preference learning and later post-training.",
        next_output="Add a preference data schema and comparison plan.",
    ),
]
