from pathlib import Path

from serving.config import load_serving_config
from serving.inference import build_inference_prompt


ROOT = Path(__file__).resolve().parents[1]


def test_serving_config_loads_local_checkpoint_defaults() -> None:
    config = load_serving_config(ROOT / "configs/serving/local_assistant.yaml")
    assert config.assistant_name == "domain-assistant"
    assert config.model_path == "models/domain-assistant-sft"
    assert config.max_new_tokens > 0


def test_inference_prompt_matches_instruction_template() -> None:
    prompt = build_inference_prompt(
        instruction="Explain LoRA in simple terms.",
        input_text="Keep it to 2 sentences.",
        system_prompt="Be concise and factual.",
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "System guidance:" in prompt
    assert prompt.endswith("### Response:\n")
