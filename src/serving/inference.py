"""Small inference helpers for local instruction-style assistants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.instruction_dataset import InstructionExample, format_instruction_prompt
from serving.config import ServingConfig


@dataclass(frozen=True)
class ModelBundle:
    tokenizer: object
    model: object
    device: torch.device


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_input_text(input_text: str = "", system_prompt: str = "") -> str:
    sections: list[str] = []
    if system_prompt.strip():
        sections.append(f"System guidance:\n{system_prompt.strip()}")
    if input_text.strip():
        sections.append(input_text.strip())
    return "\n\n".join(sections)


def build_inference_prompt(instruction: str, input_text: str = "", system_prompt: str = "") -> str:
    """Build the same prompt shape used during SFT, with optional extra context."""
    example = InstructionExample(
        instruction=instruction.strip(),
        input=_build_input_text(input_text=input_text, system_prompt=system_prompt),
        response="",
    )
    return format_instruction_prompt(example)


def _resolve_pretrained_source(model_spec: str, repo_root: Path) -> str | Path:
    direct_path = Path(model_spec)
    if direct_path.exists():
        return direct_path

    relative_path = repo_root / model_spec
    if relative_path.exists():
        return relative_path

    try:
        return snapshot_download(model_spec, local_files_only=True)
    except Exception:
        return model_spec


def load_model_bundle(config: ServingConfig, repo_root) -> ModelBundle:
    """Load a local causal LM checkpoint and tokenizer for inference."""
    model_source = _resolve_pretrained_source(config.model_path, Path(repo_root))
    if isinstance(model_source, Path) and not model_source.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_source}. "
            "Run training first, point --model-path at a saved checkpoint, or use a cached HF model id."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=config.trust_remote_code,
    )
    device = _resolve_device()
    model.to(device)
    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model, device=device)


def generate_response(
    bundle: ModelBundle,
    prompt: str,
    config: ServingConfig,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate a response string from an instruction-style prompt."""
    effective_max_new_tokens = max_new_tokens or config.max_new_tokens
    effective_temperature = config.temperature if temperature is None else temperature
    effective_top_p = config.top_p if top_p is None else top_p
    do_sample = config.do_sample if config.do_sample is not None else effective_temperature > 0.0

    inputs = bundle.tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": effective_max_new_tokens,
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "eos_token_id": bundle.tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = effective_temperature
        generation_kwargs["top_p"] = effective_top_p

    with torch.no_grad():
        output_ids = bundle.model.generate(**inputs, **generation_kwargs)

    new_token_ids = output_ids[0][prompt_length:]
    return bundle.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
