"""Show how the local serving backend loads once and generates responses."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _hf_demo_utils import build_demo_model, build_demo_tokenizer  # noqa: E402
from omlab.inference.generate import GenerationRequest, TransformersGenerationBackend  # noqa: E402
from serving.config import ServingConfig  # noqa: E402


def _write_demo_model(model_dir: Path) -> None:
    tokenizer = build_demo_tokenizer()
    model = build_demo_model(tokenizer.vocab_size)
    with torch.no_grad():
        model.lm_head.weight[18].add_(0.75)
        model.transformer.wte.weight[18].add_(0.75)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir, max_shard_size="10GB", safe_serialization=False)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="serving-backend-demo-") as tmpdir:
        model_dir = Path(tmpdir) / "model"
        _write_demo_model(model_dir)

        config = ServingConfig(
            assistant_name="demo-assistant",
            model_path=str(model_dir),
            backend="transformers",
            max_new_tokens=8,
            do_sample=False,
        )
        backend = TransformersGenerationBackend(config=config, repo_root=ROOT)
        result = backend.generate(GenerationRequest(instruction="Tell me about LoRA"))

        print("Serving backend demo")
        print(f"Backend: {result.backend}")
        print(f"Prompt starts with: {result.prompt.splitlines()[0]}")
        print(f"Response: {result.text or '<no text generated>'}")


if __name__ == "__main__":
    main()
