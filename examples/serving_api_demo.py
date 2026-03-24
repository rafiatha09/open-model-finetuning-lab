"""Show the minimal API surface without starting a separate server process."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

from fastapi.testclient import TestClient
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _hf_demo_utils import build_demo_model, build_demo_tokenizer  # noqa: E402
from omlab.inference.api import create_app  # noqa: E402
from omlab.inference.generate import TransformersGenerationBackend  # noqa: E402
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
    with tempfile.TemporaryDirectory(prefix="serving-api-demo-") as tmpdir:
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
        app = create_app(backend=backend, assistant_name=config.assistant_name)
        client = TestClient(app)

        health = client.get("/health").json()
        generated = client.post(
            "/generate",
            json={"instruction": "Tell me about LoRA", "max_new_tokens": 8},
        ).json()

        print("Serving API demo")
        print(f"Health: {health}")
        print(f"Generated backend: {generated['backend']}")
        print(f"Generated text: {generated['text'] or '<no text generated>'}")


if __name__ == "__main__":
    main()
