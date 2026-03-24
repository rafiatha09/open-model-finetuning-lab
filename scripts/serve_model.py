"""Serve a local model behind a minimal FastAPI endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omlab.inference.api import run_api  # noqa: E402
from omlab.inference.generate import build_backend_from_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local FastAPI serving layer for a saved model.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/serving/local_assistant.yaml"),
        help="Path to a serving YAML config file.",
    )
    parser.add_argument("--model-path", help="Optional override for the model or checkpoint path.")
    parser.add_argument("--backend", default="transformers", help="Backend name. Starter support is 'transformers'.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the local API server.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config, backend = build_backend_from_config(
        config_path=args.config,
        repo_root=ROOT,
        model_path=args.model_path,
        backend_name=args.backend,
    )
    print(
        f"Serving {config.assistant_name} with backend={config.backend} "
        f"at http://{args.host}:{args.port}"
    )
    run_api(
        backend=backend,
        assistant_name=config.assistant_name,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
