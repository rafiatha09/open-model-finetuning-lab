"""Run or dry-run a simple supervised fine-tuning workflow from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.config import load_training_config  # noqa: E402
from training.runner import run_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to an SFT YAML config file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the config and dataset paths without loading the model.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_training_config(args.config)
    plan = run_training(config, dry_run=args.dry_run, config_path=args.config)

    print("SFT run summary")
    print(f"Config:              {plan.config_path}")
    print(f"Method:              {plan.method}")
    print(f"Base model:          {plan.base_model}")
    print(f"Training examples:   {plan.train_examples}")
    print(f"Validation examples: {plan.validation_examples}")
    print(f"Output dir:          {plan.output_dir}")
    print(f"Notes:               {', '.join(plan.notes)}")
    if args.dry_run:
        print("Dry run only: no model weights were loaded and no training started.")


if __name__ == "__main__":
    main()
