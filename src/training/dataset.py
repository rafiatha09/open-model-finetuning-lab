"""Torch dataset helpers for instruction fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data.instruction_dataset import InstructionExample, format_instruction_prompt, format_sft_text


class InstructionSFTDataset(Dataset):
    """Tokenize instruction examples for causal language model fine-tuning."""

    def __init__(
        self,
        records: list[InstructionExample],
        tokenizer,
        max_length: int = 512,
        mask_prompt_tokens: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt_tokens = mask_prompt_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        prompt_text = format_instruction_prompt(record)
        full_text = format_sft_text(record, eos_token=self.tokenizer.eos_token or "")

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        if self.mask_prompt_tokens:
            prompt_length = min(len(prompt_ids), len(labels))
            labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class CausalLMCollator:
    tokenizer: object

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before batching.")

        input_ids = pad_sequence(
            [feature["input_ids"] for feature in features],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = pad_sequence(
            [feature["attention_mask"] for feature in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
