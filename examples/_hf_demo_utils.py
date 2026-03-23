"""Shared offline Hugging Face demo helpers."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()


DEMO_VOCAB = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    ".": 7,
    "?": 8,
    "You": 9,
    "are": 10,
    "a": 11,
    "helpful": 12,
    "careful": 13,
    "assistant": 14,
    "Tell": 15,
    "me": 16,
    "about": 17,
    "LoRA": 18,
    "adapters": 19,
    "reduce": 20,
    "memory": 21,
    "cost": 22,
    "for": 23,
    "fine": 24,
    "tuning": 25,
    "Hello": 26,
    "world": 27,
    "What": 28,
    "is": 29,
    "the": 30,
}


CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|> {{ message['content'] }}
{% elif message['role'] == 'user' %}
<|user|> {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|assistant|> {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}
""".strip()


def build_demo_tokenizer() -> PreTrainedTokenizerFast:
    model = WordLevel(vocab=DEMO_VOCAB, unk_token="<unk>")
    tokenizer_backend = Tokenizer(model)
    tokenizer_backend.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_demo_model(vocab_size: int) -> GPT2LMHeadModel:
    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=48,
        n_ctx=48,
        n_embd=24,
        n_layer=1,
        n_head=2,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@contextmanager
def load_demo_assets():
    with tempfile.TemporaryDirectory(prefix="hf-demo-") as tmpdir:
        model_dir = Path(tmpdir)
        tokenizer = build_demo_tokenizer()
        model = build_demo_model(tokenizer.vocab_size)

        quiet = io.StringIO()
        with redirect_stdout(quiet), redirect_stderr(quiet):
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir, max_shard_size="10GB", safe_serialization=False)

            loaded_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            loaded_model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        loaded_model.eval()
        yield model_dir, loaded_tokenizer, loaded_model
