# Phase 2 Self-Check Q&A

This file is a concise review sheet for `02_open_models`.

## Core Questions

### 1. What are the main Hugging Face objects you use for decoder-only inference?

- `AutoTokenizer`
- `AutoModelForCausalLM`
- sometimes `AutoConfig`

### 2. Why do the tokenizer and model need to match?

- The model expects token IDs produced by its own tokenizer setup.
- A mismatched tokenizer can break encoding, decoding, and generation quality.

### 3. What does `from_pretrained(...)` usually load?

- Tokenizer files, config, and model weights from a Hub ID or local folder.

### 4. What does a tokenizer do during inference?

- It encodes text into token IDs and decodes generated token IDs back into text.

### 5. What does `return_tensors="pt"` do?

- It returns PyTorch tensors instead of plain Python lists.

### 6. What is greedy decoding?

- Always pick the highest-probability next token.

### 7. What is sampling?

- Choose the next token probabilistically from the model’s distribution.

### 8. What does `temperature` change?

- It makes sampling more conservative when low and more random when high.

### 9. What does `top_p` change?

- It limits sampling to the smallest likely token set whose cumulative probability reaches `p`.

### 10. What does `max_new_tokens` control?

- The maximum number of tokens the model can generate beyond the prompt.

### 11. What is a chat template?

- A rule that formats role-based messages into the exact token/text layout a chat model expects.

### 12. Why can a wrong chat template hurt performance?

- The model may misread system, user, and assistant turns and generate in the wrong format.

## Applied Questions

### 13. Why should you test prompting before fine-tuning?

- Better prompting may solve the problem cheaply and give you a stronger baseline.

### 14. Why is local model loading useful even if Hub loading exists?

- It helps with offline work, fine-tuned checkpoints, reproducibility, and private artifacts.

### 15. Why can two decoding settings make the same model feel very different?

- Decoding changes how probabilities are turned into actual text.

### 16. Why is inspecting token IDs sometimes useful?

- It helps debug prompt formatting, truncation, padding, and special-token behavior.

### 17. Why is a short `max_new_tokens` value useful while debugging?

- It keeps outputs fast, cheap, and easy to inspect.

### 18. What is the practical difference between “the model is bad” and “the prompt or template is bad”?

- A bad model lacks capability, while a bad prompt or template may hide capability that is already there.

## Short Explain-Like-I’m-Teaching

### 19. Explain `temperature` in plain English.

- Temperature controls how bold or cautious the model feels when sampling.

### 20. Explain why tokenizer loading is part of the model-loading workflow.

- The model only understands token IDs, so loading the tokenizer is required to prepare valid inputs.

### 21. Explain why chat models are still next-token predictors under the hood.

- They still generate one next token at a time, just over a chat-formatted sequence.

### 22. Explain why prompt structure matters even before training.

- Clear structure gives the model better task, audience, and output guidance.

## Mini Challenge

### 23. You loaded an open model successfully, but the output is messy and ignores your system instruction. What are the first things you should check?

- Confirm the tokenizer matches the model.
- Check whether the model expects a chat template.
- Inspect the serialized prompt.
- Review decoding settings like greedy vs sampling, `temperature`, `top_p`, and `max_new_tokens`.

## Why this matters in real LLM engineering

Phase 2 is where abstract LLM knowledge turns into actual inference workflow.
If you can answer these questions clearly, you are ready to compare open models,
prompt them well, and build a real baseline before fine-tuning.
