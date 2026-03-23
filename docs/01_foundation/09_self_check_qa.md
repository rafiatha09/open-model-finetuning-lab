# Phase 1 Self-Check Q&A

This file is a concise review sheet for `01_llm_foundations`.

## Core Questions

### 1. What is the difference between raw text, tokens, and token IDs?

- Raw text is the human-readable string.
- Tokens are the text pieces created by the tokenizer.
- Token IDs are the integer IDs for those tokens.

### 2. Why is token count not the same as word count, and why does that matter in practice?

- Tokenizers split text into subwords, punctuation, whitespace patterns, and symbols, not just words.
- This matters because cost, latency, context limits, and truncation are token-based.

### 3. What does an embedding do to a token ID?

- It maps a token ID to a dense learned vector.
- That vector is the model’s starting representation of the token.

### 4. In a transformer block, what are the two main jobs of attention and the feed-forward network?

- Attention mixes information from relevant tokens.
- The feed-forward network transforms each token’s updated representation.

### 5. What do query, key, and value each represent in simple terms?

- Query: what the current token is looking for
- Key: how each token can be matched
- Value: the information each token can provide

### 6. In the attention demo, what changes if you modify the values but keep the query and keys the same?

- The attention scores stay the same.
- The attention weights stay the same.
- The final mixed output vector changes.

### 7. What is causal masking, in one or two sentences?

- Causal masking lets a token attend only to itself and earlier tokens.
- It blocks attention to future tokens.

### 8. Why is causal masking necessary for decoder-only language models?

- It prevents the model from peeking at future tokens.
- It makes training match left-to-right generation.

### 9. What does “next-token prediction” really mean?

- The model uses the current prefix to predict probabilities for the next token.

### 10. Why does next-token prediction produce probabilities instead of one guaranteed answer?

- Many continuations can be plausible.
- The model outputs a probability distribution over tokens, then decoding picks one.

### 11. What is the difference between training and inference?

- Training updates model weights from data.
- Inference uses fixed weights to generate or score outputs.

### 12. What is the difference between a context window and a KV cache?

- The context window is how many tokens the model can consider.
- The KV cache is an inference optimization that reuses past attention computations.

## Applied Questions

### 13. If a prompt is performing badly, why might tokenization be one of the first things to inspect?

- Important text may be split awkwardly or truncated.
- The real token count may be much larger than expected.

### 14. If a model can attend to future tokens during training, what goes wrong?

- It leaks answer information from the future.
- The model can cheat instead of learning valid next-token prediction.

### 15. Why is a decoder-only model a natural fit for text generation?

- It is built for left-to-right next-token prediction.
- That matches how generated text is produced.

### 16. If two prompts say the same thing in different formats, why might the model behave differently?

- Different formats create different token sequences and context patterns.
- The model is sensitive to wording and structure.

### 17. Why can a model with a large context window still need retrieval or summarization?

- Long contexts are still expensive and can contain too much irrelevant text.
- Retrieval and summarization help keep only the most useful information.

### 18. Why does KV cache help inference speed but not change what the model has learned?

- It reuses past computations during generation.
- It is a runtime optimization, not a training change.

## Short Explain-Like-I’m-Teaching

### 19. Explain causal masking to a friend in plain English.

- The model can read what came before, but it cannot peek at words that come later.

### 20. Explain the difference between keys and values in attention.

- Keys help decide where to look.
- Values are the information brought back from those places.

### 21. Explain why “chat” is still next-token prediction under the hood.

- A chat prompt is still just serialized text.
- The model predicts one next token at a time from that text.

### 22. Explain why training loss and generation quality are related but not the same thing.

- Lower loss often helps, but generation quality also depends on prompting, decoding, and whether the model handles multi-step generation well.

## Mini Challenge

### 23. Suppose a decoder-only model sees the prompt `I love you`. Explain how tokenization, embeddings, attention, causal masking, and next-token prediction all play a role in generating the next token.

- The prompt is first split into tokens and token IDs.
- Each token ID is mapped to an embedding vector.
- Attention lets each token use relevant earlier tokens.
- Causal masking blocks access to future tokens.
- The model outputs probabilities for the next token, then decoding selects one.

## Why this matters in real LLM engineering

This self-check helps confirm that the Phase 1 concepts are connected, not just
memorized in isolation. If you can answer these questions clearly, you are in a
good position to move into open-model usage, prompting, fine-tuning, and evals.
