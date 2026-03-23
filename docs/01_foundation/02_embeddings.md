# Embeddings

## Core idea

An embedding is a learned numeric representation of a token. Instead of treating
token IDs as raw labels, the model maps each token to a dense vector.

That vector is where the model starts to represent useful patterns such as:

- semantic similarity
- syntax
- rough topic information
- how a token tends to behave in context

## Intuition

A token ID like `17492` has no meaning by itself. It is just an index.

The embedding layer turns that index into a vector, which gives the model a
starting representation it can compute with.

You can think of embeddings as the model's first guess about what a token is
and how it might relate to other tokens.

## Mental model

Suppose two tokens often appear in similar contexts:

- `doctor`
- `physician`

Their embeddings may end up being similar because the training process pushes
the model to build representations that help predict the next token well.

The embedding is not a dictionary definition. It is a useful coordinate in a
learned space.

## What embeddings are not

- they are not fixed human-designed features
- they do not have one dimension per concept
- they do not fully determine model behavior on their own

Once the model starts applying attention and feed-forward layers, those initial
embeddings are transformed many times.

## Common engineering implications

Embeddings matter when:

- you add special tokens
- you fine-tune on a domain with unusual vocabulary
- you compare small vs large models on specialist terminology
- you build retrieval systems and care about vector representations

If a model struggles with domain language, one reason may be that the model's
tokenization and learned representations are weak for that vocabulary.

## Practical intuition for fine-tuning

Fine-tuning does not just teach the model new full answers. It nudges the whole
network so token representations and later hidden states become better aligned
with your task.

That is why even short, consistent examples can shift model behavior more than a
single clever prompt.

## See also

- `docs/01_foundation/01_tokenization.md`
- `docs/01_foundation/03_transformer_basics.md`
- `docs/01_foundation/06_next_token_prediction.md`

## Why this matters in real LLM engineering

Embeddings are the first learned layer between token IDs and model reasoning.
They influence how well a model handles your domain vocabulary, special tokens,
and retrieval-related representations long before decoding begins.
