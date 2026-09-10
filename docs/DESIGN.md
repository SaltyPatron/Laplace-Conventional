# Design boundary

This repository is a conventional baseline, not an implementation of Laplace cognition.

The baseline is intentionally corpus-first and source-name-agnostic. Generic readers preserve format structure; they do not claim that a JSON key, XML tag, CoNLL-U column, chess notation, lexical relation, or code AST has semantics beyond what is present in the file itself.

## Configuration derivation

The checked-in code contains algorithms and selection rules; generated dimensions live in run state.

- Tokenizer vocabulary: candidate SentencePiece unigram models are compared using held-out two-part description length (tokenizer model bits + token-id bits).
- Model parameter target: default `train_tokens / 20`, an explicit conventional scaling-law baseline, configurable by `--tokens-per-parameter`.
- Model shape: enumerated Llama-compatible widths/depths select the closest estimated parameter count.
- Context: next power of two at the selected observed record-length quantile (default p95), configurable by CLI.
- Micro-batch: physical host setting, not model identity.

Every one of those rules is inspectable and replaceable without changing corpus extraction.

## Coverage

The inventory records all bytes. Current generic trainable paths are UTF-8 text/code, JSON/JSONL, XML, CSV/TSV, and CoNLL-U. Media and opaque binary formats are not silently ignored: they remain in `unsupported_unique_bytes`, and `scripts/configure.sh` refuses a whole-corpus claim while that number is nonzero.
