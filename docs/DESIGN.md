# Design boundary

This repository is a conventional baseline, not an implementation of Laplace cognition.

The baseline is intentionally corpus-first and source-name-agnostic. Generic readers preserve format structure; they do not claim that a JSON key, XML tag, CoNLL-U column, chess notation, lexical relation, or code AST has semantics beyond what is present in the file itself.

## Configuration derivation

The checked-in code contains algorithms and selection rules; generated dimensions live in run state.

- Tokenizer vocabulary: candidate SentencePiece unigram models are compared using held-out two-part description length (tokenizer model bits + token-id bits).
- Model parameter target: default `train_tokens / tokens_per_parameter`, with the ratio explicit in `config/default.toml`.
- Model shape: enumerated Llama-compatible widths/depths select the closest estimated parameter count.
- Context: next power of two at the selected observed record-length quantile.
- Hardware: never changes model scope. It produces a separate execution plan from measured GPU/RAM state.
- Execution backend: configured `auto`, `native`, or `deepspeed_zero3_cpu_offload`. `auto` chooses a physical provider only; an unexecutable exact model is a hard failure, not a smaller-model substitution.

Every rule is inspectable and replaceable without changing corpus extraction.

## Execution-memory contract

`execution.json` records the exact generated parameter count plus conservative memory budgets. Native execution budgets full parameter/gradient/Adam state and an activation reserve. ZeRO-3 CPU-offload budgets the largest layer working set on GPU and full training state in host RAM. These are preflight bounds, not performance claims.

Training refuses stale physical overrides: `MICRO_BATCH_SIZE` must match the generated execution plan. The run records actual world size and effective global tokens per optimizer step so gradient accumulation is not described by a nominal value that differs from runtime.

## Coverage

The inventory records all bytes. Current generic trainable paths are UTF-8 text/code, JSON/JSONL, XML, CSV/TSV, and CoNLL-U. Media and opaque binary formats are not silently ignored: they remain in `unsupported_selected_bytes`, and `scripts/configure.sh` refuses a whole-corpus claim while that number is nonzero.
