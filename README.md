# Laplace-Conventional

A standalone conventional-model training system for the corpus estate. It does **not** require Laplace, PostgreSQL, a Laplace substrate, or pre-existing Laplace decomposers.

The repository separates four facts that the previous implementation incorrectly collapsed:

1. **Corpus coverage** — every file is inventoried, hashed, classified, and counted before training. Exact duplicates are recorded instead of silently overweighting duplicated source trees.
2. **Generic structural extraction** — UTF-8 text/code is preserved as text; JSON/JSONL, XML, CSV/TSV, and CoNLL-U are split into records using their file formats rather than source-name-specific parsers. Binary/image/audio/video bytes remain visible as uncovered until an explicit training path exists.
3. **Measured representation/configuration** — SentencePiece vocabulary candidates are trained against the corpus and selected by held-out minimum-description-length score. Dataset token counts and record-length quantiles are measured before the model shape is generated.
4. **Conventional training/post-training** — the neural model is a standard Hugging Face Llama causal LM, optimized by PyTorch/Accelerate. GRPO is available only when an explicit reward function and prompt dataset are supplied; there is no fabricated graph-completion reward.

## Full setup

```bash
./scripts/setup.sh
./scripts/configure.sh /vault/Data
./scripts/train.sh .state checkpoints
```

`configure.sh` deliberately stops if unique corpus bytes are classified as unsupported. That is a coverage failure to resolve, not a condition to hide behind a successful language-model run.

For the GTX 1080 Ti, setup defaults to the PyTorch 2.14 CUDA 12.6 wheel line because CUDA 13 removed Pascal library/offline compilation support. Override `TORCH_VERSION` or `TORCH_INDEX_URL` explicitly if the host has a different validated stack.

## Generated state

Nothing about corpus size, vocabulary, model dimensions, or context length is hard-coded into a checked-in `1080ti.json` or `scale.json`. Configuration is generated under `.state/` from:

- `.state/corpus/manifest.jsonl` and `summary.json`
- `.state/tokenizer/tokenizer-report.json`
- `.state/data/dataset-report.json`
- `.state/training.json`
- `.state/hardware.json`

The generated model target and the physical execution plan are separate. `MICRO_BATCH_SIZE` controls the current host's micro-batch without changing the model definition; Accelerate can launch the same training code across additional devices.

## Reinforcement learning

GRPO requires both an explicit dataset and an explicit reward implementation:

```bash
python -m laplace_conventional.rl \
  --model checkpoints/final-model \
  --dataset rewards/tasks.jsonl \
  --reward my_rewards:score \
  --output checkpoints/grpo
```

The command refuses to invent a reward from corpus relationships. A reward must be a real verifier, environment outcome, preference/reward model, test result, game result, or other explicitly chosen signal.
