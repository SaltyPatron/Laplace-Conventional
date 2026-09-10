# Laplace-Conventional

A standalone conventional-model training system for the corpus estate. It does **not** require Laplace, PostgreSQL, a Laplace substrate, or pre-existing Laplace decomposers.

The repository separates five concerns that must remain independently inspectable:

1. **Corpus coverage** — every file is inventoried, hashed, classified, and counted before training. Exact duplicates are recorded instead of silently overweighting duplicated source trees.
2. **Generic structural extraction** — UTF-8 text/code is preserved as text; JSON/JSONL, XML, CSV/TSV, and CoNLL-U are split into records using their file formats rather than source-name-specific parsers. Binary/image/audio/video bytes remain visible as uncovered until an explicit training path exists.
3. **Measured representation/configuration** — SentencePiece vocabulary candidates are trained against the corpus and selected by held-out minimum-description-length score. Dataset token counts and record-length quantiles are measured before the model shape is generated.
4. **Model scope** — the generated conventional model is derived from corpus measurements and project policy. Hardware never silently reduces its layers, width, vocabulary, or context.
5. **Execution plan** — measured GPU/RAM state selects only the physical execution provider: native Accelerate when the exact model fits, or DeepSpeed ZeRO-3 CPU offload when native state does not. If neither fits, configuration fails instead of shrinking the model and calling that success.

## Full setup

```bash
./scripts/setup.sh
./scripts/configure.sh /vault/Data .state config/default.toml
./scripts/train.sh .state checkpoints config/default.toml
```

`config/default.toml` is the checked-in project policy: deduplication, validation split, record chunking, tokenizer candidates, scaling-law ratio, context rule, optimizer settings, checkpoint cadence, shard size, coverage policy, model-scope derivation, and execution-memory policy are explicit there. Corpus-dependent dimensions are generated from measurements rather than checked in as pretend hardware/model presets.

`configure.sh` deliberately stops if unique corpus bytes are classified as unsupported, if files are inaccessible, or if the exact generated model cannot be executed under either configured native-GPU or ZeRO-3 CPU-offload budgets. These are failed boundaries to resolve, not conditions to hide behind a successful smoke test.

For the GTX 1080 Ti, setup defaults to the PyTorch 2.14 CUDA 12.6 wheel line. Offload support installs DeepSpeed 0.19.6 by default; set `INSTALL_OFFLOAD=0` only when the configured execution policy cannot select ZeRO-3.

## Generated state

Nothing about corpus size, vocabulary, model dimensions, or context length is hard-coded into a checked-in `1080ti.json` or `scale.json`. Configuration is generated under `.state/` from:

- `.state/corpus/manifest.jsonl` and `summary.json`
- `.state/tokenizer/tokenizer-report.json`
- `.state/data/dataset-report.json`
- `.state/training.json`
- `.state/hardware.json`
- `.state/execution.json`

`training.json` defines the model. `execution.json` is prohibited from changing that model; it records parameter count, GPU/RAM budgets, native-state estimate, ZeRO-3 working-set/CPU-state estimates, selected backend, and the reason for selection. Training writes `execution-receipt.json` next to checkpoints with the actual world size, gradient accumulation, requested/effective global tokens per optimizer step, and target token count.

## Scaling

The same training entrypoint works with a single device or multiple Accelerate processes. On a memory-constrained single GPU, the `auto` execution policy first evaluates native full-state training and then ZeRO-3 parameter/optimizer CPU offload. It does **not** reinterpret "fits on this GPU" as permission to make a smaller model.

The memory figures in `execution.json` are conservative launch/preflight estimates, not throughput claims. Runtime CUDA allocation, elapsed time, tokens/second, and checkpoint progress are separate receipts produced by the actual training run.

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
