# Laplace-Conventional

A standalone conventional-model training system for the corpus estate. It does **not** require Laplace, PostgreSQL, a Laplace substrate, or pre-existing Laplace decomposers.

The repository keeps the following boundaries independently inspectable:

1. **Physical accounting** — every file under the chosen corpus root is inventoried, hashed, classified, and counted before any training admission decision.
2. **Training admission** — selected/excluded files are controlled by ordered project rules. Exclusions require reasons, and exact deduplication happens after selection so an excluded copy cannot suppress an admitted one.
3. **Generic structural extraction** — UTF-8 text/code is preserved as text; JSON/JSONL, XML, CSV/TSV, and CoNLL-U are split by format rather than source name. Opaque formats remain uncovered until an executable provider exists.
4. **Conventional modality objectives** — text/code/structured records use causal language modeling; images use ViT-MAE masked-patch reconstruction; direct audio and audio streams present in selected video containers use Wav2Vec2 pretraining; video frames use VideoMAE masked spatiotemporal reconstruction.
5. **Measured representation/configuration** — SentencePiece candidates are trained against the admitted textual corpus and selected by held-out two-part description length. Text token counts and record-length quantiles are measured before its model shape is generated. Media model definitions are explicit named experimental policy and their instantiated parameter counts are measured during configuration.
6. **Model scope versus execution** — hardware never silently reduces a generated/configured model. Measured GPU/RAM state chooses only the physical provider: native Accelerate when full state fits, or DeepSpeed ZeRO-3 CPU offload when it does not. If neither fits, configuration fails.
7. **Receipts and restartability** — text and media lanes retain execution/source receipts. Long media passes checkpoint by wall time and can resume only when the selected source set and generated model/execution plan fingerprints still match.

## Full setup

```bash
./scripts/setup.sh
./scripts/configure.sh /vault/Data .state config/default.toml
./scripts/train-all.sh /vault/Data .state checkpoints config/default.toml
```

`config/default.toml` is checked-in project policy: validation split, explicit training selection, record chunking, tokenizer candidates, text scaling-law ratio/context rule, optimizer settings, modality model definitions/objectives, media checkpoint cadence, shard size, full-coverage policy, and execution-memory policy are all visible there. Corpus-dependent text dimensions are generated from measurements rather than checked in as pretend `1080ti.json`/`scale.json` presets.

`configure.sh` deliberately stops if admitted unique corpus bytes have no enabled training provider, if files are inaccessible, or if an exact configured/generated model cannot be executed under either native-GPU or configured ZeRO-3 CPU-offload budgets. A selected PDF/archive/tablebase/other opaque binary therefore remains a failed coverage boundary until a provider is implemented or the project configuration explicitly excludes it with a reason.

For the GTX 1080 Ti, setup defaults to the PyTorch 2.14 CUDA 12.6 wheel line. Setup also installs the media decode stack and DeepSpeed 0.19.6 by default. It does not use GPU fit as permission to change model scope.

## Generated state

Configuration is generated under `.state/` from the corpus and measured host:

- `.state/inventory/manifest.jsonl` — complete physical accounting
- `.state/corpus/decisions.jsonl` — every training admission decision
- `.state/corpus/manifest.jsonl` and `summary.json` — admitted, post-selection-deduplicated estate and provider coverage
- `.state/tokenizer/tokenizer-report.json`
- `.state/data/dataset-report.json`
- `.state/training.json` — generated textual model/training definition
- `.state/hardware.json`
- `.state/execution.json` — textual physical execution plan
- `.state/modalities.json` — image/audio/video model parameter counts, source counts, and per-provider execution plans

`execution.json` and the modality execution sections are prohibited from changing model dimensions. They record parameter count, GPU/RAM budgets, native-state estimate, ZeRO-3 working-set/CPU-state estimates, selected backend, and reason for selection.

## Training lanes

`train-all.sh` runs the admitted text lane and then every enabled/present media lane into independent checkpoint namespaces. Media is not flattened into text tokens:

- **Image**: every decoded frame, including animated-image frames, enters ViT-MAE.
- **Audio**: complete decoded mono PCM is divided into deterministic non-overlapping chunks for Wav2Vec2 pretraining; final partial chunks are padded with an attention mask.
- **Video**: every decoded frame enters sequential non-overlapping VideoMAE clips; the final clip is padded by its last observed frame.
- **Video audio**: every selected video container is probed. A legitimate absence of an audio stream is explicitly receipted; a container/decoder error fails the run. Present audio streams enter the same Wav2Vec2 objective as direct audio.

No filename proximity, directory proximity, shared basename, or invented graph relation is used to manufacture cross-modal supervision. Cross-modal alignment requires an explicit pairing/verification source.

## Scaling and checkpoints

The same training code can run with one device or multiple Accelerate processes. On a memory-constrained single GPU, `backend = "auto"` evaluates native full-state training and then ZeRO-3 parameter/optimizer CPU offload. An unexecutable exact model is a hard failure, not a smaller-model substitution.

Media checkpoints include the provider, exact plan fingerprint, selected source fingerprint, epoch/batch position, and accumulated work. `train-modalities.sh` resumes the latest matching checkpoint automatically; stale state is rejected.

The preflight memory figures are conservative launch bounds, not throughput claims. Runtime losses, actual batch/example counts, checkpoint progress, and final execution receipts are produced by real runs.

## Reinforcement learning

Reinforcement learning is intentionally not fabricated from corpus adjacency. The GRPO entrypoint requires both an explicit task dataset and an explicit reward implementation:

```bash
python -m laplace_conventional.rl \
  --model checkpoints/text/final-model \
  --dataset rewards/tasks.jsonl \
  --reward my_rewards:score \
  --output checkpoints/grpo
```

A reward must come from an actual verifier, environment outcome, preference/reward model, executable test, game result, or other explicitly selected signal. The repository will not relabel next-token or exact-string matching as reinforcement learning.

## Self-hosted runner on hart-server

```bash
sudo bash scripts/install-runner.sh
```

The installer uses the invoking operator's GitHub login to register a separate
`hart-server-conventional` runner as `laplace-runner:laplace-runner`. Its service
uses `UMask=0002`, its checkout is under `/build/laplace/work/conventional-runner`,
and all three temporary-directory variables point at
`/build/laplace/work/conventional-scratch`. The runner distribution is SHA-256
verified. Repeating setup retains the matching registration and repairs the
service configuration; registrations belonging to another repository are rejected.

Run the `runner-smoke` workflow to prove identity, group write, storage, and GPU
access without starting training.
