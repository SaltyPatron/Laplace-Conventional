# Laplace-Conventional

A conventional neural baseline trained against the Laplace corpus after Laplace has already decomposed it.

This repository deliberately does **not** recreate source-specific WordNet/UD/FrameNet/etc. decomposers.  The training boundary is the existing Laplace substrate:

- `laplace.entities` provides content-addressed compositional entities.
- `realize.batch(...)` turns those existing entities into model-visible text sequences.
- `laplace.consensus` provides typed subject/relation/object structure and witness counts.
- deterministic ID partitioning provides train/validation separation without source-specific rules.

The model is a conventional decoder-only Transformer optimized with AdamW/backpropagation.  Pretraining mixes realized compositional sequences and serialized consensus triples.  `laplace_conventional.rl` then performs real clipped actor-critic policy-gradient updates on graph-completion episodes whose rewards are computed from held-out substrate targets.

Model dimensions are configuration, not a hard product limit.  `configs/1080ti.json` is an execution profile for the existing Pascal card; `configs/scale.json` is a larger shape using the same corpus and objectives.

## Runner

The existing `hart-server-refactor` runner is repository-scoped to `Laplace-Refactor`.  `scripts/bootstrap-runner.sh` configures a separate runner instance for this repository without altering that registration.

## Training

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-pascal.txt
.venv/bin/pip install -e .
.venv/bin/python -m laplace_conventional.train --config configs/1080ti.json
```

Checkpoints go outside the Git worktree under `/opt/laplace/conventional/checkpoints` by default.
