from __future__ import annotations

import argparse
import json
from pathlib import Path

from .configure import write_training_config
from .corpus import build_manifest, iter_trainable_records, load_manifest, write_manifest
from .execution import validate_execution_plan, write_execution_plan
from .hardware import write_probe
from .prepare import prepare
from .settings import load_settings, section
from .tokenizer import choose_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(prog="laplace-conventional")
    ap.add_argument("--project-config", required=True, help="TOML policy/configuration file")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory")
    p.add_argument("root")
    p.add_argument("--out", required=True)

    p = sub.add_parser("tokenizer")
    p.add_argument("root")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("root")
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard-bytes", type=int)

    p = sub.add_parser("derive-config")
    p.add_argument("--dataset-report", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("hardware")
    p.add_argument("--out", required=True)

    p = sub.add_parser("plan-execution")
    p.add_argument("--training-config", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require-fit", action="store_true")

    args = ap.parse_args()
    settings = load_settings(Path(args.project_config))
    corpus_cfg = section(settings, "corpus")
    records_cfg = section(settings, "records")
    tokenizer_cfg = section(settings, "tokenizer")
    derivation_cfg = section(settings, "derivation")
    training_cfg = section(settings, "training")
    execution_cfg = section(settings, "execution")
    validation_per_10k = int(corpus_cfg.get("validation_per_10k", 100))
    max_chars = int(records_cfg.get("max_chars", 64_000))

    if args.command == "inventory":
        entries, summary = build_manifest(Path(args.root), dedupe=bool(corpus_cfg.get("dedupe", True)))
        write_manifest(entries, summary, Path(args.out))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "tokenizer":
        root = Path(args.root)
        entries = load_manifest(Path(args.manifest))

        def factory():
            return iter_trainable_records(root, entries, max_chars=max_chars)

        candidates = [int(x) for x in tokenizer_cfg.get("vocab_candidates", [])]
        print(json.dumps(choose_tokenizer(
            factory,
            Path(args.out),
            candidates,
            validation_per_10k=validation_per_10k,
        ), indent=2, sort_keys=True))
        return

    if args.command == "prepare":
        shard_bytes = int(args.shard_bytes or execution_cfg.get("shard_bytes", 512 << 20))
        print(json.dumps(prepare(
            Path(args.root),
            Path(args.manifest),
            Path(args.tokenizer),
            Path(args.out),
            shard_bytes=shard_bytes,
            validation_per_10k=validation_per_10k,
            max_chars=max_chars,
        ), indent=2, sort_keys=True))
        return

    if args.command == "derive-config":
        print(json.dumps(write_training_config(
            Path(args.dataset_report),
            Path(args.out),
            tokens_per_parameter=float(derivation_cfg.get("tokens_per_parameter", 20.0)),
            context_quantile=str(derivation_cfg.get("context_quantile", "p95")),
            training_overrides=training_cfg,
        ), indent=2, sort_keys=True))
        return

    if args.command == "hardware":
        print(json.dumps(write_probe(Path(args.out)), indent=2, sort_keys=True))
        return

    if args.command == "plan-execution":
        plan = write_execution_plan(
            Path(args.training_config),
            Path(args.hardware),
            Path(args.out),
            execution_policy=execution_cfg,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        if args.require_fit:
            validate_execution_plan(plan)
        return
