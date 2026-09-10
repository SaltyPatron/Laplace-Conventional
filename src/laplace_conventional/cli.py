from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import build_manifest, write_manifest, load_manifest, iter_trainable_records
from .tokenizer import choose_tokenizer
from .prepare import prepare
from .configure import write_training_config
from .hardware import write_probe


def main() -> None:
    ap = argparse.ArgumentParser(prog="laplace-conventional")
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
    p.add_argument("--shard-bytes", type=int, default=512 << 20)

    p = sub.add_parser("derive-config")
    p.add_argument("--dataset-report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens-per-parameter", type=float, default=20.0)
    p.add_argument("--context-quantile", choices=["p50", "p90", "p95", "p99", "max"], default="p95")

    p = sub.add_parser("hardware")
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.command == "inventory":
        entries, summary = build_manifest(Path(args.root))
        write_manifest(entries, summary, Path(args.out))
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "tokenizer":
        root = Path(args.root)
        entries = load_manifest(Path(args.manifest))
        def factory():
            return iter_trainable_records(root, entries)
        print(json.dumps(choose_tokenizer(factory, Path(args.out)), indent=2, sort_keys=True))
    elif args.command == "prepare":
        print(json.dumps(prepare(Path(args.root), Path(args.manifest), Path(args.tokenizer), Path(args.out), shard_bytes=args.shard_bytes), indent=2, sort_keys=True))
    elif args.command == "derive-config":
        print(json.dumps(write_training_config(Path(args.dataset_report), Path(args.out), tokens_per_parameter=args.tokens_per_parameter, context_quantile=args.context_quantile), indent=2, sort_keys=True))
    elif args.command == "hardware":
        print(json.dumps(write_probe(Path(args.out)), indent=2, sort_keys=True))
