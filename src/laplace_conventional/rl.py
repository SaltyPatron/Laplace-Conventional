from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def load_callable(spec: str):
    if ":" not in spec:
        raise ValueError("reward must be module:function")
    module, name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(spec)
    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO post-training with an explicit external reward function.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="JSON/JSONL dataset containing at least a prompt column")
    ap.add_argument("--reward", required=True, help="Python callable module:function; no built-in fabricated reward exists")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    reward = load_callable(args.reward)
    suffix = Path(args.dataset).suffix.lower()
    kind = "json" if suffix in {".json", ".jsonl", ".ndjson"} else None
    if kind is None:
        raise ValueError("RL dataset must be JSON/JSONL so reward inputs are explicit and inspectable")
    ds = load_dataset(kind, data_files=args.dataset, split="train")
    if "prompt" not in ds.column_names:
        raise ValueError("RL dataset must contain a prompt column")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward,
        train_dataset=ds,
        args=GRPOConfig(output_dir=args.output, max_steps=args.max_steps, report_to="none"),
    )
    trainer.train()
    trainer.save_model(args.output)
