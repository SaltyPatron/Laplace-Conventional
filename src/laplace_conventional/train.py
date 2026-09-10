from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from .model import build_model


class TokenShardDataset(IterableDataset):
    def __init__(self, data_dir: Path, split: str, context: int, dtype: str):
        self.data_dir = data_dir
        self.split = split
        self.context = context
        self.dtype = np.dtype(dtype)
        self.shards = sorted(data_dir.glob(f"{split}-*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"no {split} shards in {data_dir}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        shards = self.shards if worker is None else self.shards[worker.id::worker.num_workers]
        width = self.context + 1
        for path in shards:
            data = np.memmap(path, mode="r", dtype=self.dtype)
            usable = (len(data) // width) * width
            for start in range(0, usable, width):
                block = np.asarray(data[start:start + width], dtype=np.int64)
                yield {"input_ids": torch.from_numpy(block[:-1].copy()), "labels": torch.from_numpy(block[1:].copy())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset-report", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from accelerate import Accelerator, PartialState

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = json.loads(Path(args.dataset_report).read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    context = int(train_cfg["context_length"])
    micro_batch = int(os.environ.get("MICRO_BATCH_SIZE", "1"))
    global_tokens = int(train_cfg["global_tokens_per_step"])
    world = PartialState().num_processes
    grad_accum = max(1, math.ceil(global_tokens / (micro_batch * context * world)))
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum)
    model = build_model(Path(args.config))
    dataset = TokenShardDataset(Path(args.data), "train", context, report["dtype"])
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]), betas=(0.9, 0.95))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if args.resume:
        accelerator.load_state(args.resume)

    total_tokens = int(report["splits"]["train"]["tokens"] * float(train_cfg["epochs"]))
    seen_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    next_checkpoint = int(train_cfg["checkpoint_tokens"])
    for step, batch in enumerate(loader, start=1):
        with accelerator.accumulate(model):
            result = model(**batch)
            accelerator.backward(result.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_tokens = int(batch["labels"].numel()) * accelerator.num_processes
        seen_tokens += batch_tokens
        if accelerator.is_main_process and step % 10 == 0:
            print(json.dumps({"step": step, "loss": float(result.loss.detach()), "seen_tokens": seen_tokens, "target_tokens": total_tokens}), flush=True)
        if seen_tokens >= next_checkpoint:
            accelerator.save_state(out / f"tokens-{seen_tokens:015d}")
            next_checkpoint += int(train_cfg["checkpoint_tokens"])
        if seen_tokens >= total_tokens:
            break
    accelerator.wait_for_everyone()
    accelerator.save_state(out / "final-state")
    if accelerator.is_main_process:
        from transformers import LlamaTokenizer
        final_model = out / "final-model"
        final_model.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_model, state_dict=accelerator.get_state_dict(model), safe_serialization=True)
        tokenizer = LlamaTokenizer(vocab_file=args.tokenizer, legacy=False)
        tokenizer.save_pretrained(final_model)
