from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .model import build_model
from .tokenizer_io import save_transformers_tokenizer


class TokenShardDataset(IterableDataset):
    def __init__(self, data_dir: Path, split: str, context: int, dtype: str, pad_id: int):
        self.data_dir = data_dir
        self.split = split
        self.context = context
        self.dtype = np.dtype(dtype)
        self.pad_id = pad_id
        self.shards = sorted(data_dir.glob(f"{split}-*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"no {split} shards in {data_dir}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        shards = self.shards if worker is None else self.shards[worker.id::worker.num_workers]
        width = self.context + 1
        carry = np.empty((0,), dtype=self.dtype)
        for path in shards:
            data = np.memmap(path, mode="r", dtype=self.dtype)
            position = 0
            if len(carry):
                needed = width - len(carry)
                take = min(needed, len(data))
                carry = np.concatenate((carry, np.asarray(data[:take], dtype=self.dtype)))
                position = take
                if len(carry) == width:
                    block = carry.astype(np.int64, copy=False)
                    yield {"input_ids": torch.from_numpy(block[:-1].copy()), "labels": torch.from_numpy(block[1:].copy())}
                    carry = np.empty((0,), dtype=self.dtype)
            usable = ((len(data) - position) // width) * width
            stop = position + usable
            for start in range(position, stop, width):
                block = np.asarray(data[start:start + width], dtype=np.int64)
                yield {"input_ids": torch.from_numpy(block[:-1].copy()), "labels": torch.from_numpy(block[1:].copy())}
            if stop < len(data):
                carry = np.asarray(data[stop:], dtype=self.dtype).copy()
        if len(carry) >= 2:
            x = np.full((self.context,), self.pad_id, dtype=np.int64)
            y = np.full((self.context,), -100, dtype=np.int64)
            n = len(carry) - 1
            x[:n] = carry[:-1]
            y[:n] = carry[1:]
            yield {"input_ids": torch.from_numpy(x), "labels": torch.from_numpy(y)}


def _write_progress(path: Path, *, seen_tokens: int, batches_seen: int) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"seen_tokens": seen_tokens, "batches_seen": batches_seen}, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


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
    model.gradient_checkpointing_enable()
    pad_id = int(report["special_token_ids"]["pad"])
    dataset = TokenShardDataset(Path(args.data), "train", context, report["dtype"], pad_id)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.95),
    )
    total_tokens = int(report["splits"]["train"]["tokens"] * float(train_cfg["epochs"]))
    optimizer_steps = max(1, math.ceil(total_tokens / global_tokens))
    warmup_steps = max(1, round(optimizer_steps * float(train_cfg["warmup_ratio"])))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, optimizer_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    seen_tokens = 0
    batches_seen = 0
    if args.resume:
        resume = Path(args.resume)
        accelerator.load_state(resume)
        progress_path = resume / "progress.json"
        if not progress_path.exists():
            raise FileNotFoundError(f"resume checkpoint lacks {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        seen_tokens = int(progress["seen_tokens"])
        batches_seen = int(progress["batches_seen"])
        loader = accelerator.skip_first_batches(loader, num_batches=batches_seen)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_tokens = int(train_cfg["checkpoint_tokens"])
    next_checkpoint = ((seen_tokens // checkpoint_tokens) + 1) * checkpoint_tokens
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while seen_tokens < total_tokens:
        made_progress = False
        for batch in loader:
            made_progress = True
            batches_seen += 1
            with accelerator.accumulate(model):
                result = model(**batch)
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            valid_tokens = int((batch["labels"] != -100).sum().item()) * accelerator.num_processes
            seen_tokens += valid_tokens
            if accelerator.is_main_process and batches_seen % 10 == 0:
                print(json.dumps({
                    "batch": batches_seen,
                    "epoch": epoch,
                    "loss": float(result.loss.detach()),
                    "lr": scheduler.get_last_lr()[0],
                    "seen_tokens": seen_tokens,
                    "target_tokens": total_tokens,
                }), flush=True)
            if accelerator.sync_gradients and seen_tokens >= next_checkpoint:
                checkpoint = out / f"tokens-{seen_tokens:015d}"
                accelerator.save_state(checkpoint)
                if accelerator.is_main_process:
                    _write_progress(checkpoint / "progress.json", seen_tokens=seen_tokens, batches_seen=batches_seen)
                next_checkpoint += checkpoint_tokens
            if seen_tokens >= total_tokens and accelerator.sync_gradients:
                break
        if not made_progress:
            raise RuntimeError("training dataset produced no batches")
        epoch += 1
        loader = accelerator.prepare(DataLoader(
            dataset,
            batch_size=micro_batch,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        ))

    accelerator.wait_for_everyone()
    final_state = out / "final-state"
    accelerator.save_state(final_state)
    if accelerator.is_main_process:
        _write_progress(final_state / "progress.json", seen_tokens=seen_tokens, batches_seen=batches_seen)
        final_model = out / "final-model"
        final_model.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_model, state_dict=accelerator.get_state_dict(model), safe_serialization=True)
        save_transformers_tokenizer(Path(args.tokenizer), final_model)
