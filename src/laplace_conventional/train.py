from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from .data import mixed_token_stream, pack_blocks, batch_blocks
from .model import ConventionalTransformer, ModelConfig, parameter_count
from .tokenizer import ByteTokenizer


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(path: Path, model, optimizer, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "model_config": model.config_dict(), "train_config": cfg, "torch_version": torch.__version__}, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dsn", default=os.environ.get("LAPLACE_PG_DSN"))
    ap.add_argument("--resume")
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tok = ByteTokenizer()
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, **cfg["model"])
    model = ConventionalTransformer(model_cfg)
    device = _device()
    model.to(device)
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), betas=tuple(train_cfg.get("betas", [0.9, 0.95])), weight_decay=float(train_cfg.get("weight_decay", 0.1)), fused=bool(train_cfg.get("fused_adamw", False) and device.type == "cuda"))
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])

    max_steps = int(args.max_steps or train_cfg["max_steps"])
    warmup = int(train_cfg.get("warmup_steps", max(1, max_steps // 100)))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    grad_accum = int(train_cfg.get("gradient_accumulation", 1))
    batch_size = int(train_cfg["micro_batch_size"])
    graph_ratio = float(train_cfg.get("graph_ratio", 0.5))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    ckpt_every = int(train_cfg.get("checkpoint_every", 1000))
    out_dir = Path(train_cfg.get("output_dir", "/opt/laplace/conventional/checkpoints"))
    base_lr = float(train_cfg["learning_rate"])

    def lr_for(step: int) -> float:
        if step < warmup:
            return base_lr * (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * p))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    stream = mixed_token_stream(tok, dsn=args.dsn, graph_ratio=graph_ratio, split="train")
    blocks = pack_blocks(stream, block_size=model_cfg.max_seq_len, pad_id=tok.pad_id)
    batches = batch_blocks(blocks, batch_size=batch_size)
    print(json.dumps({"event": "start", "device": str(device), "torch": torch.__version__, "parameters": parameter_count(model), "model": model.config_dict(), "start_step": start_step, "max_steps": max_steps}), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    tokens_since = 0
    for step in range(start_step, max_steps):
        loss_sum = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(batches)
            except StopIteration:
                raise RuntimeError("substrate stream ended before max_steps")
            x, y = x.to(device), y.to(device)
            loss = model(x, y)["loss"] / grad_accum
            loss.backward()
            loss_sum += float(loss.detach()) * grad_accum
            tokens_since += int((y != -100).sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = lr_for(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % int(train_cfg.get("log_every", 10)) == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(json.dumps({"event": "train", "step": step + 1, "loss": loss_sum / grad_accum, "lr": lr, "tokens_per_second": tokens_since / elapsed, "cuda_allocated": torch.cuda.memory_allocated() if device.type == "cuda" else 0}), flush=True)
            t0, tokens_since = time.time(), 0
        if (step + 1) % ckpt_every == 0 or step + 1 == max_steps:
            _save_checkpoint(out_dir / f"step-{step+1:09d}.pt", model, optimizer, step + 1, cfg)


if __name__ == "__main__":
    main()
