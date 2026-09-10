from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .corpus import load_manifest
from .execution import validate_execution_plan
from .modality_data import AudioChunkDataset, ImageFrameDataset, VideoClipDataset, provider_entries
from .modality_models import build_provider_model
from .runtime import accelerator_for_plan


def _video_mask(batch_size: int, cfg: dict, device: torch.device) -> torch.Tensor:
    seq = (
        int(cfg["num_frames"]) // int(cfg["tubelet_size"])
        * (int(cfg["image_size"]) // int(cfg["patch_size"])) ** 2
    )
    count = max(1, min(seq - 1, round(seq * float(cfg["mask_ratio"]))))
    noise = torch.rand((batch_size, seq), device=device)
    order = noise.argsort(dim=1)
    mask = torch.zeros((batch_size, seq), dtype=torch.bool, device=device)
    mask.scatter_(1, order[:, :count], True)
    return mask


def _audio_pretraining_inputs(model, batch: dict, cfg: dict) -> dict:
    from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

    input_values = batch["input_values"]
    attention_mask = batch["attention_mask"]
    raw_length = input_values.shape[-1]
    feature_length = int(model._get_feat_extract_output_lengths(raw_length))
    feature_attention = model._get_feature_vector_attention_mask(feature_length, attention_mask)
    shape = (input_values.shape[0], feature_length)
    mask_np = _compute_mask_indices(
        shape,
        mask_prob=float(cfg["mask_time_prob"]),
        mask_length=int(cfg["mask_time_length"]),
        attention_mask=feature_attention.detach().cpu().numpy(),
        min_masks=1,
    )
    negatives_np = _sample_negative_indices(
        shape,
        num_negatives=int(cfg["num_negatives"]),
        mask_time_indices=mask_np,
    )
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "mask_time_indices": torch.as_tensor(mask_np, dtype=torch.bool, device=input_values.device),
        "sampled_negative_indices": torch.as_tensor(negatives_np, dtype=torch.long, device=input_values.device),
    }


def _dataset(modality: str, root: Path, entries, cfg: dict):
    if modality == "image":
        return ImageFrameDataset(root, entries, image_size=int(cfg["image_size"]))
    if modality == "audio":
        return AudioChunkDataset(
            root,
            entries,
            sample_rate=int(cfg["sample_rate"]),
            chunk_seconds=float(cfg["chunk_seconds"]),
        )
    if modality == "video":
        return VideoClipDataset(
            root,
            entries,
            image_size=int(cfg["image_size"]),
            num_frames=int(cfg["num_frames"]),
        )
    raise ValueError(modality)


def _forward(modality: str, model, batch: dict, cfg: dict):
    if modality == "image":
        return model(pixel_values=batch["pixel_values"])
    if modality == "audio":
        return model(**_audio_pretraining_inputs(model, batch, cfg))
    if modality == "video":
        mask = _video_mask(batch["pixel_values"].shape[0], cfg, batch["pixel_values"].device)
        return model(pixel_values=batch["pixel_values"], bool_masked_pos=mask)
    raise ValueError(modality)


def _mean_validation_loss(modality: str, model, loader, cfg: dict, accelerator) -> tuple[float | None, int]:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            result = _forward(modality, model, batch, cfg)
            gathered = accelerator.gather_for_metrics(result.loss.detach().reshape(1))
            total += float(gathered.sum().cpu())
            count += int(gathered.numel())
    model.train()
    return (total / count if count else None), count


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a selected conventional self-supervised modality provider.")
    ap.add_argument("--modality", choices=["image", "audio", "video"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--validation-per-10k", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    whole_plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    item = whole_plan[args.modality]
    if not item.get("enabled") or not item.get("present"):
        raise RuntimeError(f"{args.modality} provider is not present/enabled in modality plan")
    provider = str(item["provider"])
    cfg = dict(item["config"])
    execution = dict(item["execution"])
    validate_execution_plan(execution)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    gradient_clip = float(cfg.get("gradient_clip", 1.0))
    accelerator = accelerator_for_plan(
        execution,
        gradient_accumulation_steps=grad_accum,
        gradient_clip=gradient_clip,
    )

    entries = load_manifest(Path(args.manifest))
    train_entries = provider_entries(
        entries,
        provider,
        split="train",
        validation_per_10k=args.validation_per_10k,
    )
    validation_entries = provider_entries(
        entries,
        provider,
        split="validation",
        validation_per_10k=args.validation_per_10k,
    )
    if not train_entries:
        raise RuntimeError(f"no training files for provider {provider}")

    model = build_provider_model(provider, cfg)
    micro_batch = int(execution["micro_batch_size"])
    train_loader = DataLoader(
        _dataset(args.modality, Path(args.root), train_entries, cfg),
        batch_size=micro_batch,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        _dataset(args.modality, Path(args.root), validation_entries, cfg),
        batch_size=micro_batch,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    ) if validation_entries else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=(0.9, 0.95),
    )
    if validation_loader is None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model, optimizer, train_loader, validation_loader = accelerator.prepare(
            model, optimizer, train_loader, validation_loader
        )

    epochs = float(cfg.get("epochs", 1.0))
    if epochs <= 0 or not epochs.is_integer():
        raise ValueError("modality epochs must currently be a positive whole number to guarantee complete passes")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    total_batches = 0
    total_examples = 0
    last_loss: float | None = None

    model.train()
    for epoch in range(int(epochs)):
        epoch_batches = 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                result = _forward(args.modality, model, batch, cfg)
                accelerator.backward(result.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_batches += 1
            total_batches += 1
            total_examples += int(next(iter(batch.values())).shape[0]) * accelerator.num_processes
            last_loss = float(result.loss.detach())
            if accelerator.is_main_process and total_batches % int(cfg.get("log_every", 10)) == 0:
                print(json.dumps({
                    "event": "modality_train",
                    "modality": args.modality,
                    "provider": provider,
                    "epoch": epoch,
                    "batch": total_batches,
                    "examples": total_examples,
                    "loss": last_loss,
                    "backend": execution["backend"],
                }), flush=True)
        if epoch_batches == 0:
            raise RuntimeError(f"{provider} training dataset produced no batches")
        validation_loss, validation_batches = (None, 0)
        if validation_loader is not None:
            validation_loss, validation_batches = _mean_validation_loss(
                args.modality, model, validation_loader, cfg, accelerator
            )
        if accelerator.is_main_process:
            print(json.dumps({
                "event": "modality_epoch",
                "modality": args.modality,
                "epoch": epoch + 1,
                "train_batches": epoch_batches,
                "last_train_loss": last_loss,
                "validation_loss": validation_loss,
                "validation_batches": validation_batches,
            }), flush=True)

    accelerator.wait_for_everyone()
    state = out / "final-state"
    accelerator.save_state(state)
    if accelerator.is_main_process:
        final_model = out / "final-model"
        final_model.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            final_model,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=True,
        )
        (out / "training-receipt.json").write_text(json.dumps({
            "modality": args.modality,
            "provider": provider,
            "training_files": len(train_entries),
            "validation_files": len(validation_entries),
            "epochs": int(epochs),
            "batches": total_batches,
            "examples": total_examples,
            "last_loss": last_loss,
            "execution": execution,
        }, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
