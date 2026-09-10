from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .corpus import ManifestEntry, load_manifest
from .execution import validate_execution_plan
from .modality_data import (
    AudioChunkDataset,
    ImageFrameDataset,
    VideoClipDataset,
    probe_audio_stream,
    provider_entries,
)
from .modality_models import build_provider_model
from .providers import AUDIO, VIDEO
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


def _feature_attention_mask(base_model, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=-1).to(dtype=torch.long)
    kernels = tuple(int(x) for x in base_model.config.conv_kernel)
    strides = tuple(int(x) for x in base_model.config.conv_stride)
    if len(kernels) != len(strides):
        raise RuntimeError("Wav2Vec2 conv kernel/stride metadata mismatch")
    feature_lengths = lengths
    max_feature_length = attention_mask.shape[-1]
    for kernel, stride in zip(kernels, strides):
        feature_lengths = torch.div(feature_lengths - kernel, stride, rounding_mode="floor") + 1
        max_feature_length = (max_feature_length - kernel) // stride + 1
    feature_lengths = feature_lengths.clamp(min=0, max=max_feature_length)
    positions = torch.arange(max_feature_length, device=attention_mask.device)[None, :]
    return positions < feature_lengths[:, None]


def _audio_pretraining_inputs(model, batch: dict, cfg: dict) -> dict:
    from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

    input_values = batch["input_values"]
    attention_mask = batch["attention_mask"]
    base_model = getattr(model, "module", model)
    feature_attention = _feature_attention_mask(base_model, attention_mask)
    shape = tuple(feature_attention.shape)
    mask_np = _compute_mask_indices(
        shape,
        mask_prob=float(cfg["mask_time_prob"]),
        mask_length=int(cfg["mask_time_length"]),
        attention_mask=feature_attention.detach().cpu(),
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


def _dataset(modality: str, root: Path, entries: list[ManifestEntry], cfg: dict):
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


def _video_audio_entries(root: Path, candidates: list[ManifestEntry]) -> tuple[list[ManifestEntry], list[str]]:
    with_audio: list[ManifestEntry] = []
    without_audio: list[str] = []
    for entry in candidates:
        if probe_audio_stream(root / entry.path):
            with_audio.append(entry)
        else:
            without_audio.append(entry.path)
    return with_audio, without_audio


def _resolved_entries(
    modality: str,
    root: Path,
    entries: list[ManifestEntry],
    provider: str,
    whole_plan: dict,
    validation_per_10k: int,
) -> tuple[list[ManifestEntry], list[ManifestEntry], dict]:
    train_entries = provider_entries(
        entries, provider, split="train", validation_per_10k=validation_per_10k
    )
    validation_entries = provider_entries(
        entries, provider, split="validation", validation_per_10k=validation_per_10k
    )
    receipt: dict = {
        "direct_training_files": len(train_entries),
        "direct_validation_files": len(validation_entries),
    }
    if modality != "audio":
        return train_entries, validation_entries, receipt

    video_item = whole_plan.get("video", {})
    include_video_audio = bool(
        video_item.get("enabled")
        and video_item.get("present")
        and video_item.get("config", {}).get("include_audio_track", False)
    )
    if not include_video_audio:
        receipt["video_audio_enabled"] = False
        return train_entries, validation_entries, receipt

    candidate_train = provider_entries(
        entries, VIDEO.name, split="train", validation_per_10k=validation_per_10k
    )
    candidate_validation = provider_entries(
        entries, VIDEO.name, split="validation", validation_per_10k=validation_per_10k
    )
    video_train, no_audio_train = _video_audio_entries(root, candidate_train)
    video_validation, no_audio_validation = _video_audio_entries(root, candidate_validation)
    receipt.update({
        "video_audio_enabled": True,
        "video_training_candidates": len(candidate_train),
        "video_validation_candidates": len(candidate_validation),
        "video_training_with_audio": len(video_train),
        "video_validation_with_audio": len(video_validation),
        "video_without_audio_training": no_audio_train,
        "video_without_audio_validation": no_audio_validation,
    })
    return train_entries + video_train, validation_entries + video_validation, receipt


def _fingerprint(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_progress(path: Path, progress: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "progress.json.tmp"
    tmp.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path / "progress.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a selected conventional self-supervised modality provider.")
    ap.add_argument("--modality", choices=["image", "audio", "video"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--validation-per-10k", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume")
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

    root = Path(args.root)
    entries = load_manifest(Path(args.manifest))
    train_entries, validation_entries, source_receipt = _resolved_entries(
        args.modality, root, entries, provider, whole_plan, args.validation_per_10k
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plan_fingerprint = _fingerprint(item)
    source_fingerprint = _fingerprint(source_receipt)
    if accelerator.is_main_process:
        (out / "source-receipt.json").write_text(
            json.dumps(source_receipt, indent=2, sort_keys=True), encoding="utf-8"
        )

    if not train_entries:
        if args.modality == "audio" and not validation_entries and source_receipt.get("video_audio_enabled"):
            if accelerator.is_main_process:
                (out / "training-receipt.json").write_text(json.dumps({
                    "modality": "audio",
                    "provider": AUDIO.name,
                    "status": "no_selected_audio_streams_present",
                    "source_receipt": source_receipt,
                }, indent=2, sort_keys=True), encoding="utf-8")
            return
        raise RuntimeError(f"no training files for provider {provider}")

    model = build_provider_model(provider, cfg)
    micro_batch = int(execution["micro_batch_size"])
    train_loader = DataLoader(
        _dataset(args.modality, root, train_entries, cfg),
        batch_size=micro_batch,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        _dataset(args.modality, root, validation_entries, cfg),
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
        raise ValueError("modality epochs must be a positive whole number to guarantee complete passes")
    epochs_int = int(epochs)
    checkpoint_seconds = max(60.0, float(cfg.get("checkpoint_minutes", 30.0)) * 60.0)
    start_epoch = 0
    batch_in_epoch = 0
    total_batches = 0
    total_examples = 0

    if args.resume:
        resume = Path(args.resume)
        progress_file = resume / "progress.json"
        if not progress_file.exists():
            raise FileNotFoundError(f"resume checkpoint lacks {progress_file}")
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
        if progress.get("provider") != provider:
            raise RuntimeError("resume provider differs from current modality plan")
        if progress.get("plan_fingerprint") != plan_fingerprint:
            raise RuntimeError("resume execution/model plan differs from current generated plan")
        if progress.get("source_fingerprint") != source_fingerprint:
            raise RuntimeError("resume source set differs from current selected corpus")
        accelerator.load_state(resume)
        start_epoch = int(progress["epoch"])
        batch_in_epoch = int(progress["batch_in_epoch"])
        total_batches = int(progress["total_batches"])
        total_examples = int(progress["total_examples"])
        if start_epoch >= epochs_int:
            return

    last_loss: float | None = None
    next_checkpoint_at = time.monotonic() + checkpoint_seconds
    model.train()
    for epoch in range(start_epoch, epochs_int):
        epoch_batches = batch_in_epoch if epoch == start_epoch else 0
        active_loader = train_loader
        if epoch == start_epoch and batch_in_epoch:
            active_loader = accelerator.skip_first_batches(train_loader, batch_in_epoch)
        for batch in active_loader:
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
            if accelerator.sync_gradients and time.monotonic() >= next_checkpoint_at:
                checkpoint = out / "checkpoints" / f"epoch-{epoch:03d}-batch-{epoch_batches:09d}"
                accelerator.save_state(checkpoint)
                if accelerator.is_main_process:
                    progress = {
                        "provider": provider,
                        "plan_fingerprint": plan_fingerprint,
                        "source_fingerprint": source_fingerprint,
                        "epoch": epoch,
                        "batch_in_epoch": epoch_batches,
                        "total_batches": total_batches,
                        "total_examples": total_examples,
                    }
                    _write_progress(checkpoint, progress)
                    (out / "latest-checkpoint.txt").write_text(str(checkpoint), encoding="utf-8")
                next_checkpoint_at = time.monotonic() + checkpoint_seconds
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
        batch_in_epoch = 0

    accelerator.wait_for_everyone()
    state = out / "final-state"
    accelerator.save_state(state)
    if accelerator.is_main_process:
        _write_progress(state, {
            "provider": provider,
            "plan_fingerprint": plan_fingerprint,
            "source_fingerprint": source_fingerprint,
            "epoch": epochs_int,
            "batch_in_epoch": 0,
            "total_batches": total_batches,
            "total_examples": total_examples,
        })
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
            "epochs": epochs_int,
            "batches": total_batches,
            "examples": total_examples,
            "last_loss": last_loss,
            "source_receipt": source_receipt,
            "execution": execution,
        }, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
