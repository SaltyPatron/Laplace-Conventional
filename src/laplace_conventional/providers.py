from __future__ import annotations

from dataclasses import dataclass

from .corpus import ManifestEntry


@dataclass(frozen=True)
class TrainingProvider:
    name: str
    modality: str
    objective: str
    model_family: str


TEXT = TrainingProvider(
    name="text-causal-lm",
    modality="text",
    objective="next-token causal language modeling",
    model_family="llama",
)
IMAGE = TrainingProvider(
    name="image-vit-mae",
    modality="image",
    objective="masked patch pixel reconstruction",
    model_family="vit-mae",
)
AUDIO = TrainingProvider(
    name="audio-wav2vec2",
    modality="audio",
    objective="masked latent contrastive prediction with learned quantization",
    model_family="wav2vec2",
)
VIDEO = TrainingProvider(
    name="video-videomae",
    modality="video",
    objective="masked spatiotemporal patch pixel reconstruction",
    model_family="videomae",
)

PROVIDERS = {p.name: p for p in (TEXT, IMAGE, AUDIO, VIDEO)}


def provider_for(entry: ManifestEntry) -> TrainingProvider | None:
    if not entry.accessible:
        return None
    if entry.trainable:
        return TEXT
    if entry.kind == "image":
        return IMAGE
    if entry.kind == "audio":
        return AUDIO
    if entry.kind == "video":
        return VIDEO
    return None


def provider_summary(entries: list[ManifestEntry]) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for entry in entries:
        if entry.duplicate_of is not None:
            continue
        provider = provider_for(entry)
        name = provider.name if provider else "unsupported"
        bucket = result.setdefault(name, {"files": 0, "bytes": 0})
        bucket["files"] = int(bucket["files"]) + 1
        bucket["bytes"] = int(bucket["bytes"]) + entry.size
        if provider:
            bucket["modality"] = provider.modality
            bucket["objective"] = provider.objective
            bucket["model_family"] = provider.model_family
    return result
