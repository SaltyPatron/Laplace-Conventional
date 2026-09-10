from __future__ import annotations

import gc
import json
from pathlib import Path

from .corpus import ManifestEntry, load_manifest
from .execution import derive_execution_plan, validate_execution_plan
from .modality_models import build_provider_model, parameter_stats
from .providers import AUDIO, VIDEO, provider_for


def _provider_entries(entries: list[ManifestEntry], provider_name: str) -> list[ManifestEntry]:
    return [
        entry for entry in entries
        if entry.duplicate_of is None
        and (provider := provider_for(entry)) is not None
        and provider.name == provider_name
    ]


def _layer_count(provider_name: str, cfg: dict) -> int:
    if provider_name == "image-vit-mae":
        return int(cfg["num_hidden_layers"]) + int(cfg["decoder_num_hidden_layers"])
    if provider_name == "audio-wav2vec2":
        return int(cfg["num_hidden_layers"])
    if provider_name == "video-videomae":
        return int(cfg["num_hidden_layers"]) + int(cfg["decoder_num_hidden_layers"])
    raise ValueError(provider_name)


def plan_modalities(
    entries: list[ManifestEntry],
    modality_settings: dict,
    hardware: dict,
    execution_policy: dict,
    *,
    require_fit: bool = True,
) -> dict:
    result: dict[str, dict] = {}
    video_cfg = dict(modality_settings.get("video", {})) if isinstance(modality_settings.get("video", {}), dict) else {}
    video_audio = bool(video_cfg.get("enabled", False) and video_cfg.get("include_audio_track", False))
    video_entries = _provider_entries(entries, VIDEO.name) if video_audio else []

    for modality in ("image", "audio", "video"):
        cfg = dict(modality_settings.get(modality, {}))
        if not cfg or not bool(cfg.get("enabled", False)):
            result[modality] = {"enabled": False, "files": 0, "bytes": 0}
            continue
        provider_name = str(cfg["provider"])
        provider_entries = _provider_entries(entries, provider_name)
        source_entries = provider_entries
        source_breakdown: dict[str, int] = {"direct_files": len(provider_entries)}
        if provider_name == AUDIO.name and video_audio:
            # Audio training inspects every selected video container for an audio
            # stream. Containers without audio are receipted and skipped later;
            # containers with audio are routed through the same Wav2Vec2 objective.
            source_entries = provider_entries + video_entries
            source_breakdown["candidate_video_container_files"] = len(video_entries)
        item = {
            "enabled": True,
            "provider": provider_name,
            "files": len(source_entries),
            "bytes": sum(e.size for e in source_entries),
            "source_breakdown": source_breakdown,
            "config": cfg,
        }
        if not source_entries:
            item["present"] = False
            result[modality] = item
            continue

        item["present"] = True
        model = build_provider_model(provider_name, cfg)
        total, largest = parameter_stats(model, transformer_layers=_layer_count(provider_name, cfg))
        del model
        gc.collect()
        item["model"] = {
            "parameter_estimate": total,
            "largest_layer_parameter_estimate": largest,
        }
        provider_execution = dict(execution_policy)
        if "micro_batch_size" in cfg:
            provider_execution["micro_batch_size"] = int(cfg["micro_batch_size"])
        plan = derive_execution_plan({"model": item["model"]}, hardware, execution_policy=provider_execution)
        if require_fit:
            validate_execution_plan(plan)
        item["execution"] = plan
        result[modality] = item
    return result


def write_modality_plan(
    manifest: Path,
    hardware_path: Path,
    output: Path,
    *,
    modality_settings: dict,
    execution_policy: dict,
    require_fit: bool = True,
) -> dict:
    entries = load_manifest(manifest)
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    plan = plan_modalities(
        entries,
        modality_settings,
        hardware,
        execution_policy,
        require_fit=require_fit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan
