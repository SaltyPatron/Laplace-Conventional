from __future__ import annotations

import math
from typing import Any

import torch


def parameter_stats(model: torch.nn.Module, *, transformer_layers: int) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    # A conservative generic layer working-set estimate for ZeRO planning. It is
    # intentionally not derived from class-name matching or private HF internals.
    layer_estimate = min(total, math.ceil(total / max(1, transformer_layers) * 2.0))
    return total, layer_estimate


def build_image_model(cfg: dict[str, Any]):
    from transformers import ViTMAEConfig, ViTMAEForPreTraining

    config = ViTMAEConfig(
        image_size=int(cfg["image_size"]),
        patch_size=int(cfg["patch_size"]),
        num_channels=3,
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        intermediate_size=int(cfg["intermediate_size"]),
        decoder_hidden_size=int(cfg["decoder_hidden_size"]),
        decoder_num_hidden_layers=int(cfg["decoder_num_hidden_layers"]),
        decoder_num_attention_heads=int(cfg["decoder_num_attention_heads"]),
        decoder_intermediate_size=int(cfg["decoder_intermediate_size"]),
        mask_ratio=float(cfg["mask_ratio"]),
        norm_pix_loss=bool(cfg.get("norm_pix_loss", True)),
    )
    return ViTMAEForPreTraining(config)


def build_audio_model(cfg: dict[str, Any]):
    from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining

    config = Wav2Vec2Config(
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        intermediate_size=int(cfg["intermediate_size"]),
        hidden_dropout=float(cfg.get("hidden_dropout", 0.0)),
        attention_dropout=float(cfg.get("attention_dropout", 0.0)),
        feat_proj_dropout=float(cfg.get("feat_proj_dropout", 0.0)),
        mask_time_prob=float(cfg["mask_time_prob"]),
        mask_time_length=int(cfg["mask_time_length"]),
        num_negatives=int(cfg["num_negatives"]),
        codevector_dim=int(cfg.get("codevector_dim", 256)),
        proj_codevector_dim=int(cfg.get("proj_codevector_dim", 256)),
        num_codevector_groups=int(cfg.get("num_codevector_groups", 2)),
        num_codevectors_per_group=int(cfg.get("num_codevectors_per_group", 320)),
    )
    return Wav2Vec2ForPreTraining(config)


def build_video_model(cfg: dict[str, Any]):
    from transformers import VideoMAEConfig, VideoMAEForPreTraining

    config = VideoMAEConfig(
        image_size=int(cfg["image_size"]),
        patch_size=int(cfg["patch_size"]),
        num_channels=3,
        num_frames=int(cfg["num_frames"]),
        tubelet_size=int(cfg["tubelet_size"]),
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        intermediate_size=int(cfg["intermediate_size"]),
        decoder_hidden_size=int(cfg["decoder_hidden_size"]),
        decoder_num_hidden_layers=int(cfg["decoder_num_hidden_layers"]),
        decoder_num_attention_heads=int(cfg["decoder_num_attention_heads"]),
        decoder_intermediate_size=int(cfg["decoder_intermediate_size"]),
        norm_pix_loss=bool(cfg.get("norm_pix_loss", True)),
    )
    return VideoMAEForPreTraining(config)


def build_provider_model(provider: str, cfg: dict[str, Any]):
    if provider == "image-vit-mae":
        return build_image_model(cfg)
    if provider == "audio-wav2vec2":
        return build_audio_model(cfg)
    if provider == "video-videomae":
        return build_video_model(cfg)
    raise ValueError(f"unsupported modality model provider: {provider}")
