from __future__ import annotations

import json
import math
from pathlib import Path


def _round_multiple(x: float, m: int) -> int:
    return max(m, int(round(x / m)) * m)


def _ffn_dim(d_model: int) -> int:
    return _round_multiple((8 / 3) * d_model, 256)


def estimate_llama_params(vocab: int, layers: int, hidden: int, ffn: int) -> int:
    return vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * ffn)


def derive_shape(target_params: int, vocab: int) -> dict:
    # Expand the candidate width until the deepest conventional candidate exceeds
    # the measured target. This prevents a hidden fixed ceiling from silently turning
    # a large corpus into the largest shape the source code happened to enumerate.
    max_layers = 128
    max_hidden = 256
    while estimate_llama_params(vocab, max_layers, max_hidden, _ffn_dim(max_hidden)) < target_params * 1.25:
        max_hidden += 128
    candidates = []
    for hidden in range(256, max_hidden + 1, 128):
        heads = max(1, hidden // 64)
        if hidden % heads:
            continue
        ffn = _ffn_dim(hidden)
        for layers in range(4, max_layers + 1, 2):
            params = estimate_llama_params(vocab, layers, hidden, ffn)
            candidates.append((abs(math.log(max(params, 1) / max(target_params, 1))), params, layers, hidden, heads, ffn))
    _, params, layers, hidden, heads, ffn = min(candidates)
    return {
        "architecture": "llama",
        "parameter_estimate": params,
        "parameter_target_error": (params - target_params) / max(target_params, 1),
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "intermediate_size": ffn,
    }


def derive_training_config(dataset_report: dict, *, tokens_per_parameter: float = 20.0, context_quantile: str = "p95") -> dict:
    train_tokens = int(dataset_report["splits"]["train"]["tokens"])
    vocab = int(dataset_report["vocab_size"])
    target_params = max(vocab * 256, int(train_tokens / tokens_per_parameter))
    shape = derive_shape(target_params, vocab)
    observed = int(dataset_report["train_record_length_tokens"].get(context_quantile, 0))
    context = 256
    while context < max(observed, 256):
        context *= 2
    return {
        "derivation": {
            "train_tokens": train_tokens,
            "tokens_per_parameter": tokens_per_parameter,
            "target_parameters": target_params,
            "context_rule": f"next_power_of_two({context_quantile}_record_tokens)",
            "observed_context_tokens": observed,
        },
        "model": shape | {"vocab_size": vocab, "max_position_embeddings": context, "rms_norm_eps": 1e-5, "rope_theta": 10000.0},
        "training": {
            "context_length": context,
            "global_tokens_per_step": max(context, 131072),
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "warmup_ratio": 0.01,
            "min_lr_ratio": 0.1,
            "gradient_clip": 1.0,
            "epochs": 1.0,
            "checkpoint_tokens": max(10_000_000, train_tokens // 100 if train_tokens else 10_000_000),
        },
    }


def write_training_config(dataset_report_path: Path, output: Path, **kwargs) -> dict:
    report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    config = derive_training_config(report, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return config
