from __future__ import annotations

import json
from pathlib import Path


def build_model(config_path: Path):
    """Instantiate a standard Hugging Face Llama causal LM from a generated config."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = json.loads(config_path.read_text(encoding="utf-8"))["model"]
    hf_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_attention_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tie_word_embeddings=True,
        use_cache=False,
    )
    return LlamaForCausalLM(hf_cfg)
