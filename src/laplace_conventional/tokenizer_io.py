from __future__ import annotations

from pathlib import Path


def build_transformers_tokenizer(model_path: Path):
    """Wrap the trained SentencePiece model with Transformers' v5 SentencePiece backend."""
    from transformers import SentencePieceBackend

    return SentencePieceBackend(
        vocab_file=str(model_path),
        legacy=True,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )


def save_transformers_tokenizer(model_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_transformers_tokenizer(model_path).save_pretrained(output_dir)
