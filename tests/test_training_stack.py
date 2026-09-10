import json
from pathlib import Path

import sentencepiece as spm
import torch

from laplace_conventional.model import build_model


def test_transformer_provider_forward(tmp_path: Path):
    cfg = {
        "model": {
            "vocab_size": 512,
            "hidden_size": 128,
            "intermediate_size": 384,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
        }
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    model = build_model(path)
    ids = torch.randint(0, 512, (2, 16))
    out = model(input_ids=ids, labels=ids)
    assert out.logits.shape == (2, 16, 512)
    assert torch.isfinite(out.loss)


def test_sentencepiece_is_loadable_by_transformers(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("alpha beta gamma delta epsilon zeta eta theta\n" * 200), encoding="utf-8")
    prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=512,
        character_coverage=1.0,
        byte_fallback=True,
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer(vocab_file=str(prefix) + ".model", legacy=False)
    ids = tok.encode("alpha beta", add_special_tokens=True)
    assert ids
