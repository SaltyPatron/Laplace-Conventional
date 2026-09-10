from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

import sentencepiece as spm

from .corpus import Record, split_name


@dataclass(frozen=True)
class TokenizerCandidate:
    requested_vocab: int
    actual_vocab: int
    model_bytes: int
    validation_tokens: int
    validation_utf8_bytes: int
    description_bits: float


def candidate_vocab_sizes() -> list[int]:
    # Candidate generation is intentionally separate from selection. Selection is by
    # measured description length, not by a hard-coded "32k is good" assertion.
    return [8_192, 16_384, 32_768, 65_536]


def _sentences(records: Iterable[Record], split: str) -> Iterator[str]:
    for record in records:
        if split_name(record) == split and record.text.strip():
            yield record.text


def train_candidate(records_factory, out_dir: Path, vocab_size: int) -> TokenizerCandidate:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"sp-{vocab_size}"
    spm.SentencePieceTrainer.train(
        sentence_iterator=_sentences(records_factory(), "train"),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=1.0,
        byte_fallback=True,
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        split_digits=False,
        allow_whitespace_only_pieces=True,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )
    proc = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    tokens = 0
    utf8_bytes = 0
    for text in _sentences(records_factory(), "validation"):
        tokens += len(proc.encode(text, out_type=int))
        utf8_bytes += len(text.encode("utf-8"))
    model_bytes = (prefix.with_suffix(".model")).stat().st_size
    bits = model_bytes * 8 + tokens * math.log2(max(proc.vocab_size(), 2))
    return TokenizerCandidate(vocab_size, proc.vocab_size(), model_bytes, tokens, utf8_bytes, bits)


def choose_tokenizer(records_factory, out_dir: Path, candidates: list[int] | None = None) -> dict:
    candidates = candidates or candidate_vocab_sizes()
    results = [train_candidate(records_factory, out_dir, size) for size in candidates]
    best = min(results, key=lambda x: x.description_bits)
    src = out_dir / f"sp-{best.requested_vocab}.model"
    dst = out_dir / "tokenizer.model"
    dst.write_bytes(src.read_bytes())
    report = {"selected": asdict(best), "candidates": [asdict(x) for x in results]}
    (out_dir / "tokenizer-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
