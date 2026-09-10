from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm

from .corpus import iter_trainable_records, load_manifest, split_name


@dataclass
class SplitStats:
    records: int = 0
    tokens: int = 0
    utf8_bytes: int = 0


class LengthHistogram:
    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.total = 0
        self.maximum = 0

    def add(self, value: int) -> None:
        self.counts[value] += 1
        self.total += 1
        self.maximum = max(self.maximum, value)

    def quantile(self, q: float) -> int:
        if not self.total:
            return 0
        target = max(1, int(np.ceil(q * self.total)))
        seen = 0
        for value in sorted(self.counts):
            seen += self.counts[value]
            if seen >= target:
                return value
        return self.maximum


class ShardWriter:
    """Bounded-memory binary shard writer; it never buffers a nominal shard as Python ints."""

    def __init__(self, directory: Path, split: str, dtype, max_tokens: int):
        self.directory = directory
        self.split = split
        self.dtype = np.dtype(dtype)
        self.max_tokens = max_tokens
        self.index = 0
        self.current_tokens = 0
        self.handle = None
        self.paths: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        path = self.directory / f"{self.split}-{self.index:06d}.bin"
        self.handle = path.open("wb")
        self.paths.append(path.name)
        self.current_tokens = 0

    def _close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.index += 1
            self.current_tokens = 0

    def add(self, tokens: list[int]) -> None:
        arr = np.asarray(tokens, dtype=self.dtype)
        offset = 0
        while offset < len(arr):
            if self.handle is None:
                self._open()
            capacity = self.max_tokens - self.current_tokens
            take = min(capacity, len(arr) - offset)
            arr[offset:offset + take].tofile(self.handle)
            offset += take
            self.current_tokens += take
            if self.current_tokens >= self.max_tokens:
                self._close()

    def close(self) -> None:
        self._close()


def prepare(
    root: Path,
    manifest: Path,
    tokenizer_model: Path,
    out_dir: Path,
    *,
    shard_bytes: int = 512 << 20,
    validation_per_10k: int = 100,
    max_chars: int = 64_000,
) -> dict:
    entries = load_manifest(manifest)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    dtype = np.uint16 if sp.vocab_size() - 1 <= np.iinfo(np.uint16).max else np.uint32
    max_tokens = max(1, shard_bytes // np.dtype(dtype).itemsize)
    writers = {s: ShardWriter(out_dir, s, dtype, max_tokens) for s in ("train", "validation")}
    stats = {s: SplitStats() for s in writers}
    lengths = LengthHistogram()
    for record in iter_trainable_records(root, entries, max_chars=max_chars):
        split = split_name(record, validation_per_10k=validation_per_10k)
        ids = [sp.bos_id(), *sp.encode(record.text, out_type=int), sp.eos_id()]
        writers[split].add(ids)
        st = stats[split]
        st.records += 1
        st.tokens += len(ids)
        st.utf8_bytes += len(record.text.encode("utf-8"))
        if split == "train":
            lengths.add(len(ids))
    for writer in writers.values():
        writer.close()
    report = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": sp.vocab_size(),
        "special_token_ids": {"unk": sp.unk_id(), "bos": sp.bos_id(), "eos": sp.eos_id(), "pad": sp.pad_id()},
        "splits": {k: asdict(v) | {"shards": writers[k].paths} for k, v in stats.items()},
        "train_record_length_tokens": {
            "p50": lengths.quantile(0.50),
            "p90": lengths.quantile(0.90),
            "p95": lengths.quantile(0.95),
            "p99": lengths.quantile(0.99),
            "max": lengths.maximum,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
