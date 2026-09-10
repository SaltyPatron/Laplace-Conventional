from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import sentencepiece as spm

from .corpus import iter_trainable_records, load_manifest, split_name


@dataclass
class SplitStats:
    records: int = 0
    tokens: int = 0
    utf8_bytes: int = 0


class ShardWriter:
    def __init__(self, directory: Path, split: str, dtype, max_tokens: int):
        self.directory = directory
        self.split = split
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.buffer: list[int] = []
        self.index = 0
        self.paths: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, tokens: list[int]) -> None:
        self.buffer.extend(tokens)
        while len(self.buffer) >= self.max_tokens:
            self._flush(self.max_tokens)

    def _flush(self, n: int | None = None) -> None:
        if not self.buffer:
            return
        n = n or len(self.buffer)
        data = np.asarray(self.buffer[:n], dtype=self.dtype)
        del self.buffer[:n]
        path = self.directory / f"{self.split}-{self.index:06d}.bin"
        data.tofile(path)
        self.paths.append(path.name)
        self.index += 1

    def close(self) -> None:
        self._flush()


def prepare(root: Path, manifest: Path, tokenizer_model: Path, out_dir: Path, *, shard_bytes: int = 512 << 20) -> dict:
    entries = load_manifest(manifest)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    dtype = np.uint16 if sp.vocab_size() <= np.iinfo(np.uint16).max else np.uint32
    max_tokens = max(1, shard_bytes // np.dtype(dtype).itemsize)
    writers = {s: ShardWriter(out_dir, s, dtype, max_tokens) for s in ("train", "validation")}
    stats = {s: SplitStats() for s in writers}
    lengths: list[int] = []
    for record in iter_trainable_records(root, entries):
        split = split_name(record)
        ids = [sp.bos_id(), *sp.encode(record.text, out_type=int), sp.eos_id()]
        writers[split].add(ids)
        st = stats[split]
        st.records += 1
        st.tokens += len(ids)
        st.utf8_bytes += len(record.text.encode("utf-8"))
        if split == "train":
            lengths.append(len(ids))
    for writer in writers.values():
        writer.close()
    lengths.sort()
    def quantile(q: float) -> int:
        if not lengths:
            return 0
        i = min(len(lengths) - 1, max(0, math.ceil(q * len(lengths)) - 1))
        return lengths[i]
    report = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": sp.vocab_size(),
        "splits": {k: asdict(v) | {"shards": writers[k].paths} for k, v in stats.items()},
        "train_record_length_tokens": {"p50": quantile(0.50), "p90": quantile(0.90), "p95": quantile(0.95), "p99": quantile(0.99), "max": lengths[-1] if lengths else 0},
    }
    (out_dir / "dataset-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
