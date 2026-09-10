from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Iterator, Iterable

import torch

from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class GraphExample:
    subject: str
    relation: str
    object: str
    witness_count: int


@dataclass(frozen=True)
class TextExample:
    text: str
    entity_id_hex: str


def _connect(dsn: str | None = None):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for substrate-backed training") from exc
    return psycopg.connect(dsn or os.environ.get("LAPLACE_PG_DSN", "dbname=laplace"))


def _realized_batch(conn, ids: list[bytes | None]) -> list[str]:
    non_null = [x for x in ids if x is not None]
    if not non_null:
        return ["" for _ in ids]
    with conn.cursor() as cur:
        cur.execute("SELECT realize.batch(%s::bytea[], NULL)", (non_null,))
        labels = list(cur.fetchone()[0])
    it = iter(labels)
    return [next(it) if x is not None else "" for x in ids]


def iter_realized_entities(*, dsn: str | None = None, min_tier: int = 2, max_tier: int = 255, batch_rows: int = 1024, split: str = "train", split_modulus: int = 1000, validation_buckets: int = 10) -> Iterator[TextExample]:
    """Stream the existing compositional substrate as realized training text."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    with _connect(dsn) as conn:
        name = f"entity_stream_{os.getpid()}"
        with conn.cursor(name=name) as cur:
            cur.itersize = batch_rows
            cur.execute("SELECT id FROM laplace.entities WHERE tier BETWEEN %s AND %s ORDER BY id", (min_tier, max_tier))
            while True:
                rows = cur.fetchmany(batch_rows)
                if not rows:
                    break
                selected: list[bytes] = []
                for (entity_id,) in rows:
                    bucket = int.from_bytes(entity_id[:2], "big") % split_modulus
                    is_val = bucket < validation_buckets
                    if (split == "validation") == is_val:
                        selected.append(entity_id)
                if not selected:
                    continue
                labels = _realized_batch(conn, selected)
                for entity_id, text in zip(selected, labels):
                    if text:
                        yield TextExample(text=text, entity_id_hex=entity_id.hex())


def iter_graph_examples(*, dsn: str | None = None, batch_rows: int = 1024, split: str = "train", split_modulus: int = 1000, validation_buckets: int = 10) -> Iterator[GraphExample]:
    """Stream every realized consensus triple from the existing substrate."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    with _connect(dsn) as conn:
        name = f"graph_stream_{os.getpid()}"
        with conn.cursor(name=name) as cur:
            cur.itersize = batch_rows
            cur.execute("SELECT subject_id, type_id, object_id, witness_count FROM laplace.consensus WHERE object_id IS NOT NULL ORDER BY subject_id, type_id, object_id")
            while True:
                rows = cur.fetchmany(batch_rows)
                if not rows:
                    break
                kept = []
                flat_ids: list[bytes | None] = []
                for subject_id, type_id, object_id, witness_count in rows:
                    bucket = int.from_bytes(subject_id[:2], "big") % split_modulus
                    is_val = bucket < validation_buckets
                    if (split == "validation") != is_val:
                        continue
                    kept.append((subject_id, type_id, object_id, int(witness_count)))
                    flat_ids.extend((subject_id, type_id, object_id))
                if not kept:
                    continue
                labels = _realized_batch(conn, flat_ids)
                for i, (_, _, _, witness_count) in enumerate(kept):
                    s, r, o = labels[i * 3 : i * 3 + 3]
                    if s and r and o:
                        yield GraphExample(s, r, o, witness_count)


def mixed_token_stream(tokenizer: ByteTokenizer, *, dsn: str | None, graph_ratio: float, split: str) -> Iterator[list[int]]:
    if not 0.0 <= graph_ratio <= 1.0:
        raise ValueError("graph_ratio must be between 0 and 1")
    text_it = iter(iter_realized_entities(dsn=dsn, split=split))
    graph_it = iter(iter_graph_examples(dsn=dsn, split=split))
    selector = 0
    period = 1000
    graph_cut = round(graph_ratio * period)
    while True:
        use_graph = (selector % period) < graph_cut
        selector += 1
        try:
            if use_graph:
                g = next(graph_it)
                yield tokenizer.graph_record(g.subject, g.relation, g.object)
            else:
                t = next(text_it)
                yield tokenizer.encode(t.text, bos=True, eos=True)
        except StopIteration:
            return


def pack_blocks(examples: Iterable[list[int]], *, block_size: int, pad_id: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    buf: list[int] = []
    for ids in examples:
        buf.extend(ids)
        while len(buf) >= block_size + 1:
            chunk = buf[: block_size + 1]
            del buf[:block_size]
            yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
    if len(buf) >= 2:
        chunk = buf[: block_size + 1]
        x = torch.full((block_size,), pad_id, dtype=torch.long)
        y = torch.full((block_size,), -100, dtype=torch.long)
        n = len(chunk) - 1
        x[:n] = torch.tensor(chunk[:-1])
        y[:n] = torch.tensor(chunk[1:])
        yield x, y


def batch_blocks(blocks: Iterable[tuple[torch.Tensor, torch.Tensor]], batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    it = iter(blocks)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if len(batch) < batch_size:
            return
        yield torch.stack([x for x, _ in batch]), torch.stack([y for _, y in batch])
