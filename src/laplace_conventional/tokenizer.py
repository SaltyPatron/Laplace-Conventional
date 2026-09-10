from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer with fixed special-token ids."""

    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    graph_id: int = 259
    subject_id: int = 260
    relation_id: int = 261
    object_id: int = 262
    query_id: int = 263
    answer_id: int = 264

    @property
    def vocab_size(self) -> int:
        return 265

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytes(i for i in ids if 0 <= i < 256)
        return raw.decode("utf-8", errors="replace")

    def graph_record(self, subject: str, relation: str, object_: str) -> list[int]:
        out = [self.bos_id, self.graph_id, self.subject_id]
        out += self.encode(subject)
        out += [self.relation_id]
        out += self.encode(relation)
        out += [self.object_id]
        out += self.encode(object_)
        out += [self.eos_id]
        return out

    def graph_query(self, subject: str, relation: str) -> list[int]:
        out = [self.bos_id, self.query_id, self.subject_id]
        out += self.encode(subject)
        out += [self.relation_id]
        out += self.encode(relation)
        out += [self.answer_id]
        return out
