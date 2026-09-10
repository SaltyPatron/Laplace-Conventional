from pathlib import Path

import laplace_conventional.corpus as corpus
from laplace_conventional.corpus import build_manifest, iter_trainable_records, split_name


def test_manifest_dedup_and_generic_records(tmp_path: Path):
    (tmp_path / "Text").mkdir()
    (tmp_path / "Data").mkdir()
    (tmp_path / "Text" / "a.txt").write_text("alpha\n\nbeta\n", encoding="utf-8")
    (tmp_path / "Text" / "copy.txt").write_text("alpha\n\nbeta\n", encoding="utf-8")
    (tmp_path / "Data" / "x.jsonl").write_text('{"b":2,"a":1}\n{"x":"y"}\n', encoding="utf-8")
    (tmp_path / "Data" / "blob.bin").write_bytes(b"\x00\x01\x02")
    entries, summary = build_manifest(tmp_path)
    assert summary["duplicate_files"] == 1
    assert summary["unsupported_selected_bytes"] == 3
    assert summary["inaccessible_selected_bytes"] == 0
    assert all(e.sha256 for e in entries)
    records = list(iter_trainable_records(tmp_path, entries, max_chars=6))
    assert any('"a":1' in r.text for r in records)
    assert all(r.source_path != "Text/copy.txt" for r in records)
    assert {split_name(r, validation_per_10k=100) for r in records} <= {"train", "validation"}


def test_no_dedupe_still_hashes_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("same", encoding="utf-8")
    (tmp_path / "b.txt").write_text("same", encoding="utf-8")
    entries, summary = build_manifest(tmp_path, dedupe=False)
    assert summary["duplicate_files"] == 0
    assert entries[0].sha256 == entries[1].sha256


def test_inaccessible_file_is_receipted_not_silently_skipped(tmp_path: Path, monkeypatch):
    path = tmp_path / "locked.tsv"
    path.write_text("a\tb\n", encoding="utf-8")
    real_sha = corpus._sha256

    def denied(candidate: Path, chunk: int = 8 << 20):
        if candidate == path:
            raise PermissionError(13, "Permission denied", str(candidate))
        return real_sha(candidate, chunk)

    monkeypatch.setattr(corpus, "_sha256", denied)
    entries, summary = build_manifest(tmp_path)
    entry = entries[0]
    assert entry.path == "locked.tsv"
    assert not entry.accessible
    assert entry.sha256 is None
    assert entry.error.startswith("PermissionError:")
    assert summary["inaccessible_selected_files"] == 1
    assert summary["inaccessible_selected_bytes"] == path.stat().st_size
    assert not summary["coverage_complete"]
