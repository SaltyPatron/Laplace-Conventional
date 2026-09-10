from pathlib import Path

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
