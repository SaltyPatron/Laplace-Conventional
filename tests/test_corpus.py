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
    assert summary["unsupported_unique_bytes"] == 3
    records = list(iter_trainable_records(tmp_path, entries))
    assert any(r.text == '{"a":1,"b":2}' for r in records)
    assert all(r.source_path != "Text/copy.txt" for r in records)
    assert {split_name(r) for r in records} <= {"train", "validation"}
