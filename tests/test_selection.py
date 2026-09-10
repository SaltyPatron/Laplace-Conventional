from pathlib import Path

from laplace_conventional.corpus import build_manifest
from laplace_conventional.selection import apply_selection


def test_selection_is_explicit_and_dedupe_happens_after_selection(tmp_path: Path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "good").mkdir()
    payload = b"same bytes"
    (tmp_path / "bad" / "first.txt").write_bytes(payload)
    (tmp_path / "good" / "second.txt").write_bytes(payload)

    entries, _ = build_manifest(tmp_path, dedupe=False)
    selected, decisions, summary = apply_selection(
        entries,
        [{"glob": "bad/**", "selected": False, "reason": "fixture/superseded input"}],
    )

    assert {d.path for d in decisions if not d.selected} == {"bad/first.txt"}
    assert len(selected) == 1
    assert selected[0].path == "good/second.txt"
    assert selected[0].duplicate_of is None
    assert summary["selected_unique_files"] == 1
    assert summary["excluded_files"] == 1
    assert summary["coverage_complete"]


def test_exclusion_requires_reason(tmp_path: Path):
    (tmp_path / "x.bin").write_bytes(b"\x00")
    entries, _ = build_manifest(tmp_path, dedupe=False)
    try:
        apply_selection(entries, [{"glob": "*.bin", "selected": False}])
    except ValueError as exc:
        assert "requires a reason" in str(exc)
    else:
        raise AssertionError("silent exclusions must be rejected")


def test_selected_unsupported_bytes_still_fail_coverage(tmp_path: Path):
    (tmp_path / "x.bin").write_bytes(b"\x00\x01")
    entries, _ = build_manifest(tmp_path, dedupe=False)
    _, _, summary = apply_selection(entries, [])
    assert summary["unsupported_selected_bytes"] == 2
    assert not summary["coverage_complete"]
