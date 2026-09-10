from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".tex", ".adoc", ".org", ".html", ".htm", ".css",
    ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs",
    ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".swift", ".kt", ".kts",
    ".rb", ".php", ".pl", ".pm", ".lua", ".r", ".scala", ".sh", ".bash", ".zsh",
    ".fish", ".ps1", ".bat", ".cmd", ".sql", ".graphql", ".proto", ".cmake",
    ".make", ".mk", ".toml", ".ini", ".cfg", ".conf", ".properties", ".env",
    ".yaml", ".yml", ".jsonnet", ".nix", ".vim", ".asm", ".s", ".f", ".f90",
    ".f95", ".m", ".jl", ".dart", ".ex", ".exs", ".erl", ".hrl", ".hs", ".lhs",
    ".clj", ".cljs", ".edn", ".fs", ".fsx", ".vb", ".v", ".vhd", ".vhdl", ".sv",
    ".svh", ".sol", ".zig", ".roc", ".ml", ".mli", ".pas", ".p", ".pro", ".awk",
    ".sed", ".diff", ".patch", ".gitignore", ".gitattributes", ".dockerfile", ".pgn",
    ".fen", ".ebnf", ".bnf", ".grammar", ".lex", ".yacc", ".l", ".g4", ".ttl",
    ".rdf", ".nt", ".nq", ".sparql", ".po", ".pot", ".dtd", ".xsd",
}
STRUCTURED_EXTENSIONS = {
    ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
    ".xml": "xml", ".csv": "csv", ".tsv": "tsv", ".conllu": "conllu",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    source: str
    size: int
    sha256: str
    kind: str
    format: str
    trainable: bool
    duplicate_of: str | None = None
    utf8: bool | None = None


@dataclass(frozen=True)
class Record:
    source_path: str
    source: str
    record_index: int
    format: str
    text: str

    @property
    def stable_key(self) -> str:
        raw = f"{self.source_path}\0{self.record_index}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _utf8_probe(path: Path, sample_bytes: int = 1 << 20) -> bool:
    with path.open("rb") as f:
        sample = f.read(sample_bytes)
    if b"\0" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def classify(path: Path) -> tuple[str, str, bool, bool | None]:
    suffix = path.suffix.lower()
    if suffix in STRUCTURED_EXTENSIONS:
        ok = _utf8_probe(path)
        return "structured", STRUCTURED_EXTENSIONS[suffix], ok, ok
    if suffix in IMAGE_EXTENSIONS:
        return "image", suffix[1:], False, None
    if suffix in AUDIO_EXTENSIONS:
        return "audio", suffix[1:], False, None
    if suffix in VIDEO_EXTENSIONS:
        return "video", suffix[1:], False, None
    if suffix in TEXT_EXTENSIONS or path.name.lower() in {"makefile", "dockerfile", "license", "readme"}:
        ok = _utf8_probe(path)
        return "text", suffix[1:] or path.name.lower(), ok, ok
    ok = _utf8_probe(path)
    if ok:
        mime, _ = mimetypes.guess_type(path.name)
        return "text", mime or "utf8", True, True
    return "binary", suffix[1:] or "binary", False, False


def build_manifest(root: Path, *, hash_duplicates: bool = True) -> tuple[list[ManifestEntry], dict]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries: list[ManifestEntry] = []
    first_by_hash: dict[str, str] = {}
    totals: dict[str, dict[str, int]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        source = rel.split("/", 1)[0]
        digest = _sha256(path) if hash_duplicates else ""
        duplicate_of = first_by_hash.get(digest) if digest else None
        if digest and duplicate_of is None:
            first_by_hash[digest] = rel
        kind, fmt, trainable, utf8 = classify(path)
        entry = ManifestEntry(
            path=rel, source=source, size=path.stat().st_size, sha256=digest,
            kind=kind, format=fmt, trainable=trainable,
            duplicate_of=duplicate_of, utf8=utf8,
        )
        entries.append(entry)
        t = totals.setdefault(kind, {"files": 0, "bytes": 0, "unique_files": 0, "unique_bytes": 0})
        t["files"] += 1
        t["bytes"] += entry.size
        if duplicate_of is None:
            t["unique_files"] += 1
            t["unique_bytes"] += entry.size
    unsupported_bytes = sum(e.size for e in entries if not e.trainable and e.duplicate_of is None)
    summary = {
        "root": str(root),
        "files": len(entries),
        "bytes": sum(e.size for e in entries),
        "unique_files": sum(1 for e in entries if e.duplicate_of is None),
        "unique_bytes": sum(e.size for e in entries if e.duplicate_of is None),
        "duplicate_files": sum(1 for e in entries if e.duplicate_of is not None),
        "trainable_unique_bytes": sum(e.size for e in entries if e.trainable and e.duplicate_of is None),
        "unsupported_unique_bytes": unsupported_bytes,
        "coverage_complete": unsupported_bytes == 0,
        "kinds": totals,
        "sources": {},
    }
    for e in entries:
        s = summary["sources"].setdefault(e.source, {"files": 0, "bytes": 0, "trainable_bytes": 0, "unsupported_bytes": 0})
        s["files"] += 1
        s["bytes"] += e.size
        if e.duplicate_of is None:
            if e.trainable:
                s["trainable_bytes"] += e.size
            else:
                s["unsupported_bytes"] += e.size
    return entries, summary


def write_manifest(entries: Iterable[ManifestEntry], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> list[ManifestEntry]:
    with path.open("r", encoding="utf-8") as f:
        return [ManifestEntry(**json.loads(line)) for line in f if line.strip()]


def _read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _chunk_text(text: str, max_chars: int = 64_000) -> Iterator[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start:
                boundary = text.rfind("\n", start, end)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end]
        if chunk.strip():
            yield chunk
        start = end


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def iter_records(root: Path, entry: ManifestEntry, *, max_chars: int = 64_000) -> Iterator[Record]:
    if not entry.trainable or entry.duplicate_of is not None:
        return
    path = root / entry.path
    idx = 0
    if entry.format == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield Record(entry.path, entry.source, idx, entry.format, _canonical_json(json.loads(line)))
                idx += 1
        return
    if entry.format == "json":
        value = json.loads(_read_utf8(path))
        values = value if isinstance(value, list) else [value]
        for item in values:
            yield Record(entry.path, entry.source, idx, entry.format, _canonical_json(item))
            idx += 1
        return
    if entry.format in {"csv", "tsv"}:
        delimiter = "," if entry.format == "csv" else "\t"
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames:
                for row in reader:
                    yield Record(entry.path, entry.source, idx, entry.format, _canonical_json(row))
                    idx += 1
            else:
                f.seek(0)
                for row in csv.reader(f, delimiter=delimiter):
                    yield Record(entry.path, entry.source, idx, entry.format, _canonical_json(row))
                    idx += 1
        return
    if entry.format == "xml":
        depth = 0
        for event, elem in ET.iterparse(path, events=("start", "end")):
            if event == "start":
                depth += 1
                continue
            if depth == 2:
                text = ET.tostring(elem, encoding="unicode")
                if text.strip():
                    yield Record(entry.path, entry.source, idx, entry.format, text)
                    idx += 1
                elem.clear()
            depth -= 1
        return
    if entry.format == "conllu":
        buf: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    buf.append(line.rstrip("\n"))
                elif buf:
                    yield Record(entry.path, entry.source, idx, entry.format, "\n".join(buf))
                    idx += 1
                    buf.clear()
        if buf:
            yield Record(entry.path, entry.source, idx, entry.format, "\n".join(buf))
        return
    for chunk in _chunk_text(_read_utf8(path), max_chars=max_chars):
        yield Record(entry.path, entry.source, idx, entry.format, chunk)
        idx += 1


def iter_trainable_records(root: Path, entries: Iterable[ManifestEntry], *, max_chars: int = 64_000) -> Iterator[Record]:
    for entry in entries:
        yield from iter_records(root, entry, max_chars=max_chars)


def split_name(record: Record, *, validation_per_10k: int = 100) -> str:
    if not 0 <= validation_per_10k < 10_000:
        raise ValueError("validation_per_10k must be in [0, 10000)")
    bucket = int(record.stable_key[:8], 16) % 10_000
    return "validation" if bucket < validation_per_10k else "train"
