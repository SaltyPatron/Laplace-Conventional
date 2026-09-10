from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .corpus import ManifestEntry
from .providers import provider_for


def path_split(path: str, *, validation_per_10k: int) -> str:
    if not 0 <= validation_per_10k < 10_000:
        raise ValueError("validation_per_10k must be in [0,10000)")
    bucket = int.from_bytes(hashlib.sha256(path.encode("utf-8")).digest()[:8], "big") % 10_000
    return "validation" if bucket < validation_per_10k else "train"


def provider_entries(
    entries: list[ManifestEntry],
    provider_name: str,
    *,
    split: str,
    validation_per_10k: int,
) -> list[ManifestEntry]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    return [
        entry for entry in entries
        if entry.duplicate_of is None
        and (provider := provider_for(entry)) is not None
        and provider.name == provider_name
        and path_split(entry.path, validation_per_10k=validation_per_10k) == split
    ]


def _worker_entries(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return entries
    return entries[info.id::info.num_workers]


def image_to_tensor(image, image_size: int) -> torch.Tensor:
    from PIL import Image, ImageOps

    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.pad(
        image,
        (image_size, image_size),
        method=Image.Resampling.BICUBIC,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    array = (array - mean) / std
    return torch.from_numpy(array.copy())


class ImageFrameDataset(IterableDataset):
    """Yield every frame of every selected image; animated images are not collapsed."""

    def __init__(self, root: Path, entries: list[ManifestEntry], *, image_size: int):
        self.root = root
        self.entries = entries
        self.image_size = image_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from PIL import Image, ImageSequence

        for entry in _worker_entries(self.entries):
            path = self.root / entry.path
            try:
                with Image.open(path) as image:
                    yielded = False
                    for frame in ImageSequence.Iterator(image):
                        yielded = True
                        yield {"pixel_values": image_to_tensor(frame.copy(), self.image_size)}
                    if not yielded:
                        yield {"pixel_values": image_to_tensor(image, self.image_size)}
            except Exception as exc:
                raise RuntimeError(f"failed to decode image {entry.path}: {exc}") from exc


def _ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_audio_stream(path: Path) -> bool:
    """Return whether a media container has a decodable first audio stream.

    A legitimate no-audio container is False. Container/decoder failures are
    errors and cannot be silently reclassified as "no audio".
    """
    cmd = [
        _ffmpeg_executable(),
        "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-frames:a", "1", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if proc.returncode == 0:
        return True
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    no_stream_markers = (
        "matches no streams",
        "does not contain any stream",
        "Stream map '0:a:0' matches no streams",
    )
    if any(marker in stderr for marker in no_stream_markers):
        return False
    raise RuntimeError(f"ffmpeg audio-stream probe failed for {path}: {stderr}")


def _pcm_chunks(path: Path, *, sample_rate: int, chunk_samples: int) -> Iterator[np.ndarray]:
    cmd = [
        _ffmpeg_executable(),
        "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None
    bytes_per_chunk = chunk_samples * 4
    buffer = bytearray()
    try:
        while True:
            block = proc.stdout.read(max(64 << 10, bytes_per_chunk - len(buffer)))
            if block:
                buffer.extend(block)
            while len(buffer) >= bytes_per_chunk:
                raw = bytes(buffer[:bytes_per_chunk])
                del buffer[:bytes_per_chunk]
                yield np.frombuffer(raw, dtype="<f4").copy()
            if not block:
                break
        if buffer:
            usable = len(buffer) - (len(buffer) % 4)
            if usable:
                yield np.frombuffer(bytes(buffer[:usable]), dtype="<f4").copy()
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        proc.stderr.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg audio decode failed for {path}: {stderr.strip()}")


class AudioChunkDataset(IterableDataset):
    """Decode complete audio streams to deterministic, non-overlapping PCM chunks."""

    def __init__(self, root: Path, entries: list[ManifestEntry], *, sample_rate: int, chunk_seconds: float):
        self.root = root
        self.entries = entries
        self.sample_rate = sample_rate
        self.chunk_samples = max(1, round(sample_rate * chunk_seconds))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for entry in _worker_entries(self.entries):
            path = self.root / entry.path
            for pcm in _pcm_chunks(path, sample_rate=self.sample_rate, chunk_samples=self.chunk_samples):
                valid = len(pcm)
                if valid == 0:
                    continue
                values = np.zeros((self.chunk_samples,), dtype=np.float32)
                attention = np.zeros((self.chunk_samples,), dtype=np.int64)
                take = min(valid, self.chunk_samples)
                values[:take] = pcm[:take]
                attention[:take] = 1
                yield {
                    "input_values": torch.from_numpy(values),
                    "attention_mask": torch.from_numpy(attention),
                }


class VideoClipDataset(IterableDataset):
    """Yield every decoded video frame in sequential non-overlapping clips."""

    def __init__(self, root: Path, entries: list[ManifestEntry], *, image_size: int, num_frames: int):
        self.root = root
        self.entries = entries
        self.image_size = image_size
        self.num_frames = num_frames

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        import imageio_ffmpeg
        from PIL import Image

        for entry in _worker_entries(self.entries):
            path = self.root / entry.path
            reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
            try:
                meta = next(reader)
                width, height = meta["size"]
                clip: list[torch.Tensor] = []
                last: torch.Tensor | None = None
                for raw in reader:
                    array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    frame = image_to_tensor(Image.fromarray(array), self.image_size)
                    last = frame
                    clip.append(frame)
                    if len(clip) == self.num_frames:
                        yield {"pixel_values": torch.stack(clip)}
                        clip = []
                if clip and last is not None:
                    clip.extend(last.clone() for _ in range(self.num_frames - len(clip)))
                    yield {"pixel_values": torch.stack(clip)}
            except Exception as exc:
                raise RuntimeError(f"failed to decode video {entry.path}: {exc}") from exc
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
