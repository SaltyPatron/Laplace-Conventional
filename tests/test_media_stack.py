from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch

from laplace_conventional.corpus import build_manifest
from laplace_conventional.modality_data import (
    AudioChunkDataset,
    ImageFrameDataset,
    VideoClipDataset,
    probe_audio_stream,
)
from laplace_conventional.modality_models import build_provider_model
from laplace_conventional.modality_train import _audio_pretraining_inputs, _video_mask


def _nonzero_grad(model) -> bool:
    return any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.parameters())


def test_vit_mae_forward_backward():
    cfg = {
        'image_size': 32, 'patch_size': 8,
        'hidden_size': 32, 'num_hidden_layers': 1,
        'num_attention_heads': 4, 'intermediate_size': 64,
        'decoder_hidden_size': 16, 'decoder_num_hidden_layers': 1,
        'decoder_num_attention_heads': 4, 'decoder_intermediate_size': 32,
        'mask_ratio': 0.5, 'norm_pix_loss': True,
    }
    model = build_provider_model('image-vit-mae', cfg)
    out = model(pixel_values=torch.randn(2, 3, 32, 32))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert _nonzero_grad(model)


def test_videomae_forward_backward():
    cfg = {
        'image_size': 32, 'patch_size': 8, 'num_frames': 4, 'tubelet_size': 2,
        'hidden_size': 32, 'num_hidden_layers': 1,
        'num_attention_heads': 4, 'intermediate_size': 64,
        'decoder_hidden_size': 16, 'decoder_num_hidden_layers': 1,
        'decoder_num_attention_heads': 4, 'decoder_intermediate_size': 32,
        'mask_ratio': 0.5, 'norm_pix_loss': True,
    }
    model = build_provider_model('video-videomae', cfg)
    pixels = torch.randn(2, 4, 3, 32, 32)
    mask = _video_mask(2, cfg, pixels.device)
    out = model(pixel_values=pixels, bool_masked_pos=mask)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert _nonzero_grad(model)


def test_wav2vec2_pretraining_forward_backward():
    cfg = {
        'hidden_size': 32, 'num_hidden_layers': 1,
        'num_attention_heads': 4, 'intermediate_size': 64,
        'hidden_dropout': 0.0, 'attention_dropout': 0.0, 'feat_proj_dropout': 0.0,
        'mask_time_prob': 0.2, 'mask_time_length': 2, 'num_negatives': 5,
        'codevector_dim': 16, 'proj_codevector_dim': 16,
        'num_codevector_groups': 2, 'num_codevectors_per_group': 16,
    }
    model = build_provider_model('audio-wav2vec2', cfg)
    batch = {
        'input_values': torch.randn(2, 8000),
        'attention_mask': torch.ones(2, 8000, dtype=torch.long),
    }
    prepared = _audio_pretraining_inputs(model, batch, cfg)
    out = model(**prepared)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert _nonzero_grad(model)


def _manifest_entries(root: Path):
    entries, _ = build_manifest(root, dedupe=False)
    return entries


def test_animated_image_yields_all_frames(tmp_path: Path):
    from PIL import Image

    path = tmp_path / 'two.gif'
    frames = [Image.new('RGB', (8, 8), (255, 0, 0)), Image.new('RGB', (8, 8), (0, 255, 0))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=20, loop=0)
    entries = [e for e in _manifest_entries(tmp_path) if e.kind == 'image']
    items = list(ImageFrameDataset(tmp_path, entries, image_size=16))
    assert len(items) == 2
    assert items[0]['pixel_values'].shape == (3, 16, 16)


def test_audio_decoder_preserves_all_pcm_chunks(tmp_path: Path):
    sample_rate = 8000
    samples = np.asarray([
        int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(1000)
    ], dtype='<i2')
    path = tmp_path / 'tone.wav'
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    entries = [e for e in _manifest_entries(tmp_path) if e.kind == 'audio']
    items = list(AudioChunkDataset(tmp_path, entries, sample_rate=sample_rate, chunk_seconds=0.05))
    assert len(items) == 3
    assert [int(x['attention_mask'].sum()) for x in items] == [400, 400, 200]


def _write_video(path: Path, frames: int = 4, size=(16, 16)) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path), size, fps=4, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p'
    )
    writer.send(None)
    try:
        for i in range(frames):
            array = np.full((size[1], size[0], 3), i * 40, dtype=np.uint8)
            writer.send(array.tobytes())
    finally:
        writer.close()


def test_video_frames_and_audio_probe_are_not_silent(tmp_path: Path):
    import imageio_ffmpeg

    video = tmp_path / 'plain.mp4'
    _write_video(video)
    assert probe_audio_stream(video) is False

    entries = [e for e in _manifest_entries(tmp_path) if e.kind == 'video']
    clips = list(VideoClipDataset(tmp_path, entries, image_size=16, num_frames=2))
    assert len(clips) == 2
    assert clips[0]['pixel_values'].shape == (2, 3, 16, 16)

    wav = tmp_path / 'tone.wav'
    with wave.open(str(wav), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(np.zeros(8000, dtype='<i2').tobytes())
    with_audio = tmp_path / 'with-audio.mp4'
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
        '-i', str(video), '-i', str(wav), '-c:v', 'copy', '-c:a', 'aac',
        '-shortest', str(with_audio),
    ]
    subprocess.run(cmd, check=True)
    assert probe_audio_stream(with_audio) is True
