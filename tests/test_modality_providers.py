from laplace_conventional.corpus import ManifestEntry
from laplace_conventional.providers import AUDIO, IMAGE, VIDEO
from laplace_conventional.selection import apply_selection


def _entry(path: str, kind: str, size: int = 10) -> ManifestEntry:
    return ManifestEntry(
        path=path,
        source=path.split('/', 1)[0],
        size=size,
        sha256=(path.encode().hex() + '0' * 64)[:64],
        kind=kind,
        format=path.rsplit('.', 1)[-1],
        trainable=False,
        accessible=True,
    )


def test_media_bytes_are_unsupported_without_enabled_provider():
    entries = [_entry('x/a.png', 'image')]
    _, _, summary = apply_selection(entries, [], enabled_providers=set())
    assert summary['unsupported_selected_bytes'] == 10
    assert not summary['coverage_complete']


def test_image_audio_video_clear_only_through_enabled_providers():
    entries = [
        _entry('x/a.png', 'image', 11),
        _entry('x/a.wav', 'audio', 12),
        _entry('x/a.mp4', 'video', 13),
    ]
    enabled = {IMAGE.name, AUDIO.name, VIDEO.name}
    _, _, summary = apply_selection(
        entries,
        [],
        enabled_providers=enabled,
        require_video_audio_provider=True,
    )
    assert summary['unsupported_selected_bytes'] == 0
    assert summary['coverage_complete']
    assert summary['providers'][IMAGE.name]['bytes'] == 11
    assert summary['providers'][AUDIO.name]['bytes'] == 12
    assert summary['providers'][VIDEO.name]['bytes'] == 13
    assert summary['providers'][VIDEO.name]['audio_stream_provider_required'] is True


def test_video_with_audio_policy_fails_when_audio_provider_disabled():
    entries = [_entry('x/a.mp4', 'video', 13)]
    _, _, summary = apply_selection(
        entries,
        [],
        enabled_providers={VIDEO.name},
        require_video_audio_provider=True,
    )
    assert summary['unsupported_selected_bytes'] == 13
    assert not summary['coverage_complete']
