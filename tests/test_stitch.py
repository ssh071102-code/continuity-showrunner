"""Local, zero-cost tests for the assembly side (owner: @zihanshen41-pixel).

No API key and no money: clips are synthesized with the bundled ffmpeg.
Skips cleanly if imageio-ffmpeg isn't installed. Runs hermetically — OUTPUTS
and _TMP are redirected into pytest's tmp_path, so the real outputs/ is untouched.

    pip install pytest imageio-ffmpeg
    pytest continuity-showrunner/tests
"""
import pytest

pytest.importorskip("imageio_ffmpeg")

from src import run, stitch  # noqa: E402  (after importorskip)


def _synth(tmp, n, seconds=2, size="480x480"):
    clips = []
    for i in range(n):
        p = tmp / f"c{i}.mp4"
        col = ["red", "green", "blue", "orange"][i % 4]
        r = stitch._run(["-f", "lavfi", "-i", f"color=c={col}:s={size}:d={seconds}:r=24",
                         *stitch._X264, str(p)])
        assert r.returncode == 0, r.stderr[-400:]
        clips.append(p)
    return clips


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Redirect stitch's output + temp dirs into tmp_path."""
    monkeypatch.setattr(stitch, "OUTPUTS", tmp_path)
    monkeypatch.setattr(stitch, "_TMP", tmp_path / "_tmp")
    return tmp_path


def test_empty_clips_raises(sandbox):
    with pytest.raises(ValueError):
        stitch.assemble([])


def test_hardcut_duration(sandbox):
    clips = _synth(sandbox, 2, seconds=2)
    film = stitch.assemble(clips, "f.mp4", crossfade=0)          # no title, hard cut
    _, _, d = stitch._probe(film)
    assert film.exists() and d is not None and 3.5 < d < 4.5     # 2 + 2


def test_title_captions_xfade_duration(sandbox):
    clips = _synth(sandbox, 3, seconds=3)
    film = stitch.assemble(clips, "f.mp4", title="The Test", logline="a small proof",
                           captions=["one", "two", "three"], crossfade=0.5)
    _, _, d = stitch._probe(film)
    # title 2.6 + 3*3 - 3*0.5 crossfades = 10.1s
    assert film.exists() and d is not None and 9.0 < d < 11.0


def test_build_film_wires_manifest(sandbox):
    clips = _synth(sandbox, 2, seconds=2)
    manifest = {"plan": {"title": "T", "logline": "lg",
                         "shots": [{"id": 1, "action": "a bird at dawn"},
                                   {"id": 2, "action": "the bird lifts off"}]}}
    film = run.build_film(clips, manifest, out="f.mp4", captions=True, crossfade=0)
    assert film.exists()


def test_caption_truncates():
    long = "A tiny bird stands at the very edge of an impossibly vast field at dawn"
    out = run._caption({"action": long})
    assert out.endswith("…") and len(out) <= 48
    assert run._caption({"action": "short line"}) == "short line"
    assert run._caption({}) == ""
