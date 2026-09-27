"""Assemble shot clips into the finished film.  OWNER: @zihanshen41-pixel.

Demo-grade assembly on imageio-ffmpeg's bundled ffmpeg (no system ffmpeg,
no API, no cost):  pip install imageio-ffmpeg

Pipeline — every stage is local:
  1. NORMALIZE every clip to identical codec/size/fps/pixfmt, so join never
     chokes on mismatched inputs (the old `-c copy` concat did). Optionally
     burns a per-shot caption (lower third) during this pass.
  2. TITLE CARD rendered from the film's title + logline, prepended.
  3. JOIN with gentle crossfades (xfade); falls back to clean hard cuts if
     xfade is unavailable or a duration can't be probed.
  4. MUSIC bed laid under the whole film (optional; path only — no audio asset
     is ever committed to the repo).

Everything degrades gracefully: missing font -> captions skipped, xfade fails
-> hard cut, missing music -> silent. assemble() always yields a valid film.

drawtext on Windows can't take an absolute path (the drive colon collides with
ffmpeg's option separator), so we copy the font + write the text files into the
temp dir and run ffmpeg with cwd there, referencing them by bare basename.

Standalone (no generation side, no key, no money):
    python -m src.stitch     # synthesizes test clips if outputs/ has none
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# Stand-alone by design: stitch is pure local ffmpeg, so it must NOT pull in the
# generation side (src.qwen -> openai/dashscope). Derive ROOT ourselves — same
# value src.qwen exposes, without the API deps.
ROOT = Path(__file__).resolve().parent.parent

OUTPUTS = ROOT / "outputs"
_TMP = OUTPUTS / "_stitch_tmp"

TARGET_FPS = 24

# Fonts, in preference order (drawtext needs a real fontfile).
_FONTS = [
    Path(r"C:/Windows/Fonts/arial.ttf"),
    Path(r"C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
]
_CJK_FONT = Path(r"C:/Windows/Fonts/msyh.ttc")  # Microsoft YaHei, for non-ASCII titles

_X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([_ffmpeg(), "-hide_banner", "-y", *args],
                          capture_output=True, text=True, cwd=cwd)


def _probe(path: Path) -> tuple[int | None, int | None, float | None]:
    """(width, height, seconds) parsed from `ffmpeg -i` stderr.
    imageio-ffmpeg ships ffmpeg only (no ffprobe), so we read the banner."""
    err = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True).stderr
    wh = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", err)
    dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    w = int(wh.group(1)) if wh else None
    h = int(wh.group(2)) if wh else None
    secs = (int(dur.group(1)) * 3600 + int(dur.group(2)) * 60
            + float(dur.group(3))) if dur else None
    return w, h, secs


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def _prep_font(text: str = "") -> str | None:
    """Pick a font for `text` and copy it into _TMP; return its basename (so
    drawtext can reference it without an absolute, colon-bearing path)."""
    src = _CJK_FONT if (_has_cjk(text) and _CJK_FONT.exists()) else None
    if src is None:
        src = next((f for f in _FONTS if f.exists()),
                   _CJK_FONT if _CJK_FONT.exists() else None)
    if src is None:
        return None
    dst = _TMP / ("font" + src.suffix.lower())
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst.name


def _normalize(src: Path, dst: Path, w: int, h: int,
               caption: str | None = None, font: str | None = None) -> Path:
    """Re-encode one clip to the target grid; optionally burn a caption."""
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS},format=yuv420p")
    if caption and font:
        cap_txt = dst.with_name(dst.stem + ".cap.txt")
        cap_txt.write_text(caption, encoding="utf-8")
        vf += (f",drawtext=fontfile={font}:textfile={cap_txt.name}:"
               f"fontcolor=white:fontsize=h/26:box=1:boxcolor=black@0.55:"
               f"boxborderw=h/54:x=(w-text_w)/2:y=h-(h/7):line_spacing=8")
    r = _run(["-i", str(src), "-vf", vf, "-an", *_X264, str(dst)], cwd=str(_TMP))
    if r.returncode != 0 and caption:            # caption/font broke it — retry plain
        return _normalize(src, dst, w, h, caption=None, font=None)
    if r.returncode != 0:
        raise RuntimeError(f"normalize failed for {src.name}:\n{r.stderr[-800:]}")
    return dst


def _title_card(dst: Path, w: int, h: int, title: str,
                logline: str | None, seconds: float) -> Path:
    font = _prep_font((title or "") + (logline or ""))
    parts = ["format=yuv420p"]
    if font:
        (_TMP / "title.txt").write_text(title or "", encoding="utf-8")
        parts.append(f"drawtext=fontfile={font}:textfile=title.txt:"
                     f"fontcolor=white:fontsize=h/12:x=(w-text_w)/2:y=(h-text_h)/2-h/18")
        if logline:
            (_TMP / "logline.txt").write_text(logline, encoding="utf-8")
            parts.append(f"drawtext=fontfile={font}:textfile=logline.txt:"
                         f"fontcolor=white@0.85:fontsize=h/32:x=(w-text_w)/2:y=(h/2)+h/22")
    r = _run(["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:d={seconds}:r={TARGET_FPS}",
              "-vf", ",".join(parts), *_X264, str(dst)], cwd=str(_TMP))
    if r.returncode != 0:
        raise RuntimeError(f"title card failed:\n{r.stderr[-800:]}")
    return dst


def _xfade(segs: list[Path], durs: list[float], out: Path, cf: float) -> Path:
    inputs: list[str] = []
    for s in segs:
        inputs += ["-i", str(s)]
    fc, prev, acc = [], "0:v", durs[0]
    for i in range(1, len(segs)):
        off = max(0.0, acc - cf)
        fc.append(f"[{prev}][{i}:v]xfade=transition=fade:duration={cf}:"
                  f"offset={off:.3f}[x{i}]")
        prev, acc = f"x{i}", acc + durs[i] - cf
    r = _run([*inputs, "-filter_complex", ";".join(fc), "-map", f"[{prev}]", *_X264, str(out)])
    if r.returncode != 0:
        raise RuntimeError(f"xfade failed:\n{r.stderr[-1200:]}")
    return out


def _hardcut(segs: list[Path], out: Path) -> Path:
    lst = _TMP / "_concat.txt"
    lst.write_text("".join(f"file '{Path(s).resolve().as_posix()}'\n" for s in segs),
                   encoding="utf-8")
    r = _run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    if r.returncode != 0:                         # inputs already normalized — re-encode fallback
        r = _run(["-f", "concat", "-safe", "0", "-i", str(lst), *_X264, str(out)])
        if r.returncode != 0:
            raise RuntimeError(f"concat failed:\n{r.stderr[-800:]}")
    return out


def _add_music(video: Path, music: Path, out: Path) -> Path:
    r = _run(["-i", str(video), "-stream_loop", "-1", "-i", str(music),
              "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
              "-b:a", "192k", "-shortest", str(out)])
    if r.returncode != 0:
        raise RuntimeError(f"music mux failed:\n{r.stderr[-800:]}")
    return out


def assemble(clip_paths, out_name: str = "film.mp4", *,
             title: str | None = None, logline: str | None = None,
             captions: list[str] | None = None, music=None, crossfade: float = 0.4,
             title_seconds: float = 2.6, size: tuple[int, int] | None = None) -> Path:
    """Concatenate shot clips (in order) into one finished film. Returns Path.

    title/logline  -> a rendered title card, prepended.
    captions       -> list aligned to clips; each burned as a lower third.
    crossfade      -> seconds of xfade between segments (0 = hard cuts).
    music          -> audio file laid under the film (path only; looped/trimmed).
    size           -> (w, h); defaults to the first clip's, else 1440x1440.
    """
    clip_paths = [Path(p) for p in clip_paths]
    if not clip_paths:
        raise ValueError("no clips to assemble")
    missing = [str(c) for c in clip_paths if not c.exists()]
    if missing:
        raise FileNotFoundError("clips not found: " + ", ".join(missing))

    OUTPUTS.mkdir(exist_ok=True)
    if _TMP.exists():
        shutil.rmtree(_TMP)
    _TMP.mkdir(parents=True)

    if size:
        w, h = size
    else:
        w, h, _ = _probe(clip_paths[0])
        w, h = (w or 1440), (h or 1440)
    w -= w % 2
    h -= h % 2                                    # x264 needs even dimensions

    cap_font = _prep_font("".join(captions or []))

    # 1) normalize (+ caption) every clip
    norm = []
    for i, c in enumerate(clip_paths):
        cap = captions[i] if captions and i < len(captions) else None
        norm.append(_normalize(c, _TMP / f"n{i:02d}.mp4", w, h, caption=cap, font=cap_font))

    # 2) title card
    segs: list[Path] = []
    if title:
        try:
            segs.append(_title_card(_TMP / "title.mp4", w, h, title, logline, title_seconds))
        except RuntimeError as e:                 # never let the card block the film
            print(f"[stitch] title card skipped: {e}")
    segs += norm

    # 3) join — crossfade if we can, else hard cut
    video = _TMP / "_video.mp4"
    joined: Path | None = None
    if crossfade and crossfade > 0 and len(segs) > 1:
        try:
            durs = []
            for s in segs:
                _, _, d = _probe(s)
                if d is None:
                    raise RuntimeError(f"no duration for {s.name}")
                durs.append(d)
            joined = _xfade(segs, durs, video, crossfade)
        except RuntimeError as e:
            print(f"[stitch] crossfade unavailable, hard-cutting: {e}")
            joined = None
    if joined is None:
        joined = _hardcut(segs, video) if len(segs) > 1 else segs[0]

    # 4) music bed (optional) -> final output
    out = OUTPUTS / out_name
    if music and Path(music).exists():
        _add_music(joined, Path(music), out)
    else:
        if music:
            print(f"[stitch] music not found, silent film: {music}")
        shutil.copyfile(joined, out)

    shutil.rmtree(_TMP, ignore_errors=True)
    return out


if __name__ == "__main__":
    # Self-test with zero cost: use real shots if present, else synthesize clips.
    OUTPUTS.mkdir(exist_ok=True)
    real = sorted(OUTPUTS.glob("shot_*.mp4"))
    if real:
        clips: list[Path] = real
        caps = [f"Shot {i + 1}" for i in range(len(real))]
    else:
        print("[stitch] no real shots — synthesizing test clips (no API, no cost)...")
        clips = []
        for i, col in enumerate(["red", "green", "blue"]):
            p = OUTPUTS / f"_syn_{i}.mp4"
            _run(["-f", "lavfi", "-i", f"color=c={col}:s=640x640:d=3:r=24", *_X264, str(p)])
            clips.append(p)
        caps = ["The arrival", "The drift", "The return"]

    film = assemble(clips, "stitch_selftest.mp4", title="The Red Drift",
                    logline="a lonely chick and a lost red balloon",
                    captions=caps, crossfade=0.5)
    w, h, d = _probe(film)
    print(f"[stitch] OK -> {film}  ({w}x{h}, {d:.1f}s)")
