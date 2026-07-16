"""Image + video generation on Qwen Cloud (international / dashscope-intl).

Findings that shaped this module (see NOTES-qwen-api.md):
- Chat is OpenAI-compatible; image + video are NOT.
- Image: qwen-image-2.0 via dashscope MultiModalConversation.
- Video: wan2.7-i2v via the RAW async HTTP task API. The dashscope SDK's
  sync VideoSynthesis.call() streams and gets connection-reset across the
  border, and its `media`/`img_url` params don't map to the required
  `input.media` field. So we POST the task ourselves and poll.
- The i2v first frame must be: input.media = [{"type":"first_frame","url": <url>}]
"""
import os
import shutil
import time
from pathlib import Path

import requests
import dashscope
from dashscope import MultiModalConversation

from src.qwen import ROOT  # importing this loads secrets/.env into os.environ

INTL = "https://dashscope-intl.aliyuncs.com/api/v1"
dashscope.base_http_api_url = INTL

IMG_MODEL = "qwen-image-2.0"          # $0.035/image, fast — use during dev
IMG_MODEL_PRO = "qwen-image-2.0-pro"  # $0.075/image, higher quality
IMG_EDIT_MODEL = "qwen-image-edit"    # reference-anchored generation (D4 consistency fix)
VIDEO_MODEL = "wan2.7-i2v"            # image-to-video
OUTPUTS = ROOT / "outputs"

_VIDEO_EP = f"{INTL}/services/aigc/video-generation/video-synthesis"


def _api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set (put it in secrets/.env)")
    return key


def with_retry(fn, attempts: int = 4, wait: int = 6):
    """Cross-border resets (10054 / SSL EOF) are routine here — every network
    call goes through this. Retries on ANY exception, re-raises the last."""
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad, network layer
            last = e
            time.sleep(wait)
    raise last


def _download(url: str, out_name: str):
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / out_name
    last_err = None
    for _ in range(4):  # cross-border resets are routine; never fail on one
        try:
            path.write_bytes(requests.get(url, timeout=180).content)
            return path
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(5)
    raise RuntimeError(f"download failed after retries: {url[:80]} ({last_err})")


def gen_image_url(prompt: str, model: str = IMG_MODEL) -> str:
    """Generate one image, return its (24h) OSS URL. Feed this to gen_video."""
    rsp = with_retry(lambda: MultiModalConversation.call(
        model=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        api_key=_api_key(),
    ))
    if rsp.status_code != 200:
        raise RuntimeError(f"image gen failed: {rsp.status_code} {rsp.code} {rsp.message}")
    content = rsp.output.choices[0].message.content
    return next(i["image"] for i in content if isinstance(i, dict) and "image" in i)


def gen_image(prompt: str, out_name: str, model: str = IMG_MODEL):
    """Generate one image and save it locally. Returns the saved Path."""
    return _download(gen_image_url(prompt, model), out_name)


def gen_image_edit_url(ref, prompt: str, model: str = IMG_EDIT_MODEL) -> str:
    """Reference-anchored generation: new keyframe conditioned on `ref`.

    `ref` is a local Path (uploaded via file://) or an http(s) URL. This is the
    D4 consistency fix - visual identity is pinned by pixels, not re-imagined
    from a text contract each shot. Returns the (24h) OSS URL.
    """
    ref = str(ref)
    is_local = not ref.startswith(("http://", "https://", "file://"))

    def _attempt():  # fresh unique temp name per attempt
        uri, tmp = fresh_local_uri(ref) if is_local else (ref, None)
        try:
            return MultiModalConversation.call(
                model=model,
                messages=[{"role": "user", "content": [{"image": uri}, {"text": prompt}]}],
                api_key=_api_key(),
            )
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)

    rsp = with_retry(_attempt)
    if rsp.status_code != 200:
        raise RuntimeError(f"image edit failed: {rsp.status_code} {rsp.code} {rsp.message}")
    content = rsp.output.choices[0].message.content
    return next(i["image"] for i in content if isinstance(i, dict) and "image" in i)


def fresh_local_uri(path):
    """DashScope's staging OSS rejects re-uploading the same filename
    (FileAlreadyExists), so every upload goes through a uniquely-named temp
    copy. Returns (file_uri, temp_path) — caller deletes temp_path after the
    API call that consumes it."""
    p = Path(path)
    OUTPUTS.mkdir(exist_ok=True)
    tmp = OUTPUTS / f"_up{time.time_ns()}_{p.name}"
    shutil.copy2(p, tmp)
    return "file://" + str(tmp).replace("\\", "/"), tmp


def upload_local(path) -> str:
    """Upload a local image to DashScope's staging OSS; returns an oss:// URL
    that the video API accepts (with the OssResourceResolve header)."""
    from dashscope.utils.oss_utils import upload_file

    def _attempt():  # fresh unique name per attempt (retry + same-name = reject)
        uri, tmp = fresh_local_uri(path)
        try:
            return upload_file(VIDEO_MODEL, uri, _api_key())
        finally:
            tmp.unlink(missing_ok=True)

    url = with_retry(_attempt)
    if not url:
        raise RuntimeError(f"upload failed for {path}")
    return url


def extract_last_frame(video_path, out_name: str):
    """Extract the true final frame of a clip (the chaining anchor)."""
    import subprocess
    import imageio_ffmpeg
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / out_name
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-sseof", "-0.2",
         "-i", str(video_path), "-vsync", "0", "-q:v", "2",
         "-update", "1", str(out)],
        check=True, capture_output=True)
    return out


def gen_video(first_frame_url: str, prompt: str, out_name: str,
              last_frame_url: str | None = None,
              model: str = VIDEO_MODEL, poll_interval: int = 12,
              max_polls: int = 45):
    """Image-to-video via the async task API. Returns the saved Path.

    first/last frame URLs may be https:// (generation output) or oss://
    (upload_local). With both given, wan2.7-i2v interpolates between them —
    the first-last-frame chaining technique.
    """
    key = _api_key()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-DashScope-Async": "enable",
               "X-DashScope-OssResourceResolve": "enable"}
    media = [{"type": "first_frame", "url": first_frame_url}]
    if last_frame_url:
        media.append({"type": "last_frame", "url": last_frame_url})
    body = {"model": model,
            "input": {"prompt": prompt, "media": media},
            "parameters": {}}

    task_id = None
    for _ in range(4):
        try:
            j = requests.post(_VIDEO_EP, headers=headers, json=body, timeout=60).json()
            task_id = j.get("output", {}).get("task_id")
            if task_id:
                break
            raise RuntimeError(f"create failed: {j.get('code')} {j.get('message')}")
        except requests.exceptions.RequestException:
            time.sleep(6)  # cross-border reset — retry
    if not task_id:
        raise RuntimeError("video create failed after retries (network)")

    task_ep = f"{INTL}/tasks/{task_id}"
    for _ in range(max_polls):
        time.sleep(poll_interval)
        try:
            g = requests.get(task_ep, headers={"Authorization": f"Bearer {key}"},
                             timeout=60).json()
        except requests.exceptions.RequestException:
            continue  # transient — keep polling
        status = g["output"]["task_status"]
        if status == "SUCCEEDED":
            return _download(g["output"]["video_url"], out_name)
        if status in ("FAILED", "UNKNOWN"):
            raise RuntimeError(
                f"video failed: {g['output'].get('code')} {g['output'].get('message')}"
            )
    raise RuntimeError("video timed out")
