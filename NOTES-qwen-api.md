# Qwen Cloud API — working notes (verified 2026-07-15)

Everything below is tested against a real key. Key lives in `secrets/.env`
(gitignored) as `DASHSCOPE_API_KEY=...`.

## Endpoint & auth

- International region: `https://dashscope-intl.aliyuncs.com`
- Chat is **OpenAI-compatible**: `.../compatible-mode/v1` + the `openai` SDK.
- Image & video are **NOT** OpenAI-compatible — see below.

## Chat / LLM  (works)

```python
from openai import OpenAI
c = OpenAI(api_key=KEY, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
c.chat.completions.create(model="qwen3.6-flash", messages=[...])
```
Models: `qwen3.7-max` / `qwen3.7-plus` / `qwen3.6-flash` (flash = cheap, use for dev).
`c.models.list()` returns the full 149-model catalog.

## Image  (works — `src/media.gen_image`)

- Model: **`qwen-image-2.0`** ($0.035/img) or `-pro` ($0.075).
- Called via **`dashscope.MultiModalConversation`**, NOT `images.generate`
  (compatible route has no /images) and NOT `ImageSynthesis` (wan names 404).
- Must set `dashscope.base_http_api_url = "<intl>/api/v1"`.
- Response: `output.choices[0].message.content == [{"image": <url>}]` (URL valid 24h).

## Video / image-to-video  (works — `src/media.gen_video`)

- Model: **`wan2.7-i2v`** (also `HappyHorse-1.1-I2V`).
- Gotchas that cost us time:
  1. The SDK's sync `VideoSynthesis.call()` **streams and gets connection-reset
     across the border** (ConnectionReset 10054). The **async create POST works
     fine** — so we bypass the SDK and use raw HTTP + poll.
  2. The first frame field is `input.media`, and each item must be
     **`{"type": "first_frame", "url": <img_url>}`** — not a bare string, not
     `image_url`, not `type: image`.
- Flow: POST `.../services/aigc/video-generation/video-synthesis` with header
  `X-DashScope-Async: enable` -> get `output.task_id` -> poll
  `GET .../tasks/{task_id}` until `task_status == SUCCEEDED` -> download `video_url`.
- Takes ~1 min. Wrap create + poll in retries (cross-border jitter).

## Cost / budget

- The free $40 coupon deadline (Jul 9) **passed** — we pay per call now.
- Dev discipline: test one shot before batch runs; keep clips short; images are
  cheap ($0.035), video is the expensive part.
