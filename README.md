# Continuity — an AI showrunner for consistent short films

Give it one line of premise. The agent autonomously:

1. writes a short **script**,
2. breaks it into a **storyboard** (shot list),
3. generates **keyframes** and **video** per shot,
4. stitches a finished **short film** —

all while **locking character, costume and scene identity across every shot**.
Continuity is the whole point: most AI video drifts between shots; this doesn't.

Built for the Qwen Cloud Global AI Hackathon — Track 2 (AI Showrunner).

## Setup

```bash
pip install -r requirements.txt
# create secrets/.env from the example, paste your Qwen Cloud key:
#   DASHSCOPE_API_KEY=sk-...
python -m src.d1_hello        # smoke test: proves the API works
```

Get an API key at https://home.qwencloud.com/api-keys

## Stack

- **Qwen Cloud** (OpenAI-compatible endpoint) — `qwen3.x` for script + storyboard + self-review
- **Wan** (DashScope async task API) — keyframe images + image-to-video
- **ffmpeg** — final assembly

## Layout

```
src/            pipeline modules
prompts/        prompt templates (script, storyboard, consistency contract)
secrets/        API keys — gitignored, never committed
outputs/        generated frames + clips + final film — gitignored
```

## Status

- [x] D1: scaffold + **all three smoke tests green** — text, image (`qwen-image-2.0`), video (`wan2.7-i2v`). See `NOTES-qwen-api.md`.
- [x] D2: script + storyboard agent with the consistency contract injected per shot
- [x] D3: **first end-to-end film produced** — one premise → 4 keyframes → 4 i2v clips →
      stitched. `outputs/film.mp4`, *"The Red Drift"*, 20.2s, 1440×1440, fully automatic.
- [x] D4: **reference-anchored keyframes** — shot 1 becomes the visual anchor;
      later keyframes are generated from it via `qwen-image-edit`
- [ ] D5: demo video, public repo for Devpost, submit

### Consistency, measured by eye across film versions

**v1 — text-only contract.** 3/4 shots held; shot 3 changed background (icebergs →
rocky mountains), framing and face. Text can describe a character; it cannot pin a
visual identity.

**v2 — pixel anchoring (D4).** Environment locked **4/4** — the crack pattern on the
ice floe is literally identical in every shot. Character held 3/4: the one close-up
shot re-rendered the face at a different scale and drifted its markings.

**v3 — anchoring + framing rule.** Environment and framing fully locked (5/5);
character fine features (crest, belly patch) still faded across edits.

**v4 — segment chaining (`src/chain.py`).** The full methodology: pre-production
assets (character sheet + K0 anchor), per-segment end-beat keyframes anchored on
K0, and every segment generated with **first_frame = the previous segment's REAL
extracted tail frame** and **last_frame = the next beat keyframe** — so cuts are
pixel-continuous *by construction*. A qwen3-vl QA gate checks every keyframe and
every tail against the character sheet, with stop-loss on double failure.
**Result: same character in every sampled frame, same set, smooth joints — the
first version with no visible identity break.**

Each version is preserved in `outputs/` as `film_v<N>_*.mp4` for comparison.

### Verified building blocks (`src/media.py`, `src/qwen.py`)
- `chat(prompt)` — LLM (script/storyboard/self-review)
- `gen_image(prompt, out_name)` / `gen_image_url(prompt)` — keyframes
- `gen_video(first_frame_url, prompt, out_name)` — image-to-video

## Gallery — the consistency journey, one grid per version

| v1 text contract | v2 pixel anchor |
|---|---|
| ![v1](docs/grids/v1_text_contract.png) | ![v2](docs/grids/v2_pixel_anchor.png) |

| v3 framing rule | v4 segment chaining |
|---|---|
| ![v3](docs/grids/v3_framing_rule.png) | ![v4](docs/grids/v4_chained.png) |

## Credits

- [@ssh071102-code](https://github.com/ssh071102-code) — generation pipeline, consistency system, chain director
- [@zihanshen41-pixel](https://github.com/zihanshen41-pixel) — demo-grade assembly (title card / captions / crossfades) + hardened runner + tests
