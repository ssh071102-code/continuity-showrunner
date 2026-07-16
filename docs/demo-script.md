# Demo video script (≤5 min, Devpost Track 2)

Structure: hook → problem → live result → how it works → the honest journey → close.
Screen-record VS Code + file explorer + video player. No face needed; voiceover
(or captions only) is fine.

## 0:00–0:25 · Hook
Play the final film full-screen, no commentary for ~10s. Then:
> "This 20-second short film was made by an AI agent from one sentence:
> *'A lonely penguin on an ice floe befriends a lost red balloon.'*
> No human wrote the script, drew a frame, or edited a cut."

## 0:25–1:00 · The problem (why this is hard)
Show the v1 grid (`outputs/film_grid.png`), point at shot 3:
> "Anyone can generate one pretty clip. The moment you tell a *story* across
> shots, AI video falls apart — faces change, locations teleport, palettes
> drift. Here's our own first version: same prompt pipeline, and shot 3
> wandered from icebergs to rocky mountains. Continuity is the wall."

## 1:00–2:00 · Live run (sped up)
Screen-record `python -m src.run "..."` with the log scrolling (speed up 8-16x;
show the real ~10 min as a timer overlay):
> "One command. The agent writes the script, breaks it into shots, generates a
> keyframe per shot, animates each with Wan image-to-video, and stitches the
> film. Fully autonomous — ~10 minutes, about a dollar."

## 2:00–3:30 · How consistency actually works (the technical meat)
Show `src/showrunner.py` briefly, then the three grids side by side:
> "Three mechanisms, each one measured:
> 1. **A consistency contract** — the planner writes one canonical description
>    of character, costume, location, palette; code injects it into every shot
>    prompt. That got us 3 of 4 shots consistent.
> 2. **Pixel anchoring** — text can't pin a visual identity. So shot 1's
>    keyframe becomes the anchor, and every later keyframe is generated FROM
>    that image via qwen-image-edit. Look at the ice floe: the crack pattern is
>    literally identical across shots. Environment: 4 for 4.
> 3. **A framing rule** — the one shot that still drifted was a close-up:
>    re-rendering a face at a different scale breaks identity. The planner now
>    keeps every shot medium or wide. The motion layer never needed close-ups
>    anyway."

## 3:30–4:15 · The honest engineering story
> "We didn't guess these mechanisms — we found them by eyeballing every
> version and recording what broke. v1: text-only contract, character and
> background drifted. v2: pixel anchoring locked the environment. v3: the
> framing rule closed the last gap. Every failure and fix is in the repo's
> commit history."

## 4:15–4:45 · Close
Final film again, quiet.
> "Continuity — an AI showrunner that treats consistency as an engineering
> constraint, not a hope. Built solo on Qwen Cloud: qwen3.7 for the brain,
> qwen-image-2.0 and qwen-image-edit for keyframes, wan2.7-i2v for motion.
> One sentence in, one coherent film out."

## Assets checklist
- [ ] final film (v3)
- [ ] three grids: film_grid.png (v1) / film_grid_v2_anchored.png / film_grid_v3.png
- [ ] sped-up screen recording of a real run
- [ ] repo page for the closing shot

## Recording notes
- OBS or Xbox Game Bar (Win+G) for screen capture; 1080p is enough.
- If no voiceover: put the quotes above as on-screen captions, slightly shorter.
- Keep total ≤5:00 (Devpost limit).
