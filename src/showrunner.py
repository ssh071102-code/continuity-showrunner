"""The showrunner 'brain': premise -> structured plan, then per-shot prompts.

Continuity is enforced in CODE: the model writes one reusable consistency
contract (character / costume / location / palette / camera); we then prepend
that same contract to every shot's keyframe prompt. The model never has to
"remember" the character across shots — the lock is mechanical.
"""
import json
import re

from src.qwen import chat, LLM_PLUS

PLAN_SYSTEM = (
    "You are a film director and showrunner. You turn a one-line premise into a "
    "tight, shootable plan for a ~20-30 second short film of 4-6 shots. Above all "
    "you care about CONTINUITY: the same character, costume, location and palette "
    "in every shot."
)

PLAN_INSTRUCTION = """Given the premise, output ONLY a JSON object of this exact shape:
{{
  "title": "...",
  "logline": "one sentence",
  "style": "visual style in one line, e.g. 'soft cinematic 35mm, muted palette'",
  "consistency": {{
    "character": "exact reusable description of the main character - look, colors, distinguishing features",
    "costume": "what they wear / key body details, or 'n/a'",
    "location": "the setting, described so it can be redrawn identically",
    "palette": "color palette in a few words",
    "camera": "default camera treatment, e.g. 'locked tripod, eye level'"
  }},
  "shots": [
    {{"id": 1, "action": "the single clear image we see in this shot", "motion": "one small, physical, believable movement for image-to-video (head turn, blink, wind, ripple)", "duration": 5}}
  ]
}}
Rules:
- 4 to 6 shots, each 5 seconds.
- Each "action" is ONE clear image (it becomes a keyframe).
- "motion" is subtle and physical, NOT a camera move.
- Keep the character visually simple and distinctive so it stays easy to keep consistent.
- Framing stays MEDIUM or WIDE in every shot. Never call for a close-up or extreme
  close-up: re-rendering a face at a different scale breaks identity continuity
  (measured: the one close-up shot was the one that drifted).
Premise: {premise}"""


def _parse_json(text: str) -> dict:
    """Extract a JSON object from an LLM reply (tolerates ```json fences / prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])
    raise ValueError(f"no JSON object found in reply:\n{text[:300]}")


def write_plan(premise: str, model: str = LLM_PLUS) -> dict:
    """Premise -> structured plan (title, style, consistency contract, shots)."""
    raw = chat(PLAN_INSTRUCTION.format(premise=premise), model=model, system=PLAN_SYSTEM)
    return _parse_json(raw)


def contract_line(plan: dict) -> str:
    # Subject/scene lock for the KEYFRAME. Deliberately excludes the camera
    # field: gear words like "tripod" get drawn literally into the image.
    # Camera intent is expressed in the motion prompt instead.
    c = plan["consistency"]
    return (f'{c["character"]}. {c["costume"]}. Location: {c["location"]}. '
            f'Palette: {c["palette"]}.')


def build_shot_prompts(plan: dict) -> list[dict]:
    """Inject the consistency contract into every shot's keyframe + motion prompt."""
    contract = contract_line(plan)
    style = plan.get("style", "")
    out = []
    for s in plan["shots"]:
        out.append({
            "id": s["id"],
            "keyframe_prompt": f'{contract} {s["action"]}. Cinematic film still. {style}.',
            # D4: shots after the first are generated FROM shot 1's image, so the
            # edit prompt only asks for what changes - identity comes from pixels.
            # v3 finding: a generic "same character" instruction lets fine features
            # (crest, belly patch) fade across edits — so name every distinguishing
            # feature from the contract explicitly in each edit.
            "edit_prompt": (
                f'Keep the EXACT same main character as the reference image: '
                f'{plan["consistency"]["character"]}. {plan["consistency"]["costume"]}. '
                f'Preserve every one of these distinguishing features exactly. '
                f'Same location and same palette as the reference. Change only the '
                f'pose and composition to show: {s["action"]}. '
                f'Cinematic film still. {style}.'
            ),
            "motion_prompt": f'{s["motion"]}. Camera locked, no camera movement.',
            "duration": s.get("duration", 5),
        })
    return out


def make_film(premise: str, keyframes_only: bool = False, limit: int | None = None,
              anchored: bool = True):
    """Full generation loop (owner: @ssh071102-code). premise -> keyframes -> clips.

    anchored=True (D4): shot 1's keyframe is generated from text and becomes the
    visual anchor; every later keyframe is generated FROM that image via
    qwen-image-edit, so character/location identity is pinned by pixels. The
    first film run (text-only contract) drifted in 1 of 4 shots - this is the fix.

    Returns (clip_paths, manifest). Writes outputs/manifest.json.
    Set keyframes_only=True (and/or limit=N) for cheap partial test runs.
    """
    import json as _json
    from src.media import gen_image_url, gen_image_edit_url, gen_video, _download, OUTPUTS

    plan = write_plan(premise)
    shots = build_shot_prompts(plan)
    if limit:
        shots = shots[:limit]

    clips, manifest = [], {"premise": premise, "anchored": anchored,
                           "plan": plan, "shots": []}
    anchor_path = None
    for s in shots:
        if anchored and anchor_path is not None:
            print(f"[make_film] shot {s['id']}: keyframe (anchored on shot 1)...")
            kf_url = gen_image_edit_url(anchor_path, s["edit_prompt"])
        else:
            print(f"[make_film] shot {s['id']}: keyframe (text, anchor)...")
            kf_url = gen_image_url(s["keyframe_prompt"])
        kf_path = _download(kf_url, f"shot_{s['id']:02d}.png")
        if anchor_path is None:
            anchor_path = kf_path
        entry = {"id": s["id"], "keyframe": str(kf_path)}
        if not keyframes_only:
            print(f"[make_film] shot {s['id']}: video...")
            clip = gen_video(kf_url, s["motion_prompt"], f"shot_{s['id']:02d}.mp4")
            entry["clip"] = str(clip)
            clips.append(clip)
        manifest["shots"].append(entry)

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "manifest.json").write_text(
        _json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return clips, manifest


if __name__ == "__main__":
    import sys
    premise = sys.argv[1] if len(sys.argv) > 1 else \
        "A lonely lighthouse keeper trains a wounded seagull to deliver letters."
    print("premise:", premise, "\n")
    plan = write_plan(premise)
    print("TITLE:", plan["title"])
    print("LOGLINE:", plan["logline"])
    print("STYLE:", plan["style"])
    print("\nCONSISTENCY CONTRACT:")
    for k, v in plan["consistency"].items():
        print(f"  {k}: {v}")
    print(f"\nSHOTS ({len(plan['shots'])}):")
    for s in build_shot_prompts(plan):
        print(f"  [{s['id']}] {s['keyframe_prompt']}")
        print(f"       motion: {s['motion_prompt']}")
