"""Segment-chained film director: last-frame chaining.

Pipeline (one segment at a time, never batch):
  Phase 0   Pre-production assets: a canonical CHARACTER SHEET, then K0 - the
            opening frame that places that exact character into the environment.
  Beats     Every segment gets an END-BEAT keyframe, generated from K0 with the
            contract's distinguishing features named explicitly.
  Segment i first_frame = the REAL extracted tail frame of segment i-1
            (segment 1 starts from K0), last_frame = beat keyframe K_i.
            wan2.7-i2v interpolates between them, so cuts are pixel-continuous.
  QA gate   After every keyframe and every segment tail, a qwen3-vl vision
            check compares against the character sheet. Fail -> one retry,
            then hard stop (stop-loss; never burn segments on a broken chain).
            If the QA call itself fails (network error, unparseable reply) the
            check is recorded as "qa_unavailable" and counts as a FAIL by
            default. Set CHAIN_QA_FAIL_OPEN=1 to let it pass instead.

Run:  python -m src.chain "premise"
"""
import json
import os
import sys

import dashscope
from dashscope import MultiModalConversation

from src.qwen import ROOT
from src.media import (gen_image, gen_image_edit_url, gen_video, upload_local,
                       extract_last_frame, _download, _api_key, OUTPUTS)
from src.showrunner import write_plan, build_shot_prompts

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

QA_MODEL = "qwen3-vl-plus"

# Appended to the K0 prompt only when the plan has no "style" field
# (build_shot_prompts already appends the plan's style when there is one).
DEFAULT_STYLE = "Natural lighting, realistic detail."


def _qa_fail_open() -> bool:
    """True only when CHAIN_QA_FAIL_OPEN is set to 1/true/yes."""
    return os.getenv("CHAIN_QA_FAIL_OPEN", "").strip().lower() in {"1", "true", "yes"}


def _qa_unavailable(reason: str) -> dict:
    return {"passed": _qa_fail_open(), "status": "qa_unavailable", "issues": reason}


QA_TEXT = (
    "Image 1 is the canonical character reference. Image 2 is a {what}. "
    'Answer ONLY a JSON object '
    '{{"same_character": true/false, "artifacts": true/false, "issues": "..."}} — '
    "same_character: judge ONLY species, color scheme and major markings "
    "(e.g. eye rings, crest, belly patch) — NOT pose, framing, micro-detail "
    "or minor plumage differences. artifacts: true ONLY for defects a casual "
    "viewer would immediately notice — extra/missing limbs, two heads, melted "
    "face, heavy distortion, or a cartoon/CGI look when the reference is "
    "photographic. Minor anatomical imperfections are NOT artifacts. "
    "Calibration: block real breaks, don't nitpick.")


def vision_qa(candidate_path, ref_path, what: str) -> dict:
    """Compare a frame against the character sheet.

    Returns {passed, status, issues}. status is "ok" when the model answered
    with parseable JSON in which same_character and artifacts are both JSON
    booleans, "qa_unavailable" when the call failed after retries or the reply
    could not be parsed or lacked a boolean verdict field. qa_unavailable is fail-closed (passed=False)
    unless CHAIN_QA_FAIL_OPEN=1.
    """
    from src.media import fresh_local_uri, with_retry

    def _attempt():  # fresh temp names per attempt (same-name re-upload rejected)
        ref_uri, tmp1 = fresh_local_uri(ref_path)
        cand_uri, tmp2 = fresh_local_uri(candidate_path)
        try:
            return MultiModalConversation.call(
                model=QA_MODEL,
                messages=[{"role": "user", "content": [
                    {"image": ref_uri},
                    {"image": cand_uri},
                    {"text": QA_TEXT.format(what=what)},
                ]}],
                api_key=_api_key())
        finally:
            tmp1.unlink(missing_ok=True)
            tmp2.unlink(missing_ok=True)

    try:
        rsp = with_retry(_attempt, attempts=3)
    except Exception as e:  # noqa: BLE001 - any failure here means "no verdict"
        return _qa_unavailable(f"QA call failed after retries: {type(e).__name__}")
    if rsp.status_code != 200:
        return _qa_unavailable(f"QA call returned {rsp.status_code} {rsp.code}")
    try:
        text = "".join(i.get("text", "") for i in rsp.output.choices[0].message.content
                       if isinstance(i, dict))
        start, end = text.find("{"), text.rfind("}")
        j = json.loads(text[start:end + 1])
        # Both verdict fields must be present JSON booleans; "false", 1, null or
        # a missing field is not a verdict.
        if not isinstance(j, dict) or not all(
                isinstance(j.get(k), bool) for k in ("same_character", "artifacts")):
            raise ValueError("missing or non-boolean verdict field")
    except Exception:  # noqa: BLE001 - malformed reply means "no verdict"
        return _qa_unavailable("unparseable QA reply")
    return {"passed": j["same_character"] and not j["artifacts"],
            "status": "ok", "issues": j.get("issues", "")}


def k0_prompt(plan, shots) -> str:
    """K0 = shot 1's keyframe prompt, which already carries the plan's style.
    Falls back to a neutral style line when the plan has none."""
    prompt = shots[0]["keyframe_prompt"]
    if not str(plan.get("style") or "").strip():
        prompt = f"{prompt} {DEFAULT_STYLE}"
    return prompt


def character_sheet_prompt(plan):
    c = plan["consistency"]
    return (f'Character reference sheet, single character, full body, standing, '
            f'facing camera: {c["character"]}. {c["costume"]}. Neutral pose, plain '
            f'soft grey studio background, even lighting, crisp detail. '
            f'{plan.get("style", "")}.')


def beat_prompt(plan, action):
    c = plan["consistency"]
    return (f'Keep the EXACT same character — {c["character"]}, {c["costume"]} — '
            f'preserve every one of these distinguishing features exactly, and the '
            f'EXACT same location, palette, lighting AND photorealistic rendering '
            f'style as the reference image. Change only pose and composition to '
            f'show: {action}. Medium-wide framing. Cinematic film still. '
            f'{plan.get("style", "")}. Strictly NO 3D-render look, no cartoon, '
            f'no CGI, no toy or miniature look.')


def _qa_gate(path, ref, what):
    qa = vision_qa(path, ref, what)
    status = "PASS" if qa["passed"] else "FAIL"
    if qa.get("status") == "qa_unavailable":
        status += " (qa_unavailable)"
    print(f"  [QA] {what}: {status} {('- ' + qa['issues']) if qa['issues'] else ''}")
    return qa["passed"]


def make_film_chained(premise: str):
    plan = write_plan(premise)
    shots = build_shot_prompts(plan)
    print(f"[chain] '{plan['title']}' — {len(shots)} segments")
    manifest = {"premise": premise, "method": "last-frame chaining + first/last frame",
                "plan": plan, "segments": []}

    # Phase 0 - assets.
    # Measured: image-edit re-renders invented scenes as CGI no matter how hard
    # the prompt pins style. So K0 comes from t2i (photoreal, like v3's shot 1)
    # and BECOMES the visual anchor; edits FROM a photo keep photorealism.
    # The studio character sheet stays as the QA reference document.
    print("[chain] phase 0: character sheet...")
    char_ref = gen_image(character_sheet_prompt(plan), "ref_character.png")
    print("[chain] phase 0: K0 opening frame (t2i, becomes the anchor)...")
    k0_path = gen_image(k0_prompt(plan, shots), "beat_00.png")
    if not _qa_gate(k0_path, char_ref, "K0 opening frame"):
        k0_path = gen_image(k0_prompt(plan, shots), "beat_00.png")
        if not _qa_gate(k0_path, char_ref, "K0 retry"):
            raise SystemExit("STOP-LOSS: K0 failed QA twice — fix assets before burning segments.")

    chain_url = upload_local(k0_path)  # first_frame of segment 1
    clips = []
    for i, s in enumerate(shots, 1):
        action = plan["shots"][i - 1]["action"]
        print(f"[chain] segment {i}/{len(shots)}: beat keyframe...")
        beat_url = gen_image_edit_url(k0_path, beat_prompt(plan, action))
        beat_path = _download(beat_url, f"beat_{i:02d}.png")
        if not _qa_gate(beat_path, char_ref, f"beat {i}"):
            beat_url = gen_image_edit_url(k0_path, beat_prompt(plan, action))
            beat_path = _download(beat_url, f"beat_{i:02d}.png")
            if not _qa_gate(beat_path, char_ref, f"beat {i} retry"):
                raise SystemExit(f"STOP-LOSS: beat {i} failed QA twice.")

        print(f"[chain] segment {i}: video (first=chain tail, last=beat {i})...")
        motion = (f'{s["motion_prompt"]} Continuous take, same scene as the '
                  f'starting frame; end exactly at the final frame composition.')
        clip = gen_video(chain_url, motion, f"seg_{i:02d}.mp4", last_frame_url=beat_url)
        tail = extract_last_frame(clip, f"tail_{i:02d}.png")
        if not _qa_gate(tail, char_ref, f"segment {i} tail"):
            raise SystemExit(f"STOP-LOSS: segment {i} tail failed QA — chain broken, "
                             f"inspect outputs/seg_{i:02d}.mp4 before continuing.")
        chain_url = upload_local(tail)  # REAL tail chains into the next segment
        clips.append(clip)
        manifest["segments"].append({"i": i, "beat": str(beat_path),
                                     "clip": str(clip), "tail": str(tail)})

    (OUTPUTS / "manifest_chain.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[chain] all {len(clips)} segments done.")
    return clips, manifest


if __name__ == "__main__":
    premise = sys.argv[1] if len(sys.argv) > 1 else \
        "A lonely penguin on an ice floe befriends a lost red balloon."
    clips, manifest = make_film_chained(premise)
    from src.stitch import assemble
    film = assemble(clips, "film_chained.mp4", title=manifest["plan"].get("title"))
    print("[chain] film ->", film)
