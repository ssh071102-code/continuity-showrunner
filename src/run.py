"""End-to-end runner: premise -> finished film.

    python -m src.run "A lonely penguin befriends a lost red balloon."
    python -m src.run "..." --limit 2            # cheap: only the first 2 shots
    python -m src.run "..." --keyframes-only     # cheapest: images, no video
    python -m src.run "..." --captions --music bed.mp3
    python -m src.run "..." --dry-run            # plan only, no images/video

Owners: @zihanshen41-pixel (orchestration / logging / stitch) + @ssh071102-code (make_film).
Generation costs money (one image + one video per shot) — run deliberately;
use --limit / --keyframes-only / --dry-run to keep test runs cheap.
"""
from __future__ import annotations

import argparse
import time

from src.stitch import assemble   # local, no API — safe to import at top


def _caption(shot: dict) -> str:
    """Short lower-third text from a shot's action."""
    a = (shot.get("action") or "").strip()
    return a if len(a) <= 48 else a[:47].rstrip() + "…"


def build_film(clips, manifest: dict, *, out: str = "film.mp4", title: bool = True,
               captions: bool = False, music=None, crossfade: float = 0.4):
    """Turn make_film's (clips, manifest) into a finished film. No API here —
    this is the assembly side, testable without the generation side."""
    plan = manifest.get("plan", {})
    caps = None
    if captions:
        shots = plan.get("shots", [])
        caps = [_caption(shots[i]) if i < len(shots) else "" for i in range(len(clips))]
    return assemble(
        clips, out,
        title=plan.get("title") if title else None,
        logline=plan.get("logline") if title else None,
        captions=caps, music=music, crossfade=crossfade,
    )


def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src.run",
                                description="Premise -> finished short film (Qwen Cloud).")
    p.add_argument("premise", nargs="?",
                   default="A lonely penguin on an ice floe befriends a lost red balloon.")
    p.add_argument("--out", default="film.mp4", help="output filename in outputs/")
    p.add_argument("--limit", type=int, metavar="N", help="only the first N shots (cheap test)")
    p.add_argument("--keyframes-only", action="store_true", help="images only, no video (cheapest)")
    p.add_argument("--crossfade", type=float, default=0.4, metavar="SEC",
                   help="crossfade seconds between shots (0 = hard cut)")
    p.add_argument("--no-title", action="store_true", help="skip the title card")
    p.add_argument("--captions", action="store_true", help="burn a per-shot lower third")
    p.add_argument("--music", metavar="PATH", help="audio bed laid under the film")
    p.add_argument("--dry-run", action="store_true", help="plan only — no images/video")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    print(f"[run] premise: {args.premise}")

    from src.showrunner import make_film, write_plan   # lazy: pulls in the API stack

    if args.dry_run:
        print("[run] dry-run: planning only (one cheap LLM call, no images/video)...")
        plan = write_plan(args.premise)
        print(f"[run] TITLE: {plan.get('title')}\n[run] LOGLINE: {plan.get('logline')}")
        for s in plan.get("shots", []):
            print(f"  shot {s['id']}: {s['action']}")
        return 0

    print("[run] NOTE: generates one image + one video per shot on Qwen Cloud — real "
          "cost. Use --limit / --keyframes-only for cheap runs.")
    t0 = time.perf_counter()
    clips, manifest = make_film(args.premise, keyframes_only=args.keyframes_only, limit=args.limit)
    gen_s = time.perf_counter() - t0

    if not clips:
        n = len(manifest.get("shots", []))
        print(f"[run] {n} keyframe(s) generated, no clips (keyframes-only) — "
              f"nothing to stitch. ({gen_s:.0f}s)")
        return 0

    title = manifest.get("plan", {}).get("title", "Untitled")
    print(f"[run] {len(clips)} clips in {gen_s:.0f}s; assembling '{title}'...")
    t1 = time.perf_counter()
    film = build_film(clips, manifest, out=args.out, title=not args.no_title,
                      captions=args.captions, music=args.music, crossfade=args.crossfade)
    print(f"[run] done -> {film}  "
          f"(stitch {time.perf_counter() - t1:.0f}s, total {time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
