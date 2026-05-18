"""04_ltx_i2v.py — 远程 GPU 跑 LTX-2.3，本地只生成任务清单"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, project_dir, read_jsonl, write_jsonl

def _inline_dialogue(shot: dict, characters: dict) -> str:
    """Render dialogue as LTX-2 inline quoted speech with accent hints."""
    parts = []
    for d in shot.get("dialogue") or []:
        speaker = d.get("speaker", "")
        text = (d.get("text") or "").strip()
        if not text:
            continue
        vp = (characters.get(speaker) or {}).get("voice_profile", {})
        accent = vp.get("accent")
        emo = d.get("instruct_or_emotion")
        bits = []
        if emo:
            bits.append(emo)
        if accent:
            bits.append(f"{accent} accent")
        descriptor = ", ".join(bits) if bits else "says"
        parts.append(f'The {speaker} {descriptor}, "{text}".')
    return " ".join(parts)


def _ambient_sfx(shot: dict) -> str:
    descs = [s.get("prompt", "").strip() for s in (shot.get("sfx") or []) if s.get("prompt")]
    return "; ".join(descs)


def _ltx_prompt(shot: dict, characters: dict, style: str) -> str:
    """Assemble a LTX-2 friendly single-paragraph prompt.
    Order: style/genre + camera + scene/action (video_prompt) + dialogue + ambient audio.
    """
    base = (shot.get("video_prompt") or "").strip().rstrip(".")
    camera = (shot.get("camera") or "").strip().rstrip(".")
    dialogue = _inline_dialogue(shot, characters)
    ambient = _ambient_sfx(shot)
    pieces = [style.strip().rstrip(".")]
    if camera:
        pieces.append(camera)
    if base:
        pieces.append(base)
    if dialogue:
        pieces.append(dialogue)
    if ambient:
        pieces.append(f"Ambient sound: {ambient}")
    return ". ".join(p for p in pieces if p) + "."


def _conditioning(shot: dict, refs_dir: Path) -> dict:
    sid = shot["shot_id"]
    cond = shot.get("ltx_conditioning") or {"mode": "first_frame"}
    mode = cond.get("mode", "first_frame")
    kfs = cond.get("keyframes") or []
    images = []
    if mode == "first_frame":
        images = [{"time": 0.0, "image": str(refs_dir / f"{sid}.png")}]
    elif mode in ("first_last", "multi_keyframes"):
        # expect refs/<sid>/kf_*.png; fall back to single ref if folder missing
        kf_dir = refs_dir / sid
        if kf_dir.exists():
            for kf in sorted(kf_dir.glob("kf_*.png")):
                t = float(kf.stem.split("_")[1]) / 1000.0
                images.append({"time": t, "image": str(kf)})
        if not images:
            images = [{"time": 0.0, "image": str(refs_dir / f"{sid}.png")}]
    return {"mode": mode, "keyframes": images, "prompts": kfs}


def _character_refs(shot: dict, characters: dict, refs_dir: Path) -> list[dict]:
    """Resolve per-shot character anchor images for optional_negative_index_latents.

    Looks up images from each character's profile (ref_image / portrait), falling
    back to refs/characters/<name>.png. Only characters listed in
    shot["characters"] (or appearing in dialogue) are included.
    """
    names: list[str] = list(shot.get("characters") or [])
    if not names:
        for d in shot.get("dialogue") or []:
            sp = d.get("speaker")
            if sp and sp not in names:
                names.append(sp)
    char_root = refs_dir / "characters"
    out: list[dict] = []
    for name in names:
        prof = characters.get(name) or {}
        img = (
            prof.get("ref_image")
            or prof.get("portrait")
            or (str(char_root / f"{name}.png") if (char_root / f"{name}.png").exists() else None)
        )
        if not img:
            continue
        out.append({"name": name, "image": str(img)})
    return out


def _continuation(shot: dict, prev_sid: str | None, cfg: dict) -> dict | None:
    """How this shot connects to the previous one.

    Default: continue from prev shot's tail latents with frame_overlap.
    Explicit shot fields:
      - scene_cut: true       → start fresh (no continuation)
      - frame_overlap: int    → override default overlap (pixel frames)
    """
    if prev_sid is None or shot.get("scene_cut"):
        return None
    fps = cfg["style"]["fps"]
    default_overlap = cfg["style"].get("ltx_frame_overlap", max(8, int(fps * 0.5)))
    overlap = int(shot.get("frame_overlap", default_overlap))
    return {
        "prev_shot_id": prev_sid,
        "prev_latents": f"03_shots/{prev_sid}/latents.pt",
        "frame_overlap": overlap,
        "overlap_cond_strength": float(shot.get("overlap_cond_strength", 0.6)),
        "adain_factor": float(shot.get("adain_factor", 0.15)),
    }


def build_jobs():
    cfg = load_config()
    from _utils import load_characters
    characters = load_characters()
    shots = read_jsonl(project_dir("01_script") / "shots.jsonl")
    refs_dir = project_dir("02_assets/refs")
    style = cfg["style"]["global_prompt_suffix"]
    jobs = []
    prev_sid: str | None = None
    for shot in shots:
        sid = shot["shot_id"]
        jobs.append({
            "shot_id": sid,
            "prompt": _ltx_prompt(shot, characters, style),
            "conditioning": _conditioning(shot, refs_dir),
            "character_refs": _character_refs(shot, characters, refs_dir),
            "continuation": _continuation(shot, prev_sid, cfg),
            "duration": shot.get("duration", cfg["style"]["shot_duration_default"]),
            "fps": cfg["style"]["fps"],
            "out": str(project_dir(f"03_shots/{sid}") / "video.mp4"),
            "save_latents": str(project_dir(f"03_shots/{sid}") / "latents.pt"),
        })
        prev_sid = sid
    out = project_dir("03_shots") / "ltx_jobs.jsonl"
    write_jsonl(out, jobs)
    print(f"[ok] {len(jobs)} LTX jobs → {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.build: build_jobs()
    elif args.run: print("[todo] run on remote A100")
    else: ap.print_help()

if __name__ == "__main__":
    main()
