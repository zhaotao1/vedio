"""08_mix_shot.py — 单镜头混音：video + voice + bgm + sfx → mixed.mp4"""
from __future__ import annotations
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, project_dir, read_jsonl

def mix_one(shot, scene_bgm):
    sid = shot["shot_id"]
    base = project_dir(f"03_shots/{sid}")
    video = base / "video.mp4"
    if not video.exists(): print(f"[skip] {sid} no video"); return
    voice = base / "voice.wav"
    bgm = scene_bgm.get(shot.get("scene_id"))
    sfx_files = sorted(base.glob("sfx_*.wav"))
    inputs = ["-i", str(video)]; filters = []; amix_in = []; idx = 1
    if voice.exists():
        inputs += ["-i", str(voice)]
        filters.append(f"[{idx}:a]volume=1.0[v]"); amix_in.append("[v]"); idx += 1
    if bgm and bgm.exists():
        dur = shot.get("duration", 5)
        inputs += ["-i", str(bgm)]
        filters.append(f"[{idx}:a]atrim=0:{dur},afade=t=in:st=0:d=0.3,"
                       f"afade=t=out:st={max(0,dur-0.3)}:d=0.3,volume=0.18[b]")
        amix_in.append("[b]"); idx += 1
    for i, sfx in enumerate(sfx_files):
        inputs += ["-i", str(sfx)]
        n = int(sfx.stem.split("_")[1])
        meta = (shot.get("sfx") or [])[n] if n < len(shot.get("sfx") or []) else {}
        d = int(float(meta.get("start", 0.0)) * 1000)
        v = float(meta.get("volume", 0.6))
        filters.append(f"[{idx}:a]adelay={d}|{d},volume={v}[s{i}]")
        amix_in.append(f"[s{i}]"); idx += 1
    out = base / "mixed.mp4"
    if not amix_in:
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-c", "copy", str(out)],
                       check=True, capture_output=True); return
    filt = ";".join(filters) + (f";{''.join(amix_in)}amix=inputs={len(amix_in)}:"
                                f"duration=first:dropout_transition=0,"
                                f"loudnorm=I=-16:TP=-1.5:LRA=11[a]")
    print(f"[mix] {sid} ({len(amix_in)} tracks)")
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", filt,
                    "-map", "0:v", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)],
                   check=True)

def main():
    cfg = load_config()
    shots = read_jsonl(project_dir("01_script") / "shots.jsonl")
    bgm_dir = project_dir("02_assets/bgm")
    scene_bgm = {p.stem: p for p in bgm_dir.glob("*.wav")}
    for shot in shots:
        try: mix_one(shot, scene_bgm)
        except Exception as e: print(f"  [err] {shot['shot_id']}: {e}")

if __name__ == "__main__":
    main()
