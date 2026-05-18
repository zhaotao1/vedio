"""09_concat.py — 拼接 mixed.mp4 → movie_raw.mp4"""
from __future__ import annotations
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import project_dir, read_jsonl

def main():
    shots = read_jsonl(project_dir("01_script") / "shots.jsonl")
    parts = [project_dir(f"03_shots/{s['shot_id']}") / "mixed.mp4" for s in shots]
    parts = [p for p in parts if p.exists()]
    if not parts: print("[err] no mixed parts"); return
    out = project_dir("04_final") / "movie_raw.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 用 concat filter（解码后拼接），可容忍各段编码参数差异；强制统一为 1080p/30fps/stereo/48k
    cmd = ["ffmpeg", "-y"]
    for p in parts:
        cmd += ["-i", str(p)]
    n = len(parts)
    fc_parts = []
    for i in range(n):
        fc_parts.append(
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{i}];"
            f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}];"
        )
    fc = "".join(fc_parts) + "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "256k", str(out)]
    subprocess.run(cmd, check=True)
    print(f"[ok] → {out}")
    print(f"[ok] → {out}")

if __name__ == "__main__":
    main()
