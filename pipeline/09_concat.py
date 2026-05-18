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
    lf = project_dir("03_shots") / "concat_list.txt"
    lf.write_text("\n".join(f"file '{p}'" for p in parts))
    out = project_dir("04_final") / "movie_raw.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 不能用 -c copy：各 shot 的 AAC 通道数/采样率可能不一致，拼接后音轨会损坏
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lf),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "256k", "-ac", "2", "-ar", "48000",
        str(out)
    ], check=True)
    print(f"[ok] → {out}")

if __name__ == "__main__":
    main()
