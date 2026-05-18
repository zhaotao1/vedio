"""10_finalize.py — 字幕 + 调色 + 终混响度归一"""
from __future__ import annotations
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import project_dir, read_jsonl

def ts(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - h*3600 - m*60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def build_srt():
    shots = read_jsonl(project_dir("01_script") / "shots.jsonl")
    out = project_dir("04_final") / "movie.srt"
    out.parent.mkdir(parents=True, exist_ok=True)
    t = 0.0; lines = []; n = 1
    for s in shots:
        dur = s.get("duration", 5)
        for d in s.get("dialogue") or []:
            start = t + float(d.get("start", 0.0))
            end = min(t + dur, start + max(2.0, len(d["text"]) / 6.0))
            lines.append(f"{n}\n{ts(start)} --> {ts(end)}\n{d['text']}\n"); n += 1
        t += dur
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[srt] {out}"); return out

def main():
    raw = project_dir("04_final") / "movie_raw.mp4"
    srt = build_srt()
    final = project_dir("04_final") / "movie_final.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw),
        "-vf", f"subtitles={srt}:force_style='FontName=PingFang SC,FontSize=22,Outline=1',eq=contrast=1.05:saturation=0.95",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "256k", str(final)
    ], check=True)
    print(f"[ok] → {final}")

if __name__ == "__main__":
    main()
