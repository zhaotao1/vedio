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
    # 转义字幕路径里的特殊字符（: \ '）以适配 ffmpeg filtergraph
    srt_esc = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    vf = (f"subtitles='{srt_esc}':force_style='FontSize=22,Outline=1',"
          f"eq=contrast=1.05:saturation=0.95")
    cmd = [
        "ffmpeg", "-y", "-i", str(raw),
        "-vf", vf + ",format=yuv420p",
        "-af", "dynaudnorm=f=200:g=15:p=0.7:m=10",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", str(final),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # 字幕烧录失败（缺字体/libass），退回到无字幕版本（字幕通过 movie.srt 外挂）
        print("[warn] subtitles burn-in failed, fallback to no-subs final")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(raw),
            "-vf", "eq=contrast=1.05:saturation=0.95,format=yuv420p",
            "-af", "dynaudnorm=f=200:g=15:p=0.7:m=10",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", str(final),
        ], check=True)
    print(f"[ok] → {final}")

if __name__ == "__main__":
    main()
