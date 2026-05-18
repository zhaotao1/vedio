"""06_sao_audio.py — 远程 GPU 跑 Stable Audio Open，本地只生成任务清单"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, project_dir, read_jsonl, write_jsonl

def build_jobs():
    cfg = load_config()
    scenes = read_jsonl(project_dir("01_script") / "scenes.jsonl")
    shots = read_jsonl(project_dir("01_script") / "shots.jsonl")
    jobs = []
    sd = {}
    for s in shots: sd[s.get("scene_id")] = sd.get(s.get("scene_id"), 0) + s.get("duration", 5)
    for sc in scenes:
        sid = sc["id"]
        seconds = min(47, max(10, int(sd.get(sid, 30))))
        jobs.append({"kind": "bgm", "scene_id": sid,
                     "prompt": sc.get("bgm_prompt", "cinematic ambient, no vocals"),
                     "seconds_total": seconds,
                     "out": str(project_dir(f"02_assets/bgm/{sid}.wav"))})
    for shot in shots:
        sid = shot["shot_id"]
        for j, s in enumerate(shot.get("sfx") or []):
            jobs.append({"kind": "sfx", "shot_id": sid, "name": s.get("name"),
                         "prompt": s.get("prompt", ""),
                         "seconds_total": max(1.0, float(s.get("duration", 2.0))),
                         "out": str(project_dir(f"03_shots/{sid}") / f"sfx_{j:02d}_{s.get('name','x')}.wav")})
    op = project_dir("03_shots") / "sao_jobs.jsonl"
    write_jsonl(op, jobs)
    print(f"[ok] {len(jobs)} SAO jobs → {op}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.build: build_jobs()
    elif a.run: print("[todo] run on remote A100")
    else: ap.print_help()

if __name__ == "__main__":
    main()
