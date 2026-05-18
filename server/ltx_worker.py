"""ltx_worker.py — 消费 ltx_jobs.jsonl，逐条调用 LTX-2 distilled 生成视频。

输入 jobs 格式（04_ltx_i2v.py 产出）：
{
  "shot_id":"S01_01", "duration":5, "fps":24,
  "prompt":"...",
  "conditioning":{
    "mode":"first_frame|first_last|multi_keyframes",
    "keyframes":[{"time":0.0,"image":"/path/to.png"}, ...]
  },
  "out_path":"project/03_shots/S01_01/video.mp4",
  "seed":42
}

用法：
  cd /root/sj-tmp/LTX-2
  .venv/bin/python /root/sj-tmp/ai_film/server/ltx_worker.py \
      --jobs /root/sj-tmp/ai_film/project/03_shots/ltx_jobs.jsonl \
      --out_root /root/sj-tmp/ai_film/project/03_shots \
      --models /root/sj-tmp/models/ltx-2.3 \
      --gemma /root/sj-tmp/models/gemma-3-12b
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO_DIR = "/root/sj-tmp/LTX-2"
VENV_PY = f"{REPO_DIR}/.venv/bin/python"


def _frames_for(duration: float, fps: int) -> int:
    """LTX 约束: num_frames = 8k+1。向上取整到最近的合法帧数。"""
    target = int(round(duration * fps))
    k = max(0, (target - 1 + 7) // 8)
    return 8 * k + 1


def _resolution(cfg_str: str | None, default=(1024, 1536)) -> tuple[int, int]:
    """cfg_str 形如 '2560x1440'。LTX-2 要求 64 整除。"""
    if not cfg_str:
        return default
    try:
        w, h = cfg_str.lower().split("x")
        w, h = int(w), int(h)
        w = (w // 64) * 64
        h = (h // 64) * 64
        return (h, w)  # 返回 (height, width)
    except Exception:
        return default


def _abs_path(p: str | os.PathLike, base: Path) -> Path:
    """相对路径以 base 为基准转绝对路径。subprocess cwd 切到了 LTX-2 仓库，
    所有路径必须绝对，否则 PyAV 写 mp4 会 FileNotFoundError。"""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def build_cmd(job: dict, args, project_root: Path) -> tuple[list[str], Path]:
    sid = job["shot_id"]
    fps = int(job.get("fps", 24))
    duration = float(job.get("duration", 5))
    num_frames = _frames_for(duration, fps)
    height, width = _resolution(job.get("resolution"))
    seed = int(job.get("seed", 42))
    out_root_abs = Path(args.out_root).resolve()
    raw_out = job.get("out_path") or str(out_root_abs / sid / "video.mp4")
    out = _abs_path(raw_out, project_root)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        VENV_PY, "-m", "ltx_pipelines.distilled",
        "--distilled-checkpoint-path", f"{args.models}/ltx-2.3-22b-distilled-1.1.safetensors",
        "--spatial-upsampler-path", f"{args.models}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "--gemma-root", args.gemma,
        "--prompt", job["prompt"],
        "--output-path", str(out),
        "--num-frames", str(num_frames),
        "--frame-rate", str(fps),
        "--height", str(height),
        "--width", str(width),
        "--seed", str(seed),
    ]

    # 关键帧 → --image PATH FRAME_IDX STRENGTH
    cond = job.get("conditioning") or {}
    kfs = cond.get("keyframes") or []
    for kf in kfs:
        img = kf.get("image")
        if not img:
            continue
        img_abs = _abs_path(img, project_root)
        if not img_abs.exists():
            print(f"[warn] {sid}: missing keyframe image {img_abs}", flush=True)
            continue
        frame_idx = int(round(float(kf.get("time", 0.0)) * fps))
        frame_idx = max(0, min(frame_idx, num_frames - 1))
        strength = float(kf.get("strength", 0.9))
        cmd += ["--image", str(img_abs), str(frame_idx), str(strength)]

    if args.offload:
        cmd += ["--offload", args.offload]
    if args.quantization:
        cmd += ["--quantization", args.quantization]

    return cmd, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--models", default="/root/sj-tmp/models/ltx-2.3")
    ap.add_argument("--gemma", default="/root/sj-tmp/models/gemma-3-12b")
    ap.add_argument("--offload", default=None, choices=[None, "OffloadMode.NONE", "OffloadMode.CPU", "OffloadMode.DISK"])
    ap.add_argument("--quantization", default=None, choices=[None, "fp8-cast", "fp8-scaled-mm"])
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    jobs_path = Path(args.jobs).resolve()
    # project_root: out_root 的上两级（…/project/03_shots → …/）
    project_root = Path(args.out_root).resolve().parent.parent
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = [json.loads(l) for l in f if l.strip()]
    print(f"[ltx_worker] {len(jobs)} jobs  project_root={project_root}", flush=True)

    results = []
    for i, job in enumerate(jobs, 1):
        sid = job["shot_id"]
        cmd, out = build_cmd(job, args, project_root)
        if args.skip_existing and out.exists() and out.stat().st_size > 0:
            print(f"[{i}/{len(jobs)}] skip {sid} (already exists)", flush=True)
            results.append({"shot_id": sid, "status": "skipped", "out": str(out)})
            continue
        t0 = time.time()
        print(f"\n[{i}/{len(jobs)}] === {sid} ===", flush=True)
        print(" ".join(cmd[:6]) + " ...", flush=True)
        try:
            subprocess.run(cmd, cwd=REPO_DIR, check=True)
            elapsed = time.time() - t0
            ok = out.exists() and out.stat().st_size > 0
            print(f"[ok] {sid}  {elapsed:.1f}s  → {out}  ({out.stat().st_size if ok else 0} B)", flush=True)
            results.append({"shot_id": sid, "status": "ok" if ok else "empty",
                            "out": str(out), "elapsed_s": round(elapsed, 1)})
        except subprocess.CalledProcessError as e:
            print(f"[err] {sid} rc={e.returncode}", flush=True)
            results.append({"shot_id": sid, "status": "error", "error": f"rc={e.returncode}"})

    rep = jobs_path.parent / "ltx_report.json"
    rep.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {rep}", flush=True)


if __name__ == "__main__":
    main()
