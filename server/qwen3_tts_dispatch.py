"""qwen3_tts_dispatch.py — 读 outline.json + shots.jsonl，用 Qwen3-TTS 出每镜对话音。

输出：每个 shot 的对话音放到 project/03_shots/<sid>/dialog_<idx>_<speaker>.wav

用法：
  conda activate qwen3tts
  python /root/sj-tmp/ai_film/server/qwen3_tts_dispatch.py \
      --root /root/sj-tmp/ai_film \
      --model /root/sj-tmp/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


LANG_MAP = {
    "zh": "Chinese", "zh-CN": "Chinese", "Chinese": "Chinese",
    "en": "English", "en-US": "English", "English": "English",
    "ja": "Japanese", "ko": "Korean",
}


def build_instruct(char: dict, dialog: dict) -> str:
    vp = char.get("voice_profile") or {}
    bits = []
    if vp.get("qwen3_tts_instruct"):
        bits.append(vp["qwen3_tts_instruct"])
    else:
        for k in ("age", "gender", "accent", "timbre", "pace", "personality"):
            if vp.get(k):
                bits.append(str(vp[k]))
    emo = dialog.get("instruct_or_emotion")
    if emo:
        bits.append(emo)
    return "，".join(bits) if bits else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/sj-tmp/ai_film")
    ap.add_argument("--model", default="/root/sj-tmp/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--lang", default="Chinese")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    outline = json.loads((root / "project/01_script/outline.json").read_text(encoding="utf-8"))
    chars = outline.get("characters", {})
    shots = [json.loads(l) for l in (root / "project/01_script/shots.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out_root = root / "project/03_shots"

    # 收集所有 dialog 任务
    jobs = []
    for shot in shots:
        sid = shot["shot_id"]
        for idx, d in enumerate(shot.get("dialogue") or []):
            text = (d.get("text") or "").strip()
            if not text:
                continue
            speaker = d.get("speaker", "narrator")
            char = chars.get(speaker, {})
            out = out_root / sid / f"dialog_{idx:02d}_{speaker}.wav"
            jobs.append({
                "sid": sid, "idx": idx, "speaker": speaker,
                "text": text,
                "instruct": build_instruct(char, d),
                "language": LANG_MAP.get((char.get("voice_profile") or {}).get("language", args.lang), args.lang),
                "out": out,
            })
    print(f"[qwen3-tts] {len(jobs)} dialog jobs", flush=True)
    if not jobs:
        print("[skip] no dialogue"); return

    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=dt, attn_implementation=args.attn)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    results = []
    for i, j in enumerate(jobs, 1):
        out = j["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and out.exists() and out.stat().st_size > 1000:
            results.append({**{k: str(v) if isinstance(v, Path) else v for k, v in j.items()}, "status": "skipped"})
            print(f"[{i}/{len(jobs)}] skip {j['sid']}/{out.name}", flush=True); continue
        try:
            t0 = time.time()
            wavs, sr = model.generate_voice_design(text=j["text"], instruct=j["instruct"], language=j["language"])
            sf.write(str(out), wavs[0], sr)
            dur = len(wavs[0]) / sr
            print(f"[{i}/{len(jobs)}] {j['sid']} {j['speaker']} {dur:.2f}s ({time.time()-t0:.1f}s) → {out.name}", flush=True)
            results.append({"sid": j["sid"], "idx": j["idx"], "speaker": j["speaker"],
                            "status": "ok", "out": str(out), "duration": dur, "sample_rate": sr})
        except Exception as e:
            print(f"[err] {j['sid']}: {e}", flush=True)
            results.append({"sid": j["sid"], "idx": j["idx"], "speaker": j["speaker"],
                            "status": "error", "error": str(e)})

    rep = out_root / "qwen3_tts_report.json"
    rep.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {rep}", flush=True)


if __name__ == "__main__":
    main()
