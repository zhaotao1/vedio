"""qwen3_tts_gen.py — 用 Qwen3-TTS VoiceDesign 按文字描述生成参考音。

用法：
  python qwen3_tts_gen.py \
      --text "深夜潜入古堡的盗贼，对自己说：今晚必须拿到那本书。" \
      --instruct "30多岁英国口音男性，低沉沙哑，狡黠语气" \
      --language Chinese \
      --out /root/sj-tmp/refs/thief.wav

加 --batch jobs.jsonl 可批量（每行 {id,text,instruct,language}）。
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

MODEL_DIR_DEFAULT = "/root/sj-tmp/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def load_model(model_dir: str, dtype: str = "bfloat16", attn: str = "flash_attention_2"):
    import torch
    from qwen_tts import Qwen3TTSModel
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    print(f"[load] {model_dir} dtype={dtype} attn={attn}", flush=True)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda:0",
        dtype=dt,
        attn_implementation=attn,
    )
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)
    return model


def synth_one(model, text: str, instruct: str, language: str, out_path: Path) -> dict:
    import soundfile as sf
    t0 = time.time()
    wavs, sr = model.generate_voice_design(text=text, instruct=instruct or "", language=language or "Chinese")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wavs[0], sr)
    dur = len(wavs[0]) / sr
    print(f"[ok] {out_path}  sr={sr}  dur={dur:.2f}s  ({time.time()-t0:.1f}s)", flush=True)
    return {"path": str(out_path), "sample_rate": sr, "duration": dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--text", default=None)
    ap.add_argument("--instruct", default="")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--out", default="/root/sj-tmp/refs/out.wav")
    ap.add_argument("--batch", default=None, help="jobs.jsonl: 每行 {id,text,instruct,language,out?}")
    ap.add_argument("--out_dir", default="/root/sj-tmp/refs")
    args = ap.parse_args()

    model = load_model(args.model_dir, args.dtype, args.attn)

    if args.batch:
        results = []
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.batch, "r", encoding="utf-8") as f:
            jobs = [json.loads(l) for l in f if l.strip()]
        print(f"[batch] {len(jobs)} jobs", flush=True)
        for j in jobs:
            jid = j.get("id") or j.get("name") or "noname"
            out = Path(j.get("out") or (out_dir / f"{jid}.wav"))
            try:
                r = synth_one(model, j["text"], j.get("instruct", ""), j.get("language", "Chinese"), out)
                results.append({"id": jid, "status": "ok", **r})
            except Exception as e:
                print(f"[err] {jid}: {e}", flush=True)
                results.append({"id": jid, "status": "error", "error": str(e)})
        rep = Path(args.out_dir) / "report.json"
        rep.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] report → {rep}", flush=True)
        return

    if not args.text:
        print("ERR: 需要 --text 或 --batch", file=sys.stderr); sys.exit(2)
    synth_one(model, args.text, args.instruct, args.language, Path(args.out))


if __name__ == "__main__":
    main()
