"""05_voice_dispatch.py — 多人对白 → MiniMax T2A，按 start 错开混合"""
from __future__ import annotations
import sys, json, time, subprocess, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, project_dir, read_jsonl, load_characters

MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
SHOT_INNER_WORKERS = 4  # 同一 shot 内对白并发数


def minimax_t2a(cfg, text, voice_id, emotion=None, speed=1.0) -> bytes:
    url = f"{cfg['api_keys']['minimax_base_url']}/v1/t2a_v2"
    headers = {
        "Authorization": f"Bearer {cfg['api_keys']['minimax']}",
        "Content-Type": "application/json",
    }
    params = {}
    if cfg["api_keys"].get("minimax_group_id"):
        params["GroupId"] = cfg["api_keys"]["minimax_group_id"]
    voice_setting = {
        "voice_id": voice_id,
        "speed": speed,
        "vol": 1.0,
        "pitch": 0,
    }
    if emotion:
        voice_setting["emotion"] = emotion
    body = {
        "model": cfg["minimax"]["tts_model"],
        "text": text,
        "voice_setting": voice_setting,
        "audio_setting": {"format": "wav", "sample_rate": 32000, "channel": 1},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            js = r.json()
            base_resp = js.get("base_resp", {})
            if base_resp.get("status_code", 0) not in (0, None):
                raise RuntimeError(f"minimax err: {base_resp}")
            audio_hex = (js.get("data") or {}).get("audio")
            if not audio_hex:
                raise RuntimeError(f"no audio in response: {str(js)[:500]}")
            return bytes.fromhex(audio_hex)
        except Exception as e:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF ** attempt)
            print(f"  [retry {attempt}/{MAX_RETRIES}] minimax: {e}")
    raise RuntimeError("unreachable")


def _resolve_voice_id(speaker: str, characters: dict, registry: dict) -> str | None:
    char = characters.get(speaker, {})
    vp = char.get("voice_profile", {})
    return (
        registry.get(speaker, {}).get("voice_id")
        or vp.get("voice_id")
        or char.get("minimax_voice_id")
    )


def _engine_for(speaker: str, characters: dict) -> str:
    char = characters.get(speaker, {})
    vp = char.get("voice_profile", {})
    return vp.get("engine") or char.get("voice_engine", "minimax")


def _synth_one(cfg, characters, registry, idx, d, out_dir: Path):
    """合成单句对白，返回 (start, path) 或 None。"""
    speaker = d["speaker"]
    text = d["text"]
    start = float(d.get("start", 0.0))
    engine = _engine_for(speaker, characters)
    seg = out_dir / f"voice_{idx:02d}_{speaker}.wav"

    if engine == "minimax":
        vid = _resolve_voice_id(speaker, characters, registry)
        if not vid:
            print(f"  [skip] {speaker} no voice_id")
            return None
        audio = minimax_t2a(
            cfg, text, vid,
            emotion=d.get("instruct_or_emotion"),
            speed=float(d.get("speed", 1.0)),
        )
        seg.write_bytes(audio)
        return (start, seg)

    if engine == "qwen3-tts":
        print(f"  [todo] {speaker} → Qwen3-TTS (remote)")
        return None

    print(f"  [skip] {speaker} unknown engine: {engine}")
    return None


def _mix_parts(parts: list[tuple[float, Path]], final: Path) -> Path:
    """将多段对白按 start 错开混合为一个 wav。"""
    if len(parts) == 1 and parts[0][0] == 0.0:
        final.write_bytes(parts[0][1].read_bytes())
        return final

    inputs, filters, amix_in = [], [], []
    for i, (start, p) in enumerate(parts):
        inputs += ["-i", str(p)]
        delay_ms = int(start * 1000)
        filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
        amix_in.append(f"[a{i}]")
    filt = (
        ";".join(filters)
        + f";{''.join(amix_in)}amix=inputs={len(parts)}:duration=longest[out]"
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", *inputs, "-filter_complex", filt,
             "-map", "[out]", "-ar", "32000", "-ac", "1", str(final)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed:\n{e.stderr.decode('utf-8', errors='ignore')[:800]}"
        ) from e
    return final


def render_shot_voice(cfg, characters, registry, shot, out_dir: Path):
    dialogues = shot.get("dialogue") or []
    if not dialogues:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: list[tuple[float, Path]] = []
    workers = min(SHOT_INNER_WORKERS, len(dialogues))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_synth_one, cfg, characters, registry, i, d, out_dir): i
            for i, d in enumerate(dialogues)
        }
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res is not None:
                    parts.append(res)
            except Exception as e:
                idx = futures[fut]
                print(f"  [err] dialogue#{idx}: {e}")

    if not parts:
        return None
    parts.sort(key=lambda x: x[0])
    return _mix_parts(parts, out_dir / "voice.wav")


def main():
    cfg = load_config()
    characters = load_characters()
    shots = read_jsonl(project_dir("01_script") / "shots.jsonl")
    rp = project_dir("02_assets") / "voice_registry.json"
    registry = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}

    for shot in shots:
        sid = shot["shot_id"]
        out_dir = project_dir(f"03_shots/{sid}")
        if (out_dir / "voice.wav").exists():
            print(f"[skip] {sid}")
            continue
        try:
            print(f"[voice] {sid}")
            render_shot_voice(cfg, characters, registry, shot, out_dir)
        except Exception as e:
            print(f"  [err] {sid}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
