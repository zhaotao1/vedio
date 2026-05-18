"""03_minimax_clone.py — 一次性把角色参考音频上传到 MiniMax 复刻，写回 voice_registry.json"""
from __future__ import annotations
import sys, json, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, project_dir, write_json, load_characters

def upload_file(cfg, fp: Path) -> str:
    url = f"{cfg['api_keys']['minimax_base_url']}/v1/files/upload"
    h = {"Authorization": f"Bearer {cfg['api_keys']['minimax']}"}
    params = {}
    if cfg["api_keys"].get("minimax_group_id"):
        params["GroupId"] = cfg["api_keys"]["minimax_group_id"]
    with open(fp, "rb") as f:
        r = requests.post(url, headers=h, params=params,
                          files={"file": (fp.name, f, "audio/mpeg")},
                          data={"purpose": "voice_clone"}, timeout=120)
    r.raise_for_status()
    js = r.json()
    fid = js.get("file", {}).get("file_id") or js.get("file_id")
    if not fid: raise RuntimeError(f"no file_id: {js}")
    return str(fid)

def clone_voice(cfg, file_id: str, voice_id: str) -> dict:
    url = f"{cfg['api_keys']['minimax_base_url']}/v1/voice_clone"
    h = {"Authorization": f"Bearer {cfg['api_keys']['minimax']}",
         "Content-Type": "application/json"}
    params = {}
    if cfg["api_keys"].get("minimax_group_id"):
        params["GroupId"] = cfg["api_keys"]["minimax_group_id"]
    body = {"file_id": file_id, "voice_id": voice_id,
            "model": cfg["minimax"]["tts_model"], "need_noise_reduction": True}
    r = requests.post(url, headers=h, params=params, json=body, timeout=180)
    r.raise_for_status()
    return r.json()

def main():
    cfg = load_config()
    refs_dir = project_dir("02_assets/voice_refs")
    out_path = project_dir("02_assets") / "voice_registry.json"
    registry = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    for cid, c in load_characters().items():
        vp = c.get("voice_profile", {})
        engine = vp.get("engine") or c.get("voice_engine")
        if engine != "minimax": continue
        vid = vp.get("voice_id") or c.get("minimax_voice_id") or f"{cid}_v1"
        if registry.get(cid, {}).get("status") == "ok":
            print(f"[skip] {cid} done: {registry[cid]['voice_id']}"); continue
        ref = next((refs_dir / f"{cid}.{ext}" for ext in ("mp3","wav","m4a")
                    if (refs_dir / f"{cid}.{ext}").exists()), None)
        if not ref:
            print(f"[warn] no ref for {cid}, put audio at {refs_dir}/{cid}.mp3")
            registry[cid] = {"status": "missing_ref", "voice_id": vid}; continue
        try:
            print(f"[upload] {cid} ← {ref.name}")
            fid = upload_file(cfg, ref); print(f"  file_id={fid}")
            print(f"[clone] voice_id={vid}")
            res = clone_voice(cfg, fid, vid)
            registry[cid] = {"status": "ok", "voice_id": vid, "raw": res}
        except Exception as e:
            print(f"  [err] {e}")
            registry[cid] = {"status": "error", "voice_id": vid, "error": str(e)}
    write_json(out_path, registry)
    print(f"[ok] → {out_path}")

if __name__ == "__main__":
    main()
