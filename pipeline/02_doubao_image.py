"""02_doubao_image.py — Doubao-Seedream-4.5：T2I + i2i + 多图融合 一把梭"""
from __future__ import annotations
import argparse, sys, time, base64
from pathlib import Path
from urllib.request import urlopen
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, doubao_client, project_dir, read_jsonl, load_characters

POSE_KIT = [
    ("01_face_front_closeup",  "frontal face close-up portrait, neutral expression, soft key light"),
    ("02_face_three_quarter",  "three-quarter face portrait"),
    ("03_face_side",           "strict side profile portrait"),
    ("04_body_full_front",     "full body front view, standing, plain background"),
    ("05_body_full_back",      "full body back view, walking away"),
    ("06_action_running",      "dynamic action shot, running"),
    ("07_action_climbing",     "climbing a stone wall, looking up"),
    ("08_emotion_smirk",       "smirking expression, close-up, dramatic side light"),
]

def _save(item, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if item.get("b64_json"):
        out.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        out.write_bytes(urlopen(item["url"], timeout=120).read())
    else:
        raise RuntimeError(f"无法解析: {item}")
    print(f"  → {out}")

def _to_data_url(p_or_url):
    if isinstance(p_or_url, str) and p_or_url.startswith(("http://","https://","data:")):
        return p_or_url
    p = Path(p_or_url) if not isinstance(p_or_url, Path) else p_or_url
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

def _gen(client, cfg, prompt, ref_images=None, seed=None, size=None):
    import requests
    url = f"{cfg['api_keys']['doubao_base_url']}/images/generations"
    h = {"Authorization": f"Bearer {cfg['api_keys']['doubao_ark']}",
         "Content-Type": "application/json"}
    body = {"model": cfg["doubao"]["image_model"], "prompt": prompt,
            "size": size or cfg["style"]["resolution"],
            "response_format": "b64_json"}
    if seed is not None: body["seed"] = int(seed)
    if ref_images:
        imgs = []
        for x in ref_images:
            if isinstance(x, str) and x.startswith(("http://","https://","data:")):
                imgs.append(x)
            else:
                imgs.append(_to_data_url(x if isinstance(x, Path) else Path(x)))
        body["image"] = imgs if len(imgs) > 1 else imgs[0]
    r = requests.post(url, headers=h, json=body, timeout=300)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:500]}")
    js = r.json()
    return js["data"][0]

def cmd_seed(args, cfg, client):
    style = cfg["style"]["global_prompt_suffix"]
    out_root = project_dir("02_assets/characters")
    for cid, c in load_characters().items():
        if "appearance" not in c: continue
        sp = out_root / cid / "00_seed.png"
        if sp.exists() and not args.force:
            print(f"[skip] {sp}"); continue
        prompt = (f"Cinematic character reference photo of {c['appearance']}. "
                  f"Three-quarter angle portrait, neutral plain background, "
                  f"soft key light, sharp focus, character sheet quality. {style}")
        print(f"[seed] {cid}")
        _save(_gen(client, cfg, prompt, seed=c.get("seed", 42)), sp)

def cmd_kit(args, cfg, client):
    style = cfg["style"]["global_prompt_suffix"]
    out_root = project_dir("02_assets/characters")
    characters = load_characters()
    targets = [args.char] if args.char else list(characters.keys())
    for cid in targets:
        c = characters.get(cid, {})
        if "appearance" not in c: continue
        seed_p = out_root / cid / "00_seed.png"
        if not seed_p.exists():
            print(f"[warn] no seed for {cid}, run `seed` first"); continue
        ref = _to_data_url(seed_p)
        for fname, pose in POSE_KIT:
            tgt = out_root / cid / f"{fname}.png"
            if tgt.exists() and not args.force: continue
            prompt = (f"Same person as reference image. {c['appearance']}. "
                      f"{pose}. Keep face identity strictly consistent. {style}")
            print(f"[kit] {cid}/{fname}")
            try:
                _save(_gen(client, cfg, prompt, ref_images=[ref], seed=c.get("seed", 42)), tgt)
                time.sleep(0.5)
            except Exception as e:
                print(f"  [err] {e}")

def cmd_shots(args, cfg, client):
    style = cfg["style"]["global_prompt_suffix"]
    shots = read_jsonl(project_dir("01_script") / "shots.jsonl")
    char_dir = project_dir("02_assets/characters")
    out_root = project_dir("02_assets/refs")
    for shot in shots:
        sid = shot["shot_id"]
        tgt = out_root / f"{sid}.png"
        if tgt.exists() and not args.force: continue
        cast = shot.get("cast", [])
        refs = []
        for cid in cast:
            sp = char_dir / cid / "00_seed.png"
            if sp.exists():
                refs.append(_to_data_url(sp))
        hint = shot.get("ref_image_hint", "")
        prompt = (f"{shot.get('video_prompt','')}. Composition: {hint}. "
                  f"Frame the first frame of a 5-second cinematic shot. {style}")
        print(f"[shot] {sid} cast={cast}")
        try:
            _save(_gen(client, cfg, prompt, ref_images=refs or None, seed=42), tgt)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [err] {e}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("seed"); p1.add_argument("--force", action="store_true")
    p2 = sub.add_parser("kit"); p2.add_argument("--char", default=None); p2.add_argument("--force", action="store_true")
    p3 = sub.add_parser("shots"); p3.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(); client = doubao_client(cfg)
    print(f"[doubao-image] model={cfg['doubao']['image_model']}")
    {"seed": cmd_seed, "kit": cmd_kit, "shots": cmd_shots}[args.cmd](args, cfg, client)

if __name__ == "__main__":
    main()
