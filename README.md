# AI Film Pipeline

Hybrid pipeline for a cinematic short film:

- Script and shot list: Doubao `doubao-seed-2-0-pro-260215`
- Character seed images and per-shot references: Doubao Seedream `doubao-seedream-4-5-251128`
- Voice cloning / dialogue: MiniMax `speech-02-hd`
- Video generation: remote A100 with LTX-2.3
- BGM / SFX generation: remote A100 with Stable Audio Open 1.0
- Editing: local ffmpeg

## Current Demo

Generated assets are under `project/`.

Content lives in `project/01_script/outline.json`:

- `title` and `logline`: story identity
- `characters`: character appearance, fixed voice profile, voice id
- `scenes`: locations and BGM prompts
- `shots`: shot timing, cast, visual prompt, dialogue, SFX

`config.yaml` is only for engineering configuration: API keys, model names, global style, paths, and runtime options. It should not contain character definitions.

```bash
./run_demo.sh
```

The demo logline defaults to:

```text
一个英国盗贼夜闯古堡塔楼偷取王冠，被守卫发现后惊险跳逃
```

## Steps

```bash
python pipeline/01_doubao_script.py --logline "一个英国盗贼夜闯古堡塔楼偷取王冠，被守卫发现后惊险跳逃" --total_seconds 60 --avg_shot_seconds 5
python pipeline/02_doubao_image.py seed
python pipeline/02_doubao_image.py shots
python pipeline/03_minimax_clone.py
python pipeline/05_voice_dispatch.py
python pipeline/04_ltx_i2v.py --build
python pipeline/06_sao_audio.py --build
```

Then copy these to the A100 machine and run the GPU workers there:

```bash
project/02_assets/refs/
project/03_shots/ltx_jobs.jsonl
project/03_shots/sao_jobs.jsonl
```

After the A100 returns each shot's `video.mp4`, scene BGM, and shot SFX files, finish locally:

```bash
python pipeline/08_mix_shot.py
python pipeline/09_concat.py
python pipeline/10_finalize.py
```

## Notes

- Seedream 4.5 requires image sizes of at least 3,686,400 pixels. This config uses `2560x1440`.
- Doubao LLM model does not support `response_format={"type":"json_object"}`, so the script asks for strict JSON and parses the JSON block manually.
- Character appearance and fixed voice profiles are stored in `project/01_script/outline.json` under `characters`.
- Put MiniMax voice reference audio at `project/02_assets/voice_refs/thief.mp3` and `project/02_assets/voice_refs/guard.mp3` before running voice clone.
- If a Seedream call times out, rerun `python pipeline/02_doubao_image.py shots`; existing PNGs are skipped.
