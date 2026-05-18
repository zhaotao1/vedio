"""01_doubao_script.py — 用 doubao-seed-2-0-pro-260215 生成完整 shots.jsonl"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, doubao_client, project_dir, write_jsonl, write_json

SYSTEM_PROMPT = """你是顶级 AI 短片编剧 + 分镜师。输出严格 JSON。

【一致性铁律】
1. 任何两人或以上的对话戏，必须拆成「正反打」：首尾各 1 个 two_shot 广角双人镜头(2-3s)，中间用 single 单人特写交替(每镜 3-5s)
2. 三人以上场景标注 type=crowd
3. 同一角色全片 appearance 必须来自 characters，并在所有镜头中一字不差复用

【镜头节奏】动作 1.5-2.5s / 文戏 3-5s / 写景 4-6s / 高潮特写 1-3s

【prompt】video_prompt 用英文，画面优先；末尾必拼接 style_suffix
ref_image_hint 描述这个镜头需要哪种参考图

【LTX-2 video_prompt 写作规范（必须严格遵守，参考 ltx.io 官方 prompting guide）】
- 单段连贯英文段落，4-8 句，全部 present tense；不要分点不要换行
- 顺序：风格/genre 先放段首 → 镜头景别与机位 → 场景/光线/色调/氛围 → 角色（年龄/发型/服饰/可视化情绪）→ 动作按时间顺序展开 → 镜头运动 → 对白与环境音
- 情绪用可视线索：trembling lips / clenched fists / wide darting eyes / shoulders sinking 等；禁止使用 sad / nervous / confused 这类纯内部状态词
- 镜头语言用 LTX-2 推荐动词：dolly in、dolly back、push in、pull back、tracks、follows、pans across、tilts up、circles around、handheld、over-the-shoulder、static frame、wide establishing
- 对白直接内联在 video_prompt 中，加引号；如有口音/语气，用括号或同句标注，例：The guard shouts in a deep British accent, "Stop! Thief!"
- 环境音/氛围用具体描述："distant clock ticking", "wind howling through the stone tower"
- 不要再额外拼接 style_suffix（由 04_ltx_i2v.py 自动追加）
- 避免：可读文字/品牌/招牌；高难物理（连续翻滚/复杂跳跃）；多于 3 个清晰角色同框；冲突光源；过多动作堆叠
- 跳跃/翻滚/打斗等高难动作镜头：type 设 "action_jump"，ltx_conditioning.mode 设 "first_last"，并提供首尾两个 keyframes
- 普通对话/特写镜头：ltx_conditioning.mode 设 "first_frame"，只给 1 个 keyframe
- 复杂多阶段动作（>5s 且有明显路径变化）：ltx_conditioning.mode 设 "multi_keyframes"

【输出 JSON Schema】
{
    "title": "...", "logline": "...",
    "characters": {
        "thief": {
            "appearance":"...",
            "seed":42,
            "voice_profile": {
                "engine":"minimax|qwen3-tts",
                "voice_id":"thief_v1",
                "age":"...",
                "gender":"...",
                "accent":"...",
                "timbre":"...",
                "pace":"...",
                "personality":"...",
                "qwen3_tts_instruct":"..."
            }
        }
    },
  "scenes": [{"id":"S01","location":"...","mood":"...","bgm_prompt":"..."}],
  "shots": [{
    "shot_id":"S01_01","scene_id":"S01","duration":5,
    "type":"single|two_shot|crowd|action_jump","cast":["thief"],
    "video_prompt":"<LTX-2 4-8 句段落，按上面规范写>",
    "camera":"wide / medium / closeup / push in / dolly back / handheld / over-the-shoulder",
    "ref_image_hint":"thief, three-quarter angle, climbing pose",
    "ltx_conditioning":{
      "mode":"first_frame|first_last|multi_keyframes",
      "keyframes":[
        {"time":0.0,"prompt":"start frame composition"},
        {"time":5.0,"prompt":"end frame composition"}
      ]
    },
    "dialogue":[{"speaker":"thief","text":"...","start":0.5,"instruct_or_emotion":"low whisper, British accent"}],
    "sfx":[{"name":"wind","prompt":"cold night wind","start":0.0,"duration":5.0,"volume":0.3}],
    "transition_out":"cut|fade|xfade"
  }]
}
"""

def build_user_prompt(cfg, logline, total_seconds, avg):
    n = max(1, total_seconds // avg)
    return f"""【全片信息】
- logline: {logline}
- 总时长: {total_seconds} 秒
- 平均镜头长度: {avg} 秒
- 目标镜头数: 约 {n} 个
- 全局风格后缀（每个 video_prompt 末尾必加）: "{cfg['style']['global_prompt_suffix']}"
- 分辨率: {cfg['style']['resolution']}

【角色生成要求】
- 请先生成顶层 characters，包含主要角色的 appearance、seed、voice_profile
- voice_profile 必须包含 engine、voice_id、age、gender、accent、timbre、pace、personality、qwen3_tts_instruct
- thief 和 guard 必须存在；如果需要 narrator，也放到 characters
- 每个 shot.cast 只能引用 characters 中已定义的角色 id
- 每个 video_prompt 中提到角色时，必须复用 characters 里的 appearance

请直接输出符合上面 schema 的完整 JSON，不要任何解释文字。
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logline", required=True)
    ap.add_argument("--total_seconds", type=int, default=60)
    ap.add_argument("--avg_shot_seconds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    client = doubao_client(cfg)
    print(f"[doubao] model={cfg['doubao']['llm_model']}")
    resp = client.chat.completions.create(
        model=cfg["doubao"]["llm_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(cfg, args.logline, args.total_seconds, args.avg_shot_seconds)},
        ],
        temperature=0.8,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    # 截取首个 { 到末尾 }
    a, b = raw.find("{"), raw.rfind("}")
    if a >= 0 and b > a:
        raw = raw[a:b+1]
    data = json.loads(raw)
    out = project_dir("01_script")
    write_json(out / "outline.json", data)
    write_jsonl(out / "shots.jsonl", data.get("shots", []))
    write_jsonl(out / "scenes.jsonl", data.get("scenes", []))
    print(f"[ok] {len(data.get('shots', []))} shots → {out/'shots.jsonl'}")

if __name__ == "__main__":
    main()
