#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
LOGLINE="${LOGLINE:-一个英国盗贼夜闯古堡塔楼偷取王冠，被守卫发现后惊险跳逃}"
SECONDS_TOTAL="${SECONDS_TOTAL:-60}"
AVG="${AVG:-5}"

echo "==[1] 豆包写剧本+分镜=="
python pipeline/01_doubao_script.py --logline "$LOGLINE" --total_seconds "$SECONDS_TOTAL" --avg_shot_seconds "$AVG"

echo "==[2a] 豆包出角色种子图=="
python pipeline/02_doubao_image.py seed

echo "==[2b] 豆包按 shots 出每镜头参考图=="
python pipeline/02_doubao_image.py shots

echo "==[3] MiniMax 复刻=="
python pipeline/03_minimax_clone.py || true

echo "==[5] MiniMax 配音=="
python pipeline/05_voice_dispatch.py

echo "==[4/6] 远程 GPU 任务清单=="
python pipeline/04_ltx_i2v.py --build
python pipeline/06_sao_audio.py --build

echo
echo "下一步（远程 A100）："
echo "  scp -r project/02_assets/refs project/03_shots/ltx_jobs.jsonl project/03_shots/sao_jobs.jsonl  A100:/root/sj-tmp/jobs/"
echo "  跑 LTX + SAO worker，回传 video.mp4 / bgm.wav / sfx_*.wav"
echo "  本地: python pipeline/08_mix_shot.py && python pipeline/09_concat.py && python pipeline/10_finalize.py"
