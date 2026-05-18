#!/usr/bin/env bash
# run_30s_server.sh — 在算家云服务器上跑 30s 端到端短片测试
# 前置：工程在 /root/sj-tmp/ai_film/，conda envs: base + qwen3tts，LTX-2 在 /root/sj-tmp/LTX-2/
set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
LOGLINE="${LOGLINE:-一个英国盗贼夜闯古堡塔楼偷取王冠，被守卫发现后惊险跳逃}"
SECONDS_TOTAL="${SECONDS_TOTAL:-30}"
AVG="${AVG:-5}"

source /root/miniconda3/etc/profile.d/conda.sh

echo "==[0] 检查环境 =="
[[ -d /root/sj-tmp/LTX-2/.venv ]] || { echo "缺 LTX-2 venv"; exit 1; }
[[ -d /root/miniconda3/envs/qwen3tts ]] || { echo "缺 qwen3tts conda env"; exit 1; }

conda activate base
pip install -q openai pyyaml requests pillow 2>&1 | tail -3

echo "==[1] 豆包写剧本+分镜 (${SECONDS_TOTAL}s)=="
python pipeline/01_doubao_script.py --logline "$LOGLINE" --total_seconds "$SECONDS_TOTAL" --avg_shot_seconds "$AVG"

echo "==[2a] 豆包种子图 =="
python pipeline/02_doubao_image.py seed || true

echo "==[2b] 豆包每镜参考图 =="
python pipeline/02_doubao_image.py shots

echo "==[4] 生成 LTX 任务清单 =="
python pipeline/04_ltx_i2v.py --build

echo "==[5] Qwen3-TTS 出每镜对话音 =="
conda activate qwen3tts
python server/qwen3_tts_dispatch.py --root "$ROOT" --skip_existing
conda activate base

echo "==[6] LTX-2 渲染所有镜头 =="
/root/sj-tmp/LTX-2/.venv/bin/python server/ltx_worker.py \
    --jobs project/03_shots/ltx_jobs.jsonl \
    --out_root project/03_shots \
    --skip_existing

echo "==[8] 每镜混音 =="
python pipeline/08_mix_shot.py || true

echo "==[9] 全片拼接 =="
python pipeline/09_concat.py

echo "==[10] 终成 =="
python pipeline/10_finalize.py || true

echo
echo "==[done] 输出: $(ls -lh project/04_final/ 2>/dev/null)"
