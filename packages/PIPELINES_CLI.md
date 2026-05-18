# LTX-2 Pipelines 命令行参数手册

`ltx-pipelines` 中所有流水线统一通过 **CLI 参数**配置（**不读配置文件**）。
参数解析器集中定义在 [ltx-pipelines/src/ltx_pipelines/utils/args.py](ltx-pipelines/src/ltx_pipelines/utils/args.py)，每个 pipeline 模块以 `python -m ltx_pipelines.<name>` 运行。

> 路径参数都会经过 `Path(...).expanduser().resolve()`，支持 `~` 和相对路径。

---

## 目录

- [通用参数（所有 pipeline 共享）](#通用参数所有-pipeline-共享)
  - [模型与权重](#模型与权重)
  - [生成内容](#生成内容)
  - [LoRA 与量化](#lora-与量化)
  - [显存与性能](#显存与性能)
- [视频生成参数（new_video_gen_arg_parser）](#视频生成参数)
- [引导参数（default_1_stage_arg_parser）](#引导参数)
- [两阶段额外参数（default_2_stage_arg_parser）](#两阶段额外参数)
- [各流水线专属参数](#各流水线专属参数)
  - [ti2vid_two_stages](#1-ti2vid_two_stages-推荐生产)
  - [ti2vid_two_stages_hq](#2-ti2vid_two_stages_hq)
  - [ti2vid_one_stage](#3-ti2vid_one_stage)
  - [distilled](#4-distilled-最快)
  - [ic_lora](#5-ic_lora-video-to-video)
  - [keyframe_interpolation](#6-keyframe_interpolation)
  - [a2vid_two_stage](#7-a2vid_two_stage-audio-to-video)
  - [retake](#8-retake-视频时间片段重生)
  - [hdr_ic_lora](#9-hdr_ic_lora-hdr-视频-to-视频)
  - [lipdub](#10-lipdub-口型对配音)

---

## 通用参数（所有 pipeline 共享）

来自 `basic_arg_parser()`（[args.py L162](ltx-pipelines/src/ltx_pipelines/utils/args.py#L162)）。

### 模型与权重

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `--checkpoint-path` | path | ✅* | LTX-2 主模型 `.safetensors`。**非蒸馏** pipeline 必填 |
| `--distilled-checkpoint-path` | path | ✅* | LTX-2 **蒸馏模型** `.safetensors`。蒸馏类 pipeline（`distilled` / `ic_lora` / `retake` / `hdr_ic_lora` / `lipdub`）必填 |
| `--gemma-root` | path | ✅ | Gemma 文本编码器**根目录**（包含 tokenizer 与权重文件） |

\* 两者二选一，由 pipeline 类型决定。

### 生成内容

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--prompt` | str | ✅ | — | 视频内容描述文本 |
| `--output-path` | path | ✅ | — | 输出 MP4 文件路径 |
| `--seed` | int | ❌ | 由 params 决定 | 随机种子（用于复现） |
| `--num-inference-steps` | int | ❌ | 由 params 决定 | 去噪步数（**仅非蒸馏 pipeline**）。越大质量越高、越慢 |
| `--enhance-prompt` | flag | ❌ | false | 启用提示词增强 |

### LoRA 与量化

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--lora PATH [STRENGTH]` | 1~2 值 | `[]` | 加载 LoRA，可多次指定。默认强度 1.0。例：`--lora a.safetensors 0.8 --lora b.safetensors` |
| `--quantization` | `{fp8-cast, fp8-scaled-mm}` | 关闭 | FP8 量化策略。`fp8-cast`：FP8 存储+推理时升精度；`fp8-scaled-mm`：FP8 缩放矩阵乘（从 checkpoint 的 `.weight_scale` 自动发现层） |

### 显存与性能

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--offload` | `{none, cpu, disk}` | `none` | 权重卸载策略。`none` 全在 GPU；`cpu` 常驻 CPU RAM 按层流式拷贝；`disk` 按需从磁盘读取（最省显存） |
| `--max-batch-size N` | int ≥1 | `1` | 单次 transformer 前向的最大 batch。`4` 可把 guidance 多次前向合并，减少 PCIe 传输 |
| `--compile` | flag | false | 启用 `torch.compile` 编译 transformer 块以加速 |

---

## 视频生成参数

来自 `new_video_gen_arg_parser()`（[args.py L285](ltx-pipelines/src/ltx_pipelines/utils/args.py#L285)）。用于所有 T/I→V 类 pipeline。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--height` | int | 由 params 决定 | 视频高度（像素），需是 32 倍数（两阶段第二阶段 64 倍数） |
| `--width` | int | 由 params 决定 | 视频宽度（像素），需是 32 倍数（两阶段第二阶段 64 倍数） |
| `--num-frames` | int | 由 params 决定 | 总帧数，必须 `= 8K + 1`（如 97、193） |
| `--frame-rate` | float | 由 params 决定 | 帧率（fps） |
| `--image PATH FRAME_IDX STRENGTH [CRF]` | mixed | `[]` | 图像条件：路径、目标帧索引、条件强度，CRF 可选（H.264 压缩质量，0=无损）。可多次。例：`--image a.jpg 0 0.8 --image b.jpg 160 0.9 0` |

---

## 引导参数

来自 `default_1_stage_arg_parser()`（[args.py L390](ltx-pipelines/src/ltx_pipelines/utils/args.py#L390)）。用于带 CFG/STG 引导的 pipeline。

### 通用

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--negative-prompt` | str | 内置默认 | 负向提示词，引导模型远离不想要的内容 |

### 视频引导（针对视频流）

| 参数 | 类型 | 说明 |
|---|---|---|
| `--video-cfg-guidance-scale` | float | 视频 CFG 强度。越大越贴合 prompt，越小多样性更好。1.0=无效果 |
| `--video-stg-guidance-scale` | float | 视频 STG（时空引导）强度。0.0=无效果 |
| `--video-rescale-scale` | float | 视频引导后重标定强度。越大越能抑制过饱和。0.0=无效果 |
| `--video-stg-blocks` | int 列表 | 哪些 transformer 块参与视频 STG 扰动 |
| `--a2v-guidance-scale` | float | 音频→视频跨注意力扰动强度。提高口型同步质量。1.0=无效果 |
| `--video-skip-step N` | int | 视频扩散过程周期跳步：仅在 `step_index % (N+1) == 0` 时计算。`0`=不跳；`1`=隔一步跳 |

### 音频引导（针对音频流）

| 参数 | 类型 | 说明 |
|---|---|---|
| `--audio-cfg-guidance-scale` | float | 音频 CFG 强度。1.0=无效果 |
| `--audio-stg-guidance-scale` | float | 音频 STG 强度。0.0=无效果 |
| `--audio-rescale-scale` | float | 音频重标定（实验性）。0.0=无效果 |
| `--audio-stg-blocks` | int 列表 | 哪些 transformer 块参与音频 STG 扰动 |
| `--v2a-guidance-scale` | float | 视频→音频跨注意力扰动强度。1.0=无效果 |
| `--audio-skip-step N` | int | 音频扩散周期跳步，同 video-skip-step |

---

## 两阶段额外参数

来自 `default_2_stage_arg_parser()`（[args.py L530](ltx-pipelines/src/ltx_pipelines/utils/args.py#L530)）。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `--distilled-lora PATH [STRENGTH]` | 1~2 值 | ✅ | 第二阶段（上采样+精修）用的蒸馏 LoRA |
| `--spatial-upsampler-path` | path | ✅ | 空间上采样器权重（潜在空间放大） |

第二阶段会把视频分辨率 ×2 并用更少步数（无 CFG）的蒸馏调度做精修。

---

## 各流水线专属参数

### 1. `ti2vid_two_stages` 推荐生产

**参数集**：[通用](#通用参数所有-pipeline-共享) + [视频生成](#视频生成参数) + [引导](#引导参数) + [两阶段](#两阶段额外参数)。无额外参数。

```bash
python -m ltx_pipelines.ti2vid_two_stages \
  --checkpoint-path /models/ltxv-2.safetensors \
  --distilled-lora /models/distilled_lora.safetensors 0.8 \
  --spatial-upsampler-path /models/upsampler.safetensors \
  --gemma-root /models/gemma-3 \
  --prompt "A beautiful sunset over the ocean" \
  --output-path output.mp4
```

---

### 2. `ti2vid_two_stages_hq`

`hq_2_stage_arg_parser()`。在两阶段基础上：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--distilled-lora-strength-stage-1` | float | `0.25` | 第一阶段也叠加蒸馏 LoRA 的强度 |
| `--distilled-lora-strength-stage-2` | float | `0.5` | 第二阶段蒸馏 LoRA 的强度 |

使用 res_2s 二阶采样器，相同质量下可用更少步数。

---

### 3. `ti2vid_one_stage`

**参数集**：[通用](#通用参数所有-pipeline-共享) + [视频生成](#视频生成参数) + [引导](#引导参数)。无 `--distilled-lora` / `--spatial-upsampler-path`，仅作教学/原型用。

---

### 4. `distilled` 最快

`default_2_stage_distilled_arg_parser()`：[通用](#通用参数所有-pipeline-共享)（蒸馏） + [视频生成](#视频生成参数) + `--spatial-upsampler-path`。**无引导参数**（8+4 步固定 sigma）。

```bash
python -m ltx_pipelines.distilled \
  --distilled-checkpoint-path /models/distilled.safetensors \
  --spatial-upsampler-path /models/upsampler.safetensors \
  --gemma-root /models/gemma-3 \
  --prompt "..." --output-path out.mp4
```

---

### 5. `ic_lora` Video-to-Video

蒸馏两阶段 + 视频条件：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `--video-conditioning PATH STRENGTH` | 2 值 | ✅ | 参考视频路径 + 条件强度 |
| `--conditioning-attention-mask MASK_PATH STRENGTH` | 2 值 | ❌ | 灰度 mask 视频（[0,1] 区域强度）+ 标量倍数。0.0=忽略 IC-LoRA 条件；1.0=完全应用 |
| `--skip-stage-2` | flag | ❌ | 跳过第二阶段上采样，输出半分辨率（用于快速迭代） |

需配合 IC-LoRA 训练的蒸馏 checkpoint。

---

### 6. `keyframe_interpolation`

参数集同 [两阶段](#两阶段额外参数) + [引导](#引导参数)。无额外参数。通过 `--image PATH FRAME_IDX STRENGTH` 指定多个关键帧（在不同帧索引位置）。

---

### 7. `a2vid_two_stage` Audio-to-Video

两阶段 + 音频条件：

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--audio-path` | str | ✅ | — | 用于条件生成的音频文件 |
| `--audio-start-time` | float | ❌ | `0.0` | 从音频的哪一秒开始读取 |
| `--audio-max-duration` | float | ❌ | 视频时长 | 最大音频时长（秒） |

---

### 8. `retake` 视频时间片段重生

`video_editing_arg_parser(distilled=True)`：[通用](#通用参数所有-pipeline-共享)（蒸馏） + 

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `--video-path` | path | ✅ | 源视频路径 |
| `--start-time` | float | ✅ | 待重新生成区间起始时间（秒） |
| `--end-time` | float | ✅ | 待重新生成区间结束时间（秒） |

约束：源视频帧数需满足 `8K+1`，宽高需 32 倍数。无 `--height/--width/--num-frames`（沿用源视频）。

---

### 9. `hdr_ic_lora` HDR 视频-to-视频

**独立 parser**（不使用 `basic_arg_parser`，参数命名风格略有不同）：

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--input` | path | ✅ | — | 单个 `.mp4` 或包含 `.mp4` 的目录（批处理） |
| `--output-dir` | path | ✅ | — | 输出目录（生成 `.mov` 与 EXR 子文件夹） |
| `--hdr-lora` | path | ✅ | — | HDR IC-LoRA `.safetensors` |
| `--text-embeddings` | path | ✅ | — | 预计算的文本嵌入 `.safetensors`（无需再加载 Gemma） |
| `--distilled-checkpoint-path` | path | ✅ | — | 蒸馏 checkpoint |
| `--spatial-upsampler-path` | path | ✅ | — | 空间上采样器权重 |
| `--num-frames` | int | ❌ | 内置默认 | 输出帧数，需满足 `(n-1) % 8 == 0` |
| `--spatial-tile` | int | ❌ | 内置默认 | 分块 VAE 解码的空间块大小。低显存可设 768 |
| `--skip-mp4` | flag | ❌ | false | 不输出 H.264 MP4，仅出 EXR |
| `--exr-half` | flag | ❌ | false | EXR 用 float16 存储 |
| `--seed` | int | ❌ | `10` | 随机种子 |
| `--offload` | `{none, cpu, disk}` | ❌ | `none` | 权重卸载策略 |
| `--high-quality` | flag | ❌ | false | HQ 模式：内部按 2× 帧生成后隔帧抽取，更平滑但慢 ~2× |

---

### 10. `lipdub` 口型对配音

`lipdub_arg_parser()`：[通用](#通用参数所有-pipeline-共享)（蒸馏） + 

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--height` | int | ❌ | 第二阶段默认高度 | 输出高度，64 倍数 |
| `--width` | int | ❌ | 第二阶段默认宽度 | 输出宽度，64 倍数 |
| `--spatial-upsampler-path` | path | ✅ | — | 空间上采样器权重 |
| `--reference-video` | path | ✅ | — | 参考视频（视频 + 音轨，提供 IC-LoRA 视觉参考与音频身份） |
| `--reference-strength` | float | ❌ | `1.0` | IC-LoRA 视频参考的条件强度 |
| `--lora PATH [STRENGTH]` | — | ✅ | — | **必须且仅一个** LoRA：即 lip-dub IC-LoRA |

帧数与帧率从 `--reference-video` 自动读取并下取到 `8K+1`。无 `--num-frames` / `--frame-rate` / `--image`。

---

## 速查：各 pipeline 必填权重

| Pipeline | checkpoint | gemma | upsampler | 额外权重/输入 |
|---|---|---|---|---|
| `ti2vid_two_stages` | `--checkpoint-path` | ✅ | ✅ | `--distilled-lora` |
| `ti2vid_two_stages_hq` | `--checkpoint-path` | ✅ | ✅ | `--distilled-lora` |
| `ti2vid_one_stage` | `--checkpoint-path` | ✅ | ❌ | — |
| `distilled` | `--distilled-checkpoint-path` | ✅ | ✅ | — |
| `ic_lora` | `--distilled-checkpoint-path` | ✅ | ✅ | `--video-conditioning` + IC-LoRA via `--lora` |
| `keyframe_interpolation` | `--checkpoint-path` | ✅ | ✅ | `--distilled-lora` + 多个 `--image` |
| `a2vid_two_stage` | `--checkpoint-path` | ✅ | ✅ | `--distilled-lora` + `--audio-path` |
| `retake` | `--distilled-checkpoint-path` | ✅ | ❌ | `--video-path` + 时间区间 |
| `hdr_ic_lora` | `--distilled-checkpoint-path` | ❌（用预计算 embed） | ✅ | `--hdr-lora` + `--text-embeddings` |
| `lipdub` | `--distilled-checkpoint-path` | ✅ | ✅ | `--reference-video` + 唯一 `--lora`（IC-LoRA） |

> 用 `python -m ltx_pipelines.<name> --help` 可在任何 pipeline 上看到运行时的完整、最新参数列表。
