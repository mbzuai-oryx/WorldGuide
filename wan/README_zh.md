# WAN 模型推理

WAN Pipeline 提供了支持分布式推理的轻量级替代方案。它针对多 GPU 设置进行了优化，提供更快的推理速度和更低的内存占用。

## 下载 WAN 模型

从 HuggingFace 下载所需的 WAN 模型：

```bash
# 下载 WAN transformer 模型
huggingface-cli download tencent/HY-WorldPlay wan_transformer --local-dir /path/to/models/wan_transformer

# 下载 WAN 蒸馏模型检查点
huggingface-cli download tencent/HY-WorldPlay wan_distilled_model/model.pt --local-dir /path/to/models/wan_distilled_model
```

## 配置参数

WAN 管道接受以下参数：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--input` | 文本提示或包含提示的 txt 文件路径 | 必需 |
| `--image_path` | I2V 生成的输入图像路径 | None |
| `--num_chunk` | 块数（每块 = 16 帧） | 4 |
| `--pose` | 相机轨迹（姿态字符串或 JSON 文件） | `w-96` |
| `--ar_model_path` | WAN transformer 模型路径 | 必需 |
| `--ckpt_path` | 训练检查点路径 (model.pt) | 必需 |
| `--out` | 生成视频的输出目录 | `outputs` |

## 多 GPU 分布式推理

多 GPU 推理以获得更好的性能：

```bash
# 使用 4 个 GPU
torchrun --nproc_per_node=4 wan/generate.py \
  --input "First-person view walking around ancient Athens, with Greek architecture and marble structures" \
  --image_path /path/to/input/image.jpg \
  --num_chunk 4 \
  --pose "w-96" \
  --ar_model_path /path/to/models/wan_transformer \
  --ckpt_path /path/to/models/wan_distilled_model/model.pt \
  --out outputs
```

## 批量处理文本文件

您可以通过提供文本文件来处理多个提示：

```bash
# 创建一个 prompts.txt 文件，每行一个提示
echo "第一人称视角的中世纪城堡" > prompts.txt
echo "夜晚在赛博朋克城市中漫步" >> prompts.txt
echo "探索水下珊瑚礁" >> prompts.txt

# 对所有提示运行推理
python wan/generate.py \
  --input prompts.txt \
  --ar_model_path /path/to/models/wan_transformer \
  --ckpt_path /path/to/models/wan_distilled_model/model.pt
```

## WAN 的相机控制

WAN 使用与 HunyuanVideo 管道相同的相机控制系统：

**姿态字符串格式：**
```bash
# 前进 96 个 latents（使用 num_chunk=4 生成 961 帧）
--pose "w-96"

# 复杂轨迹
--pose "w-20, right-10, d-30, up-36"
```

**支持的动作：**
- **移动**: `w` (前进), `s` (后退), `a` (左移), `d` (右移)
- **旋转**: `up` (抬头), `down` (低头), `left` (左转), `right` (右转)
- **格式**: `动作-时长`，时长代表动作对应的latents数量。每个latent对应4帧。
