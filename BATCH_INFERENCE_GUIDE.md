# 批量推理指南 - Control-Color 颜色修复

## 概述

`batch_inference.py` 脚本用于使用训练好的 Control-Color 模型批量修复褪色照片的颜色。

## 基本用法

### 1. 找到你的 checkpoint 文件

训练完成后，checkpoint 文件保存在：
```
logs_restoration/<训练时间戳>/checkpoints/
```

例如：
```
logs_restoration/2025-11-07T19-52-10/checkpoints/
```

常见的 checkpoint 文件：
- `last.ckpt` - 最新的 checkpoint（通常使用这个）
- `epoch=000099.ckpt` - 特定 epoch 的 checkpoint

### 2. 准备输入图片

将所有需要修复的图片放在一个文件夹中，例如：
```
/path/to/faded_photos/
├── photo1.jpg
├── photo2.jpg
├── photo3.png
└── ...
```

支持的图片格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### 3. 运行批量推理

#### 场景 A: 输入图片已经是褪色图片（推荐）

如果你的输入图片本身就是褪色的老照片：

```bash
python batch_inference.py \
    --input_dir /path/to/faded_photos \
    --output_dir /path/to/restored_photos \
    --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt \
    --is_faded \
    --steps 20
```

#### 场景 B: 输入是正常图片，需要先模拟褪色

如果你想测试模型效果，可以用正常图片，脚本会先模拟褪色：

```bash
python batch_inference.py \
    --input_dir /path/to/normal_photos \
    --output_dir /path/to/restored_photos \
    --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt \
    --fade_type combined \
    --fade_intensity 0.6 \
    --steps 20 \
    --save_comparison \
    --save_faded
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--input_dir` | 输入图片文件夹 | `/data/faded_photos` |
| `--output_dir` | 输出文件夹 | `/data/restored` |
| `--ckpt` | 模型 checkpoint 路径 | `logs_restoration/.../last.ckpt` |

### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `./models/cldm_v15_inpainting_infer1.yaml` | 模型配置文件 |

### 褪色模拟参数（仅当输入不是褪色图时）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--is_faded` | False | 输入已经是褪色图片（加上此参数则不模拟褪色） |
| `--fade_type` | `combined` | 褪色类型：`uniform`, `saturation`, `brightness`, `yellow`, `sepia`, `combined` |
| `--fade_intensity` | 0.6 | 褪色强度 (0.0-1.0) |

### 推理参数

| 参数 | 默认值 | 说明 | 建议 |
|------|--------|------|------|
| `--steps` | 20 | DDIM 采样步数 | 20-50，步数越多质量越好但速度越慢 |
| `--resolution` | 512 | 推理分辨率 | 512 或 256 |
| `--seed` | 42 | 随机种子 | 固定种子确保可重复性 |
| `--guidance_scale` | 9.0 | 分类器引导强度 | 7.0-12.0，越高越遵循提示词 |
| `--sag_scale` | 0.75 | SAG 引导强度 | 0.5-1.0 |
| `--strength` | 1.0 | ControlNet 强度 | 0.5-1.5 |
| `--prompt` | `"restore photo colors"` | 文本提示词 | 可自定义 |

### 保存选项

| 参数 | 说明 |
|------|------|
| `--save_comparison` | 保存对比图（原图 \| 褪色 \| 修复） |
| `--save_faded` | 保存褪色图（如果模拟褪色） |
| `--skip_existing` | 跳过已存在的输出文件 |

## 输出文件结构

### 基本输出（默认）

```
output_dir/
├── photo1.jpg       # 修复后的图片
├── photo2.jpg
└── photo3.png
```

### 完整输出（使用 --save_comparison 和 --save_faded）

```
output_dir/
├── photo1.jpg                    # 修复后的图片
├── photo2.jpg
├── comparisons/                  # 对比图
│   ├── photo1.jpg               # 原图|褪色|修复 三图对比
│   └── photo2.jpg
└── faded/                        # 褪色图（如果模拟褪色）
    ├── photo1.jpg
    └── photo2.jpg
```

## 完整示例

### 示例 1: 修复真实的褪色老照片

```bash
# 最简单的用法
python batch_inference.py \
    --input_dir ./old_faded_photos \
    --output_dir ./restored_results \
    --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt \
    --is_faded \
    --steps 30
```

### 示例 2: 测试模型效果（用正常图片）

```bash
python batch_inference.py \
    --input_dir ./test_photos \
    --output_dir ./test_results \
    --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt \
    --fade_type combined \
    --fade_intensity 0.7 \
    --steps 30 \
    --save_comparison \
    --save_faded
```

### 示例 3: 高质量修复（更多采样步数）

```bash
python batch_inference.py \
    --input_dir ./important_photos \
    --output_dir ./high_quality_results \
    --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt \
    --is_faded \
    --steps 50 \
    --guidance_scale 10.0 \
    --resolution 512
```

### 示例 4: 快速预览（低采样步数）

```bash
python batch_inference.py \
    --input_dir ./photos \
    --output_dir ./preview_results \
    --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt \
    --is_faded \
    --steps 10 \
    --resolution 256
```

### 示例 5: 继续处理（跳过已完成的文件）

```bash
python batch_inference.py \
    --input_dir ./large_photo_collection \
    --output_dir ./restoration_progress \
    --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt \
    --is_faded \
    --steps 20 \
    --skip_existing
```

## 性能优化

### GPU 内存不足

如果遇到 CUDA Out of Memory 错误：

1. **降低分辨率**：
   ```bash
   --resolution 256  # 从 512 降到 256
   ```

2. **逐张处理**：
   将图片分成小批次分别处理

### 加速推理

1. **减少采样步数**（牺牲一些质量）：
   ```bash
   --steps 10  # 从 20 降到 10
   ```

2. **使用较低分辨率**：
   ```bash
   --resolution 256  # 速度提升约 4倍
   ```

## 参数调优指南

### 1. DDIM Steps（采样步数）

- **10 步**：快速预览，质量较低
- **20 步**：平衡质量和速度（推荐）
- **30-50 步**：高质量，速度较慢
- **> 50 步**：通常没有明显提升

### 2. Guidance Scale（引导强度）

- **5.0-7.0**：更自然，可能偏离提示词
- **9.0-10.0**：平衡（推荐）
- **12.0-15.0**：严格遵循提示词，可能过饱和

### 3. ControlNet Strength（控制强度）

- **0.5-0.7**：弱控制，更多创造性
- **1.0**：标准控制（推荐）
- **1.2-1.5**：强控制，更接近输入颜色

### 4. Resolution（分辨率）

- **256**：快速，适合预览
- **512**：标准质量（推荐）
- **> 512**：需要修改配置文件

## 常见问题

### Q1: 找不到 checkpoint 文件

**错误信息**：
```
❌ Error: Checkpoint not found at logs_restoration/...
```

**解决方法**：
```bash
# 查找所有 checkpoint 文件
find logs_restoration -name "*.ckpt"

# 使用找到的路径
python batch_inference.py --ckpt <找到的路径> ...
```

### Q2: CUDA Out of Memory

**解决方法**：
```bash
# 方法1: 降低分辨率
--resolution 256

# 方法2: 处理前清空缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### Q3: 输出图片质量不好

**可能原因和解决方法**：

1. **采样步数太少**：增加到 30-50 步
2. **模型训练不充分**：检查训练 loss，考虑训练更多 epochs
3. **褪色强度不匹配**：调整 `--fade_intensity` 或使用真实褪色图片

### Q4: 处理速度太慢

**加速方法**：

1. 减少采样步数：`--steps 10`
2. 降低分辨率：`--resolution 256`
3. 使用更少的引导强度：`--guidance_scale 7.0`

### Q5: 颜色修复不够准确

**调优方法**：

1. 增加 ControlNet 强度：`--strength 1.5`
2. 增加引导强度：`--guidance_scale 12.0`
3. 修改提示词：`--prompt "restore old faded photo, enhance colors"`

## 与单图推理的对比

| 特性 | batch_inference.py | test_restoration.py |
|------|-------------------|---------------------|
| 处理数量 | 批量处理整个文件夹 | 单张图片 |
| 进度显示 | ✅ 进度条 | ❌ 无 |
| 跳过已存在 | ✅ 支持 | ❌ 不支持 |
| 错误处理 | ✅ 继续处理其他图片 | ❌ 直接报错 |
| 适用场景 | 生产环境批量处理 | 快速测试单张图片 |

## 注意事项

1. **确保有足够的磁盘空间**：输出图片会占用与输入相近的空间
2. **GPU 内存要求**：推荐至少 8GB VRAM（分辨率 512）
3. **处理时间**：每张图片约 2-5 秒（取决于分辨率和步数）
4. **文件格式**：输出格式与输入格式相同
5. **文件命名**：输出文件名与输入文件名相同

## 技术细节

### 推理流程

```
输入图片 →
    ↓ (如果需要) 模拟褪色 →
    ↓ 提取颜色 hint →
    ↓ Resize 到推理分辨率 →
    ↓ VAE 编码 →
    ↓ ControlNet + UNet 预测 →
    ↓ DDIM 采样去噪 →
    ↓ VAE 解码 →
    ↓ Resize 回原始尺寸 →
输出修复图片
```

### 内存使用

| 分辨率 | 批次大小 | 约需 GPU 内存 |
|--------|----------|---------------|
| 256    | 1        | ~4 GB         |
| 512    | 1        | ~8 GB         |

### 典型处理时间（RTX 3090）

| 分辨率 | Steps | 单张耗时 |
|--------|-------|----------|
| 256    | 10    | ~1 秒    |
| 256    | 20    | ~2 秒    |
| 512    | 20    | ~4 秒    |
| 512    | 50    | ~10 秒   |

## 进阶用法

### 自定义提示词

针对不同类型的照片使用不同的提示词：

```bash
# 人像照片
--prompt "restore portrait photo, natural skin tone, vivid colors"

# 风景照片
--prompt "restore landscape photo, vibrant nature colors, clear sky"

# 老照片
--prompt "restore vintage photo, enhance faded colors, remove yellowing"
```

### 批量处理不同参数

对不同类型的照片使用不同的参数：

```bash
# 轻度褪色
python batch_inference.py --input_dir ./light_faded --ckpt ... --strength 0.7 --steps 20

# 重度褪色
python batch_inference.py --input_dir ./heavy_faded --ckpt ... --strength 1.5 --steps 50
```

## 相关文档

- `train_restoration.py` - 训练脚本
- `test_restoration.py` - 单图推理脚本
- `CONTROL_AND_CONDITIONING_EXPLAINED.md` - 模型架构说明
- `TRAINING_OOM_SOLUTION.md` - 训练问题解决

## 获取帮助

查看完整参数列表：
```bash
python batch_inference.py --help
```
