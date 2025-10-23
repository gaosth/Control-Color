# 🎨 颜色修复/恢复任务指南

使用Control-Color进行照片颜色修复，恢复褪色老照片的原始色彩。

---

## 📖 目录

- [任务说明](#任务说明)
- [快速开始](#快速开始)
- [褪色效果展示](#褪色效果展示)
- [训练模型](#训练模型)
- [使用模型](#使用模型)
- [原理解释](#原理解释)
- [高级配置](#高级配置)

---

## 🎯 任务说明

### 什么是颜色修复？

颜色修复是将**褪色/变色的照片恢复成原始色彩**的任务。

**与传统着色的区别**：

| 特性 | 传统着色 | 颜色修复 |
|------|---------|---------|
| **输入** | 灰度图 | 褪色的彩色图 |
| **颜色信息** | 无颜色信息 | 有褪色的颜色信息 |
| **任务难度** | 高（完全创造颜色） | 中（恢复/增强颜色） |
| **应用场景** | 黑白照片上色 | 老照片修复、褪色图片增强 |

### 核心创新：利用褪色颜色作为引导

**关键思想**：褪色图片中的颜色虽然不鲜艳，但仍然包含了原始颜色的信息！

```
例如：
- 褪色的蓝色天空 → 淡蓝色 → 引导模型生成鲜艳的蓝色
- 褪色的红色花朵 → 粉红色 → 引导模型生成鲜艳的红色
```

---

## 🚀 快速开始

### 方法1：测试褪色效果（无需训练）

```bash
# 1. 测试不同的褪色效果
python -c "from ldm.data.color_restoration import test_fading_effects; test_fading_effects()"

# 会生成 fading_effects_test.png 展示不同褪色效果
```

### 方法2：使用预训练模型修复图片

```bash
# 修复褪色图片
python test_restoration.py \
    --input path/to/faded_image.jpg \
    --output restored.png \
    --ckpt pretrained_models/main_model.ckpt \
    --is_faded \
    --save_comparison

# 或者从原始图片模拟褪色再修复（测试）
python test_restoration.py \
    --input path/to/original_image.jpg \
    --output restored.png \
    --ckpt pretrained_models/main_model.ckpt \
    --fade_type yellow \
    --fade_intensity 0.6 \
    --save_comparison
```

### 方法3：训练自己的修复模型

```bash
# 1. 准备数据（彩色图片）
mkdir -p data/train/color
cp your_images/*.jpg data/train/color/

# 2. 生成文件列表
find data/train/color -name "*.jpg" > file_lists/train.txt

# 3. 开始训练（自动模拟褪色）
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 50 \
    --fade_type mixed
```

---

## 🎨 褪色效果展示

### 支持的褪色类型

我们实现了5种常见的照片褪色效果：

#### 1. **uniform** - 均匀褪色
向灰色褪色，保持色相但降低饱和度
```python
simulate_color_fading(image, 'uniform', 0.5)
```
- **效果**：整体颜色变淡，像蒙了一层灰
- **应用**：通用褪色效果

#### 2. **saturation** - 饱和度降低
只降低饱和度，亮度不变
```python
simulate_color_fading(image, 'saturation', 0.6)
```
- **效果**：颜色不鲜艳，偏灰
- **应用**：模拟长期存放的照片

#### 3. **brightness** - 亮度降低
整体变暗
```python
simulate_color_fading(image, 'brightness', 0.4)
```
- **效果**：照片变暗，细节减少
- **应用**：模拟曝光不足的老照片

#### 4. **yellow** - 泛黄效果
老照片常见的泛黄
```python
simulate_color_fading(image, 'yellow', 0.7)
```
- **效果**：整体偏黄，像老照片
- **应用**：80-90年代老照片修复

#### 5. **sepia** - 棕褐色调
经典的老照片色调
```python
simulate_color_fading(image, 'sepia', 0.5)
```
- **效果**：棕褐色调，复古
- **应用**：非常老的照片（50-70年代）

#### 6. **mixed** - 混合效果（推荐训练时使用）
随机混合上述效果
```python
simulate_color_fading(image, 'mixed', 0.5)
```
- **效果**：每次随机选择一种效果
- **应用**：训练时增加数据多样性

---

## 📚 训练模型

### 数据准备

**好消息**：只需要**原始彩色图片**！

数据集会自动：
1. ✅ 模拟褪色效果
2. ✅ 提取褪色颜色作为hint
3. ✅ 生成训练对（褪色图 + 原图）

```bash
# 数据组织
data/
├── train/
│   └── color/           # 只需要彩色图片！
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── val/
    └── color/
        └── ...

# 生成列表
find data/train/color -name "*.jpg" > file_lists/train.txt
find data/val/color -name "*.jpg" > file_lists/val.txt
```

### 训练配置

#### 基础训练（推荐）

```bash
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --val_list file_lists/val.txt \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 50 \
    --fade_type mixed \
    --fade_intensity 0.3 0.7
```

**参数说明**：
- `--fade_type mixed`: 混合多种褪色效果，提高泛化能力
- `--fade_intensity 0.3 0.7`: 褪色强度范围，训练时随机采样

#### 特定褪色类型

如果你的照片都是**特定类型的褪色**（比如都是泛黄的老照片）：

```bash
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --fade_type yellow \
    --fade_intensity 0.5 0.8 \
    --max_epochs 30
```

#### 快速训练（只训练ControlNet）

```bash
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --freeze_unet \
    --train_list file_lists/train.txt \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 20
```

### 监控训练

```bash
tensorboard --logdir logs_restoration/
```

**关键可视化**：
- `train/faded_input`: 褪色的输入图片
- `train/reconstruction`: 修复后的图片
- `train/samples`: 生成的样本

---

## 🔧 使用模型

### 命令行使用

#### 修复真实的褪色照片

```bash
python test_restoration.py \
    --input my_faded_photo.jpg \
    --output restored.png \
    --ckpt logs_restoration/YYYY-MM-DD/checkpoints/last.ckpt \
    --is_faded \
    --save_comparison
```

#### 测试模型效果（模拟褪色）

```bash
python test_restoration.py \
    --input test_image.jpg \
    --output restored.png \
    --ckpt logs_restoration/YYYY-MM-DD/checkpoints/last.ckpt \
    --fade_type yellow \
    --fade_intensity 0.7 \
    --save_comparison
```

### Python API使用

```python
from ldm.data.color_restoration import simulate_color_fading
from test_restoration import restore_faded_image
from cldm.model import create_model, load_state_dict
from cldm.ddim_haced_sag_step import DDIMSampler
import cv2

# 1. 加载模型
model = create_model('./models/cldm_v15_inpainting_infer1.yaml').cpu()
model.load_state_dict(load_state_dict('path/to/checkpoint.ckpt', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# 2. 读取褪色图片
faded_image = cv2.imread('faded_photo.jpg')
faded_image = cv2.cvtColor(faded_image, cv2.COLOR_BGR2RGB)

# 3. 修复
restored_images = restore_faded_image(
    model,
    ddim_sampler,
    faded_image,
    prompt="restore old photo colors",
    ddim_steps=20,
    seed=42
)

# 4. 保存
restored = restored_images[0]
cv2.imwrite('restored.jpg', cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
```

---

## 🧠 原理解释

### 为什么这个方法有效？

#### 传统着色的问题

```
灰度图 → 模型 → 彩色图
         ⬆️
    完全猜测颜色
```

**问题**：模型不知道天空应该是蓝色还是红色

#### 颜色修复的优势

```
褪色图（淡蓝色天空）→ 模型 → 彩色图（鲜艳蓝色）
         ⬆️
    有颜色引导！
```

**优势**：
1. ✅ 褪色颜色提供了**色相（Hue）**信息
2. ✅ 模型只需恢复**饱和度（Saturation）**和**亮度（Value）**
3. ✅ 降低了任务难度，提高了准确性

### 技术实现

#### 1. 褪色模拟

我们模拟了真实照片的褪色过程：

```python
# 均匀褪色：向灰色插值
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faded = image * (1 - intensity) + gray * intensity

# 饱和度降低：HSV空间操作
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity)  # S通道
faded = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# 泛黄：增加红绿，减少蓝色
yellow_tint = [1.0 + 0.2*intensity, 1.0 + 0.15*intensity, 1.0 - 0.3*intensity]
faded = image * yellow_tint
```

#### 2. 颜色Hint提取

最简单有效的方式：**直接使用褪色图片的颜色**

```python
# 褪色图片已经包含了颜色信息
color_hint = faded_image  # RGB, 3通道
```

这个hint输入到ControlNet，引导模型生成正确的颜色。

#### 3. 模型架构

**无需修改原始架构**！

```yaml
ControlNet:
  hint_channels: 3  # RGB褪色图

UNet:
  in_channels: 9    # 4(latent) + 1(mask) + 4(masked_latent)
```

**数据流**：
```
褪色图（RGB） → ControlNet → 控制信号
                              ↓
latent noise → UNet ← 控制信号 → 修复的latent
                ↓
              VAE Decoder
                ↓
            修复的彩色图
```

---

## ⚙️ 高级配置

### 自定义褪色效果

在 `ldm/data/color_restoration.py` 中添加新的褪色类型：

```python
def simulate_color_fading(image, fade_type, intensity):
    # ...现有代码...

    elif fade_type == 'my_custom_fade':
        # 你的自定义褪色逻辑
        faded = custom_fade_function(image, intensity)
        return faded
```

### 使用真实的褪色数据对

如果你有**真实的褪色前后配对数据**：

```python
# 修改数据集类
class RealFadedPairDataset(Dataset):
    def __init__(self, faded_list, original_list):
        self.faded_paths = faded_list
        self.original_paths = original_list

    def __getitem__(self, idx):
        # 直接读取真实的配对数据
        faded = Image.open(self.faded_paths[idx])
        original = Image.open(self.original_paths[idx])

        # 其他处理...
        return {
            "jpg": original,
            "hint": faded,
            # ...
        }
```

### 调整褪色强度分布

```python
# 在数据集类中
def generate_faded_image(self, original_image):
    # 使用Beta分布而非均匀分布
    import scipy.stats as stats
    intensity = stats.beta.rvs(2, 5)  # 偏向较小的褪色

    faded_image = simulate_color_fading(
        original_image,
        fade_type=self.fade_type,
        intensity=intensity
    )
    return faded_image, intensity
```

---

## 📊 效果评估

### 定量指标

```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# PSNR（越高越好）
psnr_value = psnr(original, restored)

# SSIM（越接近1越好）
ssim_value = ssim(original, restored, multichannel=True)

print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.4f}")
```

### 定性评估

关注：
1. ✅ **颜色准确性**：蓝天是否真的是蓝色
2. ✅ **饱和度恢复**：颜色是否鲜艳
3. ✅ **细节保留**：是否保留了原始细节
4. ✅ **自然度**：看起来是否自然

---

## 🎯 应用场景

### 1. 老照片修复

**场景**：修复家庭老照片

```bash
python test_restoration.py \
    --input old_family_photo.jpg \
    --output restored_family_photo.png \
    --is_faded \
    --steps 30 \
    --save_comparison
```

### 2. 褪色艺术作品修复

**场景**：修复褪色的绘画、海报

```bash
# 使用更多步数获得更好效果
python test_restoration.py \
    --input faded_artwork.jpg \
    --output restored_artwork.png \
    --is_faded \
    --steps 50 \
    --seed 42
```

### 3. 批量处理

```bash
# 创建批处理脚本
for img in faded_photos/*.jpg; do
    python test_restoration.py \
        --input "$img" \
        --output "restored/$(basename $img)" \
        --is_faded
done
```

---

## 💡 最佳实践

### 训练建议

1. **数据多样性**：使用 `fade_type='mixed'` 训练
2. **褪色强度**：覆盖广泛的强度范围 `[0.3, 0.8]`
3. **微调策略**：从预训练模型开始，小学习率 `1e-5`
4. **验证集**：使用真实褪色照片（如果有）

### 推理建议

1. **采样步数**：20-30步通常足够
2. **CFG Scale**：7-9 获得较好效果
3. **随机种子**：尝试多个种子，选择最佳结果
4. **后处理**：可以轻微调整饱和度

---

## 🐛 常见问题

### Q: 修复结果颜色过于鲜艳

**A**: 降低褪色强度或调整CFG scale

```bash
--fade_intensity 0.3 0.5  # 降低褪色强度
# 或推理时
--scale 5.0  # 降低引导强度
```

### Q: 修复结果仍然偏灰

**A**:
1. 增加训练轮数
2. 使用更强的褪色效果训练
3. 检查数据质量

### Q: 颜色不准确（蓝天变绿天）

**A**:
1. 检查褪色模拟是否合理
2. 增加训练数据
3. 使用文本提示："blue sky, vibrant colors"

---

## 📚 参考资源

- [Control-Color 论文](https://arxiv.org/abs/2402.10855)
- [ControlNet 论文](https://arxiv.org/abs/2302.05543)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

---

## 🎉 总结

颜色修复通过利用褪色图片中的颜色信息作为引导，显著降低了任务难度：

✅ **无需修改模型架构** - 直接使用现有ControlNet
✅ **自动数据生成** - 从彩色图模拟褪色
✅ **更高准确性** - 有颜色引导，不会猜错
✅ **实用价值高** - 老照片修复、艺术品保护

立即开始修复你的褪色照片吧！🚀
