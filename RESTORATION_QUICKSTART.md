# 🚀 颜色修复快速开始

3分钟上手照片颜色修复！

---

## 什么是颜色修复？

**颜色修复** = 恢复褪色照片的原始色彩

```
褪色的蓝天（淡蓝） → 模型 → 鲜艳的蓝天（深蓝）
褪色的红花（粉红） → 模型 → 鲜艳的红花（鲜红）
```

**核心优势**：利用褪色图片中的颜色信息作为引导，比从灰度图着色更准确！

---

## 第一步：环境准备

```bash
# 运行准备脚本
bash prepare_restoration.sh
```

这会：
- ✅ 创建必要目录
- ✅ 检查预训练模型
- ✅ 测试褪色效果

---

## 第二步：选择使用方式

### 方式A：直接测试（无需训练）⚡

**适合**：快速测试，看看效果

```bash
# 下载预训练模型到 pretrained_models/main_model.ckpt

# 在任意图片上测试褪色修复
python test_restoration.py \
    --input your_image.jpg \
    --output restored.png \
    --ckpt pretrained_models/main_model.ckpt \
    --fade_type yellow \
    --fade_intensity 0.6 \
    --save_comparison
```

**结果**：生成3张对比图
- 原图 | 模拟褪色 | 修复结果

### 方式B：训练专属模型 🎯

**适合**：有特定领域照片，想要最佳效果

#### B1. 准备数据

```bash
# 只需要彩色图片！
cp your_photos/*.jpg data/train/color/

# 生成列表（已由prepare_restoration.sh完成）
find data/train/color -name "*.jpg" > file_lists/train.txt
```

**数据要求**：
- ✅ 格式：JPG, PNG
- ✅ 数量：建议500+张
- ✅ 内容：原始彩色图片（不需要褪色图）

#### B2. 开始训练

```bash
# 推荐：从预训练模型微调
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 50
```

**预计时间**：
- 500张图片：1-2小时（RTX 3090）
- 5000张图片：10-20小时

#### B3. 监控训练

```bash
tensorboard --logdir logs_restoration/
# 打开 http://localhost:6006
```

查看：
- `faded_input`: 模拟的褪色图
- `reconstruction`: 修复结果
- `samples`: 生成样本

#### B4. 使用训练好的模型

```bash
python test_restoration.py \
    --input faded_photo.jpg \
    --output restored.png \
    --ckpt logs_restoration/YYYY-MM-DD/checkpoints/last.ckpt \
    --is_faded \
    --save_comparison
```

---

## 第三步：高级选项

### 选择褪色类型

不同类型模拟不同年代的照片褪色：

| 类型 | 效果 | 适用场景 |
|------|------|----------|
| `uniform` | 整体褪色 | 通用 |
| `saturation` | 饱和度降低 | 长期存放的照片 |
| `yellow` | 泛黄 | 80-90年代老照片 |
| `sepia` | 棕褐色 | 50-70年代照片 |
| `mixed` | 混合（推荐训练） | 提高泛化能力 |

```bash
# 训练时使用混合类型
python train_restoration.py \
    --fade_type mixed \
    --fade_intensity 0.3 0.7

# 测试时指定特定类型
python test_restoration.py \
    --fade_type yellow \
    --fade_intensity 0.6
```

### 调整褪色强度

```bash
# 轻微褪色
--fade_intensity 0.2 0.4

# 中等褪色（推荐）
--fade_intensity 0.3 0.7

# 严重褪色
--fade_intensity 0.6 0.9
```

### 快速训练模式

只训练ControlNet，速度快2-3倍：

```bash
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --freeze_unet \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 20
```

---

## 常见用例

### 用例1: 修复家庭老照片

```bash
# 老照片通常泛黄
python test_restoration.py \
    --input old_family_photo.jpg \
    --output restored_family.png \
    --ckpt pretrained_models/main_model.ckpt \
    --fade_type yellow \
    --fade_intensity 0.7 \
    --is_faded \
    --save_comparison
```

### 用例2: 修复艺术作品

```bash
# 使用更多采样步数获得更好质量
python test_restoration.py \
    --input faded_painting.jpg \
    --output restored_painting.png \
    --ckpt pretrained_models/main_model.ckpt \
    --is_faded \
    --steps 50
```

### 用例3: 批量处理

```bash
# 创建批处理脚本
for img in old_photos/*.jpg; do
    python test_restoration.py \
        --input "$img" \
        --output "restored/$(basename $img)" \
        --ckpt pretrained_models/main_model.ckpt \
        --is_faded
done
```

---

## 参数速查

### 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--batch_size` | 批量大小 | 4 |
| `--learning_rate` | 学习率 | 1e-5 (微调) / 1e-4 (从零) |
| `--max_epochs` | 训练轮数 | 50 |
| `--fade_type` | 褪色类型 | mixed |
| `--fade_intensity` | 褪色强度范围 | 0.3 0.7 |
| `--freeze_unet` | 只训练ControlNet | 快速实验时使用 |

### 推理参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--steps` | DDIM采样步数 | 20-30 |
| `--seed` | 随机种子 | 42（可复现） / -1（随机） |
| `--fade_type` | 褪色类型（测试用） | yellow |
| `--fade_intensity` | 褪色强度（测试用） | 0.5-0.7 |
| `--is_faded` | 输入已是褪色图 | 修复真实褪色照片时使用 |
| `--save_comparison` | 保存对比图 | 推荐使用 |

---

## 常见问题

### Q: 修复后颜色太鲜艳？

```bash
# 降低褪色强度
--fade_intensity 0.3 0.5

# 或使用更温和的褪色类型
--fade_type saturation
```

### Q: 修复后还是偏灰？

- 增加训练轮数
- 使用更强的褪色训练
- 检查数据质量

### Q: GPU显存不足？

```bash
# 减小batch size
--batch_size 2

# 或冻结UNet
--freeze_unet
```

### Q: 颜色不准（蓝天变绿）？

- 增加训练数据量
- 使用文本提示："blue sky, vibrant colors"
- 检查褪色模拟是否合理

---

## 效果对比

### 传统着色 vs 颜色修复

| 特性 | 传统着色 | 颜色修复 |
|------|---------|---------|
| 输入信息 | 只有灰度 | 有褪色颜色 |
| 颜色准确性 | 中等（猜测） | 高（引导） |
| 训练难度 | 高 | 中 |
| 应用场景 | 黑白照片 | 褪色照片 |

---

## 下一步

- 📖 阅读详细指南：[COLOR_RESTORATION_GUIDE.md](COLOR_RESTORATION_GUIDE.md)
- 🎨 查看原理解释：模型如何工作
- 🔧 高级配置：自定义褪色效果

---

## 需要帮助？

- 查看详细文档：`COLOR_RESTORATION_GUIDE.md`
- GitHub Issues: [Control-Color Issues](https://github.com/ZhexinLiang/Control-Color/issues)
- Email: zhexinliang@gmail.com

---

**开始修复你的褪色照片吧！** 🎉

只需3步：
1. `bash prepare_restoration.sh` - 准备环境
2. `python test_restoration.py ...` - 测试效果
3. `python train_restoration.py ...` - 训练模型（可选）
