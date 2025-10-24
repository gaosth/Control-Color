# 验证集配置指南

## 1. 数据目录结构

推荐的目录结构：

```
Control-Color/
├── data/
│   ├── train/          # 训练集图片
│   │   └── *.jpg/png
│   └── val/            # 验证集图片
│       └── *.jpg/png
├── file_lists/         # 文件列表目录
│   ├── train.txt       # 训练集图片路径列表
│   └── val.txt         # 验证集图片路径列表
```

## 2. 准备数据

### 方式1：手动组织数据

```bash
# 创建数据目录
mkdir -p data/train data/val

# 将您的训练图片放入 data/train/
# 将您的验证图片放入 data/val/
```

### 方式2：从现有数据集划分

如果您有一个图片目录，可以随机划分：

```bash
# 假设所有图片在 my_images/ 目录下
# 将80%用于训练，20%用于验证

# 创建目录
mkdir -p data/train data/val

# 使用 Python 脚本划分（推荐）
python -c "
import os
import shutil
import random
from pathlib import Path

# 源目录
source_dir = 'my_images'
train_dir = 'data/train'
val_dir = 'data/val'

# 获取所有图片
images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
    images.extend(Path(source_dir).glob(ext))

# 随机打乱
random.seed(42)
random.shuffle(images)

# 划分（80% 训练，20% 验证）
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

# 复制文件
for img in train_images:
    shutil.copy(img, train_dir)
for img in val_images:
    shutil.copy(img, val_dir)

print(f'训练集: {len(train_images)} 张图片')
print(f'验证集: {len(val_images)} 张图片')
"
```

## 3. 创建文件列表

### 自动生成文件列表

```bash
# 创建 file_lists 目录
mkdir -p file_lists

# 生成训练集列表
find data/train -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | sort > file_lists/train.txt

# 生成验证集列表
find data/val -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | sort > file_lists/val.txt

# 查看文件数量
echo "训练集图片数量: $(wc -l < file_lists/train.txt)"
echo "验证集图片数量: $(wc -l < file_lists/val.txt)"
```

### 文件列表格式

`file_lists/train.txt` 和 `file_lists/val.txt` 的格式：
```
data/train/image001.jpg
data/train/image002.jpg
data/train/image003.png
...
```

每行一个图片路径，路径相对于项目根目录。

## 4. 训练命令（只训练ControlNet）

### 使用验证集训练

```bash
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --val_list file_lists/val.txt \
    --gpus 0,1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_epochs 50 \
    --freeze_unet
```

### 不使用验证集训练

如果没有验证集，可以：

```bash
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --val_list file_lists/val_DOES_NOT_EXIST.txt \
    --gpus 0,1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_epochs 50 \
    --freeze_unet
```

脚本会检测到验证集文件不存在，自动禁用验证。

## 5. 参数说明

### 核心参数

- `--pretrained`：预训练模型路径（用于微调）
- `--train_list`：训练集图片列表文件
- `--val_list`：验证集图片列表文件
- `--freeze_unet`：**只训练ControlNet，冻结UNet**（推荐用于微调）

### GPU和批次大小

- `--gpus 0,1`：使用GPU 0和1（双卡训练）
- `--batch_size 2`：每张GPU的batch size（总batch size = 2×2 = 4）

### 训练参数

- `--learning_rate 1e-5`：学习率（微调推荐1e-5）
- `--max_epochs 50`：训练轮数

### 褪色效果参数

- `--fade_type mixed`：褪色类型
  - `uniform`：均匀褪色
  - `saturation`：饱和度降低
  - `brightness`：亮度降低
  - `yellow`：泛黄效果
  - `sepia`：棕褐色调
  - `mixed`：随机混合（默认）

- `--fade_intensity 0.3 0.7`：褪色强度范围（最小值 最大值）

## 6. 监控训练

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir logs_restoration/

# 访问 http://localhost:6006
```

### 查看训练日志

```bash
# 查看最新的训练日志
ls -lt logs_restoration/
```

### 可视化内容

训练过程中会记录：
- 训练损失曲线
- 验证损失曲线（如果有验证集）
- 每500步的修复效果对比图：
  - 褪色图 → 修复图 → 原图

## 7. Checkpoint管理

训练的模型保存在：
```
logs_restoration/YYYY-MM-DDTHH-MM-SS/checkpoints/
├── last.ckpt                    # 最新的checkpoint
├── epoch00-step000500.ckpt      # 定期保存的checkpoint
├── epoch01-step001000.ckpt
└── ...
```

### 从checkpoint恢复训练

```bash
python train_restoration.py \
    --resume logs_restoration/2024-XX-XXTHH-MM-SS/checkpoints/last.ckpt \
    --train_list file_lists/train.txt \
    --val_list file_lists/val.txt \
    --gpus 0,1 \
    --batch_size 2
```

## 8. 最佳实践

### 验证集大小

- **推荐比例**：训练集80%，验证集20%
- **最小验证集**：至少50-100张图片
- **最大验证集**：不超过训练集的30%

### 只训练ControlNet vs 训练整个模型

**只训练ControlNet（推荐用于微调）**：
```bash
--freeze_unet  # 加这个参数
--learning_rate 1e-5
--batch_size 2  # 可以用较大的batch size
```

优点：
- ✅ 训练速度快
- ✅ 显存占用少
- ✅ 不容易过拟合
- ✅ 适合数据量较小的情况

**训练整个模型**：
```bash
# 不加 --freeze_unet
--learning_rate 5e-6  # 更小的学习率
--batch_size 1  # 可能需要更小的batch size
```

优点：
- ✅ 更强的适应能力
- ✅ 适合大规模数据集
- ⚠️ 需要更多显存和训练时间

## 9. 常见问题

### Q: 没有验证集可以训练吗？
A: 可以，脚本会自动禁用验证。但建议至少准备一小部分验证集来监控过拟合。

### Q: 验证集图片要求？
A: 与训练集相同，任何彩色图片即可。脚本会自动生成褪色版本。

### Q: 验证频率？
A: 默认每个epoch结束后验证一次。

### Q: 如何判断训练效果？
A:
1. 查看TensorBoard中的损失曲线
2. 观察验证集上的修复效果图
3. 验证损失应该逐渐下降且稳定

### Q: 什么时候停止训练？
A:
- 验证损失不再下降（早停）
- 验证损失开始上升（过拟合）
- 视觉效果达到满意程度

## 10. 快速开始示例

假设您有500张图片在 `my_photos/` 目录：

```bash
# 1. 创建数据目录
mkdir -p data/train data/val file_lists

# 2. 手动或脚本方式划分数据（400训练，100验证）
# ... 移动文件到 data/train 和 data/val ...

# 3. 生成文件列表
find data/train -type f \( -name "*.jpg" -o -name "*.png" \) > file_lists/train.txt
find data/val -type f \( -name "*.jpg" -o -name "*.png" \) > file_lists/val.txt

# 4. 查看统计
echo "训练集: $(wc -l < file_lists/train.txt) 张"
echo "验证集: $(wc -l < file_lists/val.txt) 张"

# 5. 开始训练（只训练ControlNet）
python train_restoration.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --val_list file_lists/val.txt \
    --gpus 0,1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_epochs 50 \
    --freeze_unet

# 6. 监控训练
tensorboard --logdir logs_restoration/
```
