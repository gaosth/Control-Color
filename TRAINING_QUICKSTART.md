# 🚀 训练快速开始指南

3步开始训练Control-Color模型！

## 第一步：准备数据

```bash
# 1. 将你的彩色图片放到这个目录
mkdir -p data/train/color
# 复制你的图片到 data/train/color/

# 2. （可选）准备验证集
mkdir -p data/val/color
# 复制验证图片到 data/val/color/

# 3. 运行准备脚本
bash prepare_training.sh
```

**数据要求**：
- 格式：JPG, PNG
- 建议尺寸：512x512 或更大
- 最少数量：建议1000+张
- 内容：多样化的场景和对象

## 第二步：下载预训练模型（推荐）

从 [Google Drive](https://drive.google.com/drive/folders/1lgqstNwrMCzymowRsbGM-4hk0-7L-eOT) 下载：

```bash
pretrained_models/
├── main_model.ckpt                        # 主模型 (必需)
└── content-guided_deformable_vae.ckpt     # VAE模型 (可选)
```

## 第三步：开始训练

### 方案A：微调预训练模型（推荐）⭐

适合：数据量较小（<10k张），想快速得到结果

```bash
python train.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/train.txt \
    --val_list file_lists/val.txt \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 50 \
    --gpus 0
```

**预计训练时间**：
- 1k 图片：1-2小时（RTX 3090）
- 10k 图片：10-20小时

### 方案B：只训练ControlNet（最快）⚡

适合：只想调整控制能力，数据量小

```bash
python train.py \
    --pretrained pretrained_models/main_model.ckpt \
    --freeze_unet \
    --train_list file_lists/train.txt \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 30 \
    --gpus 0
```

**预计训练时间**：
- 1k 图片：30分钟-1小时

### 方案C：从零训练（完整训练）

适合：大规模数据集（>50k张），特定领域

```bash
python train.py \
    --train_list file_lists/train.txt \
    --val_list file_lists/val.txt \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_epochs 200 \
    --gpus 0
```

**预计训练时间**：
- 50k 图片：3-5天（多GPU）

## 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 在浏览器打开
# http://localhost:6006
```

**关键指标**：
- `train/loss`：应该持续下降
- `val/loss`：不应该上升（过拟合）
- `samples`：查看生成效果

## 调整超参数

### GPU显存不足？

```bash
# 减小batch size
--batch_size 2

# 或使用梯度累积（等效于batch_size=8）
--batch_size 2 --accumulate_grad_batches 4
```

### 训练效果不好？

```python
# 降低学习率
--learning_rate 5e-6

# 增加训练轮数
--max_epochs 100

# 增加数据
# 收集更多训练图片
```

### 过拟合？

```python
# 使用更多数据增强（修改 ldm/data/colorization.py）
# 添加dropout
# 减少训练轮数
```

## 使用训练好的模型

```python
# 修改 test.py 中的模型路径
ckpt_path = "./logs/YYYY-MM-DDTHH-MM-SS/checkpoints/last.ckpt"

# 运行测试
python test.py
```

## 常见问题

### Q: CUDA out of memory

```bash
# 方案1: 减小batch size
--batch_size 1

# 方案2: 减小图片尺寸（修改数据集类）
size=256  # 在 ldm/data/colorization.py
```

### Q: 训练很慢

```bash
# 方案1: 使用多GPU
--gpus 0,1,2,3

# 方案2: 增加num_workers
# 修改train.py中的 num_workers=8
```

### Q: 生成结果不理想

1. **检查数据质量**：图片是否多样化？
2. **增加训练时间**：尝试更多epochs
3. **调整学习率**：太大会不稳定，太小会太慢
4. **使用预训练模型**：从pretrained开始微调

### Q: 如何恢复训练？

```bash
python train.py \
    --resume logs/YYYY-MM-DDTHH-MM-SS/checkpoints/last.ckpt \
    --train_list file_lists/train.txt
```

## 进阶技巧

### 1. 特定领域微调

```bash
# 例如：人像着色
# 只用人像数据训练，使用较小学习率
python train.py \
    --pretrained pretrained_models/main_model.ckpt \
    --train_list file_lists/portraits_train.txt \
    --learning_rate 5e-6 \
    --max_epochs 100
```

### 2. 风格化着色

```python
# 修改数据集类，添加特定风格的颜色变换
# 例如：复古风格、动漫风格等
```

### 3. 混合训练

```bash
# 使用多个数据集
# 在 file_lists/train.txt 中混合不同来源的数据
cat dataset1/list.txt dataset2/list.txt > file_lists/train.txt
```

## 性能优化

### 加速训练

```yaml
# 使用更高效的数据加载
num_workers: 8
pin_memory: True
persistent_workers: True

# 使用混合精度
precision: 16

# 使用梯度累积
accumulate_grad_batches: 4
```

### 内存优化

```python
# 使用gradient checkpointing
use_checkpoint: True

# 减小模型大小
# 修改配置文件中的 model_channels
```

## 评估模型

```python
# TODO: 添加评估脚本
# 计算FID, LPIPS等指标
```

## 资源

- 📖 [完整训练指南](TRAINING_GUIDE.md)
- 📄 [论文](https://arxiv.org/abs/2402.10855)
- 🌐 [项目页面](https://zhexinliang.github.io/Control_Color/)

## 需要帮助？

遇到问题？
1. 查看 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) 详细文档
2. 检查 [Issues](https://github.com/ZhexinLiang/Control-Color/issues)
3. 联系：zhexinliang@gmail.com

---

**祝训练顺利！** 🎉

记得定期保存checkpoint，监控训练进度，及时调整超参数。
