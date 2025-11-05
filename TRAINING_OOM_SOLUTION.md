# 训练显存不足(OOM)解决方案

## 问题诊断

你遇到的错误：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 23.57 GiB of which 1.54 GiB is free.
```

**根本原因**：
- 虽然指定了2张GPU (`--gpus 0,1`)
- 但DDP模式下，**每张GPU都加载完整模型**
- 模型总大小 1.4B参数 ≈ 5.7GB
- batch_size=4 + 512x512图片 + 梯度 ≈ 18GB
- 总需求: ~24GB > 23.57GB 可用显存

**为什么只用了GPU 0？**
- DDP训练时，每张卡分别处理batch的一部分
- 但都需要完整模型副本
- 显存溢出发生在GPU 0，导致训练中止

---

## 解决方案

### 方案 1: 减小Batch Size（推荐）

```bash
# 将batch_size从4改成2（每个GPU）
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 2 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**效果**：
- 每张GPU: batch_size=2
- 总effective batch size = 2 × 2 = 4（与之前相同）
- 显存需求: ~15GB（可以运行）

### 方案 2: 减小图片分辨率

修改 `train_restoration.py` 第261-262行：

```python
"params": {
    "size": 256,  # 从512改成256
    "training_images_list_file": opt.train_list,
}
```

然后运行：

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 4 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**权衡**：
- ✅ 显存需求大幅降低
- ❌ 图片分辨率降低（256x256）
- ❌ 训练出的模型只能处理256x256图片

### 方案 3: 使用梯度累积

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 1 \
    --accumulate_grad_batches 2 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**效果**：
- 每张GPU: batch_size=1（显存需求最小）
- 梯度累积2步 = effective batch_size 4
- 速度稍慢，但显存安全

### 方案 4: 启用梯度检查点（不推荐DDP）

之前禁用了`use_checkpoint`以兼容DDP，但如果单卡训练可以启用。

修改 `models/cldm_v15_inpainting_infer.yaml`:

```yaml
control_stage_config:
  params:
    use_checkpoint: True  # 改成True

unet_config:
  params:
    use_checkpoint: True  # 改成True
```

然后**单卡训练**：

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0 \
    --batch_size 4 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**权衡**：
- ✅ 大幅减少显存
- ❌ 只能用单卡（放弃双卡优势）
- ❌ 训练速度变慢（需要重计算）

---

## 推荐配置

### 配置 A: 双卡训练 + 小batch（推荐）

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 2 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**优势**：
- ✅ 利用双卡加速
- ✅ 512x512高分辨率
- ✅ 显存安全
- ✅ Effective batch_size=4（合理）

### 配置 B: 双卡 + 梯度累积

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 1 \
    --accumulate_grad_batches 4 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**优势**：
- ✅ 显存需求最小（batch=1）
- ✅ 512x512高分辨率
- ✅ Effective batch_size=8（更大）
- ⚠ 稍慢（累积4步才更新一次）

### 配置 C: 降低分辨率（如果显存仍不够）

```bash
# 1. 修改 train_restoration.py 第261行: size: 256

# 2. 运行训练
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 4 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

---

## 验证显存使用

训练前可以先测试显存占用：

```bash
# 方法1：使用nvidia-smi监控
watch -n 1 nvidia-smi

# 方法2：在Python中测试
python -c "
import torch
from cldm.model import create_model

model = create_model('models/cldm_v15_inpainting_infer.yaml').cuda()
print(f'Model loaded, GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

# 模拟一个batch
batch_size = 2
x = torch.randn(batch_size, 9, 64, 64).cuda()
t = torch.randint(0, 1000, (batch_size,)).cuda()
context = torch.randn(batch_size, 77, 768).cuda()

print(f'Before forward, GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
"
```

---

## 添加--accumulate_grad_batches支持

如果 `train_restoration.py` 没有这个参数，添加它：

```python
# 在argparse部分添加
parser.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    help="梯度累积步数"
)

# 在trainer_kwargs中添加
trainer_kwargs = {
    # ...
    "accumulate_grad_batches": opt.accumulate_grad_batches,
    # ...
}
```

---

## 快速解决（立即可用）

**最简单的方法 - 减小batch_size**：

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 2 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

如果还是OOM，进一步减小：

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 1 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

---

## 显存优化检查清单

训练前确认：

- [ ] batch_size ≤ 2（双卡）或 ≤ 4（单卡）
- [ ] 图片分辨率 = 512（可降到256节省显存）
- [ ] use_checkpoint = False（DDP兼容性）
- [ ] precision = 16（使用混合精度）
- [ ] 关闭其他占用GPU的程序

---

## 监控显存使用

训练时在另一个终端运行：

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用更详细的工具
nvitop
```

观察：
- GPU 0和GPU 1的显存应该**基本相等**（DDP正常）
- 如果只有GPU 0显存满，说明DDP有问题
- 正常情况：每张GPU用~15-18GB（batch_size=2）

---

## 总结

**立即执行**（推荐）：

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 2 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**如果还OOM**：

```bash
python train_restoration.py \
    --config models/cldm_v15_inpainting_infer.yaml \
    --gpus 0,1 \
    --batch_size 1 \
    --max_epochs 50 \
    --resume_from_checkpoint pretrained_models/main_model.ckpt
```

**DDP正常表现**：
- 两张GPU显存使用量应该接近
- 训练速度是单卡的~1.8倍
- loss正常下降

现在重新开始训练吧！🚀
