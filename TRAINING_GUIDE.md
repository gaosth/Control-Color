# Control-Color 训练/微调指南

本指南详细介绍如何在自己的数据集上训练或微调 Control-Color 模型。

## 📋 目录
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [创建数据集类](#创建数据集类)
- [配置训练参数](#配置训练参数)
- [编写训练脚本](#编写训练脚本)
- [微调策略](#微调策略)
- [监控训练](#监控训练)

---

## 🔧 环境准备

```bash
# 安装依赖
conda env create -f CtrlColor_environ.yaml
conda activate CtrlColor

# 安装PyTorch Lightning (如果未安装)
pip install pytorch-lightning
```

---

## 📁 数据准备

### 数据集结构

Control-Color 需要**成对的数据**：灰度图/参考图 + 彩色图

```
your_dataset/
├── train/
│   ├── color/              # 彩色图像（ground truth）
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── gray/               # 灰度图像（输入）- 可选，也可以程序自动生成
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
├── val/
│   ├── color/
│   └── gray/
└── file_lists/
    ├── train.txt           # 训练图片路径列表
    └── val.txt             # 验证图片路径列表
```

### 生成文件列表

创建 `file_lists/train.txt`:
```bash
# 方式1: 如果只有彩色图（推荐）
find /path/to/your_dataset/train/color -name "*.jpg" -o -name "*.png" > file_lists/train.txt

# 方式2: 如果有灰度+彩色配对
# 需要在数据集类中处理配对关系
```

---

## 🗂️ 创建数据集类

在 `ldm/data/colorization.py` 中创建自定义数据集：

```python
import os
import cv2
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset

class ColorizationDataset(Dataset):
    """
    着色任务数据集

    数据增强策略：
    - 随机裁剪
    - 水平翻转
    - 颜色抖动
    """
    def __init__(self,
                 image_list_file,      # 图片路径列表文件
                 size=512,              # 图片尺寸
                 random_crop=True,      # 是否随机裁剪
                 use_mask=True,         # 是否使用mask（用于局部着色训练）
                 mask_ratio=0.3):       # mask比例
        super().__init__()

        # 读取图片路径
        with open(image_list_file, "r") as f:
            self.image_paths = f.read().splitlines()

        self.size = size
        self.random_crop = random_crop
        self.use_mask = use_mask
        self.mask_ratio = mask_ratio

        # 图像预处理pipeline
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if self.random_crop:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)

            # 数据增强
            self.augmentation = albumentations.Compose([
                self.rescaler,
                self.cropper,
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.5
                ),
            ])
        else:
            self.augmentation = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.image_paths)

    def load_and_preprocess(self, image_path):
        """加载并预处理图像"""
        # 读取彩色图像（ground truth）
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # 数据增强
        image = self.augmentation(image=image)["image"]

        # 转换为LAB色彩空间
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L_channel = image_lab[:, :, 0]

        # 生成灰度图（hint）
        gray_image = cv2.merge([L_channel, L_channel, L_channel])

        return image, gray_image, image_lab

    def generate_mask(self, height, width):
        """生成随机mask用于训练"""
        if not self.use_mask or np.random.rand() > 0.5:
            # 50%概率不使用mask（全图着色）
            return np.ones((height, width, 1), dtype=np.float32)

        # 生成随机mask
        mask = np.zeros((height, width, 1), dtype=np.float32)
        num_strokes = np.random.randint(3, 10)

        for _ in range(num_strokes):
            # 随机画笔参数
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(10, 50)
            cv2.circle(mask, (x, y), radius, 1.0, -1)

        # 膨胀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # 加载并预处理
        color_image, gray_image, image_lab = self.load_and_preprocess(image_path)

        # 生成mask
        mask = self.generate_mask(color_image.shape[0], color_image.shape[1])

        # 生成masked image (用于inpainting)
        masked_image = mask * gray_image + (1 - mask) * color_image

        # 归一化到[-1, 1]
        color_image = (color_image / 127.5 - 1.0).astype(np.float32)
        gray_image = (gray_image / 127.5 - 1.0).astype(np.float32)
        masked_image = (masked_image / 127.5 - 1.0).astype(np.float32)

        # 准备输出字典
        example = {
            "jpg": color_image,           # Ground truth彩色图
            "hint": gray_image,           # 灰度图（ControlNet输入）
            "mask": mask,                 # Mask
            "mask_img": masked_image,     # 带mask的图像
            "txt": ""                     # 文本提示（可选）
        }

        return example


class ColorizationTrain(ColorizationDataset):
    """训练集"""
    def __init__(self, size=512, training_images_list_file="file_lists/train.txt"):
        super().__init__(
            image_list_file=training_images_list_file,
            size=size,
            random_crop=True,
            use_mask=True
        )


class ColorizationValidation(ColorizationDataset):
    """验证集"""
    def __init__(self, size=512, validation_images_list_file="file_lists/val.txt"):
        super().__init__(
            image_list_file=validation_images_list_file,
            size=size,
            random_crop=False,  # 验证时使用中心裁剪
            use_mask=False      # 验证时不使用mask
        )
```

---

## ⚙️ 配置训练参数

创建训练配置文件 `models/cldm_colorization_train.yaml`:

```yaml
model:
  target: cldm.cldm.ControlLDM
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    control_key: "hint"
    masked_image: "mask_img"
    mask: "mask"
    image_size: 64
    channels: 4
    cond_stage_trainable: false
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: True
    only_mid_control: False

    # ControlNet配置
    control_stage_config:
      target: cldm.cldm.ControlNet
      params:
        image_size: 32
        in_channels: 4
        hint_channels: 3
        model_channels: 320
        attention_resolutions: [4, 2, 1]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    # UNet配置
    unet_config:
      target: cldm.cldm.ControlledUnetModel
      params:
        image_size: 32
        in_channels: 9  # 4(latent) + 1(mask) + 4(masked_latent)
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2, 1]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    # VAE配置（第一阶段模型）
    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [1, 2, 4, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    # 文本编码器配置
    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder

# 数据配置
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 4
    num_workers: 8
    wrap: false
    train:
      target: ldm.data.colorization.ColorizationTrain
      params:
        size: 512
        training_images_list_file: "file_lists/train.txt"

    validation:
      target: ldm.data.colorization.ColorizationValidation
      params:
        size: 512
        validation_images_list_file: "file_lists/val.txt"

# Lightning配置
lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 500
        max_images: 8
        increase_log_steps: False

  trainer:
    max_epochs: 100
    accelerator: "gpu"
    devices: 1
    precision: 16  # 使用混合精度训练
    accumulate_grad_batches: 4
    gradient_clip_val: 1.0
```

---

## 🚀 编写训练脚本

创建 `train.py`:

```python
import argparse
import datetime
import os
import sys
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from cldm.model import create_model, load_state_dict
from ldm.data.base import DataModuleFromConfig


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b", "--base",
        type=str,
        default="models/cldm_colorization_train.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "-r", "--resume",
        type=str,
        default="",
        help="从checkpoint恢复训练"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="预训练模型路径（用于微调）"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "-l", "--logdir",
        type=str,
        default="logs",
        help="日志目录"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="根据batch size缩放学习率"
    )
    return parser


def main():
    parser = get_parser()
    opt = parser.parse_args()

    # 设置随机种子
    seed_everything(opt.seed)

    # 加载配置
    config = OmegaConf.load(opt.base)

    # 创建数据模块
    data = DataModuleFromConfig(**config.data.params)
    data.prepare_data()
    data.setup()

    # 创建模型
    model = create_model(opt.base)

    # 加载预训练权重（微调）
    if opt.pretrained:
        print(f"Loading pretrained model from {opt.pretrained}")
        sd = load_state_dict(opt.pretrained, location='cpu')
        model.load_state_dict(sd, strict=False)

    # 设置学习率
    bs = config.data.params.batch_size
    base_lr = config.model.base_learning_rate
    if opt.scale_lr:
        model.learning_rate = bs * base_lr
        print(f"Scaled learning rate to {model.learning_rate}")
    else:
        model.learning_rate = base_lr

    # 创建callbacks
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir = os.path.join(opt.logdir, now)

    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logdir, "checkpoints"),
        filename="epoch{epoch:02d}-step{step:06d}",
        save_top_k=3,
        monitor="val/loss_simple_ema",
        mode="min",
        save_last=True,
        every_n_train_steps=1000
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Logger
    logger = TensorBoardLogger(save_dir=logdir, name="tensorboard")

    # 创建Trainer
    trainer = Trainer(
        max_epochs=config.lightning.trainer.max_epochs,
        accelerator=config.lightning.trainer.accelerator,
        devices=config.lightning.trainer.devices,
        precision=config.lightning.trainer.precision,
        accumulate_grad_batches=config.lightning.trainer.accumulate_grad_batches,
        gradient_clip_val=config.lightning.trainer.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1000,
        resume_from_checkpoint=opt.resume if opt.resume else None
    )

    # 开始训练
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
```

---

## 🎯 微调策略

### 1. 从预训练模型微调（推荐）

```bash
# 下载预训练模型
# 放在 ./pretrained_models/main_model.ckpt

# 微调主模型
python train.py \
    --base models/cldm_colorization_train.yaml \
    --pretrained ./pretrained_models/main_model.ckpt \
    --logdir logs/finetune
```

### 2. 只训练ControlNet（冻结UNet）

修改配置文件，在 `cldm/cldm.py:531` 的 `configure_optimizers`:

```python
def configure_optimizers(self):
    lr = self.learning_rate

    # 只优化ControlNet
    params = list(self.control_model.parameters())

    # 如果要微调UNet的输出层
    # params += list(self.model.diffusion_model.output_blocks.parameters())
    # params += list(self.model.diffusion_model.out.parameters())

    opt = torch.optim.AdamW(params, lr=lr)
    return opt
```

### 3. 训练Deformable VAE

如果要训练内容引导的可变形VAE:

```bash
# 使用 ldm/models/autoencoder_train.py 中的训练逻辑
python train_vae.py \
    --base configs/autoencoder_colorization.yaml \
    --gpus 0,
```

---

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir logs/
```

### 关键指标

- `train/loss_simple`: 训练损失
- `val/loss_simple_ema`: 验证损失（EMA）
- `train/loss_vlb`: VLB损失
- `lr_abs`: 学习率

### 可视化

训练过程中会自动保存生成的图像到 TensorBoard：
- `inputs`: 输入灰度图
- `reconstruction`: 重建图像
- `samples`: 生成的彩色图

---

## 💡 训练技巧

### 1. 学习率调度

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

def configure_optimizers(self):
    optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=100000,  # 总步数
        eta_min=1e-7
    )
    return [optimizer], [scheduler]
```

### 2. 数据增强

建议的增强策略：
- 随机裁剪
- 水平翻转
- 轻微的颜色抖动（小心不要改变太多颜色分布）
- 随机缩放

### 3. Batch Size调整

```bash
# GPU显存不足时，使用梯度累积
# 在配置文件中设置: accumulate_grad_batches: 4
# 等效于 batch_size * 4
```

### 4. 混合精度训练

```yaml
# 在配置文件中
lightning:
  trainer:
    precision: 16  # 或 "bf16"
```

---

## 🐛 常见问题

### Q: CUDA out of memory
**A**:
- 减小batch_size
- 增加accumulate_grad_batches
- 使用更小的图像尺寸（256而非512）
- 启用gradient checkpointing

### Q: 生成结果颜色溢出
**A**:
- 增加训练时mask的使用比例
- 调整ControlNet的控制强度
- 确保训练了Deformable VAE

### Q: 训练不稳定
**A**:
- 降低学习率
- 使用gradient clipping
- 检查数据归一化是否正确

---

## 📚 参考资源

- [PyTorch Lightning文档](https://pytorch-lightning.readthedocs.io/)
- [ControlNet论文](https://arxiv.org/abs/2302.05543)
- [Control-Color论文](https://arxiv.org/abs/2402.10855)

---

## 🚦 快速开始

```bash
# 1. 准备数据
mkdir -p your_dataset/train/color
# 复制你的彩色图片到这个目录

# 2. 生成文件列表
find your_dataset/train/color -name "*.jpg" > file_lists/train.txt

# 3. 创建数据集类
# 参考上面的 ColorizationDataset

# 4. 开始训练
python train.py \
    --base models/cldm_colorization_train.yaml \
    --pretrained ./pretrained_models/main_model.ckpt \
    --seed 42 \
    --logdir logs/my_experiment
```

祝训练顺利！🎉
