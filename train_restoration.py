"""
颜色修复/恢复任务训练脚本

训练模型恢复褪色照片的颜色
输入：褪色图片 + 褪色颜色hint
输出：原始彩色图片
"""
import argparse
import datetime
import os
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from cldm.model import create_model, load_state_dict


class DataModuleFromConfig(pl.LightningDataModule):
    """从配置创建数据模块"""
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 num_workers=4, shuffle_train=True, shuffle_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def setup(self, stage=None):
        """实例化数据集"""
        if stage == 'fit' or stage is None:
            if "train" in self.dataset_configs:
                self.datasets_train = self.instantiate_from_config(self.dataset_configs["train"])
            if "validation" in self.dataset_configs:
                self.datasets_val = self.instantiate_from_config(self.dataset_configs["validation"])

    def instantiate_from_config(self, config):
        """从配置实例化对象"""
        if not "target" in config:
            raise KeyError("Expected key `target` to instantiate.")
        module, cls = config["target"].rsplit(".", 1)
        cls = getattr(__import__(module, fromlist=[cls]), cls)
        return cls(**config.get("params", dict()))

    def train_dataloader(self):
        return DataLoader(
            self.datasets_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_val,
            pin_memory=True
        )


class RestorationImageLogger(pl.Callback):
    """记录修复结果到TensorBoard"""
    def __init__(self, batch_frequency=500, max_images=8):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.batch_freq == 0:
            self.log_images(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self.log_images(trainer, pl_module, batch, batch_idx, split="val")

    @torch.no_grad()
    def log_images(self, trainer, pl_module, batch, batch_idx, split="train"):
        """记录图像对比：褪色图 -> 修复图 -> 原图"""
        logger = trainer.logger.experiment

        # 获取输入
        images = pl_module.log_images(batch, split=split)

        # 额外记录褪色图（如果有）
        if 'faded' in batch:
            faded = batch['faded'][:self.max_images]
            faded = torch.clamp(faded, -1., 1.)
            faded = (faded + 1.0) / 2.0
            logger.add_images(f"{split}/faded_input", faded, global_step=trainer.global_step)

        for k in images:
            grid = images[k]
            if len(grid.shape) == 4:
                grid = grid[0:self.max_images]
                grid = torch.clamp(grid, -1., 1.)
                grid = (grid + 1.0) / 2.0

                tag = f"{split}/{k}"
                logger.add_images(tag, grid, global_step=trainer.global_step)


def get_parser(**parser_kwargs):
    """命令行参数解析"""
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-b", "--base",
        type=str,
        default="models/cldm_v15_inpainting_infer1.yaml",
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
        default="logs_restoration",
        help="日志目录"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="最大训练轮数"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="使用的GPU ID"
    )
    parser.add_argument(
        "--train_list",
        type=str,
        default="file_lists/train.txt",
        help="训练图片列表文件"
    )
    parser.add_argument(
        "--val_list",
        type=str,
        default="file_lists/val.txt",
        help="验证图片列表文件"
    )
    parser.add_argument(
        "--fade_type",
        type=str,
        default="mixed",
        choices=['uniform', 'saturation', 'brightness', 'yellow', 'sepia', 'mixed'],
        help="褪色类型"
    )
    parser.add_argument(
        "--fade_intensity",
        type=float,
        nargs=2,
        default=[0.3, 0.7],
        help="褪色强度范围 [min, max]"
    )
    parser.add_argument(
        "--freeze_unet",
        action="store_true",
        help="冻结UNet，只训练ControlNet"
    )

    return parser


def main():
    parser = get_parser()
    opt = parser.parse_args()

    # 设置随机种子
    seed_everything(opt.seed)

    # 创建日志目录
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir = os.path.join(opt.logdir, now)
    os.makedirs(logdir, exist_ok=True)

    # 保存配置
    config_save_path = os.path.join(logdir, "config.txt")
    with open(config_save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Color Restoration Training Configuration\n")
        f.write("=" * 80 + "\n\n")
        for arg in vars(opt):
            f.write(f"{arg}: {getattr(opt, arg)}\n")
        f.write("\n" + "=" * 80 + "\n")

    print(f"Logging to {logdir}")

    # 检查文件是否存在
    if not os.path.exists(opt.train_list):
        print(f"Error: Training list file not found: {opt.train_list}")
        print("Please create this file with image paths, one per line.")
        return

    # 创建数据模块
    from ldm.data.color_restoration import ColorRestorationTrain, ColorRestorationValidation

    data = DataModuleFromConfig(
        batch_size=opt.batch_size,
        num_workers=4,
        train={
            "target": "ldm.data.color_restoration.ColorRestorationTrain",
            "params": {
                "size": 512,
                "training_images_list_file": opt.train_list,
                # 注意：这些参数会被类的__init__中的默认值覆盖
                # 如果需要自定义，需要修改数据集类
            }
        },
        validation={
            "target": "ldm.data.color_restoration.ColorRestorationValidation",
            "params": {
                "size": 512,
                "validation_images_list_file": opt.val_list
            }
        } if os.path.exists(opt.val_list) else None
    )

    # 设置数据
    data.setup()

    # 创建模型
    print(f"Creating model from config: {opt.base}")
    model = create_model(opt.base).cpu()

    # 加载预训练权重（微调）
    if opt.pretrained:
        print(f"Loading pretrained model from {opt.pretrained}")
        try:
            sd = load_state_dict(opt.pretrained, location='cpu')
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            print("✓ Pretrained model loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")

    # 设置学习率
    model.learning_rate = opt.learning_rate

    # 冻结UNet（如果需要）
    if opt.freeze_unet:
        print("Freezing UNet parameters...")
        model.sd_locked = True  # 设置sd_locked，避免configure_optimizers添加UNet参数
        for param in model.model.diffusion_model.parameters():
            param.requires_grad = False
        print("✓ UNet frozen, only ControlNet will be trained")
    else:
        # 确保sd_locked设置正确
        model.sd_locked = False

    # Note: Do not manually move model to GPU when using PyTorch Lightning
    # Lightning will handle device placement automatically in multi-GPU setup

    # 创建callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logdir, "checkpoints"),
        filename="epoch{epoch:02d}-step{step:06d}",
        save_top_k=3,
        monitor="train/loss" if not os.path.exists(opt.val_list) else "val/loss",
        mode="min",
        save_last=True,
        every_n_train_steps=1000,
        save_on_train_epoch_end=True
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Image logger (专门用于颜色修复的可视化)
    image_logger = RestorationImageLogger(batch_frequency=500, max_images=4)
    callbacks.append(image_logger)

    # Logger
    logger = TensorBoardLogger(save_dir=logdir, name="tensorboard")

    # 解析GPU设置
    gpus = [int(x) for x in opt.gpus.split(',')]

    # 创建Trainer
    # 验证策略：
    # - 有验证集：每个epoch验证一次 (val_check_interval=1.0)
    # - 无验证集：禁用验证 (limit_val_batches=0)
    trainer_kwargs = {
        "max_epochs": opt.max_epochs,
        "accelerator": "gpu",
        "devices": gpus,
        "precision": 16,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1.0,
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": 50,
        "num_sanity_val_steps": 0,
    }

    # 多卡训练提示和策略设置
    if len(gpus) > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
        print(f"\n使用 {len(gpus)} 张GPU进行分布式训练 (DDP模式)")
        print(f"GPU设备: {gpus}")
        print(f"每张GPU的batch size: {opt.batch_size}")
        print(f"实际总batch size: {opt.batch_size * len(gpus)}\n")

    if os.path.exists(opt.val_list):
        # 有验证集：每个epoch结束时验证
        trainer_kwargs["val_check_interval"] = 1.0  # float表示每个epoch验证
    else:
        # 无验证集：禁用验证
        trainer_kwargs["limit_val_batches"] = 0

    trainer = Trainer(**trainer_kwargs)

    # 开始训练
    print("\n" + "=" * 80)
    print("Starting Color Restoration Training...")
    print("=" * 80)
    print(f"Task: Photo Color Restoration (Fading Recovery)")
    print(f"Fade type: {opt.fade_type}")
    print(f"Fade intensity range: {opt.fade_intensity}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Learning rate: {opt.learning_rate}")
    print(f"Max epochs: {opt.max_epochs}")
    print(f"Training samples: {len(data.datasets_train)}")
    if os.path.exists(opt.val_list):
        print(f"Validation samples: {len(data.datasets_val)}")
    print("=" * 80 + "\n")

    trainer.fit(model, data, ckpt_path=opt.resume if opt.resume else None)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Logs saved to: {logdir}")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
