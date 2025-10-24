#!/usr/bin/env python3
"""
数据集准备脚本
帮助快速设置训练和验证数据集

使用方法：
    python prepare_dataset.py --source my_images/ --train_ratio 0.8
"""

import os
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='准备训练和验证数据集')
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='源图片目录路径'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='训练集比例（默认0.8，即80%训练，20%验证）'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='data/train',
        help='训练集输出目录'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default='data/val',
        help='验证集输出目录'
    )
    parser.add_argument(
        '--file_lists_dir',
        type=str,
        default='file_lists',
        help='文件列表输出目录'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='复制文件（默认是移动文件）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    return parser.parse_args()


def collect_images(source_dir):
    """收集所有图片文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []

    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"源目录不存在: {source_dir}")

    print(f"\n正在扫描目录: {source_dir}")
    for ext in image_extensions:
        images.extend(source_path.glob(f'**/*{ext}'))

    print(f"找到 {len(images)} 张图片")
    return images


def split_dataset(images, train_ratio, seed=42):
    """划分训练集和验证集"""
    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_images)} 张 ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_images)} 张 ({(1-train_ratio)*100:.0f}%)")

    return train_images, val_images


def copy_or_move_files(images, target_dir, copy_mode=False):
    """复制或移动文件到目标目录"""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    operation = shutil.copy2 if copy_mode else shutil.move
    operation_name = "复制" if copy_mode else "移动"

    print(f"\n{operation_name}文件到 {target_dir}...")
    for img in tqdm(images):
        dest = target_path / img.name
        # 如果目标文件已存在，添加序号
        if dest.exists():
            stem = img.stem
            suffix = img.suffix
            counter = 1
            while dest.exists():
                dest = target_path / f"{stem}_{counter}{suffix}"
                counter += 1
        operation(str(img), str(dest))


def create_file_lists(train_dir, val_dir, file_lists_dir):
    """创建文件列表"""
    file_lists_path = Path(file_lists_dir)
    file_lists_path.mkdir(parents=True, exist_ok=True)

    # 训练集列表
    train_list_file = file_lists_path / 'train.txt'
    train_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        train_images.extend(Path(train_dir).glob(f'*{ext}'))

    train_images.sort()
    with open(train_list_file, 'w') as f:
        for img in train_images:
            f.write(f"{img}\n")

    print(f"\n训练集列表: {train_list_file}")
    print(f"  包含 {len(train_images)} 张图片")

    # 验证集列表
    val_list_file = file_lists_path / 'val.txt'
    val_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        val_images.extend(Path(val_dir).glob(f'*{ext}'))

    val_images.sort()
    with open(val_list_file, 'w') as f:
        for img in val_images:
            f.write(f"{img}\n")

    print(f"验证集列表: {val_list_file}")
    print(f"  包含 {len(val_images)} 张图片")


def main():
    args = parse_args()

    print("=" * 80)
    print("数据集准备工具")
    print("=" * 80)

    # 1. 收集图片
    images = collect_images(args.source)

    if len(images) == 0:
        print("\n错误: 没有找到图片文件")
        return

    # 2. 划分数据集
    train_images, val_images = split_dataset(images, args.train_ratio, args.seed)

    if len(val_images) < 10:
        print(f"\n警告: 验证集只有 {len(val_images)} 张图片，建议至少50-100张")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return

    # 3. 复制/移动文件
    copy_or_move_files(train_images, args.train_dir, args.copy)
    copy_or_move_files(val_images, args.val_dir, args.copy)

    # 4. 创建文件列表
    create_file_lists(args.train_dir, args.val_dir, args.file_lists_dir)

    print("\n" + "=" * 80)
    print("✓ 数据集准备完成!")
    print("=" * 80)
    print("\n下一步:")
    print(f"  1. 查看数据: ls {args.train_dir}  和  ls {args.val_dir}")
    print(f"  2. 查看列表: cat {args.file_lists_dir}/train.txt")
    print("  3. 开始训练:")
    print("\n     python train_restoration.py \\")
    print("         --pretrained pretrained_models/main_model.ckpt \\")
    print(f"         --train_list {args.file_lists_dir}/train.txt \\")
    print(f"         --val_list {args.file_lists_dir}/val.txt \\")
    print("         --gpus 0,1 \\")
    print("         --batch_size 2 \\")
    print("         --learning_rate 1e-5 \\")
    print("         --max_epochs 50 \\")
    print("         --freeze_unet")
    print()


if __name__ == '__main__':
    main()
