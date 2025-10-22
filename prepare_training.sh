#!/bin/bash
# Control-Color 训练准备脚本

echo "=================================="
echo "Control-Color 训练环境准备"
echo "=================================="
echo ""

# 创建必要的目录
echo "1. 创建必要的目录..."
mkdir -p file_lists
mkdir -p logs
mkdir -p data/train/color
mkdir -p data/val/color
echo "✓ 目录创建完成"
echo ""

# 检查预训练模型
echo "2. 检查预训练模型..."
if [ -f "pretrained_models/main_model.ckpt" ]; then
    echo "✓ 找到预训练模型: pretrained_models/main_model.ckpt"
else
    echo "⚠ 未找到预训练模型"
    echo "请从以下链接下载并放到 pretrained_models/ 目录:"
    echo "Google Drive: https://drive.google.com/drive/folders/1lgqstNwrMCzymowRsbGM-4hk0-7L-eOT"
fi
echo ""

# 检查VAE模型
echo "3. 检查VAE模型..."
if [ -f "pretrained_models/content-guided_deformable_vae.ckpt" ]; then
    echo "✓ 找到VAE模型: pretrained_models/content-guided_deformable_vae.ckpt"
else
    echo "⚠ 未找到VAE模型"
    echo "如果需要使用Deformable VAE，请下载此模型"
fi
echo ""

# 生成训练数据列表
echo "4. 生成训练数据列表..."
if [ -d "data/train/color" ]; then
    find data/train/color -name "*.jpg" -o -name "*.png" | sort > file_lists/train.txt
    train_count=$(wc -l < file_lists/train.txt)
    echo "✓ 训练集: $train_count 张图片"

    if [ $train_count -eq 0 ]; then
        echo "⚠ 警告: 训练集为空!"
        echo "请将训练图片放到 data/train/color/ 目录"
    fi
fi

if [ -d "data/val/color" ]; then
    find data/val/color -name "*.jpg" -o -name "*.png" | sort > file_lists/val.txt
    val_count=$(wc -l < file_lists/val.txt)
    echo "✓ 验证集: $val_count 张图片"

    if [ $val_count -eq 0 ]; then
        echo "⚠ 警告: 验证集为空 (可选)"
    fi
fi
echo ""

# 显示使用说明
echo "=================================="
echo "准备完成！使用以下命令开始训练:"
echo ""
echo "# 从预训练模型微调 (推荐):"
echo "python train.py \\"
echo "    --pretrained pretrained_models/main_model.ckpt \\"
echo "    --train_list file_lists/train.txt \\"
echo "    --val_list file_lists/val.txt \\"
echo "    --batch_size 4 \\"
echo "    --learning_rate 1e-5 \\"
echo "    --max_epochs 100 \\"
echo "    --gpus 0"
echo ""
echo "# 从零开始训练:"
echo "python train.py \\"
echo "    --train_list file_lists/train.txt \\"
echo "    --val_list file_lists/val.txt \\"
echo "    --batch_size 4 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --max_epochs 200 \\"
echo "    --gpus 0"
echo ""
echo "# 只训练ControlNet (冻结UNet):"
echo "python train.py \\"
echo "    --pretrained pretrained_models/main_model.ckpt \\"
echo "    --freeze_unet \\"
echo "    --train_list file_lists/train.txt \\"
echo "    --batch_size 8 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --gpus 0"
echo ""
echo "监控训练进度:"
echo "tensorboard --logdir logs/"
echo "=================================="
