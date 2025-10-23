#!/bin/bash
# 颜色修复任务准备脚本

echo "=========================================="
echo "颜色修复/恢复任务环境准备"
echo "=========================================="
echo ""

# 创建必要的目录
echo "1. 创建必要的目录..."
mkdir -p file_lists
mkdir -p logs_restoration
mkdir -p data/train/color
mkdir -p data/val/color
mkdir -p test_images/faded
mkdir -p test_images/restored
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

# 生成训练数据列表
echo "3. 生成训练数据列表..."
if [ -d "data/train/color" ]; then
    find data/train/color -name "*.jpg" -o -name "*.png" | sort > file_lists/train.txt
    train_count=$(wc -l < file_lists/train.txt)
    echo "✓ 训练集: $train_count 张图片"

    if [ $train_count -eq 0 ]; then
        echo "⚠ 警告: 训练集为空!"
        echo "请将原始彩色图片放到 data/train/color/ 目录"
        echo "脚本会自动模拟褪色效果进行训练"
    fi
fi

if [ -d "data/val/color" ]; then
    find data/val/color -name "*.jpg" -o -name "*.png" | sort > file_lists/val.txt
    val_count=$(wc -l < file_lists/val.txt)
    echo "✓ 验证集: $val_count 张图片"
fi
echo ""

# 测试褪色效果
echo "4. 测试褪色效果生成..."
echo "运行测试脚本生成褪色效果对比图..."
python3 -c "
try:
    from ldm.data.color_restoration import test_fading_effects
    test_fading_effects()
    print('✓ 褪色效果测试完成，查看 fading_effects_test.png')
except Exception as e:
    print(f'⚠ 测试失败: {e}')
    print('这是正常的，训练时会自动应用褪色效果')
" 2>/dev/null
echo ""

# 显示使用说明
echo "=========================================="
echo "准备完成！"
echo "=========================================="
echo ""
echo "📖 快速开始："
echo ""
echo "方式1: 测试褪色效果（无需训练）"
echo "--------------------------------------"
echo "# 在现有图片上模拟褪色并修复"
echo "python test_restoration.py \\"
echo "    --input test_image.jpg \\"
echo "    --output restored.png \\"
echo "    --ckpt pretrained_models/main_model.ckpt \\"
echo "    --fade_type yellow \\"
echo "    --fade_intensity 0.6 \\"
echo "    --save_comparison"
echo ""
echo "方式2: 训练颜色修复模型"
echo "--------------------------------------"
echo "# 从预训练模型微调（推荐）"
echo "python train_restoration.py \\"
echo "    --pretrained pretrained_models/main_model.ckpt \\"
echo "    --train_list file_lists/train.txt \\"
echo "    --val_list file_lists/val.txt \\"
echo "    --batch_size 4 \\"
echo "    --learning_rate 1e-5 \\"
echo "    --max_epochs 50 \\"
echo "    --fade_type mixed"
echo ""
echo "# 快速训练（只训练ControlNet）"
echo "python train_restoration.py \\"
echo "    --pretrained pretrained_models/main_model.ckpt \\"
echo "    --freeze_unet \\"
echo "    --train_list file_lists/train.txt \\"
echo "    --batch_size 8 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --max_epochs 20"
echo ""
echo "方式3: 使用训练好的模型修复真实褪色照片"
echo "--------------------------------------"
echo "python test_restoration.py \\"
echo "    --input faded_photo.jpg \\"
echo "    --output restored.png \\"
echo "    --ckpt logs_restoration/YYYY-MM-DD/checkpoints/last.ckpt \\"
echo "    --is_faded \\"
echo "    --save_comparison"
echo ""
echo "监控训练进度："
echo "--------------------------------------"
echo "tensorboard --logdir logs_restoration/"
echo ""
echo "详细文档："
echo "--------------------------------------"
echo "查看 COLOR_RESTORATION_GUIDE.md 了解更多"
echo ""
echo "=========================================="
echo ""
echo "💡 提示："
echo "- 训练只需要原始彩色图片，会自动模拟褪色"
echo "- 支持5种褪色类型：uniform, saturation, yellow, sepia, mixed"
echo "- 推荐使用 'mixed' 类型训练以获得更好泛化能力"
echo "- 褪色强度范围推荐 0.3-0.7"
echo ""
echo "=========================================="
