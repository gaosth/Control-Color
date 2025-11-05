"""
测试数据集参数配置
直接从 ColorRestorationTrain 数据集类生成样本，验证褪色参数效果

使用方法:
    # 基本测试 - 生成几个样本看看效果
    python test_dataset_params.py --images img1.jpg img2.jpg img3.jpg

    # 使用文件列表测试
    python test_dataset_params.py --file_list file_lists/train.txt --num_samples 10

    # 批量测试 - 生成多个样本观察一致性
    python test_dataset_params.py --images img1.jpg --num_generations 5

    # 对比不同参数配置
    python test_dataset_params.py --images img1.jpg --compare_configs
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from ldm.data.color_restoration import ColorRestorationTrain, simulate_color_fading


class DatasetTester:
    """数据集参数测试器"""

    def __init__(self, output_dir='dataset_test_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def test_single_image(self, image_path, num_generations=1):
        """测试单张图片的褪色效果

        Args:
            image_path: 图片路径
            num_generations: 生成多少个不同的褪色版本（测试随机性）
        """
        print(f"\n{'='*60}")
        print(f"测试图片: {image_path}")
        print(f"{'='*60}")

        # 读取原图
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # 创建临时数据集实例（不需要file list）
        dataset = ColorRestorationTrain(size=512, training_images_list_file=None)
        # 手动设置图片列表
        dataset.image_paths = [image_path]

        # 准备原图
        preprocessed = dataset.load_and_preprocess(image_path)

        # 生成多个褪色版本
        results = []
        for i in range(num_generations):
            faded, intensity = dataset.generate_faded_image(preprocessed)
            results.append({
                'faded': faded,
                'intensity': intensity,
            })

        # 可视化
        if num_generations == 1:
            self._visualize_single(preprocessed, results[0], image_path)
        else:
            self._visualize_multiple(preprocessed, results, image_path)

        return results

    def _visualize_single(self, original, result, image_path):
        """可视化单个结果"""
        faded = result['faded']
        intensity = result['intensity']

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 原图
        axes[0].imshow(original)
        axes[0].set_title('原始图片', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 褪色图
        axes[1].imshow(faded)
        axes[1].set_title(f'数据集生成的褪色效果\n(intensity={intensity:.3f})',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()

        # 保存
        output_filename = Path(image_path).stem + '_dataset_test.png'
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存: {output_path}")
        plt.close()

    def _visualize_multiple(self, original, results, image_path):
        """可视化多个生成结果（观察随机性和一致性）"""
        num_results = len(results)

        # 计算网格大小
        cols = min(4, num_results + 1)
        rows = (num_results + 1 + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        # 原图
        axes[0].imshow(original)
        axes[0].set_title('原始图片', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 各个褪色版本
        for i, result in enumerate(results):
            axes[i+1].imshow(result['faded'])
            axes[i+1].set_title(f'版本 {i+1}\n(intensity={result["intensity"]:.3f})',
                               fontsize=11, fontweight='bold')
            axes[i+1].axis('off')

        # 隐藏多余的子图
        for i in range(num_results + 1, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'数据集褪色效果测试 - {num_results}个随机生成版本',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存
        output_filename = Path(image_path).stem + f'_dataset_test_{num_results}versions.png'
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存: {output_path}")
        plt.close()

    def test_with_dataset_pipeline(self, image_path):
        """完整测试数据集pipeline（包括mask和其他处理）"""
        print(f"\n{'='*60}")
        print(f"测试完整数据集pipeline: {image_path}")
        print(f"{'='*60}")

        # 创建数据集
        dataset = ColorRestorationTrain(size=512, training_images_list_file=None)
        dataset.image_paths = [image_path]

        # 获取数据集输出
        sample = dataset[0]

        # 提取各个组件
        original = sample['jpg']  # [-1, 1]
        hint = sample['hint']     # [-1, 1]
        faded = sample['faded']   # [-1, 1]
        mask = sample['mask']     # [0, 1]
        masked_img = sample['mask_img']  # [-1, 1]
        text = sample['txt']

        # 反归一化到[0, 255]用于显示
        def denorm(x):
            return ((x + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

        # 可视化完整pipeline
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        axes[0].imshow(denorm(original))
        axes[0].set_title('Ground Truth\n(训练目标)', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(denorm(faded))
        axes[1].set_title('Faded Image\n(褪色图)', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(denorm(hint))
        axes[2].set_title('Color Hint\n(颜色提示)', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Mask可视化（扩展到3通道）
        mask_vis = np.repeat(mask, 3, axis=2)
        axes[3].imshow(mask_vis, cmap='gray')
        axes[3].set_title('Mask\n(1=需要修复)', fontsize=12, fontweight='bold')
        axes[3].axis('off')

        axes[4].imshow(denorm(masked_img))
        axes[4].set_title('Masked Image\n(带mask的图像)', fontsize=12, fontweight='bold')
        axes[4].axis('off')

        # 文本提示
        axes[5].text(0.5, 0.5, f'Text Prompt:\n"{text}"',
                    ha='center', va='center', fontsize=11, wrap=True)
        axes[5].set_title('Text Condition', fontsize=12, fontweight='bold')
        axes[5].axis('off')

        plt.suptitle('完整数据集Pipeline输出', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存
        output_filename = Path(image_path).stem + '_full_pipeline.png'
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"完整pipeline测试结果已保存: {output_path}")
        plt.close()

        # 打印统计信息
        print(f"\n数据统计:")
        print(f"  原图 shape: {original.shape}, range: [{original.min():.3f}, {original.max():.3f}]")
        print(f"  褪色图 shape: {faded.shape}, range: [{faded.min():.3f}, {faded.max():.3f}]")
        print(f"  颜色提示 shape: {hint.shape}, range: [{hint.min():.3f}, {hint.max():.3f}]")
        print(f"  Mask shape: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]")
        print(f"  Mask覆盖率: {mask.mean():.1%}")
        print(f"  文本提示: {text}")

    def compare_different_configs(self, image_path):
        """对比不同参数配置的效果"""
        print(f"\n{'='*60}")
        print(f"对比不同参数配置: {image_path}")
        print(f"{'='*60}")

        # 读取原图
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # 定义不同配置
        configs = [
            {
                'name': '当前数据集配置\n(ColorRestorationTrain)',
                'use_dataset': True,
            },
            {
                'name': '轻度褪色\n(intensity=0.4)',
                'intensity': 0.4,
                'effects': {'saturation': 0.5, 'yellow': 0.7},
            },
            {
                'name': '中度褪色\n(intensity=0.6)',
                'intensity': 0.6,
                'effects': {'saturation': 0.6, 'yellow': 1.0, 'sepia': 0.5},
            },
            {
                'name': '重度褪色\n(intensity=0.8)',
                'intensity': 0.8,
                'effects': {'saturation': 0.8, 'yellow': 1.0, 'sepia': 0.7},
            },
        ]

        # 生成结果
        results = []
        for config in configs:
            if config.get('use_dataset'):
                # 使用数据集配置
                dataset = ColorRestorationTrain(size=512, training_images_list_file=None)
                dataset.image_paths = [image_path]
                preprocessed = dataset.load_and_preprocess(image_path)
                faded, intensity = dataset.generate_faded_image(preprocessed)
                config['intensity_actual'] = intensity
            else:
                # 使用指定配置
                faded = simulate_color_fading(
                    image_np,
                    fade_type='combined',
                    intensity=config['intensity'],
                    combined_effects=config['effects']
                )

            results.append({
                'config': config,
                'faded': faded,
            })

        # 可视化对比
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # 原图
        axes[0].imshow(image_np)
        axes[0].set_title('原始图片', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 各个配置
        for i, result in enumerate(results):
            config = result['config']
            faded = result['faded']

            title = config['name']
            if 'intensity_actual' in config:
                title += f"\n(实际intensity={config['intensity_actual']:.3f})"

            axes[i+1].imshow(faded)
            axes[i+1].set_title(title, fontsize=11, fontweight='bold')
            axes[i+1].axis('off')

        # 隐藏多余的子图
        for i in range(len(results) + 1, len(axes)):
            axes[i].axis('off')

        plt.suptitle('不同参数配置对比', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存
        output_filename = Path(image_path).stem + '_config_comparison.png'
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"配置对比结果已保存: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='测试数据集参数配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本测试
  python test_dataset_params.py --images img1.jpg img2.jpg

  # 测试随机性（生成5个版本）
  python test_dataset_params.py --images img1.jpg --num_generations 5

  # 测试完整pipeline
  python test_dataset_params.py --images img1.jpg --full_pipeline

  # 对比不同配置
  python test_dataset_params.py --images img1.jpg --compare_configs

  # 批量测试文件列表中的图片
  python test_dataset_params.py --file_list file_lists/train.txt --num_samples 10
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--images', nargs='+', help='要测试的图片路径')
    group.add_argument('--file_list', type=str, help='图片列表文件（每行一个路径）')

    parser.add_argument('--num_samples', type=int, default=None,
                       help='从file_list中随机选择多少张图片测试（默认全部）')
    parser.add_argument('--num_generations', type=int, default=1,
                       help='每张图片生成多少个褪色版本（测试随机性）')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='测试完整的数据集pipeline（包括mask等）')
    parser.add_argument('--compare_configs', action='store_true',
                       help='对比不同参数配置的效果')
    parser.add_argument('--output_dir', type=str, default='dataset_test_results',
                       help='输出目录')

    args = parser.parse_args()

    # 获取要测试的图片列表
    if args.images:
        image_paths = args.images
    else:
        with open(args.file_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]

        if args.num_samples:
            import random
            image_paths = random.sample(image_paths, min(args.num_samples, len(image_paths)))

    # 创建测试器
    tester = DatasetTester(args.output_dir)

    print("="*60)
    print("数据集参数测试")
    print("="*60)
    print(f"测试图片数量: {len(image_paths)}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)

    # 执行测试
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在，跳过: {img_path}")
            continue

        if args.compare_configs:
            # 对比不同配置
            tester.compare_different_configs(img_path)
        elif args.full_pipeline:
            # 测试完整pipeline
            tester.test_with_dataset_pipeline(img_path)
        else:
            # 基本测试
            tester.test_single_image(img_path, args.num_generations)

    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"所有结果已保存至: {args.output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
