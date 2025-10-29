"""
测试褪色效果脚本
用你自己的图片测试不同的褪色效果

使用方法:
    python test_fading.py --image path/to/your/image.jpg

    # 指定褪色类型
    python test_fading.py --image path/to/your/image.jpg --fade_type yellow

    # 指定褪色强度
    python test_fading.py --image path/to/your/image.jpg --intensity 0.7
"""
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入褪色函数
sys.path.insert(0, os.path.dirname(__file__))
from ldm.data.color_restoration import simulate_color_fading, extract_color_mask


def test_single_image(image_path, fade_type='mixed', intensity=0.5, output_dir='fading_test_results',
                     combined_effects=None):
    """
    测试单张图片的褪色效果

    Args:
        image_path: 图片路径
        fade_type: 褪色类型 ('uniform', 'saturation', 'brightness', 'yellow', 'sepia', 'mixed', 'combined')
        intensity: 褪色强度 (0-1)
        output_dir: 输出目录
        combined_effects: 组合效果配置（当fade_type='combined'时使用）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图片
    print(f"Reading image: {image_path}")
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)

    print(f"Image shape: {image.shape}")

    # 应用褪色
    if fade_type == 'combined' and combined_effects:
        print(f"Applying combined fading (intensity={intensity})...")
        print(f"  Effects: {combined_effects}")
    else:
        print(f"Applying fading (type={fade_type}, intensity={intensity})...")

    faded_image = simulate_color_fading(image, fade_type=fade_type, intensity=intensity,
                                       combined_effects=combined_effects)

    # 提取颜色hint
    color_hint = extract_color_mask(faded_image, image, 'faded_color')

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(faded_image)
    axes[1].set_title(f'Faded Image\n({fade_type}, intensity={intensity:.2f})',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(color_hint)
    axes[2].set_title('Color Hint (Input to Model)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # 保存结果
    output_filename = os.path.basename(image_path).rsplit('.', 1)[0]
    output_path = os.path.join(output_dir, f'{output_filename}_{fade_type}_intensity{intensity:.2f}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved result to: {output_path}")

    # 同时保存单独的褪色图片
    faded_only_path = os.path.join(output_dir, f'{output_filename}_faded.png')
    Image.fromarray(faded_image).save(faded_only_path)
    print(f"Saved faded image to: {faded_only_path}")

    plt.close()

    return image, faded_image, color_hint


def test_all_fade_types(image_path, intensity=0.5, output_dir='fading_test_results'):
    """
    测试所有褪色类型

    Args:
        image_path: 图片路径
        intensity: 褪色强度
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图片
    print(f"Reading image: {image_path}")
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)

    print(f"Image shape: {image.shape}")

    # 所有褪色类型
    fade_types = ['uniform', 'saturation', 'brightness', 'yellow', 'sepia']

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 各种褪色效果
    for i, fade_type in enumerate(fade_types):
        print(f"Applying {fade_type} fading...")
        faded = simulate_color_fading(image, fade_type=fade_type, intensity=intensity)
        axes[i + 1].imshow(faded)
        axes[i + 1].set_title(f'{fade_type.capitalize()}\n(intensity={intensity:.2f})',
                             fontsize=12, fontweight='bold')
        axes[i + 1].axis('off')

    plt.tight_layout()

    # 保存对比图
    output_filename = os.path.basename(image_path).rsplit('.', 1)[0]
    output_path = os.path.join(output_dir, f'{output_filename}_all_types_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {output_path}")

    plt.close()


def test_multiple_intensities(image_path, fade_type='mixed', output_dir='fading_test_results'):
    """
    测试不同褪色强度

    Args:
        image_path: 图片路径
        fade_type: 褪色类型
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图片
    print(f"Reading image: {image_path}")
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)

    print(f"Image shape: {image.shape}")

    # 不同强度
    intensities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, intensity in enumerate(intensities):
        print(f"Applying fading with intensity={intensity:.2f}...")
        if intensity == 0.0:
            faded = image  # 无褪色
        else:
            faded = simulate_color_fading(image, fade_type=fade_type, intensity=intensity)

        axes[i].imshow(faded)
        axes[i].set_title(f'Intensity = {intensity:.1f}', fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle(f'Fade Type: {fade_type.capitalize()}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存对比图
    output_filename = os.path.basename(image_path).rsplit('.', 1)[0]
    output_path = os.path.join(output_dir, f'{output_filename}_{fade_type}_intensity_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved intensity comparison to: {output_path}")

    plt.close()


def test_combined_effects(image_path, intensity=0.5, output_dir='fading_test_results'):
    """
    测试组合褪色效果

    Args:
        image_path: 图片路径
        intensity: 褪色强度
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图片
    print(f"Reading image: {image_path}")
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)

    print(f"Image shape: {image.shape}")

    # 不同的组合配置
    combinations = [
        ('Original', None, None),
        ('Brightness + Yellow', 'combined',
         {'brightness': intensity * 0.5, 'yellow': intensity * 0.6}),
        ('Yellow + Sepia', 'combined',
         {'yellow': intensity * 0.6, 'sepia': intensity * 0.4}),
        ('Brightness + Sepia', 'combined',
         {'brightness': intensity * 0.4, 'sepia': intensity * 0.5}),
        ('All Three (B+Y+S)', 'combined',
         {'brightness': intensity * 0.4, 'yellow': intensity * 0.6, 'sepia': intensity * 0.3}),
        ('Complex Aging', 'combined',
         {'brightness': intensity * 0.3, 'saturation': intensity * 0.4, 'yellow': intensity * 0.5, 'sepia': intensity * 0.2}),
    ]

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (title, fade_type, effects) in enumerate(combinations):
        print(f"Processing: {title}...")
        if fade_type is None:
            # 原图
            result = image
        else:
            result = simulate_color_fading(image, fade_type=fade_type,
                                          intensity=intensity,
                                          combined_effects=effects)

        axes[i].imshow(result)
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle(f'Combined Fading Effects (Base Intensity={intensity:.2f})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存对比图
    output_filename = os.path.basename(image_path).rsplit('.', 1)[0]
    output_path = os.path.join(output_dir, f'{output_filename}_combined_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined effects comparison to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test color fading effects on your images')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--fade_type', type=str, default='mixed',
                       choices=['uniform', 'saturation', 'brightness', 'yellow', 'sepia', 'mixed', 'combined'],
                       help='Type of fading effect')
    parser.add_argument('--intensity', type=float, default=0.5,
                       help='Fading intensity (0.0-1.0)')
    parser.add_argument('--output_dir', type=str, default='fading_test_results',
                       help='Output directory for results')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'all_types', 'all_intensities', 'combined'],
                       help='Test mode: single, all_types, all_intensities, or combined')
    parser.add_argument('--effects', type=str, default=None,
                       help='Combined effects in JSON format, e.g., \'{"brightness":0.3,"yellow":0.5}\'\')

    args = parser.parse_args()

    # 检查输入图片
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    print("=" * 60)
    print("Color Fading Test")
    print("=" * 60)

    # 解析组合效果参数
    combined_effects = None
    if args.effects:
        import json
        try:
            combined_effects = json.loads(args.effects)
            print(f"Using custom combined effects: {combined_effects}")
        except json.JSONDecodeError as e:
            print(f"Error parsing effects JSON: {e}")
            print("Using default combined effects")

    if args.mode == 'single':
        # 测试单个配置
        test_single_image(args.image, args.fade_type, args.intensity, args.output_dir,
                         combined_effects=combined_effects)

    elif args.mode == 'all_types':
        # 测试所有褪色类型
        test_all_fade_types(args.image, args.intensity, args.output_dir)

    elif args.mode == 'all_intensities':
        # 测试所有强度
        test_multiple_intensities(args.image, args.fade_type, args.output_dir)

    elif args.mode == 'combined':
        # 测试各种组合效果
        test_combined_effects(args.image, args.intensity, args.output_dir)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
