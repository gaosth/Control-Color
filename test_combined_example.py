"""
组合褪色效果演示脚本
展示如何同时应用多种褪色效果

这个脚本演示了如何组合多种褪色效果来模拟真实的老照片效果
"""
import sys
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入褪色函数
sys.path.insert(0, os.path.dirname(__file__))
from ldm.data.color_restoration import simulate_color_fading


def demo_combined_effects(image_path):
    """
    演示组合效果的不同配置
    """
    # 读取图片
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)

    # 定义不同的组合配置
    configs = [
        {
            'name': '原图',
            'fade_type': None,
        },
        {
            'name': '单独：泛黄',
            'fade_type': 'yellow',
            'intensity': 0.6,
        },
        {
            'name': '组合：亮度+泛黄',
            'fade_type': 'combined',
            'intensity': 0.6,
            'effects': {'brightness': 0.3, 'yellow': 0.5}
        },
        {
            'name': '组合：泛黄+棕褐色',
            'fade_type': 'combined',
            'intensity': 0.6,
            'effects': {'yellow': 0.6, 'sepia': 0.4}
        },
        {
            'name': '组合：亮度+泛黄+棕褐色',
            'fade_type': 'combined',
            'intensity': 0.6,
            'effects': {'brightness': 0.4, 'yellow': 0.6, 'sepia': 0.3}
        },
        {
            'name': '组合：复杂老化\n(B+S+Y+Sepia)',
            'fade_type': 'combined',
            'intensity': 0.6,
            'effects': {
                'brightness': 0.3,
                'saturation': 0.4,
                'yellow': 0.5,
                'sepia': 0.2
            }
        },
    ]

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, config in enumerate(configs):
        print(f"\nProcessing: {config['name']}")

        if config['fade_type'] is None:
            # 原图
            result = image
        else:
            # 应用褪色
            if config['fade_type'] == 'combined':
                print(f"  Combining effects: {config['effects']}")
                result = simulate_color_fading(
                    image,
                    fade_type='combined',
                    intensity=config.get('intensity', 0.5),
                    combined_effects=config['effects']
                )
            else:
                result = simulate_color_fading(
                    image,
                    fade_type=config['fade_type'],
                    intensity=config.get('intensity', 0.5)
                )

        axes[i].imshow(result)
        axes[i].set_title(config['name'], fontsize=13, fontweight='bold', pad=10)
        axes[i].axis('off')

    plt.suptitle('组合褪色效果对比 - Combined Fading Effects Demo',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存
    output_dir = 'fading_test_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_effects_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Demo saved to: {output_path}")
    print(f"{'='*60}")

    plt.show()


if __name__ == "__main__":
    # 使用项目中的示例图片
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认使用项目的示例图片
        image_path = './assets/teaser_aligned.png'
        if not os.path.exists(image_path):
            print(f"Error: Default image not found at {image_path}")
            print("Usage: python test_combined_example.py [path/to/your/image.jpg]")
            sys.exit(1)

    print("="*60)
    print("组合褪色效果演示")
    print("Combined Fading Effects Demo")
    print("="*60)

    demo_combined_effects(image_path)
