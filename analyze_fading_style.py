"""
褪色照片特征分析工具
从真实褪色照片中学习褪色规律，自动推荐模拟参数

使用方法:
    # 分析单个文件夹中的图片
    python analyze_fading_style.py --folder /path/to/faded_photos

    # 分析多个图片
    python analyze_fading_style.py --images img1.jpg img2.jpg img3.jpg

    # 生成详细报告和可视化
    python analyze_fading_style.py --folder /path/to/faded_photos --detailed

输出:
    1. 颜色特征统计报告（JSON格式）
    2. 推荐的褪色参数配置
    3. 可视化图表（可选）
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class FadingStyleAnalyzer:
    """褪色风格分析器"""

    def __init__(self):
        self.features = {
            'saturation': [],      # 饱和度
            'hue': [],             # 色调
            'brightness': [],      # 亮度（V通道）
            'rgb_ratio': [],       # RGB通道比例
            'yellow_shift': [],    # 泛黄程度
            'sepia_score': [],     # 棕褐色调分数
            'contrast': [],        # 对比度
        }

    def analyze_image(self, image_path):
        """分析单张图片的褪色特征"""
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. 分析饱和度
        saturation = image_hsv[:, :, 1].mean() / 255.0
        self.features['saturation'].append(saturation)

        # 2. 分析色调（加权平均，忽略低饱和度像素）
        mask = image_hsv[:, :, 1] > 30  # 只考虑有颜色的像素
        if mask.sum() > 0:
            hue = image_hsv[:, :, 0][mask].mean() / 180.0  # 归一化到0-1
            self.features['hue'].append(hue)

        # 3. 分析亮度
        brightness = image_hsv[:, :, 2].mean() / 255.0
        self.features['brightness'].append(brightness)

        # 4. RGB通道比例分析
        r_mean = image_rgb[:, :, 0].mean()
        g_mean = image_rgb[:, :, 1].mean()
        b_mean = image_rgb[:, :, 2].mean()
        total = r_mean + g_mean + b_mean
        if total > 0:
            rgb_ratio = [r_mean/total, g_mean/total, b_mean/total]
            self.features['rgb_ratio'].append(rgb_ratio)

        # 5. 泛黄程度（红色和绿色通道高于蓝色）
        yellow_shift = (r_mean + g_mean) / 2 - b_mean
        yellow_shift = max(0, yellow_shift) / 255.0  # 归一化
        self.features['yellow_shift'].append(yellow_shift)

        # 6. 棕褐色调分数（接近棕褐色的程度）
        # 标准棕褐色的RGB比例约为 [0.44, 0.39, 0.17]
        if total > 0:
            sepia_target = np.array([0.44, 0.39, 0.17])
            sepia_current = np.array(rgb_ratio)
            sepia_score = 1.0 - np.linalg.norm(sepia_current - sepia_target)
            sepia_score = max(0, sepia_score)
            self.features['sepia_score'].append(sepia_score)

        # 7. 对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std() / 128.0  # 归一化
        self.features['contrast'].append(contrast)

        return {
            'saturation': saturation,
            'hue': hue if mask.sum() > 0 else None,
            'brightness': brightness,
            'rgb_ratio': rgb_ratio if total > 0 else None,
            'yellow_shift': yellow_shift,
            'sepia_score': sepia_score if total > 0 else 0,
            'contrast': contrast,
        }

    def analyze_folder(self, folder_path, extensions=['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
        """分析文件夹中的所有图片"""
        folder = Path(folder_path)
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))

        if not image_files:
            print(f"No images found in {folder_path}")
            return

        print(f"Found {len(image_files)} images in {folder_path}")
        print("Analyzing images...")

        for img_path in tqdm(image_files):
            self.analyze_image(img_path)

    def analyze_images(self, image_paths):
        """分析指定的多张图片"""
        print(f"Analyzing {len(image_paths)} images...")
        for img_path in tqdm(image_paths):
            self.analyze_image(img_path)

    def get_statistics(self):
        """获取统计结果"""
        stats = {}

        for feature_name, values in self.features.items():
            if not values:
                continue

            if feature_name == 'rgb_ratio':
                # RGB比例特殊处理
                rgb_array = np.array(values)
                stats[feature_name] = {
                    'mean': rgb_array.mean(axis=0).tolist(),
                    'std': rgb_array.std(axis=0).tolist(),
                    'median': np.median(rgb_array, axis=0).tolist(),
                }
            else:
                stats[feature_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        return stats

    def recommend_parameters(self, stats):
        """根据统计结果推荐褪色参数"""
        recommendations = {
            'fade_type': 'combined',
            'intensity_range': [0.4, 0.7],  # 默认
            'combined_effects': {},
            'explanation': {}
        }

        # 1. 根据饱和度推荐 saturation 参数
        if 'saturation' in stats:
            sat_mean = stats['saturation']['mean']
            # 饱和度越低，说明褪色越严重
            if sat_mean < 0.3:
                # 严重褪色
                recommendations['combined_effects']['saturation'] = 0.8
                recommendations['intensity_range'] = [0.6, 0.8]
                recommendations['explanation']['saturation'] = \
                    f"检测到极低饱和度({sat_mean:.3f})，推荐强力降低饱和度(0.8)"
            elif sat_mean < 0.5:
                # 中度褪色
                recommendations['combined_effects']['saturation'] = 0.6
                recommendations['intensity_range'] = [0.5, 0.7]
                recommendations['explanation']['saturation'] = \
                    f"检测到中等饱和度({sat_mean:.3f})，推荐中度降低饱和度(0.6)"
            else:
                # 轻微褪色或无需降低饱和度
                recommendations['combined_effects']['saturation'] = 0.3
                recommendations['intensity_range'] = [0.3, 0.5]
                recommendations['explanation']['saturation'] = \
                    f"检测到较高饱和度({sat_mean:.3f})，推荐轻度降低饱和度(0.3)"

        # 2. 根据泛黄程度推荐 yellow 参数
        if 'yellow_shift' in stats:
            yellow_mean = stats['yellow_shift']['mean']
            if yellow_mean > 0.15:
                # 明显泛黄
                recommendations['combined_effects']['yellow'] = 1.0
                recommendations['explanation']['yellow'] = \
                    f"检测到明显泛黄({yellow_mean:.3f})，推荐强力泛黄效果(1.0)"
            elif yellow_mean > 0.08:
                # 中度泛黄
                recommendations['combined_effects']['yellow'] = 0.7
                recommendations['explanation']['yellow'] = \
                    f"检测到中度泛黄({yellow_mean:.3f})，推荐中度泛黄效果(0.7)"
            elif yellow_mean > 0.03:
                # 轻微泛黄
                recommendations['combined_effects']['yellow'] = 0.4
                recommendations['explanation']['yellow'] = \
                    f"检测到轻微泛黄({yellow_mean:.3f})，推荐轻度泛黄效果(0.4)"

        # 3. 根据棕褐色分数推荐 sepia 参数
        if 'sepia_score' in stats:
            sepia_mean = stats['sepia_score']['mean']
            if sepia_mean > 0.7:
                # 明显棕褐色调
                recommendations['combined_effects']['sepia'] = 0.8
                recommendations['explanation']['sepia'] = \
                    f"检测到明显棕褐色调({sepia_mean:.3f})，推荐强力棕褐色效果(0.8)"
            elif sepia_mean > 0.5:
                # 中度棕褐色调
                recommendations['combined_effects']['sepia'] = 0.5
                recommendations['explanation']['sepia'] = \
                    f"检测到中度棕褐色调({sepia_mean:.3f})，推荐中度棕褐色效果(0.5)"
            elif sepia_mean > 0.3:
                # 轻微棕褐色调
                recommendations['combined_effects']['sepia'] = 0.3
                recommendations['explanation']['sepia'] = \
                    f"检测到轻微棕褐色调({sepia_mean:.3f})，推荐轻度棕褐色效果(0.3)"

        # 4. 根据亮度推荐 brightness 参数
        if 'brightness' in stats:
            bright_mean = stats['brightness']['mean']
            if bright_mean < 0.4:
                # 图片偏暗
                recommendations['combined_effects']['brightness'] = 0.6
                recommendations['explanation']['brightness'] = \
                    f"检测到偏暗图片({bright_mean:.3f})，推荐降低亮度(0.6)"
            elif bright_mean < 0.6:
                # 正常亮度
                recommendations['combined_effects']['brightness'] = 0.4
                recommendations['explanation']['brightness'] = \
                    f"检测到正常亮度({bright_mean:.3f})，推荐轻度降低亮度(0.4)"
            else:
                # 图片偏亮，可能不需要降低亮度
                pass

        # 5. 如果没有检测到明显特征，使用默认配置
        if not recommendations['combined_effects']:
            recommendations['combined_effects'] = {
                'brightness': 0.4,
                'saturation': 0.5,
                'yellow': 0.6,
            }
            recommendations['explanation']['default'] = \
                "未检测到明显特征，使用默认配置"

        return recommendations

    def generate_report(self, output_file='fading_analysis_report.json'):
        """生成分析报告"""
        stats = self.get_statistics()
        recommendations = self.recommend_parameters(stats)

        report = {
            'statistics': stats,
            'recommendations': recommendations,
            'sample_count': len(self.features['saturation']),
        }

        # 保存JSON报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("分析报告")
        print(f"{'='*60}")
        print(f"分析图片数量: {report['sample_count']}")
        print()

        print("颜色特征统计:")
        for feature_name, feature_stats in stats.items():
            print(f"\n  {feature_name}:")
            if feature_name == 'rgb_ratio':
                print(f"    平均RGB比例: R={feature_stats['mean'][0]:.3f}, "
                      f"G={feature_stats['mean'][1]:.3f}, B={feature_stats['mean'][2]:.3f}")
            else:
                print(f"    平均值: {feature_stats['mean']:.3f}")
                print(f"    标准差: {feature_stats['std']:.3f}")
                print(f"    范围: [{feature_stats['min']:.3f}, {feature_stats['max']:.3f}]")

        print(f"\n{'-'*60}")
        print("推荐的褪色参数配置:")
        print(f"{'-'*60}")
        print(f"fade_type: {recommendations['fade_type']}")
        print(f"intensity_range: {recommendations['intensity_range']}")
        print(f"combined_effects:")
        for effect, value in recommendations['combined_effects'].items():
            print(f"  {effect}: {value}")
            if effect in recommendations['explanation']:
                print(f"    → {recommendations['explanation'][effect]}")

        print(f"\n{'-'*60}")
        print("使用示例:")
        print(f"{'-'*60}")
        effects_str = json.dumps(recommendations['combined_effects'])
        intensity = sum(recommendations['intensity_range']) / 2
        print(f"python test_fading.py \\")
        print(f"    --image your_photo.jpg \\")
        print(f"    --fade_type combined \\")
        print(f"    --intensity {intensity:.2f} \\")
        print(f"    --effects '{effects_str}'")

        print(f"\n报告已保存至: {output_file}")
        print(f"{'='*60}\n")

        return report

    def visualize(self, output_file='fading_analysis_visualization.png'):
        """生成可视化图表"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        plot_features = [
            ('saturation', '饱和度分布'),
            ('brightness', '亮度分布'),
            ('yellow_shift', '泛黄程度分布'),
            ('sepia_score', '棕褐色调分数分布'),
            ('contrast', '对比度分布'),
        ]

        for i, (feature_name, title) in enumerate(plot_features):
            if feature_name in self.features and self.features[feature_name]:
                values = self.features[feature_name]
                axes[i].hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                axes[i].axvline(np.mean(values), color='red', linestyle='--',
                               linewidth=2, label=f'均值: {np.mean(values):.3f}')
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].set_xlabel('值', fontsize=10)
                axes[i].set_ylabel('频数', fontsize=10)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        # RGB比例饼图
        if 'rgb_ratio' in self.features and self.features['rgb_ratio']:
            rgb_mean = np.array(self.features['rgb_ratio']).mean(axis=0)
            axes[5].pie(rgb_mean, labels=['Red', 'Green', 'Blue'],
                       autopct='%1.1f%%', startangle=90,
                       colors=['#ff6b6b', '#51cf66', '#4dabf7'])
            axes[5].set_title('平均RGB通道比例', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"可视化图表已保存至: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='分析褪色照片特征，自动推荐模拟参数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析文件夹
  python analyze_fading_style.py --folder /path/to/faded_photos

  # 分析指定图片
  python analyze_fading_style.py --images img1.jpg img2.jpg img3.jpg

  # 生成详细可视化
  python analyze_fading_style.py --folder /path/to/faded_photos --visualize
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', type=str, help='褪色照片所在文件夹')
    group.add_argument('--images', nargs='+', type=str, help='指定要分析的图片路径')

    parser.add_argument('--output', type=str, default='fading_analysis_report.json',
                       help='输出报告文件名 (默认: fading_analysis_report.json)')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    parser.add_argument('--viz-output', type=str, default='fading_analysis_visualization.png',
                       help='可视化图表输出文件名')

    args = parser.parse_args()

    # 创建分析器
    analyzer = FadingStyleAnalyzer()

    # 分析图片
    if args.folder:
        analyzer.analyze_folder(args.folder)
    else:
        analyzer.analyze_images(args.images)

    # 生成报告
    report = analyzer.generate_report(args.output)

    # 生成可视化（可选）
    if args.visualize:
        analyzer.visualize(args.viz_output)

    return report


if __name__ == "__main__":
    main()
