"""
验证褪色参数效果
使用推荐的参数对测试图片应用褪色，与真实褪色照片对比

使用方法:
    # 使用分析报告中的推荐参数
    python validate_fading_params.py \\
        --report fading_analysis_report.json \\
        --test_images img1.jpg img2.jpg \\
        --reference_faded ref1.jpg ref2.jpg

    # 手动指定参数进行验证
    python validate_fading_params.py \\
        --test_images img1.jpg img2.jpg \\
        --reference_faded ref1.jpg ref2.jpg \\
        --intensity 0.6 \\
        --effects '{"brightness":0.4,"yellow":0.8,"sepia":0.5}'
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

sys.path.insert(0, os.path.dirname(__file__))
from ldm.data.color_restoration import simulate_color_fading


def compare_color_features(image1, image2):
    """比较两张图片的颜色特征相似度"""
    # 转换到HSV
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)

    # 计算特征
    features1 = {
        'saturation': hsv1[:, :, 1].mean() / 255.0,
        'hue': hsv1[:, :, 0].mean() / 180.0,
        'brightness': hsv1[:, :, 2].mean() / 255.0,
    }

    features2 = {
        'saturation': hsv2[:, :, 1].mean() / 255.0,
        'hue': hsv2[:, :, 0].mean() / 180.0,
        'brightness': hsv2[:, :, 2].mean() / 255.0,
    }

    # RGB通道比例
    rgb1 = image1.astype(np.float32)
    rgb2 = image2.astype(np.float32)

    r1, g1, b1 = rgb1[:,:,0].mean(), rgb1[:,:,1].mean(), rgb1[:,:,2].mean()
    r2, g2, b2 = rgb2[:,:,0].mean(), rgb2[:,:,1].mean(), rgb2[:,:,2].mean()

    total1 = r1 + g1 + b1
    total2 = r2 + g2 + b2

    if total1 > 0 and total2 > 0:
        ratio1 = np.array([r1/total1, g1/total1, b1/total1])
        ratio2 = np.array([r2/total2, g2/total2, b2/total2])
        rgb_similarity = 1.0 - np.linalg.norm(ratio1 - ratio2)
    else:
        rgb_similarity = 0.0

    # 计算相似度分数
    similarity_scores = {
        'saturation': 1.0 - abs(features1['saturation'] - features2['saturation']),
        'hue': 1.0 - abs(features1['hue'] - features2['hue']),
        'brightness': 1.0 - abs(features1['brightness'] - features2['brightness']),
        'rgb_ratio': max(0, rgb_similarity),
    }

    # 总体相似度（加权平均）
    overall_similarity = (
        similarity_scores['saturation'] * 0.35 +
        similarity_scores['hue'] * 0.15 +
        similarity_scores['brightness'] * 0.20 +
        similarity_scores['rgb_ratio'] * 0.30
    )

    return similarity_scores, overall_similarity, features1, features2


def validate_parameters(test_images, reference_faded, intensity, combined_effects,
                       output_dir='validation_results'):
    """验证参数效果"""

    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"\n{'='*60}")
    print("验证褪色参数效果")
    print(f"{'='*60}")
    print(f"参数配置:")
    print(f"  intensity: {intensity}")
    print(f"  combined_effects: {json.dumps(combined_effects, indent=4)}")
    print(f"{'='*60}\n")

    # 对每张测试图片应用褪色
    for i, test_img_path in enumerate(test_images):
        print(f"处理测试图片 {i+1}/{len(test_images)}: {test_img_path}")

        # 读取测试图片
        test_img = Image.open(test_img_path).convert('RGB')
        test_img_np = np.array(test_img)

        # 应用褪色
        faded_img_np = simulate_color_fading(
            test_img_np,
            fade_type='combined',
            intensity=intensity,
            combined_effects=combined_effects
        )

        # 与参考褪色照片对比
        best_similarity = 0
        best_ref = None
        best_ref_img = None

        for ref_path in reference_faded:
            ref_img = Image.open(ref_path).convert('RGB')
            ref_img_np = np.array(ref_img)

            # 比较特征
            sim_scores, overall_sim, feat1, feat2 = compare_color_features(
                faded_img_np, ref_img_np
            )

            if overall_sim > best_similarity:
                best_similarity = overall_sim
                best_ref = ref_path
                best_ref_img = ref_img_np
                best_sim_scores = sim_scores
                best_feat1 = feat1
                best_feat2 = feat2

        # 保存对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(test_img_np)
        axes[0].set_title('原始测试图片', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(faded_img_np)
        axes[1].set_title(f'模拟褪色效果\n相似度: {best_similarity:.2%}',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(best_ref_img)
        axes[2].set_title(f'参考褪色照片\n{Path(best_ref).name}',
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        # 保存
        output_filename = Path(test_img_path).stem + '_validation.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  → 与最相似的参考图片: {Path(best_ref).name}")
        print(f"  → 总体相似度: {best_similarity:.2%}")
        print(f"  → 各项指标相似度:")
        for metric, score in best_sim_scores.items():
            print(f"      {metric}: {score:.2%}")
        print(f"  → 对比图已保存: {output_path}\n")

        results.append({
            'test_image': str(test_img_path),
            'best_reference': str(best_ref),
            'overall_similarity': float(best_similarity),
            'similarity_scores': {k: float(v) for k, v in best_sim_scores.items()},
            'simulated_features': best_feat1,
            'reference_features': best_feat2,
        })

    # 生成总结报告
    avg_similarity = np.mean([r['overall_similarity'] for r in results])

    print(f"\n{'='*60}")
    print("验证总结")
    print(f"{'='*60}")
    print(f"测试图片数量: {len(test_images)}")
    print(f"平均相似度: {avg_similarity:.2%}")
    print()

    if avg_similarity > 0.85:
        print("✅ 参数效果很好！模拟的褪色效果与真实照片非常相似")
    elif avg_similarity > 0.70:
        print("✓ 参数效果不错，模拟的褪色效果与真实照片较为相似")
    elif avg_similarity > 0.55:
        print("⚠ 参数效果一般，建议调整参数以获得更好的效果")
    else:
        print("✗ 参数效果不佳，建议重新分析或手动调整参数")

    print(f"\n所有对比图已保存至: {output_dir}/")
    print(f"{'='*60}\n")

    # 保存JSON报告
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'parameters': {
                'intensity': intensity,
                'combined_effects': combined_effects,
            },
            'average_similarity': float(avg_similarity),
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    print(f"详细报告已保存至: {report_path}\n")

    return results, avg_similarity


def main():
    parser = argparse.ArgumentParser(
        description='验证褪色参数效果',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--test_images', nargs='+', required=True,
                       help='用于测试的原始图片')
    parser.add_argument('--reference_faded', nargs='+', required=True,
                       help='参考的真实褪色照片')

    # 参数来源：报告或手动指定
    parser.add_argument('--report', type=str,
                       help='使用分析报告中的推荐参数')
    parser.add_argument('--intensity', type=float,
                       help='手动指定intensity（如果不使用报告）')
    parser.add_argument('--effects', type=str,
                       help='手动指定combined_effects（JSON格式）')

    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='输出目录')

    args = parser.parse_args()

    # 确定参数来源
    if args.report:
        # 从报告中读取
        print(f"从报告中读取参数: {args.report}")
        with open(args.report, 'r', encoding='utf-8') as f:
            report = json.load(f)

        recommendations = report['recommendations']
        intensity = sum(recommendations['intensity_range']) / 2
        combined_effects = recommendations['combined_effects']

    elif args.intensity is not None and args.effects is not None:
        # 手动指定
        intensity = args.intensity
        combined_effects = json.loads(args.effects)

    else:
        print("错误: 必须指定 --report 或者同时指定 --intensity 和 --effects")
        return

    # 执行验证
    validate_parameters(
        args.test_images,
        args.reference_faded,
        intensity,
        combined_effects,
        args.output_dir
    )


if __name__ == "__main__":
    main()
