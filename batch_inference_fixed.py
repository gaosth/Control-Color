"""
批量图片推理脚本（修复版）- Control-Color颜色修复

使用训练时的采样逻辑，修复了维度顺序问题
"""
import os
import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm
import argparse
from pathlib import Path

from cldm.model import create_model, load_state_dict
from ldm.data.color_restoration import simulate_color_fading


def prepare_batch_for_inference(image_rgb, is_faded=True, fade_type='combined', fade_intensity=0.6, target_size=512):
    """
    准备单张图片的batch数据，模仿训练时的数据格式

    Args:
        image_rgb: RGB图像 [H, W, 3], 0-255, numpy array
        is_faded: 是否已经是褪色图
        fade_type: 褪色类型（如果需要模拟）
        fade_intensity: 褪色强度（如果需要模拟）
        target_size: 目标尺寸

    Returns:
        batch: 符合模型输入的batch字典
        faded_image: 褪色图像（用于保存）
        original_image: 原始图像（用于对比）
    """
    from PIL import Image

    # 确保图片是3通道RGB格式
    if len(image_rgb.shape) == 2:
        # 灰度图转RGB
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    elif len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {image_rgb.shape}")

    # Resize到目标尺寸
    pil_image = Image.fromarray(image_rgb.astype(np.uint8))
    pil_image = pil_image.resize((target_size, target_size), Image.LANCZOS)
    resized_image = np.array(pil_image).astype(np.float32)  # [H, W, 3]

    # 归一化到 [-1, 1]
    original_image = (resized_image / 127.5 - 1.0).astype(np.float32)  # [H, W, 3]

    # 生成褪色图像（如果需要）
    if not is_faded:
        # 反归一化到 [0, 255]
        img_uint8 = (original_image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        faded_uint8, intensity = simulate_color_fading(
            img_uint8,
            fade_type=fade_type,
            intensity=fade_intensity,
            return_intensity=True
        )
        # 重新归一化到 [-1, 1]
        faded_image = (faded_uint8.astype(np.float32) / 127.5 - 1.0).astype(np.float32)
    else:
        # 输入已经是褪色图
        faded_image = original_image.copy()
        intensity = fade_intensity

    # 提取颜色hint（从褪色图）
    color_hint = faded_image.copy()  # [H, W, 3]

    # 创建mask（全图修复）
    H, W, C = faded_image.shape
    mask = np.ones((H, W, 1), dtype=np.float32)

    # 转换为torch tensors并添加batch维度
    # 重要：保持 [batch, height, width, channels] 格式
    # get_input() 会自动使用 rearrange 转换为 [batch, channels, height, width]

    jpg_tensor = torch.from_numpy(original_image).unsqueeze(0)  # [H, W, 3] -> [1, H, W, 3]
    hint_tensor = torch.from_numpy(color_hint).unsqueeze(0)  # [H, W, 3] -> [1, H, W, 3]
    mask_img_tensor = torch.from_numpy(faded_image).unsqueeze(0)  # [H, W, 3] -> [1, H, W, 3]
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [H, W, 1] -> [1, H, W, 1]

    # 构建batch - 所有图像张量都是 [1, H, W, C] 格式！
    batch = {
        'jpg': jpg_tensor.contiguous(),  # [1, H, W, 3]
        'txt': [f"restore faded photo, intensity {intensity:.2f}"],
        'hint': hint_tensor.contiguous(),  # [1, H, W, 3]
        'mask_img': mask_img_tensor.contiguous(),  # [1, H, W, 3]
        'mask': mask_tensor.contiguous(),  # [1, H, W, 1]
        'faded': mask_img_tensor.contiguous(),  # [1, H, W, 3]
    }

    return batch, faded_image, original_image


@torch.no_grad()
def inference_single_image(model, batch, ddim_steps=20, guidance_scale=9.0, eta=0.0):
    """
    使用模型的log_images方法进行推理
    """
    # 将batch移到GPU
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_gpu[k] = v.cuda()
        else:
            batch_gpu[k] = v

    # 调用模型的log_images方法
    images = model.log_images(
        batch_gpu,
        N=1,
        sample=True,
        ddim_steps=ddim_steps,
        ddim_eta=eta,
        unconditional_guidance_scale=guidance_scale,
        plot_diffusion_rows=False,
        plot_progressive_rows=False,
        plot_denoise_rows=False
    )

    # 提取采样结果
    key_name = f"samples_cfg_scale_{guidance_scale:.2f}"
    if key_name in images:
        x_samples = images[key_name]
    elif "samples" in images:
        x_samples = images["samples"]
    else:
        raise ValueError(f"No samples found in model output. Available keys: {images.keys()}")

    # 转换为numpy数组 [0, 255]
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()  # [1, 3, H, W] -> [1, H, W, 3]
    x_samples = (x_samples * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    return x_samples[0]  # [H, W, 3]


def get_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
    """获取目录中所有图片文件"""
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description="批量图片颜色修复（修复版）")

    # 必需参数
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出文件夹")
    parser.add_argument("--ckpt", type=str, required=True, help="模型checkpoint路径")

    # 模型配置
    parser.add_argument("--config", type=str, default="./models/cldm_v15_inpainting_infer1.yaml", help="模型配置")

    # 褪色模拟
    parser.add_argument("--is_faded", action="store_true", help="输入已是褪色图")
    parser.add_argument("--fade_type", type=str, default="combined", help="褪色类型")
    parser.add_argument("--fade_intensity", type=float, default=0.6, help="褪色强度")

    # 推理参数
    parser.add_argument("--steps", type=int, default=20, help="DDIM采样步数")
    parser.add_argument("--resolution", type=int, default=512, help="推理分辨率")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="引导强度")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 其他选项
    parser.add_argument("--save_comparison", action="store_true", help="保存对比图")
    parser.add_argument("--save_faded", action="store_true", help="保存褪色图")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在文件")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_comparison:
        comparison_dir = output_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)

    if args.save_faded and not args.is_faded:
        faded_dir = output_dir / "faded"
        faded_dir.mkdir(exist_ok=True)

    # 检查checkpoint
    if not os.path.exists(args.ckpt):
        print(f"❌ Checkpoint not found: {args.ckpt}")
        return

    # 设置随机种子
    seed_everything(args.seed)

    # 加载模型
    print("=" * 80)
    print("🔧 Loading model...")
    print(f"   Checkpoint: {args.ckpt}")
    print("=" * 80)

    try:
        model = create_model(args.config).cpu()
        model.load_state_dict(load_state_dict(args.ckpt, location='cuda'), strict=False)
        model = model.cuda()
        model.eval()
        print("✅ Model loaded!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 获取图片列表
    image_files = get_image_files(args.input_dir)
    if len(image_files) == 0:
        print(f"❌ No images found in {args.input_dir}")
        return

    print(f"📁 Found {len(image_files)} images\n")
    print("=" * 80)
    print("🚀 Starting batch inference...")
    print("=" * 80)

    # 批量处理
    success_count = 0
    skip_count = 0
    error_count = 0

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            output_path = output_dir / img_path.name
            if args.skip_existing and output_path.exists():
                skip_count += 1
                continue

            # 读取图片
            input_image = cv2.imread(str(img_path))
            if input_image is None:
                print(f"\n⚠️  Failed to read {img_path.name}")
                error_count += 1
                continue

            # 转换为RGB
            if len(input_image.shape) == 2:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
            elif input_image.shape[2] == 4:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGB)
            else:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # 准备batch
            batch, faded_image, original_image = prepare_batch_for_inference(
                input_image,
                is_faded=args.is_faded,
                fade_type=args.fade_type,
                fade_intensity=args.fade_intensity,
                target_size=args.resolution
            )

            # 推理
            restored_image = inference_single_image(
                model,
                batch,
                ddim_steps=args.steps,
                guidance_scale=args.guidance_scale,
                eta=args.eta
            )

            # 保存结果
            cv2.imwrite(str(output_path), cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))

            # 保存褪色图
            if args.save_faded and not args.is_faded:
                faded_uint8 = (faded_image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(faded_dir / img_path.name), cv2.cvtColor(faded_uint8, cv2.COLOR_RGB2BGR))

            # 保存对比图
            if args.save_comparison and not args.is_faded:
                original_uint8 = (original_image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                faded_uint8 = (faded_image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

                comparison = np.hstack([
                    cv2.resize(original_uint8, (512, 512)),
                    cv2.resize(faded_uint8, (512, 512)),
                    cv2.resize(restored_image, (512, 512))
                ])

                font = cv2.FONT_HERSHEY_SIMPLEX
                comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                cv2.putText(comparison_bgr, "Original", (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(comparison_bgr, "Faded", (522, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(comparison_bgr, "Restored", (1034, 30), font, 1, (255, 255, 255), 2)

                cv2.imwrite(str(comparison_dir / img_path.name), comparison_bgr)

            success_count += 1

        except Exception as e:
            print(f"\n❌ Error: {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # 总结
    print("\n" + "=" * 80)
    print("🎉 Completed!")
    print("=" * 80)
    print(f"✅ Success: {success_count}/{len(image_files)}")
    if skip_count > 0:
        print(f"⏭️  Skipped: {skip_count}")
    if error_count > 0:
        print(f"❌ Failed: {error_count}")
    print(f"\n📁 Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
