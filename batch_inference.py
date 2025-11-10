"""
批量图片推理脚本 - Control-Color颜色修复

使用训练好的checkpoint批量修复褪色照片的颜色
"""
import os
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

from cldm.model import create_model, load_state_dict
from cldm.ddim_haced_sag_step import DDIMSampler
from annotator.util import resize_image
from ldm.data.color_restoration import simulate_color_fading, extract_color_mask


def prepare_mask_and_masked_image(image, mask):
    """准备mask和masked image"""
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def restore_faded_image(
    model,
    ddim_sampler,
    faded_image,
    color_hint=None,
    prompt="",
    a_prompt="high quality, detailed, vibrant colors",
    n_prompt="low quality, blurry, grayscale",
    num_samples=1,
    image_resolution=512,
    ddim_steps=20,
    strength=1.0,
    scale=9.0,
    sag_scale=0.75,
    seed=42,
    eta=0.0,
    use_full_restoration=True
):
    """
    修复褪色图片

    Args:
        model: 加载的模型
        ddim_sampler: DDIM采样器
        faded_image: 褪色图片 [H, W, 3], RGB, 0-255
        color_hint: 颜色提示（如果为None，从faded_image提取）
        prompt: 文本提示
        use_full_restoration: True=全图修复，False=局部修复

    Returns:
        restored_images: 修复后的图片列表
    """
    torch.cuda.empty_cache()

    with torch.no_grad():
        # 1. 准备颜色hint
        if color_hint is None:
            color_hint = extract_color_mask(faded_image, mask_type='faded_color')

        H_ori, W_ori, C_ori = faded_image.shape

        # 2. Resize
        faded_resized = resize_image(faded_image, image_resolution)
        hint_resized = resize_image(color_hint, image_resolution)
        H, W, C = faded_resized.shape

        # 3. 准备mask
        if use_full_restoration:
            # 全图修复
            mask = np.ones((H, W, 1), dtype=np.float32)
        else:
            # 可以自定义局部修复mask
            mask = np.ones((H, W, 1), dtype=np.float32)

        # 4. 准备masked image
        mask_tensor, masked_image = prepare_mask_and_masked_image(
            Image.fromarray(hint_resized),
            Image.fromarray((mask * 255).astype(np.uint8))
        )

        # 5. Encode mask
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=(mask_tensor.shape[2] // 8, mask_tensor.shape[3] // 8)
        )
        mask_tensor = mask_tensor.to(device="cuda")
        masked_image_latents = model.get_first_stage_encoding(
            model.encode_first_stage(masked_image.cuda())
        ).detach()

        # 6. 准备ControlNet输入（颜色hint）
        control = torch.from_numpy(hint_resized.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # 7. 设置随机种子
        if seed == -1:
            seed = np.random.randint(0, 65535)
        seed_everything(seed)

        # 8. 准备条件
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }

        shape = (4, H // 8, W // 8)

        # 9. 采样
        model.control_scales = [strength] * 13

        samples, intermediates = ddim_sampler.sample(
            model,
            ddim_steps,
            num_samples,
            shape,
            cond,
            mask=mask_tensor,
            masked_image_latents=masked_image_latents,
            verbose=False,
            eta=eta,
            x_T=None,
            unconditional_guidance_scale=scale,
            sag_scale=sag_scale,
            SAG_influence_step=600,
            noise=None,
            unconditional_conditioning=un_cond
        )

        # 10. 解码
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # 11. Resize回原始尺寸
        results = [x_samples[i] for i in range(num_samples)]
        results = [cv2.resize(i, (W_ori, H_ori), interpolation=cv2.INTER_LANCZOS4) for i in results]

        return results


def get_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
    """获取目录中所有图片文件"""
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description="批量图片颜色修复")

    # 必需参数
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出文件夹路径")
    parser.add_argument("--ckpt", type=str, required=True, help="模型checkpoint路径（例如：logs_restoration/.../checkpoints/last.ckpt）")

    # 模型配置
    parser.add_argument("--config", type=str, default="./models/cldm_v15_inpainting_infer1.yaml", help="模型配置文件")

    # 褪色模拟（如果输入是原始图片）
    parser.add_argument("--is_faded", action="store_true", help="输入图片已经是褪色图片（不需要模拟褪色）")
    parser.add_argument("--fade_type", type=str, default="combined", help="褪色类型（如果输入是原始图）")
    parser.add_argument("--fade_intensity", type=float, default=0.6, help="褪色强度（如果输入是原始图）")

    # 推理参数
    parser.add_argument("--steps", type=int, default=20, help="DDIM采样步数")
    parser.add_argument("--resolution", type=int, default=512, help="推理分辨率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="分类器引导强度")
    parser.add_argument("--sag_scale", type=float, default=0.75, help="SAG引导强度")
    parser.add_argument("--strength", type=float, default=1.0, help="ControlNet强度")

    # 其他选项
    parser.add_argument("--save_comparison", action="store_true", help="保存对比图（原图|褪色|修复）")
    parser.add_argument("--save_faded", action="store_true", help="保存褪色图（如果模拟褪色）")
    parser.add_argument("--prompt", type=str, default="restore photo colors", help="文本提示词")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在的输出文件")

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

    # 检查checkpoint是否存在
    if not os.path.exists(args.ckpt):
        print(f"❌ Error: Checkpoint not found at {args.ckpt}")
        print("\n请确保checkpoint路径正确，例如：")
        print("  --ckpt logs_restoration/2025-11-07T19-52-10/checkpoints/last.ckpt")
        return

    # 加载模型
    print("=" * 80)
    print("🔧 Loading model...")
    print(f"   Config: {args.config}")
    print(f"   Checkpoint: {args.ckpt}")
    print("=" * 80)

    try:
        model = create_model(args.config).cpu()
        model.load_state_dict(load_state_dict(args.ckpt, location='cuda'), strict=False)
        model = model.cuda()
        model.eval()
        ddim_sampler = DDIMSampler(model)
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 获取所有图片文件
    image_files = get_image_files(args.input_dir)

    if len(image_files) == 0:
        print(f"❌ No image files found in {args.input_dir}")
        return

    print(f"📁 Found {len(image_files)} images in {args.input_dir}\n")
    print("=" * 80)
    print("🚀 Starting batch inference...")
    print("=" * 80)

    # 批量处理
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # 检查是否跳过已存在的文件
            output_path = output_dir / img_path.name
            if args.skip_existing and output_path.exists():
                skip_count += 1
                continue

            # 读取图片
            input_image = cv2.imread(str(img_path))
            if input_image is None:
                print(f"\n⚠️  Warning: Failed to read {img_path.name}, skipping...")
                error_count += 1
                continue

            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # 如果输入不是褪色图，先模拟褪色
            if not args.is_faded:
                faded_image = simulate_color_fading(
                    input_image,
                    args.fade_type,
                    args.fade_intensity
                )
                original_image = input_image

                # 保存褪色图
                if args.save_faded:
                    faded_path = faded_dir / img_path.name
                    cv2.imwrite(
                        str(faded_path),
                        cv2.cvtColor(faded_image, cv2.COLOR_RGB2BGR)
                    )
            else:
                faded_image = input_image
                original_image = None

            # 修复褪色图片
            restored_images = restore_faded_image(
                model,
                ddim_sampler,
                faded_image,
                color_hint=None,
                prompt=args.prompt,
                ddim_steps=args.steps,
                seed=args.seed,
                scale=args.guidance_scale,
                sag_scale=args.sag_scale,
                strength=args.strength,
                image_resolution=args.resolution
            )
            restored_image = restored_images[0]

            # 保存修复结果
            output_bgr = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), output_bgr)

            # 保存对比图
            if args.save_comparison and original_image is not None:
                comparison = np.hstack([
                    cv2.resize(original_image, (512, 512)),
                    cv2.resize(faded_image, (512, 512)),
                    cv2.resize(restored_image, (512, 512))
                ])
                comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

                # 添加标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(comparison_bgr, "Original", (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(comparison_bgr, "Faded", (522, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(comparison_bgr, "Restored", (1034, 30), font, 1, (255, 255, 255), 2)

                comparison_path = comparison_dir / img_path.name
                cv2.imwrite(str(comparison_path), comparison_bgr)

            success_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            error_count += 1
            continue

    # 打印总结
    print("\n" + "=" * 80)
    print("🎉 Batch inference completed!")
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count}/{len(image_files)}")
    if skip_count > 0:
        print(f"⏭️  Skipped (already exists): {skip_count}")
    if error_count > 0:
        print(f"❌ Failed: {error_count}")
    print(f"\n📁 Output directory: {output_dir}")
    if args.save_comparison:
        print(f"📁 Comparison directory: {comparison_dir}")
    if args.save_faded and not args.is_faded:
        print(f"📁 Faded images directory: {faded_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
