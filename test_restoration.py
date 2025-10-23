"""
颜色修复/恢复任务测试脚本

使用训练好的模型修复褪色照片的颜色
"""
import os
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from PIL import Image

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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径（原始彩色图或褪色图）")
    parser.add_argument("--output", type=str, default="output_restoration.png", help="输出路径")
    parser.add_argument("--ckpt", type=str, default="./pretrained_models/main_model.ckpt", help="模型checkpoint路径")
    parser.add_argument("--config", type=str, default="./models/cldm_v15_inpainting_infer1.yaml", help="模型配置文件")
    parser.add_argument("--fade_type", type=str, default="uniform", help="褪色类型（如果输入是原始图）")
    parser.add_argument("--fade_intensity", type=float, default=0.6, help="褪色强度（如果输入是原始图）")
    parser.add_argument("--is_faded", action="store_true", help="输入已经是褪色图片")
    parser.add_argument("--steps", type=int, default=20, help="DDIM采样步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_comparison", action="store_true", help="保存对比图")

    args = parser.parse_args()

    # 加载模型
    print(f"Loading model from {args.ckpt}...")
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    print("✓ Model loaded")

    # 读取输入图片
    print(f"Loading input image from {args.input}...")
    input_image = cv2.imread(args.input)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {input_image.shape}")

    # 如果输入不是褪色图，先模拟褪色
    if not args.is_faded:
        print(f"Simulating color fading ({args.fade_type}, intensity={args.fade_intensity})...")
        faded_image = simulate_color_fading(input_image, args.fade_type, args.fade_intensity)
        original_image = input_image
        print("✓ Fading simulated")
    else:
        faded_image = input_image
        original_image = None
        print("Using input as faded image")

    # 修复褪色图片
    print(f"Restoring colors (steps={args.steps}, seed={args.seed})...")
    restored_images = restore_faded_image(
        model,
        ddim_sampler,
        faded_image,
        color_hint=None,  # 自动从褪色图提取
        prompt="restore photo colors",
        ddim_steps=args.steps,
        seed=args.seed
    )
    restored_image = restored_images[0]
    print("✓ Restoration complete")

    # 保存结果
    if args.save_comparison and original_image is not None:
        # 保存对比图：原图 | 褪色图 | 修复图
        comparison = np.hstack([
            cv2.resize(original_image, (512, 512)),
            cv2.resize(faded_image, (512, 512)),
            cv2.resize(restored_image, (512, 512))
        ])
        comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Faded", (522, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Restored", (1034, 30), font, 1, (255, 255, 255), 2)

        comparison_path = args.output.replace(".png", "_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        print(f"✓ Comparison saved to: {comparison_path}")

    # 保存修复图
    output_bgr = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_bgr)
    print(f"✓ Restored image saved to: {args.output}")

    # 同时保存褪色图（如果是模拟的）
    if not args.is_faded:
        faded_path = args.output.replace(".png", "_faded.png")
        cv2.imwrite(faded_path, cv2.cvtColor(faded_image, cv2.COLOR_RGB2BGR))
        print(f"✓ Faded image saved to: {faded_path}")

    print("\n" + "=" * 80)
    print("Color restoration completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
