"""
颜色修复/恢复任务数据集
支持颜色褪色模拟和成对数据生成

用于训练模型恢复褪色的照片颜色
"""
import os
import cv2
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset


def simulate_color_fading(image, fade_type='uniform', intensity=0.5, combined_effects=None):
    """
    模拟照片颜色褪色效果

    Args:
        image: RGB图像 [H, W, 3], uint8
        fade_type: 褪色类型
            - 'uniform': 均匀褪色
            - 'saturation': 饱和度降低
            - 'brightness': 亮度降低
            - 'yellow': 泛黄效果（老照片）
            - 'sepia': 棕褐色调（老照片）
            - 'mixed': 随机选择一种效果
            - 'combined': 组合多种效果（需要提供combined_effects参数）
        intensity: 褪色强度 [0, 1]，0=无褪色，1=完全褪色
        combined_effects: 组合效果配置，字典格式，例如：
            {'brightness': 0.3, 'yellow': 0.5, 'sepia': 0.4}
            表示同时应用三种效果，每种效果有独立的强度

    Returns:
        faded_image: 褪色后的图像 [H, W, 3], uint8
    """
    image = image.astype(np.float32)

    if fade_type == 'uniform':
        # 均匀降低颜色强度，向灰色靠近
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
        faded = image * (1 - intensity) + gray * intensity

    elif fade_type == 'saturation':
        # 降低饱和度
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity)  # 降低饱和度
        faded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    elif fade_type == 'brightness':
        # 降低亮度
        faded = image * (1 - intensity * 0.3)  # 轻微降低亮度

    elif fade_type == 'yellow':
        # 老照片泛黄效果
        # 增加红色和绿色通道，模拟泛黄
        yellow_tint = np.array([1.0 + intensity * 0.2, 1.0 + intensity * 0.15, 1.0 - intensity * 0.3])
        faded = image * yellow_tint
        # 降低饱和度
        hsv = cv2.cvtColor(faded.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity * 0.5)
        faded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    elif fade_type == 'sepia':
        # 棕褐色调（老照片）
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia = cv2.transform(image, sepia_matrix)
        faded = image * (1 - intensity) + sepia * intensity

    elif fade_type == 'mixed':
        # 随机选择一种效果
        effects = ['uniform', 'saturation', 'yellow']
        chosen = np.random.choice(effects)
        return simulate_color_fading(image.astype(np.uint8), chosen, intensity)

    elif fade_type == 'combined':
        # 组合多种效果
        if combined_effects is None:
            # 默认组合：模拟真实老照片（亮度降低 + 泛黄 + 棕褐色调）
            combined_effects = {
                'brightness': intensity * 0.4,
                'yellow': intensity * 0.6,
                'sepia': intensity * 0.3
            }

        faded = image.copy()

        # 按顺序应用每种效果
        for effect_type, effect_intensity in combined_effects.items():
            if effect_intensity > 0:
                if effect_type == 'brightness':
                    # 降低亮度
                    faded = faded * (1 - effect_intensity * 0.3)

                elif effect_type == 'saturation':
                    # 降低饱和度
                    hsv = cv2.cvtColor(faded.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 1] = hsv[:, :, 1] * (1 - effect_intensity)
                    faded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

                elif effect_type == 'yellow':
                    # 泛黄效果
                    yellow_tint = np.array([
                        1.0 + effect_intensity * 0.2,
                        1.0 + effect_intensity * 0.15,
                        1.0 - effect_intensity * 0.3
                    ])
                    faded = faded * yellow_tint
                    # 同时降低饱和度
                    hsv = cv2.cvtColor(faded.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 1] = hsv[:, :, 1] * (1 - effect_intensity * 0.5)
                    faded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

                elif effect_type == 'sepia':
                    # 棕褐色调
                    sepia_matrix = np.array([
                        [0.393, 0.769, 0.189],
                        [0.349, 0.686, 0.168],
                        [0.272, 0.534, 0.131]
                    ])
                    sepia_img = cv2.transform(faded.astype(np.uint8), sepia_matrix).astype(np.float32)
                    faded = faded * (1 - effect_intensity) + sepia_img * effect_intensity

                elif effect_type == 'uniform':
                    # 向灰色靠近
                    gray = cv2.cvtColor(faded.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
                    faded = faded * (1 - effect_intensity) + gray * effect_intensity

        # 裁剪到有效范围
        faded = np.clip(faded, 0, 255)

    else:
        raise ValueError(f"Unknown fade_type: {fade_type}")

    # 添加轻微噪声（老照片效果）
    noise = np.random.randn(*faded.shape) * (intensity * 3)
    faded = faded + noise

    # 裁剪到有效范围
    faded = np.clip(faded, 0, 255).astype(np.uint8)

    return faded


def extract_color_mask(faded_image, original_image=None, mask_type='faded_color'):
    """
    从褪色图片中提取颜色mask/hint

    Args:
        faded_image: 褪色图片 [H, W, 3]
        original_image: 原始图片（可选）
        mask_type: mask类型
            - 'faded_color': 直接使用褪色图片的颜色
            - 'saturation_map': 饱和度图
            - 'color_diff': 颜色差异图（需要original_image）

    Returns:
        color_mask: 颜色提示 [H, W, 3]
    """
    if mask_type == 'faded_color':
        # 直接返回褪色图片的颜色
        return faded_image

    elif mask_type == 'saturation_map':
        # 提取饱和度图作为mask
        hsv = cv2.cvtColor(faded_image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        # 转换为3通道
        color_mask = cv2.cvtColor(saturation, cv2.COLOR_GRAY2RGB)
        return color_mask

    elif mask_type == 'color_diff':
        # 原始颜色和褪色颜色的差异
        if original_image is None:
            return faded_image
        diff = np.abs(original_image.astype(np.float32) - faded_image.astype(np.float32))
        return diff.astype(np.uint8)

    return faded_image


class ColorRestorationDataset(Dataset):
    """
    颜色修复/恢复任务数据集

    从原始彩色图片生成褪色版本，训练模型恢复颜色

    数据流程：
    1. 加载原始彩色图片（ground truth）
    2. 模拟褪色效果（输入）
    3. 提取褪色图片的颜色作为hint
    4. 训练模型：褪色图 + 颜色hint -> 原始图
    """
    def __init__(self,
                 image_list_file,
                 size=512,
                 random_crop=True,
                 fade_type='mixed',           # 褪色类型
                 fade_intensity_range=(0.3, 0.7),  # 褪色强度范围
                 use_color_hint=True,         # 是否使用颜色hint
                 use_augmentation=True):
        super().__init__()

        # 读取图片路径
        with open(image_list_file, "r") as f:
            self.image_paths = f.read().splitlines()

        print(f"Loaded {len(self.image_paths)} images for color restoration")

        self.size = size
        self.random_crop = random_crop
        self.fade_type = fade_type
        self.fade_intensity_range = fade_intensity_range
        self.use_color_hint = use_color_hint
        self.use_augmentation = use_augmentation

        # 图像预处理pipeline
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if self.random_crop:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)

            # 数据增强
            if self.use_augmentation:
                self.augmentation = albumentations.Compose([
                    self.rescaler,
                    self.cropper,
                    albumentations.HorizontalFlip(p=0.5),
                    # 注意：不要对颜色做太多增强，会影响褪色效果
                ])
            else:
                self.augmentation = albumentations.Compose([
                    self.rescaler,
                    self.cropper,
                ])
        else:
            self.augmentation = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.image_paths)

    def load_and_preprocess(self, image_path):
        """加载并预处理图像"""
        # 读取原始彩色图像（ground truth）
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # 确保是3通道图像
        if len(image.shape) == 2:
            # 灰度图，转换为3通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            # 单通道图，转换为3通道
            image = np.repeat(image, 3, axis=2)

        # 数据增强
        image = self.augmentation(image=image)["image"]

        # 再次确保是3通道（数据增强后）
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] != 3:
            raise ValueError(f"Image has wrong number of channels: {image.shape}")

        return image

    def generate_faded_image(self, original_image):
        """生成褪色图片"""
        # 随机褪色强度
        intensity = np.random.uniform(*self.fade_intensity_range)

        # 应用褪色效果
        faded_image = simulate_color_fading(
            original_image,
            fade_type=self.fade_type,
            intensity=intensity
        )

        return faded_image, intensity

    def generate_random_mask(self, height, width):
        """生成随机mask（可选，用于局部修复）"""
        # 70%概率全图修复，30%概率局部修复
        if np.random.rand() > 0.3:
            return np.ones((height, width, 1), dtype=np.float32)

        # 生成随机mask (先用2D)
        mask = np.zeros((height, width), dtype=np.float32)
        num_regions = np.random.randint(1, 5)

        for _ in range(num_regions):
            # 随机区域
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            w = np.random.randint(width // 4, width // 2)
            h = np.random.randint(height // 4, height // 2)

            x1 = max(0, x - w // 2)
            y1 = max(0, y - h // 2)
            x2 = min(width, x + w // 2)
            y2 = min(height, y + h // 2)

            mask[y1:y2, x1:x2] = 1.0

        # 膨胀操作 (在2D上操作)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 扩展到3D (H, W, 1)
        mask = mask[:, :, np.newaxis]

        return mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            # 1. 加载原始图片（ground truth）
            original_image = self.load_and_preprocess(image_path)

            # 2. 生成褪色图片
            faded_image, fade_intensity = self.generate_faded_image(original_image)

            # 3. 提取颜色hint（从褪色图片）
            if self.use_color_hint:
                color_hint = extract_color_mask(faded_image, original_image, 'faded_color')
            else:
                # 如果不使用颜色hint，使用灰度图
                gray = cv2.cvtColor(faded_image, cv2.COLOR_RGB2GRAY)
                color_hint = cv2.merge([gray, gray, gray])

            # 4. 生成mask（用于局部修复）
            mask = self.generate_random_mask(original_image.shape[0], original_image.shape[1])

            # 5. 生成masked image
            # mask=1的区域需要修复，mask=0的区域保持原样
            masked_image = mask * faded_image + (1 - mask) * original_image

            # 6. 归一化到[-1, 1]
            original_image = (original_image / 127.5 - 1.0).astype(np.float32)
            faded_image = (faded_image / 127.5 - 1.0).astype(np.float32)
            color_hint = (color_hint / 127.5 - 1.0).astype(np.float32)
            masked_image = (masked_image / 127.5 - 1.0).astype(np.float32)
            mask = mask.astype(np.float32)

            # 准备输出字典
            example = {
                "jpg": original_image,        # Ground truth原始图 [H, W, 3]
                "hint": color_hint,           # 颜色提示（褪色图的颜色） [H, W, 3]
                "faded": faded_image,         # 褪色图（用于可视化）
                "mask": mask,                 # Mask [H, W, 1]
                "mask_img": masked_image,     # 带mask的图像 [H, W, 3]
                "txt": f"restore faded photo, intensity {fade_intensity:.2f}"  # 文本提示
            }

            return example

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回下一张图片
            return self.__getitem__((idx + 1) % len(self))


class ColorRestorationTrain(ColorRestorationDataset):
    """训练集"""
    def __init__(self, size=512, training_images_list_file="file_lists/train.txt"):
        super().__init__(
            image_list_file=training_images_list_file,
            size=size,
            random_crop=True,
            fade_type='mixed',              # 混合多种褪色效果
            fade_intensity_range=(0.3, 0.8),  # 中等到强烈褪色
            use_color_hint=True,
            use_augmentation=True
        )


class ColorRestorationValidation(ColorRestorationDataset):
    """验证集"""
    def __init__(self, size=512, validation_images_list_file="file_lists/val.txt"):
        super().__init__(
            image_list_file=validation_images_list_file,
            size=size,
            random_crop=False,
            fade_type='uniform',            # 验证时使用统一的褪色效果
            fade_intensity_range=(0.5, 0.5),  # 固定褪色强度
            use_color_hint=True,
            use_augmentation=False
        )


# 用于测试：可视化褪色效果
def test_fading_effects():
    """测试不同的褪色效果"""
    import matplotlib.pyplot as plt

    # 创建测试图像
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # 或者加载真实图片
    # test_img = cv2.imread('test.jpg')
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    fade_types = ['uniform', 'saturation', 'brightness', 'yellow', 'sepia']
    intensities = [0.3, 0.5, 0.7]

    fig, axes = plt.subplots(len(intensities), len(fade_types) + 1, figsize=(15, 9))

    for i, intensity in enumerate(intensities):
        # 原图
        axes[i, 0].imshow(test_img)
        axes[i, 0].set_title(f'Original\nIntensity: {intensity}')
        axes[i, 0].axis('off')

        # 不同褪色效果
        for j, fade_type in enumerate(fade_types):
            faded = simulate_color_fading(test_img, fade_type, intensity)
            axes[i, j + 1].imshow(faded)
            axes[i, j + 1].set_title(fade_type)
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig('fading_effects_test.png', dpi=150, bbox_inches='tight')
    print("Saved fading effects test to: fading_effects_test.png")


if __name__ == "__main__":
    # 运行测试
    print("Testing color fading effects...")
    test_fading_effects()
