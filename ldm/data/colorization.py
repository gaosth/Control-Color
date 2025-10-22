"""
着色任务数据集
用于Control-Color模型训练
"""
import os
import cv2
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    """
    着色任务数据集

    支持：
    1. 自动从彩色图生成灰度图
    2. 随机mask生成（模拟笔触）
    3. 数据增强

    Args:
        image_list_file: 图片路径列表文件
        size: 图片尺寸
        random_crop: 是否随机裁剪
        use_mask: 是否使用mask（用于局部着色训练）
        mask_ratio: mask比例
    """
    def __init__(self,
                 image_list_file,
                 size=512,
                 random_crop=True,
                 use_mask=True,
                 mask_ratio=0.3,
                 use_augmentation=True):
        super().__init__()

        # 读取图片路径
        with open(image_list_file, "r") as f:
            self.image_paths = f.read().splitlines()

        print(f"Loaded {len(self.image_paths)} images from {image_list_file}")

        self.size = size
        self.random_crop = random_crop
        self.use_mask = use_mask
        self.mask_ratio = mask_ratio
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
                    # 轻微的颜色抖动（不要太强，避免改变颜色分布）
                    albumentations.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05,
                        p=0.3
                    ),
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
        # 读取彩色图像（ground truth）
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # 数据增强
        image = self.augmentation(image=image)["image"]

        # 转换为LAB色彩空间
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L_channel = image_lab[:, :, 0]

        # 生成灰度图（hint）
        gray_image = cv2.merge([L_channel, L_channel, L_channel])

        return image, gray_image, image_lab

    def generate_random_mask(self, height, width):
        """生成随机mask用于训练（模拟用户笔触）"""
        if not self.use_mask or np.random.rand() > 0.5:
            # 50%概率不使用mask（全图着色）
            return np.ones((height, width, 1), dtype=np.float32)

        # 生成随机mask
        mask = np.zeros((height, width, 1), dtype=np.float32)
        num_strokes = np.random.randint(3, 10)

        for _ in range(num_strokes):
            # 随机画笔参数
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(10, 50)

            # 随机选择形状
            if np.random.rand() > 0.5:
                # 圆形
                cv2.circle(mask, (x, y), radius, 1.0, -1)
            else:
                # 矩形
                w = np.random.randint(20, 100)
                h = np.random.randint(20, 100)
                x1 = max(0, x - w // 2)
                y1 = max(0, y - h // 2)
                x2 = min(width, x + w // 2)
                y2 = min(height, y + h // 2)
                mask[y1:y2, x1:x2] = 1.0

        # 膨胀操作（使mask边缘更自然）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            # 加载并预处理
            color_image, gray_image, image_lab = self.load_and_preprocess(image_path)

            # 生成mask
            mask = self.generate_random_mask(color_image.shape[0], color_image.shape[1])

            # 生成masked image (用于inpainting)
            # mask=1的地方保留灰度图，mask=0的地方保留彩色图
            masked_image = mask * gray_image + (1 - mask) * color_image

            # 归一化到[-1, 1]
            color_image = (color_image / 127.5 - 1.0).astype(np.float32)
            gray_image = (gray_image / 127.5 - 1.0).astype(np.float32)
            masked_image = (masked_image / 127.5 - 1.0).astype(np.float32)

            # 转换为HWC格式
            mask = mask.astype(np.float32)

            # 准备输出字典
            example = {
                "jpg": color_image,           # Ground truth彩色图 [H, W, 3]
                "hint": gray_image,           # 灰度图（ControlNet输入） [H, W, 3]
                "mask": mask,                 # Mask [H, W, 1]
                "mask_img": masked_image,     # 带mask的图像 [H, W, 3]
                "txt": ""                     # 文本提示（可选）
            }

            return example

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回下一张图片
            return self.__getitem__((idx + 1) % len(self))


class ColorizationTrain(ColorizationDataset):
    """训练集"""
    def __init__(self, size=512, training_images_list_file="file_lists/train.txt"):
        super().__init__(
            image_list_file=training_images_list_file,
            size=size,
            random_crop=True,
            use_mask=True,
            use_augmentation=True
        )


class ColorizationValidation(ColorizationDataset):
    """验证集"""
    def __init__(self, size=512, validation_images_list_file="file_lists/val.txt"):
        super().__init__(
            image_list_file=validation_images_list_file,
            size=size,
            random_crop=False,  # 验证时使用中心裁剪
            use_mask=False,     # 验证时不使用mask
            use_augmentation=False  # 验证时不使用数据增强
        )


# 兼容性：支持只提供文本文件路径的方式
class ColorizationBase(Dataset):
    """基础着色数据集（向后兼容）"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example
