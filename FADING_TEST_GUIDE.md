# 褪色效果测试指南

这个指南帮助你使用自己的图片测试褪色效果。

## 快速开始

### 1. 基本用法 - 测试单张图片

```bash
python test_fading.py --image /path/to/your/image.jpg
```

这会生成褪色效果并保存到 `fading_test_results/` 目录。

### 2. 测试所有褪色类型

```bash
python test_fading.py --image /path/to/your/image.jpg --mode all_types
```

会生成一个对比图，展示所有褪色类型：
- **uniform**: 均匀褪色（向灰色靠近）
- **saturation**: 饱和度降低
- **brightness**: 亮度降低
- **yellow**: 泛黄效果（老照片）
- **sepia**: 棕褐色调（老照片）

### 3. 测试不同褪色强度

```bash
python test_fading.py --image /path/to/your/image.jpg --mode all_intensities --fade_type yellow
```

会生成一个对比图，展示从 0.0 到 1.0 的不同强度效果。

### 4. 自定义褪色参数

```bash
# 指定褪色类型和强度
python test_fading.py --image /path/to/your/image.jpg --fade_type yellow --intensity 0.7

# 指定输出目录
python test_fading.py --image /path/to/your/image.jpg --output_dir my_test_results
```

## 褪色类型说明

### uniform（均匀褪色）
- 最常见的照片褪色效果
- 颜色逐渐向灰色靠近
- 适合模拟自然老化的照片

### saturation（饱和度降低）
- 只降低颜色饱和度，不改变亮度
- 颜色变得暗淡但结构清晰
- 适合模拟轻微褪色

### brightness（亮度降低）
- 整体变暗
- 模拟曝光不足或年代久远的照片

### yellow（泛黄）
- 老照片典型效果
- 偏黄色调，像是被时间氧化
- 适合模拟 70-80 年代的老照片

### sepia（棕褐色）
- 经典的老照片色调
- 温暖的棕褐色调
- 适合模拟更古老的照片（50-60年代）

### mixed（混合）
- 随机选择上述效果之一
- 训练时使用，增加数据多样性

## 褪色强度说明

- **0.0**: 无褪色（原图）
- **0.3**: 轻微褪色（颜色稍微暗淡）
- **0.5**: 中等褪色（明显的颜色损失）
- **0.7**: 强烈褪色（颜色严重损失）
- **1.0**: 完全褪色（接近灰度图）

训练时使用的范围：**0.3 - 0.8**

## 输出文件

运行测试后，`fading_test_results/` 目录会包含：

1. **对比图**: 显示原图、褪色图、颜色提示（模型输入）
2. **褪色图**: 单独保存的褪色图片
3. **类型对比图**: 所有褪色类型的对比（使用 `--mode all_types`）
4. **强度对比图**: 不同强度的对比（使用 `--mode all_intensities`）

## 示例

### 示例 1: 测试老照片泛黄效果

```bash
python test_fading.py \
    --image photos/my_photo.jpg \
    --fade_type yellow \
    --intensity 0.6
```

### 示例 2: 对比所有褪色类型

```bash
python test_fading.py \
    --image photos/my_photo.jpg \
    --mode all_types \
    --intensity 0.5
```

### 示例 3: 查看不同强度的棕褐色效果

```bash
python test_fading.py \
    --image photos/my_photo.jpg \
    --mode all_intensities \
    --fade_type sepia
```

## 理解结果

测试生成的图片会显示三部分：

1. **Original Image**: 原始图片（训练的目标输出）
2. **Faded Image**: 褪色后的图片（这是实际看到的褪色照片）
3. **Color Hint**: 颜色提示（这是输入给模型的条件信号）

在训练中：
- **输入**: Color Hint（褪色图片的颜色） + Mask
- **输出**: 恢复后的彩色图片（接近 Original Image）
- **目标**: 学习如何从褪色的颜色信息中恢复原始颜色

## 数据集准备建议

根据测试结果选择合适的褪色参数：

1. **如果你的真实数据是老照片泛黄**:
   ```python
   fade_type='yellow'  或 'sepia'
   fade_intensity_range=(0.4, 0.7)
   ```

2. **如果你的真实数据是颜色暗淡**:
   ```python
   fade_type='saturation' 或 'uniform'
   fade_intensity_range=(0.3, 0.6)
   ```

3. **如果数据多样（推荐）**:
   ```python
   fade_type='mixed'  # 混合多种效果
   fade_intensity_range=(0.3, 0.8)  # 训练时的默认设置
   ```

## 修改褪色参数

如果需要调整训练时的褪色效果，修改配置文件：

```yaml
# models/cldm_v15_inpainting_infer.yaml
data:
  target: ldm.data.color_restoration.ColorRestorationTrain
  params:
    size: 512
    training_images_list_file: file_lists/train.txt
    fade_type: 'yellow'  # 修改这里
    fade_intensity_range: [0.4, 0.7]  # 修改这里
```

或者直接修改 `ldm/data/color_restoration.py` 中的 `ColorRestorationTrain` 类。

## 故障排除

### 问题：找不到模块
```
ModuleNotFoundError: No module named 'albumentations'
```

解决：安装依赖
```bash
pip install albumentations opencv-python matplotlib pillow
```

### 问题：图片路径错误
```
Error: Image not found
```

解决：使用绝对路径或确保相对路径正确
```bash
# 使用绝对路径
python test_fading.py --image /home/user/photos/test.jpg

# 或使用相对路径
python test_fading.py --image ./photos/test.jpg
```

### 问题：中文路径问题

如果图片路径包含中文，可能会出现读取错误。建议：
1. 使用英文路径
2. 或修改脚本使用 cv2.imdecode 读取

## 下一步

测试完褪色效果后：
1. 根据测试结果调整 `fade_type` 和 `fade_intensity_range`
2. 准备你的训练数据集
3. 参考 `VALIDATION_SETUP.md` 配置验证集
4. 开始训练！
