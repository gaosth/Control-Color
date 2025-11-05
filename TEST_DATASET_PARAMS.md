# 测试数据集参数配置指南

当你修改了 `ldm/data/color_restoration.py` 中的褪色参数后，使用这个工具来验证效果。

---

## 为什么需要这个工具？

修改数据集参数后，你需要确认：
1. ✅ 褪色效果是否符合预期
2. ✅ 随机性是否合理（每次生成的效果不同但风格一致）
3. ✅ 完整的数据pipeline是否正常工作
4. ✅ 与其他配置对比，选择最佳参数

---

## 快速开始

### 测试 1: 基本褪色效果

```bash
python test_dataset_params.py --images test1.jpg test2.jpg
```

**输出**：
- 对比原图和褪色图
- 显示使用的intensity值
- 保存到 `dataset_test_results/`

### 测试 2: 观察随机性

```bash
python test_dataset_params.py --images test.jpg --num_generations 5
```

**作用**：生成5个不同的褪色版本，观察：
- 是否每次都略有不同（应该有随机性）
- 整体风格是否一致
- intensity范围是否合理

### 测试 3: 完整Pipeline测试

```bash
python test_dataset_params.py --images test.jpg --full_pipeline
```

**输出**：显示数据集返回的所有组件
- Ground Truth（原图）
- Faded Image（褪色图）
- Color Hint（颜色提示）
- Mask（需要修复的区域）
- Masked Image（带mask的图像）
- Text Condition（文本提示）

### 测试 4: 对比不同配置

```bash
python test_dataset_params.py --images test.jpg --compare_configs
```

**输出**：对比4种配置：
1. 当前数据集配置（ColorRestorationTrain）
2. 轻度褪色
3. 中度褪色
4. 重度褪色

---

## 详细使用指南

### 1. 测试单张或多张图片

```bash
# 测试1张
python test_dataset_params.py --images photo.jpg

# 测试多张
python test_dataset_params.py --images photo1.jpg photo2.jpg photo3.jpg
```

### 2. 从文件列表中随机测试

```bash
# 从train.txt中随机选10张测试
python test_dataset_params.py \
    --file_list file_lists/train.txt \
    --num_samples 10
```

### 3. 测试参数的随机性

```bash
# 生成10个不同版本，观察随机性
python test_dataset_params.py \
    --images test.jpg \
    --num_generations 10
```

**应该看到什么**：
- 10个版本的intensity略有不同
- 整体褪色风格保持一致
- 泛黄、饱和度等效果在合理范围内变化

### 4. 完整Pipeline验证

```bash
python test_dataset_params.py \
    --images test.jpg \
    --full_pipeline
```

**检查项**：
- ✅ 所有tensor的shape正确
- ✅ 值的范围正确（[-1, 1]或[0, 1]）
- ✅ mask覆盖率合理（通常30%-70%）
- ✅ 颜色提示与褪色图一致
- ✅ 文本提示包含intensity信息

**示例输出**：
```
数据统计:
  原图 shape: (512, 512, 3), range: [-1.000, 1.000]
  褪色图 shape: (512, 512, 3), range: [-1.000, 1.000]
  颜色提示 shape: (512, 512, 3), range: [-1.000, 1.000]
  Mask shape: (512, 512, 1), range: [0.000, 1.000]
  Mask覆盖率: 45.3%
  文本提示: restore faded photo, intensity 0.58
```

### 5. 配置对比

```bash
python test_dataset_params.py \
    --images test.jpg \
    --compare_configs
```

**输出图表**：
- 位置1：原始图片
- 位置2：**当前数据集配置**（这是你修改后的配置）
- 位置3：轻度褪色（intensity=0.4）
- 位置4：中度褪色（intensity=0.6）
- 位置5：重度褪色（intensity=0.8）

**如何判断**：
- 当前数据集配置的效果应该在3-5之间
- 如果看起来太弱→增大intensity_range
- 如果看起来太强→减小intensity_range

---

## 修改参数后的测试流程

假设你从 `analyze_fading_style.py` 得到了推荐参数：

```python
# 推荐参数
fade_type: combined
intensity_range: [0.5, 0.7]
combined_effects:
  saturation: 0.6
  yellow: 1.0
  sepia: 0.8
```

### 步骤 1: 修改 color_restoration.py

编辑 `ldm/data/color_restoration.py`：

```python
class ColorRestorationTrain(ColorRestorationDataset):
    """训练集"""
    def __init__(self, size=512, training_images_list_file="file_lists/train.txt"):
        super().__init__(
            image_list_file=training_images_list_file,
            size=size,
            random_crop=True,
            fade_type='combined',              # 已修改为combined
            fade_intensity_range=(0.5, 0.7),   # 修改为推荐范围
            use_color_hint=True,
            use_augmentation=True
        )

    def generate_faded_image(self, original_image):
        """生成褪色图片"""
        intensity = np.random.uniform(*self.fade_intensity_range)

        # 添加这段代码：使用推荐的combined_effects
        if self.fade_type == 'combined':
            combined_effects = {
                'saturation': 0.6,
                'yellow': 1.0,
                'sepia': 0.8,
            }

            faded_image = simulate_color_fading(
                original_image,
                fade_type='combined',
                intensity=intensity,
                combined_effects=combined_effects
            )
        else:
            faded_image = simulate_color_fading(
                original_image,
                fade_type=self.fade_type,
                intensity=intensity
            )

        return faded_image, intensity
```

### 步骤 2: 测试修改后的效果

```bash
# 基本测试
python test_dataset_params.py --images test1.jpg test2.jpg

# 观察随机性（应该生成不同的版本）
python test_dataset_params.py --images test.jpg --num_generations 5

# 对比配置（看看你的配置与其他配置的差异）
python test_dataset_params.py --images test.jpg --compare_configs
```

### 步骤 3: 验证与真实照片的相似度

```bash
python validate_fading_params.py \
    --test_images test1.jpg test2.jpg \
    --reference_faded your_real_faded_photos/ref1.jpg \
    --intensity 0.6 \
    --effects '{"saturation":0.6,"yellow":1.0,"sepia":0.8}'
```

### 步骤 4: 如果相似度不够高

根据相似度分数调整：

**相似度 < 80%** → 需要调整

```python
# 在 color_restoration.py 中调整 combined_effects
combined_effects = {
    'saturation': 0.7,  # 原来0.6，增大
    'yellow': 1.0,
    'sepia': 0.6,       # 原来0.8，减小
}
```

然后重新运行步骤2和3。

---

## 常见问题和调试

### Q1: 褪色效果太弱

**症状**：生成的图片颜色还很鲜艳

**解决**：
```python
# 方法1: 增大intensity范围
fade_intensity_range=(0.6, 0.8)  # 原来是(0.5, 0.7)

# 方法2: 增大效果比例
combined_effects = {
    'saturation': 0.8,  # 增大
    'yellow': 1.0,
    'sepia': 0.8,
}
```

### Q2: 褪色效果太强

**症状**：生成的图片几乎变成灰色/黑白

**解决**：
```python
# 方法1: 减小intensity范围
fade_intensity_range=(0.3, 0.5)  # 原来是(0.5, 0.7)

# 方法2: 减小效果比例
combined_effects = {
    'saturation': 0.4,  # 减小
    'yellow': 0.7,      # 减小
    'sepia': 0.4,       # 减小
}
```

### Q3: 泛黄效果不够

**症状**：照片褪色了但不够黄

**解决**：
```python
combined_effects = {
    'saturation': 0.6,
    'yellow': 1.0,      # 已经是最大，确保不要更小
    'sepia': 0.5,       # 增大sepia也会增加泛黄
}
```

### Q4: 随机性太大

**症状**：每次生成的效果差异太大

**解决**：
```python
# 缩小intensity范围
fade_intensity_range=(0.55, 0.65)  # 原来是(0.5, 0.7)
```

### Q5: 随机性不够

**症状**：每次生成的效果几乎一样

**解决**：
```python
# 扩大intensity范围
fade_intensity_range=(0.4, 0.8)  # 原来是(0.5, 0.7)

# 或者在generate_faded_image中添加随机配置选择
configs = [
    {'saturation': 0.6, 'yellow': 1.0, 'sepia': 0.8},
    {'saturation': 0.7, 'yellow': 0.8, 'sepia': 0.6},
    {'saturation': 0.5, 'yellow': 1.0, 'sepia': 0.5},
]
combined_effects = random.choice(configs)
```

---

## 输出文件说明

测试完成后，`dataset_test_results/` 目录包含：

### 基本测试
- `image_name_dataset_test.png` - 原图 vs 褪色图

### 随机性测试
- `image_name_dataset_test_5versions.png` - 原图 + 5个褪色版本

### 完整Pipeline测试
- `image_name_full_pipeline.png` - 6张图：
  1. Ground Truth
  2. Faded Image
  3. Color Hint
  4. Mask
  5. Masked Image
  6. Text Condition

### 配置对比测试
- `image_name_config_comparison.png` - 原图 + 4种配置

---

## 最佳实践

### 1. 先测试单张，再测试多张

```bash
# 第一步：选一张代表性图片
python test_dataset_params.py --images representative.jpg --compare_configs

# 第二步：满意后测试多张
python test_dataset_params.py --file_list file_lists/train.txt --num_samples 20
```

### 2. 结合验证工具

```bash
# 测试数据集生成
python test_dataset_params.py --images test.jpg

# 验证与真实照片的相似度
python validate_fading_params.py \
    --test_images test.jpg \
    --reference_faded real_faded.jpg \
    --intensity 0.6 \
    --effects '{"saturation":0.6,"yellow":1.0,"sepia":0.8}'
```

### 3. 记录参数和效果

创建一个配置文件记录你的实验：

```yaml
# my_fading_config.yaml
experiment_1:
  date: 2024-01-15
  intensity_range: [0.5, 0.7]
  combined_effects:
    saturation: 0.6
    yellow: 1.0
    sepia: 0.8
  similarity_score: 87%
  notes: 效果很好，接近1960年代照片

experiment_2:
  date: 2024-01-16
  intensity_range: [0.6, 0.8]
  combined_effects:
    saturation: 0.7
    yellow: 1.0
    sepia: 0.6
  similarity_score: 75%
  notes: 稍微偏强，降低intensity
```

---

## 完整工作流示例

```bash
# 1. 从真实照片学习参数
python analyze_fading_style.py --folder my_faded_photos --visualize

# 2. 修改 color_restoration.py 应用推荐参数
# （手动编辑文件）

# 3. 测试数据集生成效果
python test_dataset_params.py --images test1.jpg test2.jpg --compare_configs

# 4. 观察随机性
python test_dataset_params.py --images test1.jpg --num_generations 10

# 5. 完整pipeline验证
python test_dataset_params.py --images test1.jpg --full_pipeline

# 6. 验证与真实照片的相似度
python validate_fading_params.py \
    --test_images test1.jpg test2.jpg \
    --reference_faded my_faded_photos/ref1.jpg \
    --intensity 0.6 \
    --effects '{"saturation":0.6,"yellow":1.0,"sepia":0.8}'

# 7. 如果相似度>85%，开始训练！
python train_restoration.py --config models/cldm_v15_inpainting_infer.yaml
```

---

## 总结

这个工具帮你：

1. ✅ **快速验证**修改后的参数效果
2. ✅ **可视化对比**不同配置
3. ✅ **检查pipeline**确保数据正确
4. ✅ **观察随机性**确保数据多样性

**推荐流程**：
```
修改参数 → 测试效果 → 验证相似度 → 微调 → 再测试 → 开始训练
```

记住：好的训练数据是模型成功的关键！花时间调整褪色参数是值得的。
