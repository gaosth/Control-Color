# 从真实褪色照片学习褪色规律

这个工具可以从你的真实褪色照片中**自动学习褪色特征**，然后生成适合的模拟参数配置。

## 为什么需要这个工具？

当你有一批真实的褪色老照片，但不知道应该用什么参数来模拟类似的效果时，这个工具可以：

1. **自动分析**你的褪色照片的颜色特征
2. **统计总结**褪色规律（饱和度、泛黄程度、色调等）
3. **推荐参数**用于模拟相似的褪色效果
4. **验证效果**对比模拟结果与真实照片的相似度

---

## 工作流程

```
真实褪色照片 → 分析特征 → 推荐参数 → 验证效果 → 应用到训练
```

---

## 快速开始

### 步骤 1: 分析褪色照片

将你的褪色照片放在一个文件夹中，然后运行分析：

```bash
python analyze_fading_style.py --folder /path/to/your/faded_photos
```

**输出**：
- `fading_analysis_report.json` - 详细的分析报告
- 推荐的参数配置（打印在终端）

### 步骤 2: 验证参数效果

使用推荐的参数对一些测试图片应用褪色，与真实褪色照片对比：

```bash
python validate_fading_params.py \
    --report fading_analysis_report.json \
    --test_images test1.jpg test2.jpg \
    --reference_faded faded1.jpg faded2.jpg
```

**输出**：
- `validation_results/` - 对比图和相似度报告
- 相似度分数（越高越好，>85% 为优秀）

### 步骤 3: 应用到训练

如果验证效果满意，将推荐的参数应用到训练配置中（参见下文）。

---

## 详细使用指南

### 1. 分析褪色照片

#### 基本用法

```bash
# 分析整个文件夹
python analyze_fading_style.py --folder /path/to/faded_photos

# 分析指定的几张图片
python analyze_fading_style.py --images img1.jpg img2.jpg img3.jpg img4.jpg
```

#### 生成可视化图表

```bash
python analyze_fading_style.py \
    --folder /path/to/faded_photos \
    --visualize
```

这会生成 `fading_analysis_visualization.png`，包含：
- 饱和度分布直方图
- 亮度分布
- 泛黄程度分布
- 棕褐色调分数分布
- 对比度分布
- RGB通道比例饼图

#### 自定义输出文件

```bash
python analyze_fading_style.py \
    --folder /path/to/faded_photos \
    --output my_analysis.json \
    --visualize \
    --viz-output my_visualization.png
```

---

### 2. 理解分析报告

运行分析后，你会看到类似这样的输出：

```
============================================================
分析报告
============================================================
分析图片数量: 50

颜色特征统计:

  saturation:
    平均值: 0.342
    标准差: 0.089
    范围: [0.180, 0.520]

  yellow_shift:
    平均值: 0.125
    标准差: 0.045
    范围: [0.050, 0.230]

  sepia_score:
    平均值: 0.680
    标准差: 0.120
    范围: [0.420, 0.850]

------------------------------------------------------------
推荐的褪色参数配置:
------------------------------------------------------------
fade_type: combined
intensity_range: [0.5, 0.7]
combined_effects:
  saturation: 0.6
    → 检测到中等饱和度(0.342)，推荐中度降低饱和度(0.6)
  yellow: 1.0
    → 检测到明显泛黄(0.125)，推荐强力泛黄效果(1.0)
  sepia: 0.8
    → 检测到明显棕褐色调(0.680)，推荐强力棕褐色效果(0.8)
```

**关键指标解释**：

| 指标 | 说明 | 典型范围 |
|------|------|----------|
| **saturation** | 饱和度 | 褪色照片通常 0.2-0.5 |
| **yellow_shift** | 泛黄程度 | >0.1 表示明显泛黄 |
| **sepia_score** | 棕褐色调相似度 | >0.6 表示明显棕褐色 |
| **brightness** | 亮度 | <0.5 表示偏暗 |
| **rgb_ratio** | RGB通道比例 | 泛黄照片: R>G>B |

---

### 3. 验证参数效果

#### 使用报告中的推荐参数

```bash
python validate_fading_params.py \
    --report fading_analysis_report.json \
    --test_images normal1.jpg normal2.jpg normal3.jpg \
    --reference_faded faded1.jpg faded2.jpg faded3.jpg
```

#### 手动指定参数验证

```bash
python validate_fading_params.py \
    --test_images normal1.jpg normal2.jpg \
    --reference_faded faded1.jpg faded2.jpg \
    --intensity 0.6 \
    --effects '{"saturation":0.6,"yellow":1.0,"sepia":0.8}'
```

#### 理解相似度分数

验证完成后会显示相似度分数：

```
总体相似度: 87.5%
各项指标相似度:
  saturation: 92%
  hue: 85%
  brightness: 88%
  rgb_ratio: 86%
```

**评分标准**：
- **> 85%**: ✅ 优秀，模拟效果非常接近真实照片
- **70-85%**: ✓ 良好，效果较为相似
- **55-70%**: ⚠ 一般，建议调整参数
- **< 55%**: ✗ 不佳，需要重新调整

---

### 4. 调整和优化参数

如果相似度不够高，可以手动微调参数：

#### 方法 1: 调整 intensity

```bash
# 增大intensity使效果更强
python validate_fading_params.py \
    --report fading_analysis_report.json \
    --test_images test.jpg \
    --reference_faded ref.jpg \
    --intensity 0.7  # 原本是0.6，增大到0.7
```

#### 方法 2: 调整各效果的比例

```bash
# 增强泛黄，减弱棕褐色
python validate_fading_params.py \
    --test_images test.jpg \
    --reference_faded ref.jpg \
    --intensity 0.6 \
    --effects '{"saturation":0.6,"yellow":1.0,"sepia":0.5}'  # sepia从0.8降到0.5
```

#### 常见调整建议

| 问题 | 调整方法 |
|------|----------|
| 模拟的颜色太鲜艳 | 增大 `saturation` 比例或 `intensity` |
| 泛黄效果不够 | 增大 `yellow` 比例（最大1.0） |
| 棕褐色调太重 | 减小 `sepia` 比例 |
| 整体效果太弱 | 增大 `intensity` 值 |
| 整体效果太强 | 减小 `intensity` 值 |

---

### 5. 应用到训练

验证满意后，将参数应用到训练中：

#### 方法 1: 修改数据集类

编辑 `ldm/data/color_restoration.py`，在 `ColorRestorationTrain` 类中：

```python
class ColorRestorationTrain(ColorRestorationDataset):
    def __init__(self, size=512, training_images_list_file="file_lists/train.txt"):
        super().__init__(
            image_list_file=training_images_list_file,
            size=size,
            random_crop=True,
            fade_type='combined',
            fade_intensity_range=(0.5, 0.7),  # 使用推荐的intensity范围
            use_color_hint=True,
            use_augmentation=True
        )

    def generate_faded_image(self, original_image):
        """使用分析得到的参数生成褪色图片"""
        intensity = np.random.uniform(*self.fade_intensity_range)

        # 使用推荐的combined_effects
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

        return faded_image, intensity
```

#### 方法 2: 在配置文件中指定（如果支持）

如果你的配置文件支持，可以直接在YAML中指定：

```yaml
data:
  target: ldm.data.color_restoration.ColorRestorationTrain
  params:
    size: 512
    fade_type: 'combined'
    fade_intensity_range: [0.5, 0.7]
    combined_effects:
      saturation: 0.6
      yellow: 1.0
      sepia: 0.8
```

---

## 完整示例工作流

### 场景：你有50张1960年代的老照片

```bash
# 1. 将所有老照片放到一个文件夹
mkdir my_1960s_photos
cp *.jpg my_1960s_photos/

# 2. 分析这些照片
python analyze_fading_style.py \
    --folder my_1960s_photos \
    --visualize \
    --output 1960s_analysis.json

# 3. 准备一些测试图片和参考图片
# test_images: 正常的彩色照片（用于测试模拟效果）
# reference_faded: 从1960s照片中选几张作为参考

# 4. 验证推荐的参数
python validate_fading_params.py \
    --report 1960s_analysis.json \
    --test_images test1.jpg test2.jpg test3.jpg \
    --reference_faded my_1960s_photos/ref1.jpg my_1960s_photos/ref2.jpg

# 5. 如果相似度不够高（<80%），手动调整
python validate_fading_params.py \
    --test_images test1.jpg \
    --reference_faded my_1960s_photos/ref1.jpg \
    --intensity 0.65 \
    --effects '{"saturation":0.7,"yellow":1.0,"sepia":0.6}'

# 6. 找到满意的参数后，应用到训练代码中
# 修改 ldm/data/color_restoration.py
```

---

## 技术细节

### 分析的颜色特征

1. **Saturation (饱和度)**
   - 使用HSV色彩空间的S通道
   - 褪色照片通常饱和度较低

2. **Hue (色调)**
   - HSV色彩空间的H通道
   - 泛黄照片色调偏向黄色区域（30-60度）

3. **Brightness (亮度)**
   - HSV色彩空间的V通道
   - 老照片可能偏暗

4. **RGB Ratio (RGB比例)**
   - 分析RGB三通道的相对比例
   - 泛黄：R和G高于B
   - 棕褐色：接近 [0.44, 0.39, 0.17]

5. **Yellow Shift (泛黄偏移)**
   - 计算 (R+G)/2 - B
   - 值越大表示越泛黄

6. **Sepia Score (棕褐色分数)**
   - 与标准棕褐色RGB比例的相似度
   - 值越高表示越接近棕褐色调

7. **Contrast (对比度)**
   - 灰度图的标准差
   - 老照片可能对比度降低

### 参数推荐算法

根据统计结果自动推荐：

- **Saturation**: 饱和度越低 → saturation参数越大
- **Yellow Shift**: 泛黄程度越高 → yellow参数越大
- **Sepia Score**: 棕褐色分数越高 → sepia参数越大
- **Brightness**: 亮度越低 → brightness参数越大
- **Intensity Range**: 根据整体褪色程度确定

### 相似度计算

验证时比较模拟照片和真实照片的相似度：

```
overall_similarity =
    0.35 × saturation_similarity +
    0.15 × hue_similarity +
    0.20 × brightness_similarity +
    0.30 × rgb_ratio_similarity
```

权重可以根据实际需求调整。

---

## 常见问题

### Q1: 需要多少张褪色照片才能得到准确的分析？

**建议至少20-30张**，越多越好。照片应该具有相似的褪色风格（同一年代、相似的保存条件）。

### Q2: 我的褪色照片风格不统一怎么办？

可以将不同风格的照片分组，分别分析：

```bash
# 1960年代照片
python analyze_fading_style.py --folder 1960s_photos --output 1960s.json

# 1980年代照片
python analyze_fading_style.py --folder 1980s_photos --output 1980s.json
```

### Q3: 验证相似度只有60%，是否正常？

60%相似度偏低，建议：
1. 检查参考照片是否真的是褪色照片（不是刻意处理的艺术效果）
2. 尝试调整intensity值（增大或减小0.1）
3. 手动微调各效果的比例
4. 确保测试图片和参考照片的内容相似（都是人像、风景等）

### Q4: 能否同时学习多种不同的褪色风格？

可以！分别分析不同风格，然后在训练时随机选择：

```python
configs = [
    # 配置1：1960年代风格
    {'saturation': 0.7, 'yellow': 1.0, 'sepia': 0.8},
    # 配置2：1980年代风格
    {'saturation': 0.5, 'yellow': 0.6, 'brightness': 0.5},
]
# 随机选择
effects = random.choice(configs)
```

### Q5: 这个工具适用于所有类型的褪色吗？

主要适用于：
- ✅ 老照片自然褪色（泛黄、饱和度降低）
- ✅ 棕褐色调老照片
- ✅ 曝光不足/过度的照片
- ❌ 水渍、霉斑等物理损伤（需要其他处理）
- ❌ 严重撕裂、折痕（需要inpainting）

---

## 总结

这个工具的核心优势：

1. **自动化**：无需手动尝试各种参数组合
2. **基于真实数据**：从你的实际照片中学习
3. **可验证**：提供量化的相似度评分
4. **可调整**：支持手动微调优化

**推荐工作流程**：
```
分析 → 验证 → 微调 → 再验证 → 应用到训练
```

祝你训练顺利！如果有任何问题，请查看输出的JSON报告获取详细信息。
