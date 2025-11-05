# 组合褪色效果使用指南

## 什么是组合褪色？

组合褪色允许你**同时应用多种褪色效果**，更真实地模拟老照片的自然老化过程。

真实的老照片通常不是单一效果，而是多种效果叠加：
- **亮度降低** + **泛黄** + **饱和度下降**
- **棕褐色调** + **泛黄**
- **亮度降低** + **棕褐色** + **噪声**

## 快速开始

### 1. 运行演示脚本

```bash
# 使用默认示例图片
python test_combined_example.py

# 使用你自己的图片
python test_combined_example.py /path/to/your/photo.jpg
```

这会生成一个对比图，展示6种不同的组合配置。

### 2. 测试组合效果

```bash
# 测试多种预设的组合配置
python test_fading.py --image your_photo.jpg --mode combined
```

这会生成6种不同的组合效果对比。

### 3. 自定义组合效果

```bash
# 亮度 + 泛黄 (泛黄为主，亮度辅助)
python test_fading.py \
    --image your_photo.jpg \
    --fade_type combined \
    --intensity 0.6 \
    --effects '{"brightness":0.5,"yellow":1.0}'

# 泛黄 + 棕褐色 (泛黄为主)
python test_fading.py \
    --image your_photo.jpg \
    --fade_type combined \
    --intensity 0.6 \
    --effects '{"yellow":1.0,"sepia":0.6}'

# 复杂老化效果（4种效果组合）
python test_fading.py \
    --image your_photo.jpg \
    --fade_type combined \
    --intensity 0.6 \
    --effects '{"brightness":0.5,"saturation":0.7,"yellow":1.0,"sepia":0.4}'
```

## 组合效果配置

### 基本语法

使用 JSON 格式指定多种效果及其**相对比例**：

```json
{
    "effect_type": ratio_value,
    "effect_type2": ratio_value2
}
```

**重要**：这些值表示相对比例（0-1），实际强度 = `intensity × ratio_value`

示例：
```bash
--intensity 0.6 --effects '{"brightness":0.5,"yellow":1.0}'
```
- `brightness` 实际强度：0.6 × 0.5 = 0.3
- `yellow` 实际强度：0.6 × 1.0 = 0.6

### 可用的效果类型

这些值表示**相对比例**（0-1），会乘以 `intensity` 得到实际强度。

- **brightness**: 亮度降低 (比例建议: 0.4-0.8)
- **saturation**: 饱和度降低 (比例建议: 0.5-1.0)
- **yellow**: 泛黄效果 (比例建议: 0.8-1.0，通常是主效果)
- **sepia**: 棕褐色调 (比例建议: 0.3-0.7)
- **uniform**: 向灰色靠近 (比例建议: 0.5-1.0)

### 推荐配置

#### 1. 轻微老化（70-80年代照片）

```bash
python test_fading.py \
    --image your_photo.jpg \
    --fade_type combined \
    --intensity 0.4 \
    --effects '{"brightness":0.5,"yellow":1.0}'
```

**效果**：轻微变暗（0.4×0.5=0.2），略微泛黄（0.4×1.0=0.4）

#### 2. 中度老化（50-60年代照片）

```bash
python test_fading.py \
    --image your_photo.jpg \
    --fade_type combined \
    --intensity 0.6 \
    --effects '{"brightness":0.5,"yellow":1.0,"sepia":0.5}'
```

**效果**：明显泛黄，带有棕褐色调

#### 3. 严重老化（30-40年代照片）

```bash
python test_fading.py \
    --image your_photo.jpg \
    --fade_type combined \
    --intensity 0.7 \
    --effects '{"brightness":0.6,"saturation":0.7,"yellow":1.0,"sepia":0.6}'
```

**效果**：严重褪色，强烈的老照片感

#### 4. 极度老化（接近黑白）

```bash
python test_fading.py \
    --image your_photo.jpg \
    --fade_type combined \
    --intensity 0.8 \
    --effects '{"saturation":1.0,"sepia":0.8,"brightness":0.5}'
```

**效果**：颜色几乎消失，只剩微弱的棕褐色调

## 在训练中使用组合效果

### 方法 1: 修改数据集类

编辑 `ldm/data/color_restoration.py`：

```python
class ColorRestorationTrain(ColorRestorationDataset):
    """训练集"""
    def __init__(self, size=512, training_images_list_file="file_lists/train.txt"):
        super().__init__(
            image_list_file=training_images_list_file,
            size=size,
            random_crop=True,
            fade_type='combined',  # 使用组合效果
            fade_intensity_range=(0.4, 0.7),
            use_color_hint=True,
            use_augmentation=True
        )

    def generate_faded_image(self, original_image):
        """生成褪色图片 - 使用组合效果"""
        # 随机褪色强度
        intensity = np.random.uniform(*self.fade_intensity_range)

        # 定义组合效果比例配置（随机选择）
        # 注意：这些是相对比例，实际强度 = intensity × 比例
        configs = [
            # 配置1: 亮度 + 泛黄 (泛黄为主)
            {'brightness': 0.5, 'yellow': 1.0},
            # 配置2: 泛黄 + 棕褐色 (泛黄为主)
            {'yellow': 1.0, 'sepia': 0.6},
            # 配置3: 全面老化 (泛黄为主，其他辅助)
            {'brightness': 0.6, 'saturation': 0.7,
             'yellow': 1.0, 'sepia': 0.5},
        ]

        # 随机选择一个配置
        combined_effects = configs[np.random.randint(len(configs))]

        # 应用组合褪色
        faded_image = simulate_color_fading(
            original_image,
            fade_type='combined',
            intensity=intensity,
            combined_effects=combined_effects
        )

        return faded_image, intensity
```

### 方法 2: 在配置文件中指定

如果你的数据加载支持，可以在配置文件中指定：

```yaml
data:
  target: ldm.data.color_restoration.ColorRestorationTrain
  params:
    size: 512
    training_images_list_file: file_lists/train.txt
    fade_type: 'combined'
    fade_intensity_range: [0.4, 0.7]
    # 默认会使用自动组合配置
```

## 理解效果强度

### intensity vs 效果比例（重要！）

**新的设计**：`intensity` 作为**全局缩放因子**，`combined_effects` 中的值表示**相对比例**。

- **intensity**: 全局强度（0-1），控制整体褪色程度
- **combined_effects 中的值**: 各效果的相对比例（0-1）

**实际强度 = intensity × 效果比例**

例如：
```python
intensity = 0.6
combined_effects = {'brightness': 0.5, 'yellow': 1.0, 'sepia': 0.4}
```

实际应用的强度：
- `brightness`: 0.6 × 0.5 = **0.3** (30% 亮度降低)
- `yellow`: 0.6 × 1.0 = **0.6** (60% 泛黄)
- `sepia`: 0.6 × 0.4 = **0.24** (24% 棕褐色调)
- 噪声强度也是 0.6

**优势**：
- 可以通过调整 `intensity` 来**整体调节**所有效果的强度
- `combined_effects` 只需定义**比例关系**，更容易调整

### 调整建议

1. **先定义比例，再调整intensity**
   - 第1步：设定各效果的相对比例 `{'brightness': 0.5, 'yellow': 1.0, 'sepia': 0.4}`
   - 第2步：调整 `intensity` 控制整体强度（0.3-0.8）

2. **比例值的含义**
   - `1.0`: 主要效果（会最强）
   - `0.5-0.8`: 次要效果（中等强度）
   - `0.2-0.4`: 辅助效果（轻微）

3. **推荐比例设置**
   - 泛黄为主: `{'yellow': 1.0, 'brightness': 0.5, 'sepia': 0.3}`
   - 棕褐色为主: `{'sepia': 1.0, 'yellow': 0.6, 'brightness': 0.4}`
   - 饱和度降低为主: `{'saturation': 1.0, 'brightness': 0.6, 'yellow': 0.3}`

4. **匹配真实数据**: 观察你的真实老照片，调整参数来匹配

## 与单一效果对比

| 方法 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| **单一效果** | 简单、可控 | 不够真实 | 特定类型的褪色（纯泛黄、纯饱和度降低） |
| **mixed模式** | 多样性 | 随机性强，效果不可控 | 训练数据增强 |
| **combined组合** | 真实、可控 | 配置稍复杂 | 模拟真实老照片，最推荐！ |

## 常见问题

### Q1: 组合效果和 mixed 模式有什么区别？

- **mixed**: 随机**选择一种**效果（uniform、saturation、yellow 三选一）
- **combined**: **同时应用多种**效果，更接近真实老化

### Q2: 多种效果是按什么顺序应用的？

按照字典遍历顺序依次应用。效果是累积的：
1. brightness → 降低亮度
2. saturation → 在降低亮度的基础上降低饱和度
3. yellow → 在前两者基础上添加泛黄
4. sepia → 最后叠加棕褐色调

### Q3: 如何选择合适的组合？

1. **分析真实数据**: 观察你的真实老照片有哪些特征
2. **运行测试**: 使用 `--mode combined` 看预设效果
3. **微调参数**: 根据测试结果调整各效果的强度
4. **A/B对比**: 生成多组配置，选择最接近真实的

### Q4: 可以添加自定义效果吗？

可以！在 `ldm/data/color_restoration.py` 的 `simulate_color_fading` 函数中添加新的 `elif` 分支：

```python
elif effect_type == 'my_custom_effect':
    # 你的自定义效果实现
    faded = apply_my_effect(faded, effect_intensity)
```

## 示例脚本

创建一个 Python 脚本来测试不同配置：

```python
import sys
sys.path.insert(0, '.')
from ldm.data.color_restoration import simulate_color_fading
from PIL import Image
import numpy as np

# 读取图片
image = Image.open('your_photo.jpg').convert('RGB')
image = np.array(image)

# 测试配置1
config1 = {'brightness': 0.3, 'yellow': 0.5}
faded1 = simulate_color_fading(image, 'combined', 0.6, config1)
Image.fromarray(faded1).save('result_config1.jpg')

# 测试配置2
config2 = {'yellow': 0.6, 'sepia': 0.4}
faded2 = simulate_color_fading(image, 'combined', 0.6, config2)
Image.fromarray(faded2).save('result_config2.jpg')

# 测试配置3
config3 = {'brightness': 0.4, 'saturation': 0.5, 'yellow': 0.6, 'sepia': 0.3}
faded3 = simulate_color_fading(image, 'combined', 0.6, config3)
Image.fromarray(faded3).save('result_config3.jpg')
```

## 总结

组合褪色效果让你能够：
- ✅ 更真实地模拟老照片
- ✅ 完全控制每种效果的强度
- ✅ 匹配你的真实数据特征
- ✅ 创建更具挑战性的训练样本

**推荐工作流程**：
1. 使用 `python test_combined_example.py` 快速预览
2. 使用 `python test_fading.py --mode combined` 看预设配置
3. 使用 `--effects` 参数微调自定义配置
4. 更新训练数据集类来应用最佳配置
5. 开始训练！
