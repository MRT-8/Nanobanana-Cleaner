# 🍌 Nanobanana Cleaner

<div align="center">

**智能背景去除工具 - 让 AI 生成的图片更完美**

[English](README.md) | 简体中文

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


</div>

---

## 🌟 项目简介

Nanobanana Cleaner 是一个轻量快速的 Python 工具，用于去除 AI 生成图片的背景并添加透明通道。

### 为什么需要 Nanobanana Cleaner？

您是否遇到过这样的情况：用 AI 生成的图片看起来很棒，但当您将它插入到文档、PPT 或网页时，却发现背景不是纯白色，或者根本没有透明通道？

**常见问题：**
- ❌ 背景带有轻微的泛黄、泛灰色调
- ❌ PNG 格式但没有透明通道
- ❌ 粘贴到其他地方时造成遮挡
- ❌ 浅蓝色、浅橙色等有色背景被误删

**Nanobanana Cleaner 的解决方案：**
- ✅ 智能识别并去除背景色
- ✅ 自动添加透明通道
- ✅ 保护有色背景元素不被误删
- ✅ 边缘平滑处理，消除锯齿
- ✅ 支持批量处理，提升效率

---

## ✨ 功能特性

### 核心功能

| 功能 | 说明 |
|------|------|
| 🎨 **智能背景检测** | 支持白色、黑色及自定义颜色背景 |
| 🛡️ **颜色保护机制** | 自动保护浅蓝色、浅橙色等有色背景元素 |
| ✨ **边缘平滑处理** | 可调节的羽化效果，消除锯齿 |
| 🔍 **边缘检测保护** | Sobel 算子保护主体边界不被误删 |
| 🧹 **形态学优化** | 自动去除噪点和填充空洞 |
| 🚀 **批量处理能力** | 一次处理数百张图片 |
| 🎯 **自动背景检测** | K-means 聚类自动识别背景色 |

### 技术亮点

- **CIELAB 颜色空间**：更符合人眼感知的颜色差异计算
- **Sobel 边缘检测**：精准识别并保护主体边界
- **高斯模糊羽化**：平滑边缘，提升视觉质量
- **K-means 自动检测**：智能识别图片主背景色
- **形态学操作**：开运算去噪，闭运算填洞

---

## 📸 效果展示

| 处理前 | 处理后 |
|:------:|:------:|
| ![Before](assets/sample.png) | ![After](assets/sample_cleaned.png) |

*左侧图片带有浅灰色背景，右侧图片已去除背景并添加透明通道*

---

## 📦 安装指南

### 环境要求

- Python 3.10 或更高版本
- NumPy >= 2.2.6
- Pillow >= 12.0.0
- SciPy >= 1.10.0
- scikit-learn >= 1.3.0

### 快速安装

#### 方法 1：使用 uv（推荐）

```bash
git clone https://github.com/MRT-8/Nanobanana-Cleaner.git
cd Nanobanana-Cleaner
uv sync
```

#### 方法 2：使用 pip

```bash
git clone https://github.com/MRT-8/Nanobanana-Cleaner.git
cd Nanobanana-Cleaner
pip install numpy pillow scipy scikit-learn
```

---

## 🚀 快速开始

### 最简单的用法

```bash
# 处理单张图片（所有增强功能自动启用）
python cleaner.py -i your_image.png

# 输出：your_image_cleaned.png
```

就这么简单！工具会自动：
- 检测背景色
- 应用颜色保护
- 平滑边缘
- 输出高质量 PNG

### 批量处理

```bash
# 处理多张图片
python cleaner.py -i img1.png img2.png img3.png

# 处理整个目录
python cleaner.py -i /path/to/images/
# 输出保存到 /path/to/images/output/
```

### 自定义处理

```bash
# 去除黑色背景
python cleaner.py -i dark_image.png --background 0,0,0

# 自动检测背景色
python cleaner.py -i unknown_bg.png --auto-detect-bg

# 更激进的背景去除
python cleaner.py -i image.png --transparent 0.05
```

---

## ⚙️ 参数详解

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --input` | 必需 | 输入图片路径或目录 |
| `-o, --output` | 自动生成 | 输出图片路径 |
| `--transparent` | 0.1 | 透明阈值 (0-1) |
| `--opaque` | 1.0 | 不透明阈值 (0-1) |
| `--background` | 255,255,255 | 背景色 (R,G,B) |

### 高级参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feather` | 2 | 边缘羽化半径 (像素) |
| `--no-lab` | False | 使用 RGB 颜色空间 |
| `--no-edge-protection` | False | 禁用边缘保护 |
| `--no-morphological` | False | 禁用形态学优化 |
| `--no-color-protection` | False | 禁用颜色保护 |
| `--auto-detect-bg` | False | 自动检测背景色 |

---

## 💡 使用技巧

### 1. 处理带浅色边框的图片

```bash
# 颜色保护默认启用，会自动保留浅蓝色、浅橙色等元素
python cleaner.py -i diagram.png
```

**适用场景**：技术图表、流程图、带装饰性边框的图片

### 2. 获得更平滑的边缘

```bash
# 增加羽化半径
python cleaner.py -i portrait.png --feather 4
```

**适用场景**：人物肖像、需要柔和边缘的图片

### 3. 处理纯白色背景图片

```bash
# 禁用颜色保护以获得更干净的背景
python cleaner.py -i pure_white.png --no-color-protection
```

**适用场景**：纯白色背景的商品图、图标

### 4. 处理未知背景色的图片

```bash
# 让工具自动检测背景色
python cleaner.py -i mystery.png --auto-detect-bg
```

**适用场景**：不确定背景色的图片

---

## 🔧 全局命令（可选）

将 cleaner 添加到系统路径，随时随地使用：

**Zsh 用户：**
```bash
echo 'alias cleaner="python /path/to/Nanobanana-Cleaner/cleaner.py"' >> ~/.zshrc
source ~/.zshrc
```

**Bash 用户：**
```bash
echo 'alias cleaner="python /path/to/Nanobanana-Cleaner/cleaner.py"' >> ~/.bashrc
source ~/.bashrc
```

现在你可以直接使用：
```bash
cleaner -i ~/Downloads/image.png
```

---

## 📐 技术原理

Nanobanana Cleaner 采用先进的图像处理流程：

```
图像加载 → 颜色空间转换 → 颜色距离计算 → 颜色保护
    ↓
边缘检测 → 透明度应用 → 形态学优化 → 边缘羽化
    ↓
高质量输出
```

### 关键技术详解

#### 1. CIELAB 颜色空间
- 更符合人眼感知的颜色差异计算
- 提供感知均匀的颜色距离

#### 2. Sobel 边缘检测
- 精准识别主体边界
- 保护主体不被误删

#### 3. 高斯模糊羽化
- 平滑边缘，消除锯齿
- 可调节羽化半径

#### 4. K-means 聚类
- 自动识别图片主背景色
- 适合未知背景色的图片

#### 5. 形态学操作
- 开运算：去除小噪点
- 闭运算：填充小空洞

---

## 🎯 适用场景

- ✅ AI 生成图片（Gemini、GPT、Claude 等）
- ✅ 技术图表、流程图
- ✅ 商品图片、产品图
- ✅ 人物肖像、头像
- ✅ 图标、Logo
- ✅ 需要透明背景的任何图片

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

特别感谢 [Melmaphother/Nanobanana-Peel](https://github.com/Melmaphother/Nanobanana-Peel) 项目提供的灵感。Nanobanana Cleaner 在其基础上进行了功能增强和算法优化，提供了更强大的背景去除能力。

---

<div align="center">

**用 ❤️ 打造的 Nanobanana Cleaner**

[⬆ 返回顶部](#-nanobanana-cleaner)

</div>