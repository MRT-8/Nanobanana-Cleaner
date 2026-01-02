# 🍌 Nanobanana Cleaner

<div align="center">

**Intelligent Background Removal Tool - Make AI-Generated Images Perfect**

[English](README.md) | [简体中文](README_zh.md)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


</div>

---

## 🌟 Project Overview

Nanobanana Cleaner is a lightweight and fast Python tool for removing backgrounds from AI-generated images and adding transparency channels.

### Why Do You Need Nanobanana Cleaner?

Have you ever encountered this situation: AI-generated images look great, but when you insert them into documents, PPTs, or web pages, you find that the background is not pure white, or there's no transparency channel at all?

**Common Problems:**
- ❌ Background has slight yellowish or grayish tones
- ❌ PNG format but no transparency channel
- ❌ Causes occlusion when pasted elsewhere
- ❌ Light blue, light orange, and other colored backgrounds are accidentally deleted

**Nanobanana Cleaner's Solution:**
- ✅ Intelligently identify and remove background colors
- ✅ Automatically add transparency channels
- ✅ Protect colored background elements from accidental deletion
- ✅ Edge smoothing processing, eliminate jagged edges
- ✅ Support batch processing, improve efficiency

---

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| 🎨 **Smart Background Detection** | Supports white, black, and custom color backgrounds |
| 🛡️ **Color Protection Mechanism** | Automatically protects light blue, light orange, and other colored background elements |
| ✨ **Edge Smoothing Processing** | Adjustable feathering effect, eliminates jagged edges |
| 🔍 **Edge Detection Protection** | Sobel operator protects subject boundaries from accidental deletion |
| 🧹 **Morphological Optimization** | Automatically removes noise and fills holes |
| 🚀 **Batch Processing Capability** | Process hundreds of images at once |
| 🎯 **Auto Background Detection** | K-means clustering automatically identifies background color |

### Technical Highlights

- **CIELAB Color Space**: Color difference calculation that better matches human perception
- **Sobel Edge Detection**: Accurately identify and protect subject boundaries
- **Gaussian Blur Feathering**: Smooth edges, improve visual quality
- **K-means Auto Detection**: Intelligently identify main background color of images
- **Morphological Operations**: Opening removes noise, closing fills holes

---

## 📸 Results

| Before | After |
|:------:|:------:|
| ![Before](assets/sample.png) | ![After](assets/sample_cleaned.png) |

*Left image has a light gray background, right image has background removed with transparent channel*

---

## 📦 Installation

### Requirements

- Python 3.10 or higher
- NumPy >= 2.2.6
- Pillow >= 12.0.0
- SciPy >= 1.10.0
- scikit-learn >= 1.3.0

### Quick Install

#### Method 1: Using uv (Recommended)

```bash
git clone https://github.com/MRT-8/Nanobanana-Cleaner.git
cd Nanobanana-Cleaner
uv sync
```

#### Method 2: Using pip

```bash
git clone https://github.com/MRT-8/Nanobanana-Cleaner.git
cd Nanobanana-Cleaner
pip install numpy pillow scipy scikit-learn
```

---

## 🚀 Quick Start

### Simplest Usage

```bash
# Process a single image (all enhanced features automatically enabled)
python cleaner.py -i your_image.png

# Output: your_image_cleaned.png
```

It's that simple! The tool will automatically:
- Detect background color
- Apply color protection
- Smooth edges
- Output high-quality PNG

### Batch Processing

```bash
# Process multiple images
python cleaner.py -i img1.png img2.png img3.png

# Process entire directory
python cleaner.py -i /path/to/images/
# Output saved to /path/to/images/output/
```

### Custom Processing

```bash
# Remove black background
python cleaner.py -i dark_image.png --background 0,0,0

# Auto-detect background color
python cleaner.py -i unknown_bg.png --auto-detect-bg

# More aggressive background removal
python cleaner.py -i image.png --transparent 0.05
```

---

## ⚙️ Parameters

### Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i, --input` | Required | Input image path or directory |
| `-o, --output` | Auto-generated | Output image path |
| `--transparent` | 0.1 | Transparency threshold (0-1) |
| `--opaque` | 1.0 | Opacity threshold (0-1) |
| `--background` | 255,255,255 | Background color (R,G,B) |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--feather` | 2 | Edge feathering radius (pixels) |
| `--no-lab` | False | Use RGB color space |
| `--no-edge-protection` | False | Disable edge protection |
| `--no-morphological` | False | Disable morphological optimization |
| `--no-color-protection` | False | Disable color protection |
| `--auto-detect-bg` | False | Auto-detect background color |

---

## 💡 Usage Tips

### 1. Process Images with Light-Colored Borders

```bash
# Color protection is enabled by default, automatically preserving light blue, light orange elements
python cleaner.py -i diagram.png
```

**Use Case**: Technical diagrams, flowcharts, images with decorative borders

### 2. Get Smoother Edges

```bash
# Increase feathering radius
python cleaner.py -i portrait.png --feather 4
```

**Use Case**: Portrait photos, images that need soft edges

### 3. Process Pure White Background Images

```bash
# Disable color protection for cleaner background
python cleaner.py -i pure_white.png --no-color-protection
```

**Use Case**: Pure white background product images, icons

### 4. Process Images with Unknown Background Color

```bash
# Let the tool auto-detect background color
python cleaner.py -i mystery.png --auto-detect-bg
```

**Use Case**: Images with uncertain background color

---

## 🔧 Global Installation (Optional)

Add cleaner to your system path for use anywhere:

**Zsh users:**
```bash
echo 'alias cleaner="python /path/to/Nanobanana-Cleaner/cleaner.py"' >> ~/.zshrc
source ~/.zshrc
```

**Bash users:**
```bash
echo 'alias cleaner="python /path/to/Nanobanana-Cleaner/cleaner.py"' >> ~/.bashrc
source ~/.bashrc
```

Now you can use it directly:
```bash
cleaner -i ~/Downloads/image.png
```

---

## 📐 Technical Details

Nanobanana Cleaner uses advanced image processing pipeline:

```
Image Loading → Color Space Conversion → Color Distance Calculation → Color Protection
    ↓
Edge Detection → Transparency Application → Morphological Optimization → Edge Feathering
    ↓
High Quality Output
```

### Key Technologies Detailed

#### 1. CIELAB Color Space
- Color difference calculation that better matches human perception
- Provides perceptually uniform color distance

#### 2. Sobel Edge Detection
- Accurately identify subject boundaries
- Protect subject from accidental deletion

#### 3. Gaussian Blur Feathering
- Smooth edges, eliminate jagged lines
- Adjustable feathering radius

#### 4. K-means Clustering
- Automatically identify main background color of images
- Suitable for images with unknown background color

#### 5. Morphological Operations
- Opening: Remove small noise
- Closing: Fill small holes

---

## 🎯 Use Cases

- ✅ AI-generated images (Gemini, GPT, Claude, etc.)
- ✅ Technical diagrams, flowcharts
- ✅ Product images, product photos
- ✅ Portraits, avatars
- ✅ Icons, logos
- ✅ Any images that need transparent backgrounds

---

## 🤝 Contributing Guide

Contributions, issue reporting, and suggestions are welcome!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

Special thanks to [Melmaphother/Nanobanana-Peel](https://github.com/Melmaphother/Nanobanana-Peel) for the inspiration. Nanobanana Cleaner has been enhanced with additional features and algorithm optimizations on top of their excellent work.

---

<div align="center">

**Made with ❤️ | Nanobanana Cleaner**

[⬆ Back to Top](#-nanobanana-cleaner)

</div>