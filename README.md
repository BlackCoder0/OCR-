

# OCR-手写数字识别

由于某大学讲座丧心病狂的作业过于不当人，因而怒写下<br>
C2动漫社专用（骗你的，不是也能用）

---

## 🧠 项目简介

这是一个基于 TensorFlow 和 CNN（卷积神经网络）的手写数字识别项目。你可以训练模型识别 MNIST 手写数字集，然后通过图形界面上传自己的数字图片，模型将预测你写的是哪个数字！

它包含了：
- 手写数字的训练模型（基于经典的 MNIST 数据集）
- 图形界面（Tkinter 实现）
- 支持图片上传和实时识别

---

## 📦 依赖安装

在运行此项目之前，请确保你已经安装了以下依赖（一行行，不要一口气粘贴）：

```bash
pip install tensorflow
pip install pillow
pip install numpy
```
如果你不便使用VPN，请使用阿里云镜像
```bash
pip install tensorflow -i https://mirrors.aliyun.com/pypi/simple/
pip install pillow -i https://mirrors.aliyun.com/pypi/simple/
pip install numpy -i https://mirrors.aliyun.com/pypi/simple/
 ```
或者也可以用清华大学镜像
```bash
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy -i https://mirrors.aliyun.com/pypi/simple/
```
<br>
对于 Windows 或 macOS，通常在安装 Python 时会自动包含 Tkinter。

---

## 📁 文件结构说明

```plaintext
.
├── data/                  # 可选：我从互联网收集的一组手写数字图片（非必须，模型训练不依赖）
│   └── ...               
├── MNIST/                 # 可选：本地 MNIST 数据集（第一次运行main.py后生成）
│   └── ...                
├── mnist_cnn_model.h5     # 可选：已训练好的 CNN 模型（第一次运行main.py后生成）
├── main.py                # 主程序文件（含模型训练 + GUI 界面）
```

### 📂 `data/` 文件夹

图片库，存放手写的数字图像方便寻找。格式应为 `.png` 或 `.jpg`，建议：

* 图像清晰
* 图片应该为黑底白字或者白底黑字
* 单个数字居中


### 📂 `MNIST/` 文件夹


如果你希望使用本地数据集进行训练，可以将数据放在此文件夹内。但项目默认会从 TensorFlow 自动下载数据，因此**不是必需的**。

### 📄 `mnist_cnn_model.h5`

* 模型训练完成后自动生成。
* 这是一个包含 CNN 权重和结构的模型文件。
* 后续运行将直接加载该模型，无需重复训练。

---

## 🛠️ 使用方法

1. 运行程序：

   ```bash
   python main.py
   ```
   
3. 等待训练完毕后，GUI 界面将会弹出，点击 **“上传图像”** 按钮，选择你写的数字图片。

4. 模型将识别图片中的数字并显示预测结果。

---

## 🧪 模型结构简介

训练使用的是一个简洁但有效的卷积神经网络结构：

```text
Input: 28x28 grayscale image
↓ Conv2D (32 filters, 3x3, ReLU)
↓ MaxPooling2D (2x2)
↓ Flatten
↓ Dense (128, ReLU)
↓ Dense (10, Softmax)
Output: Probability of digits 0~9
```

* 使用 `ImageDataGenerator` 进行轻度数据增强（旋转、平移、缩放）
* 训练集划分 10% 作为验证集
* 使用 `Adam` 优化器，训练 10 个 epoch

---

## ✨ 展示效果

> ✅ 输入图像 → 🤖 模型识别 → 🎉 输出结果！

程序会自动对上传图片进行以下处理：

* 转换为灰度
* 二值化 + 反转（如有需要）
* 缩放至 28x28 尺寸
* 归一化输入数据

识别结果将在 GUI 界面上实时显示！

---

## 💡 小贴士

* 上传图像应尽量清晰、单个数字居中。
* 支持任意图片尺寸，程序会自动适配。
* 上传图片应该为黑底白字或者白底黑字
* 训练时间大概在十分钟以内
