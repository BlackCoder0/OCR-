from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 调整形状
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)



model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_size = int(0.1 * x_train.shape[0])
x_val = x_train[:val_size]
y_val = y_train[:val_size]
x_train2 = x_train[val_size:]
y_train2 = y_train[val_size:]

model.fit(
    datagen.flow(x_train2, y_train2, batch_size=64),
    epochs=10,
    validation_data=(x_val, y_val)
)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'测试准确率: {test_acc}')

model.save('mnist_cnn_model.h5')



# 加载模型
model = load_model('mnist_cnn_model.h5')


def predict_digit(img_path):
    # 加载图像并转换为灰度
    img = Image.open(img_path).convert('L')

    # 二值化处理，阈值可调节
    img = img.point(lambda x: 0 if x < 128 else 255, '1')

    # 反转颜色：MNIST是白底黑字，如果图像是黑底白字，则反转
    # 先检查平均像素，判断是否需要反转
    if np.array(img).mean() < 128:
        img = ImageOps.invert(img.convert('L'))

    # 调整大小为 28x28
    img = img.resize((28, 28))

    # 转换为 NumPy 数组并归一化
    img_array = np.array(img).astype('float32') / 255.0

    # 调整形状以匹配模型输入
    img_array = img_array.reshape(1, 28, 28, 1)

    # 预测
    prediction = model.predict(img_array)
    return np.argmax(prediction)


def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        # 显示上传的图像
        img = Image.open(file_path)
        img = img.resize((100, 100))  # 调整显示大小
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # 保持引用，防止被垃圾回收

        # 预测数字
        digit = predict_digit(file_path)
        result_label.config(text=f'预测结果: {digit}')

# 创建主窗口
app = tk.Tk()
app.title('手写数字识别')
app.geometry('300x200')

# 创建并排显示的框架
frame = tk.Frame(app)
frame.pack(pady=10)

# 图像显示标签
image_label = tk.Label(frame)
image_label.pack(side='left', padx=10)

# 预测结果标签
result_label = tk.Label(frame, text='预测结果: ', font=('Arial', 14))
result_label.pack(side='right', padx=10)

# 上传按钮
upload_button = tk.Button(app, text='上传图像', command=upload_and_predict)
upload_button.pack(pady=10)

app.mainloop()


