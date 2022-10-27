# Tensorflow / Keras
from tensorflow import keras  # 用于构建神经网络

print('Tensorflow/Keras: %s' % keras.__version__)  # 打印版本
from keras.models import Sequential  # 用于组装神经网络模型
from keras.layers import Dense  # 给神经网络模型增添一些层
from tensorflow.keras.utils import plot_model  # 绘制模型图
# 数据操纵
import numpy as np  # 用于数据操纵

print('numpy: %s' % np.__version__)  # 打印版本
import pandas as pd  # 用于数据操纵

print('pandas: %s' % pd.__version__)  # 打印版本
import math  # 用于生成真实数据（本例中指向一个圆）
# 可视化
import matplotlib
import matplotlib.pyplot as plt  # 用于数据可视化

print('matplotlib: %s' % matplotlib.__version__)  # 打印版本
import graphviz  # for showing model diagram

print('graphviz: %s' % graphviz.__version__)  # 打印版本
import plotly
import plotly.express as px  # 用于数据可视化

print('plotly: %s' % plotly.__version__)  # 打印版本
# 其他工具
import sys
import os

# 把主目录赋值给一个变量
main_dir = os.path.dirname(sys.path[0])


# 获取其边缘（圆周）上点的坐标的函数
def PointsInCircum(r, n=100):
    return [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)]


# 保存组成半径为2的圆的一组实数点的坐标
circle = np.array(PointsInCircum(r=2, n=1000))
# 绘制图表
plt.figure(figsize=(15, 15), dpi=400)
plt.title(label='Real circle to be learned by the GAN generator', loc='center')
plt.scatter(circle[:, 0], circle[:, 1], s=5, color='black')
plt.show()


# 定义生成器模型
def generator(latent_dim, n_outputs=2):
    model = Sequential(name="Generator")  # 模型
    # 添加层
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim,
                    name='Generator-Hidden-Layer-1'))  # 隐藏层
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', name='Generator-Hidden-Layer-2'))  # 隐藏层
    model.add(Dense(n_outputs, activation='linear', name='Generator-Output-Layer'))  # 输出层
    return model


# 实例化
latent_dim = 3
gen_model = generator(latent_dim)
# 显示模型总结信息并绘制模型图
gen_model.summary()
plot_model(gen_model, show_shapes=True, show_layer_names=True, dpi=400)


# 建立鉴别器模型
def discriminator(n_inputs=2):
    model = Sequential(name="Discriminator")  # 模型
    # 添加层
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs,
                    name='Discriminator-Hidden-Layer-1'))  # 隐藏层
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', name='Discriminator-Hidden-Layer-2'))  # 隐藏层
    model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer'))  # 输出层
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 实例化
dis_model = discriminator()
# 显示模型总结信息并绘制模型图
dis_model.summary()
plot_model(dis_model, show_shapes=True, show_layer_names=True, dpi=400)


def def_gan(generator, discriminator):
    # 我们不想在这个阶段训练鉴别器的权重。因此，使其不可训练
    discriminator.trainable = False
    # 结合这两个模型
    model = Sequential(name="GAN")  # GAN 模型
    model.add(generator)  # 添加生成器
    model.add(discriminator)  # 添加鉴别器
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# 实例化
gan_model = def_gan(gen_model, dis_model)
# 显示模型总结信息并绘制模型图
gan_model.summary()
plot_model(gan_model, show_shapes=True, show_layer_names=True, dpi=400)


def real_samples(n):
    # 真实样本数据
    X = circle[np.random.choice(circle.shape[0], n, replace=True), :]
    # 类标签
    y = np.ones((n, 1))
    return X, y


# 生成潜在空间上的点数据；我们将用于后面的生成器的输入数据
def latent_points(latent_dim, n):
    # 生成潜在空间上的点数据
    latent_input = np.random.randn(latent_dim * n)
    # 重新构造形状：使成为网络的批输出
    latent_input = latent_input.reshape(n, latent_dim)
    return latent_input


# 使用生成器生成n个伪样本数据，结合类标签信息
def fake_samples(generator, latent_dim, n):
    # 生成潜在空间中的点
    latent_output = latent_points(latent_dim, n)
    # 预测输出（例如生成伪样本数据）
    X = generator.predict(latent_output)
    # 创建类标签
    y = np.zeros((n, 1))
    return X, y


def performance_summary(epoch, generator, discriminator, latent_dim, n=100):
    # 获取真实数据的样本
    x_real, y_real = real_samples(n)
    # 在真实数据上评估鉴别器
    _, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=1)
    # 获取假的（生成的）样本
    x_fake, y_fake = fake_samples(generator, latent_dim, n)
    # 在虚假（生成的）数据上评估鉴别器
    _, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=1)
    # 总结鉴别器性能
    print("Epoch number: ", epoch)
    print("Discriminator Accuracy on REAL points: ", real_accuracy)
    print("Discriminator Accuracy on FAKE (generated) points: ", fake_accuracy)
    # 创建二维散点图以显示真实和虚假（生成的）数据点
    plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(x_real[:, 0], x_real[:, 1], s=5, color='black')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], s=5, color='red')
    plt.show()


def train(g_model, d_model, gan_model, latent_dim, n_epochs=10001, n_batch=256, n_eval=1000):
    # 我们训练鉴别器的批次将包括一半真实点和一半假（生成的）点
    half_batch = int(n_batch / 2)
    # 我们使用手动方式枚举世代（ epochs ）
    for i in range(n_epochs):
        # 训练鉴别器
        # 准备真实样品数据
        x_real, y_real = real_samples(half_batch)
        # 准备假（生成）样本数据
        x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)
        # 使用真实和虚假样本训练鉴别器
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # 生成器训练
        # 从潜在空间中获取用作生成器输入的点
        x_gan = latent_points(latent_dim, n_batch)
        # 当我们生成假样本时，我们希望GAN生成器模型创建与真实样本相似的样本
        # 因此，我们希望传递与真实样本对应的标签，即y=1，而不是0。
        y_gan = np.ones((n_batch, 1))
        # Train the generator via a composite GAN model
        gan_model.train_on_batch(x_gan, y_gan)
        # Evaluate the model at every n_eval epochs
        if (i) % n_eval == 0:
            performance_summary(i, g_model, d_model, latent_dim)
