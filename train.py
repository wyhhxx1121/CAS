import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from keras.losses import binary_crossentropy  # 二元交叉熵用来评判一个二分类模型预测结果的好坏程度的
from skimage.io import imread  # 基于scikit-image的图像读取、（保存和显示imsave, imshow）
import keras.backend as K  # 导入后端模块
from sklearn.model_selection import train_test_split  # 将数组或矩阵分割成随机的序列和测试子集
import os
import matplotlib.pyplot as plt  # 可视化
from keras.optimizers import *  # 优化器
from tensorflow.keras.backend import *  # 后端
import glob  # glob是python自带的一个操作文件的相关模块
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
# 回调函数（callback）是在调用fit 时传入模型的一个对象（即实现特定方法的类实例），它在训练过程中的不同时间点都会被模型调用。
# 它可以访问关于模型状态与性能的所有可用数据，还可以采取行动：中断训练、保存模型、加载一组不同的权重或改变模型的状态。
# keras中也提供了丰富的回调API，我们可以根据需求自定义相关的对象。
# ModelCheckpoint在每个训练期之后保存模型;LearningRateScheduler学习速率定时器;
# EarlyStopping当被监测的数量不再提升，则停止训练;ReduceLROnPlateau当标准评估停止提升时，降低学习速率
from skimage.transform import resize
from segmentation.segmodel import ENet
from segmentation.segmodel import SegNet
from fengemx import Unet
from classificationsegmentation.segmentation.ENet import ENet
from classificationsegmentation.segmentation.FCN8 import FCN8
from classificationsegmentation.segmentation.Unet import Unet
from classificationsegmentation.segmentation.CRAEnet import CRAENet
# from segmentation.segmodel import CRAENet
# skimage的transform模块实现图片缩放与形变
# 改变图片尺寸resize;按比例缩放rescale;旋转rotate;
# 图像金字塔skimage.transform.pyramid_gaussian(image,downscale=2)downscale控制着金字塔的缩放比例
import cv2
# OpenCV2（Open Source Computer Vision Library），是一个开源的库平台计算机视觉库。有很强大的图片处理功能，可实现图像处理和计算机视觉方面的很多通用算法。
from sklearn.metrics import *  # 评价指标函数名称
from pathlib import Path
# Path将文件或者文件夹路径（str）转换为Path对象，
# 可以实现不同OS路径连接符问题不同、以及对该路径的其他操作，如判断是否为文件、文件夹，根据路径创建创建文件夹（包括是否递归创建缺少的文件夹）等。
np.random.seed(2)
BATCH_SIZE = 12
EPOCHS = 65
SEED = 42  # 随机数
reg_param = 1.0  # 正则化参数，规范化协方差
lr = 6e-4  # 学习率
use_dice = True  # 是否使用dice_loss作为网络的损失函数，只能用于两类分割
dice_bce_param = 0.0


img_list1 = sorted(glob.glob('G:/czy/data/BUSI/benign/*.png'))

# sorted（）对所有可迭代的对象进行排序操作

IMG_SIZE = 256
image = np.empty((len(img_list1), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
# empty(shape[, dtype, order])
# 依给定的shape, 和数据类型 dtype,  返回一个一维或者多维数组，数组的元素不为空，为随机产生的数据。
# print(x_data.shape)
for i, img_path in enumerate(img_list1):  # enumerate(iteration, start)在遍历中可以获得索引和元素值
    #     # load image
    img1 = imread(img_path)  # 读取图片
    # resize image with 1 channel
    img1 = resize(img1, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    # preserve_range=True的话表示保持原有的 取值范围，false的话就成0-1了
    # save to x_data
    image[i] = img1
image = image / 255.0  # 对图像进行归一化，范围为[0,1]


img_list2 = sorted(glob.glob('G:/czy/data/BUSI/benign_mask/*.png'))

IMG_SIZE = 256

mask = np.empty((len(img_list2), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
# print(x_data.shape)
for i, img_path in enumerate(img_list2):
    # load image
    img2 = imread(img_path)
    # resize image with 1 channel
    img2 = resize(img2, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    # save to x_data
    mask[i] = img2
mask = mask / 255.0

x_train, x_test, y_train, y_test = train_test_split(image, mask, test_size=0.2, random_state=2021)



print(x_train.shape)
print(x_test.shape)
# print(x_valid.shape)

model = CRAENet(1)
model.summary()

reduceLROnPlat = ReduceLROnPlateau(factor=0.5,
                                   patience=3,
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
# reduceLROnPlat调整学习率
# monitor：监测的值，可以是accuracy，val_loss,val_accuracy
# factor：缩放学习率的值，学习率将以lr=lr*factor的形式被减少
# patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
# mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
# min_lr：学习率最小值，能缩小到的下限
# verbose: 详细信息模式，0或者1


model.compile(optimizer=Adam(lr=lr),
              loss="binary_crossentropy",
              metrics=['accuracy'])
# model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
# model.compile(optimizer = 优化器，
#               loss = 损失函数，
#               metrics = ["准确率”])

loss_history = model.fit(x=x_train,
                         y=y_train,
                         batch_size=BATCH_SIZE,
                         steps_per_epoch=len(x_train) // BATCH_SIZE,
                         epochs=EPOCHS,
                         )
# 将训练数据在模型中训练一定次数，返回loss和测量指标
# x：输入
# y：输出
# batch_siz：每一个batch的大小（批尺寸）；即训练一次网络所用的样本数
# epoch：迭代次数，即全部样本数据将被“轮”多少次，轮完训练停止
# steps_per_epoch：用于指定每个epoch所使用的迭代次数
# 当fit方法的输入数据是张量类型时，steps_per_epoch 默认为数据集中的样本数量除以批次大小
model.save_weights('G:/czy/segmentation/weights/b-weights.h5')


