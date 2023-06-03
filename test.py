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
# skimage的transform模块实现图片缩放与形变
# 改变图片尺寸resize;按比例缩放rescale;旋转rotate;
# 图像金字塔skimage.transform.pyramid_gaussian(image,downscale=2)downscale控制着金字塔的缩放比例
import cv2
# OpenCV2（Open Source Computer Vision Library），是一个开源的库平台计算机视觉库。有很强大的图片处理功能，可实现图像处理和计算机视觉方面的很多通用算法。
from sklearn.metrics import *  # 评价指标函数名称
from pathlib import Path
# Path将文件或者文件夹路径（str）转换为Path对象，
# 可以实现不同OS路径连接符问题不同、以及对该路径的其他操作，如判断是否为文件、文件夹，根据路径创建创建文件夹（包括是否递归创建缺少的文件夹）等。
from segmentation.segmodel import ENet
from segmentation.segmodel import SegNet
from fengemx import FCN8
from fengemx import Unet

from segmentation.segmodel import ENet
from classificationsegmentation.segmentation.ENet import ENet
from classificationsegmentation.segmentation.FCN8 import FCN8
from classificationsegmentation.segmentation.Unet import Unet
from classificationsegmentation.segmentation.CEnet import ENet
from classificationsegmentation.segmentation.CRAEnet import CRAENet
np.random.seed(2)
BATCH_SIZE = 12
EPOCHS = 60
SEED = 42
reg_param = 1.0
lr = 6e-4
use_dice = True
dice_bce_param = 0.0


img_list1 = sorted(glob.glob('G:/czy/data/BUSI/benign/*.png'))

# sorted（）对所有可迭代的对象进行排序操作

IMG_SIZE = 256
image = np.empty((len(img_list1), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
# print(x_data.shape)
for i, img_path in enumerate(img_list1):  # enumerate(iteration, start)在遍历中可以获得索引和元素值
    #     # load image
    img1 = imread(img_path)
    # resize image with 1 channel
    img1 = resize(img1, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    # preserve_range=True的话表示保持原有的 取值范围，false的话就成0-1了
    # save to x_data
    image[i] = img1
image = image / 255.0   # 对图像进行归一化，范围为[0,1]


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

x_train, x_test, y_train, y_test = train_test_split(image,mask,test_size = 0.2,random_state = 2021)
model = CRAENet(1)

model.load_weights('G:/czy/segmentation/weights/b-weights.h5')

y_pred = model.predict(x_test, batch_size=12, verbose=1)


y_pred_threshold = []
i = 0


for y in y_pred:
    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    y = y * 255
    cv2.imwrite('G:/czy/data/BUSI/output/%d.png' % i, y)
    # G:\czy\segmentation\output
    i += 1

a = []
b = []
Y_test = list(np.ravel(y_test))
y_pred_threshold = list(np.ravel(y_pred_threshold))
for i in range(len(Y_test)):
    a.append(int(Y_test[i]))
for i in range((len(y_pred_threshold))):
    b.append(int(y_pred_threshold[i]))
tn, fp, fn, tp = confusion_matrix(a, b).ravel()
#

IOU1 = tp / (tp + fp + fn)
IOU2 = tn / (tn + fp + fn)
miou1 = (IOU1 + IOU2) / 2
DSC = (2 * tp) / (tp + fp + tp + fn)

print('Accuracy:', accuracy_score(a, b))
print('Sensitivity:', jaccard_score(a, b))
print('Specificity:', tn / (tn + fp))
print('miou:', miou1)
print('dsc:', DSC)

y_predict = model.predict(x_test, batch_size=12, verbose=1)
y_predict = y_predict > 0.5

for i in range(x_test.shape[0]):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].set_title('Original')
    ax[1].set_title('Result')
    ax[2].set_title('Predicted Result')
    ax[0].imshow(x_test[i, :, :, 0], cmap='gray')
    ax[1].imshow(y_test[i, :, :, 0])
    ax[2].imshow(y_predict[i, :, :, 0])
plt.savefig("Predict.png")
plt.show()
