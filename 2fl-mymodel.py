import tensorflow as tf  # 引入tensorflow，并取别名为tf
gpus = tf.config.experimental.list_physical_devices('GPU')  # 获得当前主机上某种特定运算设备类型（如GPU或CPU）的列表
for gpu in gpus:
    tf.compat.v1.config.experimental.set_memory_growth(gpu, True)  # 动态申请内存，需要多少，申请多少


import matplotlib.pyplot as plt  # 引入matplotlib绘图

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np  # 引入numpy数值计算扩展工具
import pandas as pd  # 引入pandas解决数据分析任务
from sklearn import *  # 引入sklearn数据集,sklearn包含众多数据集
import tensorflow as tf

import keras
from scipy import interp  # 引入scipy插值和拟合
import random  # 引入random随机函数
import os  # os处理文件和目录
import time  # 时间
import datetime

import keras.backend as K  # keras后端
from itertools import cycle  # 迭代器
from sklearn.preprocessing import label_binarize  # 数据预处理，label_binarize对图像标签的独热编码
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from tensorflow.keras.regularizers import l2  #L2正则化


starttime = datetime.datetime.now()

print('--------------------------程序开始运行------------------------')

train_tfrecord = 'XRay_train.tfrecords' # 训练集
test_tfrecord = 'XRay_test.tfrecords'  # 测试集
train_percentage = 0.8  # Proportion of training set 训练集比例
random.seed(2021)  # 随机数种子
input_path = 'G:/czy/data/zy/bm/'  # 输入
learning_rate = 0.001  # 学习率
buffer_size = 512  # 存放数据集中部分数据的缓冲区大小
batch_size = 16  # 在读取TFRecord文件的时候，每次读取图片的数量
epochs = 55  # 轮次
img_size = 224  # 图片尺寸

# 测试集、训练集
def read_directory():
    # 定义一个读取字典
    data_filenames = []  # 文件名
    data_labels = []   # 标签


    for filename in os.listdir(input_path + 'benign'):
        data_filenames.append(input_path + 'benign/' + filename)
        data_labels.append(1)

    for filename in os.listdir(input_path + 'malignant'):
        data_filenames.append(input_path + 'malignant/' + filename)
        data_labels.append(0)

    data_size = len(data_labels)  # 获取数据类型的长度

    tmp_uni = list(zip(data_filenames, data_labels))  # 将文件名和标签进行打包

    random.shuffle(tmp_uni)  # 打乱

    train_size = int(data_size * train_percentage)  # 训练集数量
    print('Size of training set：', train_size)
    print('Size of test set：', data_size - train_size)

    train_list = tmp_uni[0:train_size]  # 训练集列表
    test_list = tmp_uni[train_size:]    # 测试集列表

    train_filenames, train_labels = zip(*train_list)
    test_filenames, test_labels = zip(*test_list)
    print(len(train_filenames))
    print(len(train_labels))
    print(len(test_list))
    print(len(test_labels))


    return train_filenames, train_labels, test_filenames, test_labels

def build_train_tfrecord(train_filenames, train_labels):
    # 生成训练集的TFRecord，TFRecord是将图像数据和标签统一存储的二进制文件
    # 将数据转换为特征
    with tf.io.TFRecordWriter(train_tfrecord)as writer:  # 将tfrecord文件写入到output_filename
        for filename, label in zip(train_filenames, train_labels):  # 遍历训练集的文件名和标签
            image = open(filename, 'rb').read()  # 读取数据集图片到内存，image为一个Byte类型的字符串
            # 建立tf.train.Feature字典
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个Bytes对象
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个Int对象
            }
            # 生成protocol数据类型
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立Example
            # 将特征写入TFRecord文件中
            writer.write(example.SerializeToString())  # 将Example序列化并写入TFRecord文件

def build_test_tfrecord(test_filenames, test_labels):
    # 生成测试集的TFRecord
    with tf.io.TFRecordWriter(test_tfrecord)as writer:
        for filename, label in zip(test_filenames, test_labels):
            image = open(filename, 'rb').read()

            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def _parse_example(example_string):
    # 将TFRecord文件中的每一个序列化的tf.train.Example解码
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # 字符串特征
        'label': tf.io.FixedLenFeature([], tf.int64),   # 数值特征
    }
    # 解析TFRecord中的数据和标签
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_png(feature_dict['image'], channels=3)  # 解码PNG图片
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [img_size, img_size]) / 255.0  # 根据模型调整图片大小
    return feature_dict['image'], feature_dict['label']


def get_train_dataset(train_tfrecord):  # 读取TFRecord
    raw_train_dataset = tf.data.TFRecordDataset(train_tfrecord)
    train_dataset = raw_train_dataset.map(_parse_example)

    return train_dataset

def get_test_dataset(test_tfrecord):
    raw_test_dataset = tf.data.TFRecordDataset(test_tfrecord)
    test_dataset = raw_test_dataset.map(_parse_example)

    return test_dataset

def data_Preprocessing(train_dataset, test_dataset):
    # 数据预处理
    train_dataset = train_dataset.shuffle(buffer_size)
    # buffer_size的作用就是存放数据集中部分数据的缓冲区大小，每次取数据是从缓冲区中随机取出一个item，
    # 该item是一个batch，取出后再拿数据集中未在缓冲区出现过的数据（依次）去填充该缓冲区的空缺位置。
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset.prefetch()的作用是会在第n个epoch的training的同时预先fetch第n+1个epoch的data,
    # 这个操作的实现是在background开辟一个新的线程，将数据读取在cache中，这也大大的缩减了总的训练的时间。
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset
#######################################################################################################################
##model
models = []
histories = []

def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    # 卷积+池化
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


# Inception Block
def Inception(x, nb_filter):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (5, 5), padding='same', strides=(1, 1), name=None)

    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

    return x

def relu6(x):
    return K.relu(x, max_value=6)

def depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def mobilenetv2model():
    inpt = Input(shape=(224, 224, 3))
    x1 = Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')  # （ 112, 112, 64)
    x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x1)  # (None, 56, 56, 64)
    x3 = Conv2d_BN(x2, 192, (3, 3), strides=(1, 1), padding='same')  # (None, 56, 56, 192)
    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x3)  # (None, 28, 28, 192)
    x5 = Inception(x4, 64)  # 256  (None, 28, 28, 256)
    x6 = Inception(x5, 120)  # 480 (None, 28, 28, 480)
    x7 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x6)  # (None, 14, 14, 480)
    x8 = Inception(x7, 128)  # 512 (None, 14, 14, 512)
    x9 = Inception(x8, 128)  # (None, 14, 14, 512)
    x10 = Inception(x9, 128)  # (None, 14, 14, 512)
    x11 = Inception(x10, 132)  # 528  (None, 14, 14, 528)
    x12 = Inception(x11, 208)  # 823  (None, 14, 14, 832)
    x13 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x12)
    x14 = Inception(x13, 208)  # (None, 7, 7, 832)
    x15 = Inception(x14, 256)  # 1024  (None, 7, 7, 1024)
    mobilenetv2 = GlobalAveragePooling2D()(x15)
    mobilenetv2 = BatchNormalization()(mobilenetv2)
    mobilenetv2 = Dense(2, activation="softmax")(mobilenetv2)
    model = Model(inputs=inpt, outputs=mobilenetv2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    return model
def dwconv2d_bn(x,  kernel_size=3,strides=1, padding='same', activation='relu',depth_multiplier=1, name=None):
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=strides, padding=padding,depth_multiplier=depth_multiplier, use_bias=False,name=name+'_sepconv2d')(x)
    x = keras.layers.BatchNormalization(axis=-1, scale=True, momentum=0.95,name=name+'_dwconv2d_bn')(x)
    if activation is not None:
        x = keras.layers.Activation(activation, name=name+'_dwconv2d_bn_relu')(x)
    return x

def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras import backend as K
import math



def spatial_attention( input_feature, kernel_size=7):
        avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
        max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        return multiply([input_feature, cbam_feature])


def eca_block(input_feature, b=1, gamma=2, name=""):
    channel = input_feature.shape[-1]
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    avg_pool = GlobalAveragePooling2D()(input_feature)

    x = Reshape((-1, 1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", name="eca_layer_" + str(name), use_bias=False, )(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1, 1, -1))(x)

    output = multiply([input_feature, x])
    return output



def AENet():
    inp = Input(shape=(img_size, img_size, 3))

    name = 'ghost'

    i1 = Conv2d_BN(inp, 64, (7, 7), strides=(2, 2), padding='same')  # （112, 112, 64)
    i2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(i1)  # (None, 56, 56, 64)
    i3 = Conv2d_BN(i2, 192, (3, 3), strides=(1, 1), padding='same')  # (None, 56, 56, 192)
    a1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(i3)  # (None, 28, 28, 192)

    x1 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(a1)
    x1 = BatchNormalization(axis=3)(x1)

    x2 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(a1)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = tf.nn.atrous_conv2d(value=x2, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x2 = dwconv2d_bn(x2, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw10')
    x2 = BatchNormalization(axis=3)(x2)

    x3 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(a1)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = tf.nn.atrous_conv2d(value=x3, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x3 = dwconv2d_bn(x3, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw11')
    x3 = BatchNormalization(axis=3)(x3)

    x4 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(a1)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = tf.nn.atrous_conv2d(value=x4, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x4 = dwconv2d_bn(x4, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw12')
    x4 = BatchNormalization(axis=3)(x4)

    x5 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(a1)
    x5 = BatchNormalization(axis=3)(x5)
    x5 = tf.nn.atrous_conv2d(value=x5, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x5 = dwconv2d_bn(x5, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw13')
    x5 = BatchNormalization(axis=3)(x5)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(a1)
    x6 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(x6)
    x6 = BatchNormalization(axis=3)(x6)



    dw = tf.nn.atrous_conv2d(value=a1, filters=tf.constant(value=1, shape=[3, 3, 192, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    dw = dwconv2d_bn(dw, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw')


    i2 = concatenate([x1, x2, x3, x4, x5, x6, dw], axis=3)
    i2 = eca_block(i2, name="eca_layer_7")
    i2 = spatial_attention(i2)




    x1 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i2)
    x1 = BatchNormalization(axis=3)(x1)

    x2 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i2)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = tf.nn.atrous_conv2d(value=x2, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x2 = dwconv2d_bn(x2, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw14')
    x2 = BatchNormalization(axis=3)(x2)

    x3 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i2)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = tf.nn.atrous_conv2d(value=x3, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x3 = dwconv2d_bn(x3, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw15')
    x3 = BatchNormalization(axis=3)(x3)

    x4 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i2)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = tf.nn.atrous_conv2d(value=x4, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x4 = dwconv2d_bn(x4, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw16')
    x4 = BatchNormalization(axis=3)(x4)

    x5 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i2)
    x5 = BatchNormalization(axis=3)(x5)
    x5 = tf.nn.atrous_conv2d(value=x5, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x5 = dwconv2d_bn(x5, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw17')
    x5 = BatchNormalization(axis=3)(x5)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(i2)
    x6 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(x6)
    x6 = BatchNormalization(axis=3)(x6)


    dw = tf.nn.atrous_conv2d(value=dw, filters=tf.constant(value=1, shape=[3, 3, 16, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    dw = dwconv2d_bn(dw, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw1')


    i3 = concatenate([x1, x2, x3, x4, x5, x6, dw], axis=3)
    i3 = eca_block(i3, name="eca_layer_6")
    i3 = spatial_attention(i3)

    i3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(i3)



    x1 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i3)
    x1 = BatchNormalization(axis=3)(x1)

    x2 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i3)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = tf.nn.atrous_conv2d(value=x2, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x2 = dwconv2d_bn(x2, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw18')
    x2 = BatchNormalization(axis=3)(x2)

    x3 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i3)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = tf.nn.atrous_conv2d(value=x3, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x3 = dwconv2d_bn(x3, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw19')
    x3 = BatchNormalization(axis=3)(x3)

    x4 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i3)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = tf.nn.atrous_conv2d(value=x4, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x4 = dwconv2d_bn(x4, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw20')
    x4 = BatchNormalization(axis=3)(x4)

    x5 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i3)
    x5 = BatchNormalization(axis=3)(x5)
    x5 = tf.nn.atrous_conv2d(value=x5, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x5 = dwconv2d_bn(x5, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw21')
    x5 = BatchNormalization(axis=3)(x5)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(i3)
    x6 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(x6)
    x6 = BatchNormalization(axis=3)(x6)



    dw = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(dw)
    dw = BatchNormalization(axis=3)(dw)
    dw = tf.nn.atrous_conv2d(value=dw, filters=tf.constant(value=1, shape=[3, 3, 16, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    dw = dwconv2d_bn(dw, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw2')


    i4 = concatenate([x1, x2, x3, x4, x5, x6, dw], axis=3)
    i4 = eca_block(i4, name="eca_layer_5")
    i4 = spatial_attention(i4)



    x1 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i4)
    x1 = BatchNormalization(axis=3)(x1)

    x2 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i4)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = tf.nn.atrous_conv2d(value=x2, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x2 = dwconv2d_bn(x2, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw22')
    x2 = BatchNormalization(axis=3)(x2)

    x3 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i4)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = tf.nn.atrous_conv2d(value=x3, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x3 = dwconv2d_bn(x3, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw23')
    x3 = BatchNormalization(axis=3)(x3)

    x4 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i4)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = tf.nn.atrous_conv2d(value=x4, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x4 = dwconv2d_bn(x4, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw24')
    x4 = BatchNormalization(axis=3)(x4)

    x5 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i4)
    x5 = BatchNormalization(axis=3)(x5)
    x5 = tf.nn.atrous_conv2d(value=x5, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x5 = dwconv2d_bn(x5, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw25')
    x5 = BatchNormalization(axis=3)(x5)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(i4)
    x6 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(x6)
    x6 = BatchNormalization(axis=3)(x6)


    dw = tf.nn.atrous_conv2d(value=dw, filters=tf.constant(value=1, shape=[3, 3, 16, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    dw = dwconv2d_bn(dw, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw3')


    i5 = concatenate([x1, x2, x3, x4, x5, x6, dw], axis=3)
    i5 = eca_block(i5, name="eca_layer_4")
    i5 = spatial_attention(i5)



    x1 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i5)
    x1 = BatchNormalization(axis=3)(x1)

    x2 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i5)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = tf.nn.atrous_conv2d(value=x2, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x2 = dwconv2d_bn(x2, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw26')
    x2 = BatchNormalization(axis=3)(x2)

    x3 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i5)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = tf.nn.atrous_conv2d(value=x3, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x3 = dwconv2d_bn(x3, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw27')
    x3 = BatchNormalization(axis=3)(x3)

    x4 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i5)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = tf.nn.atrous_conv2d(value=x4, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x4 = dwconv2d_bn(x4, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw28')
    x4 = BatchNormalization(axis=3)(x4)

    x5 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i5)
    x5 = BatchNormalization(axis=3)(x5)
    x5 = tf.nn.atrous_conv2d(value=x5, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x5 = dwconv2d_bn(x5, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw281')
    x5 = BatchNormalization(axis=3)(x5)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(i5)
    x6 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(x6)
    x6 = BatchNormalization(axis=3)(x6)


    dw = tf.nn.atrous_conv2d(value=dw, filters=tf.constant(value=1, shape=[3, 3, 16, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    dw = dwconv2d_bn(dw, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw4')


    i6 = concatenate([x1, x2, x3, x4, x5, x6, dw], axis=3)
    i6 = eca_block(i6, name="eca_layer_3")
    i6 = spatial_attention(i6)
    x1 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i6)
    x1 = BatchNormalization(axis=3)(x1)

    x2 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i6)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = tf.nn.atrous_conv2d(value=x2, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x2 = dwconv2d_bn(x2, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw29')
    x2 = BatchNormalization(axis=3)(x2)

    x3 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i6)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = tf.nn.atrous_conv2d(value=x3, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x3 = dwconv2d_bn(x3, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw30')
    x3 = BatchNormalization(axis=3)(x3)

    x4 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i6)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = tf.nn.atrous_conv2d(value=x4, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x4 = dwconv2d_bn(x4, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw31')
    x4 = BatchNormalization(axis=3)(x4)

    x5 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i6)
    x5 = BatchNormalization(axis=3)(x5)
    x5 = tf.nn.atrous_conv2d(value=x5, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x5 = dwconv2d_bn(x5, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw32')
    x5 = BatchNormalization(axis=3)(x5)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(i6)
    x6 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(x6)
    x6 = BatchNormalization(axis=3)(x6)

    #
    dw = tf.nn.atrous_conv2d(value=dw, filters=tf.constant(value=1, shape=[3, 3, 16, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    dw = dwconv2d_bn(dw, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw5')
    #
    #
    i7 = concatenate([x1, x2, x3, x4, x5, x6,dw], axis=3)
    i7 = eca_block(i7, name="eca_layer_2")
    i7 = spatial_attention(i7)

    x1 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i7)
    x1 = BatchNormalization(axis=3)(x1)

    x2 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i7)
    x2 = BatchNormalization(axis=3)(x2)
    x2 = tf.nn.atrous_conv2d(value=x2, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x2 = dwconv2d_bn(x2, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw33')
    x2 = BatchNormalization(axis=3)(x2)

    x3 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i7)
    x3 = BatchNormalization(axis=3)(x3)
    x3 = tf.nn.atrous_conv2d(value=x3, filters=tf.constant(value=1, shape=[3, 3, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x3 = dwconv2d_bn(x3, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw34')
    x3 = BatchNormalization(axis=3)(x3)

    x4 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i7)
    x4 = BatchNormalization(axis=3)(x4)
    x4 = tf.nn.atrous_conv2d(value=x4, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=1)
    x4 = dwconv2d_bn(x4, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw35')
    x4 = BatchNormalization(axis=3)(x4)

    x5 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(i7)
    x5 = BatchNormalization(axis=3)(x5)
    x5 = tf.nn.atrous_conv2d(value=x5, filters=tf.constant(value=1, shape=[5, 5, 64, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    x5 = dwconv2d_bn(x5, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw36')
    x5 = BatchNormalization(axis=3)(x5)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(i7)
    x6 = Conv2D(64, 1, padding='SAME', strides=(1, 1), activation='relu')(x6)
    x6 = BatchNormalization(axis=3)(x6)

    #
    dw = tf.nn.atrous_conv2d(value=dw, filters=tf.constant(value=1, shape=[3, 3, 16, 16], dtype=tf.float32),
                             padding='SAME', rate=2)
    dw = dwconv2d_bn(dw, kernel_size=3, strides=1, padding='same', activation='relu', depth_multiplier=1,
                     name=name + '_dw6')
    #
    i8 = concatenate([x1, x2, x3, x4, x5, x6,dw], axis=3)
    #
    i8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(i8)
    i8 = eca_block(i8)
    i8 = spatial_attention(i8)

    x = GlobalAveragePooling2D()(i8)
    x = BatchNormalization()(x)
    y = Dense(2, bias_initializer='zeros',activation="softmax",kernel_regularizer=l2(0.0001))(x)

    model = Model(inputs=inp, outputs=y)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    return model

#######################################################################################################################



def scheduler(epoch, learning_rate):
    if epoch < 7:    # 1个epoch等于使用训练集中的全部样本训练一次
        return learning_rate
    else:
        return learning_rate * 0.9

# schedule：函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
# 学习率自动调节

class_weight = {0: 2277, 1: 1650}
# 分类问题中，当不同类别的样本量差异很大，即类分布不平衡时，很容易影响分类结果。因此需要进行校正。class_weight为每一个类增加权重
# 类多的就把权重减少。类少的就把权重增大

def train():
# 定义模型训练和测试的方法
# 模型的训练状态
    time_start = time.time()
    model1 = AENet()
    models.append(model1)


    model1.summary()
    train_history1 = model1.fit(train_dataset, epochs=epochs,class_weight=class_weight, callbacks=[callback])
    # model.fit()将训练数据在模型中训练一定次数，返回loss和测量指标
    histories.append(train_history1)
    time_end = time.time()
    print('Model1 Training Time:', time_end - time_start)
    print('\n')



    return histories

def show_train_history(train_history, index):
    # 显示训练过程
    # train_history - 训练结果存储的参数位置
    # index - 训练数据的执行结果
    plt.plot(train_history.history[index])
    plt.title('Train History')
    plt.ylabel(index)
    plt.xlabel('Epoch')
    plt.show()

labeldict = {}
labeldict['benign'] = 0
labeldict['malignant'] = 1


def test(test_labels):
    test_labels = np.array(test_labels)  # np.array()把列表转换为数组，列表不存在维度，而数组是有维度的
    print('Testing:')
    probabilities = (models[0].predict(test_dataset)) / 1

    import seaborn as sns  # 导入seaborn库，取别名sns
    # Seaborn是一个用Python制作统计图形的库。它建立在matplotlib之上，并与pandas数据结构紧密集成。
    # Seaborn可帮助探索和理解数据。它的绘图功能在包含整个数据集的数据框和数组上运行，并在内部执行必要的语义映射和统计汇总，以生成有用的图。
    predIdxs = np.argmax(probabilities, axis=1)  # np.argmax()取得数组中每一行或者每一列的的最大值

    print('Accuracy score is :{:.4f}', metrics.accuracy_score(predIdxs, test_labels))
    print('Precision score is :{:.4f}', metrics.precision_score(predIdxs, test_labels, average='weighted'))
    print('Recall score is :{:.4f}', metrics.recall_score(predIdxs, test_labels, average='weighted'))
    print('F1 Score is :{:.4f}', metrics.f1_score(predIdxs, test_labels, average='weighted'))
    print('Cohen Kappa Score:{:.4f}', metrics.cohen_kappa_score(predIdxs, test_labels))   # 评价指标

    confusion_mtx = confusion_matrix(predIdxs, test_labels)
    confusionMatrix = pd.DataFrame(confusion_mtx,
                                   index=['malignant', 'benign'],
                                   columns=['malignant', 'benign'], ) # index-行标签、columns-列标签
    plt.figure(figsize=(4, 4))  # 尺寸
    ax = sns.heatmap(confusionMatrix, cmap='Blues', linecolor='black', linewidth=1, annot_kws={"size": 10}, annot=True,
                     fmt='')
    # 热力图：用颜色编码的矩阵来绘制矩形数据
    # cmap：matplotlib 颜色条名称或者对象，或者是颜色列表，可选参数从数据值到颜色空间的映射。
    # annot_kws：字典或者键值对，可选参数
    # 当annot为True时，注入数据
    # fmt：字符串，可选参数，添加注释时要使用的字符串格式代码
    # ax：绘制图的坐标轴
    ax.set_ylabel("True label")
    ax.set_xlabel("Predict label")
    ax.set_title('Confusion matrix')
    plt.show()


    lw=2

    nb_classes=2


    fpr1, tpr1, threshold = roc_curve(test_labels, predIdxs)
    roc_auc1 = auc(fpr1, tpr1)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 0.01, 1], [0, 0.01, 1], 'k--')
    plt.plot(fpr1, tpr1, color='black',
             lw=2, label='AE-Net  (AUC = %0.4f)' % roc_auc1)
    #         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #                  label='ROC curve of class {0} (AUC = {1:0.2f})'
    #                        ''.format(i, roc_auc[i]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


train_filenames, train_labels, test_filenames, test_labels = read_directory()

build_train_tfrecord(train_filenames, train_labels)
build_test_tfrecord(test_filenames, test_labels)

train_dataset = get_train_dataset(train_tfrecord)
test_dataset = get_test_dataset(test_tfrecord)

print('Info of train_dataset', type(train_dataset))
print('Info of test_dataset', type(test_dataset))

train_dataset, test_dataset = data_Preprocessing(train_dataset, test_dataset)

histories = train()

test(test_labels)

for i in range(1):
    show_train_history(histories[i], 'sparse_categorical_accuracy')


endtime = datetime.datetime.now()

print ('程序总共运行的时间是：',endtime - starttime)

print('------------------------------------------程序结束-----------------------------------------------')
