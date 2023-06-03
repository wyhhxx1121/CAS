# coding=utf-8
from keras.models import *
from keras.layers import *
import os
from tensorflow.keras.regularizers import l2
import keras.backend as K
# crop o1 wrt o2
def crop(o1, o2, i):
    # 裁剪
    o_shape2 = Model(i, o2).output_shape

    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape

    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

    return o1, o2


def FCN8(input_size=(256, 256, 1)):
    img_input = Input(input_size)
    nClasses = 1


    x = Conv2D(16, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f1 = x
    # 112 x 112
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f2 = x

    # 56 x 56
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f3 = x

    # 28 x 28
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f4 = x

    # 14 x 14
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f5 = x
    # 7 x 7

    o = f5

    o = (Conv2D(256, (7, 7), activation='relu', padding='same'))(o)
    o = BatchNormalization()(o)

    o = (Conv2D(nClasses, (1, 1)))(o)
    # W = (N - 1) * S - 2P + F = 6 * 2 - 0 + 4 = 16
    o = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(2, 2), padding="valid")(o)

    o2 = f4
    o2 = (Conv2D(nClasses, (1, 1)))(o2)

    o, o2 = crop(o, o2, img_input)
    o = Add()([o, o2])
    # W = (N - 1) * S - 2P + F = 13 * 2 - 0 + 2 = 28
    o = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(2, 2), padding="valid")(o)
    o2 = f3
    o2 = (Conv2D(nClasses, (1, 1)))(o2)

    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])

    # W = (N - 1) * S - 2P + F = 27 * 8 + 8 = 224
    o = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), padding="valid")(o)
    o = Conv2D(1, (3, 3), padding='same')(o)

    model = Model(inputs=[img_input], outputs=[o])
    return model
####
#model = FCN8()
##############################################
# 设置图像大小
img_w = 256
img_h = 256

# 分类
n_label = 1
def SegNet():
    model = Sequential()
    # encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, 1), padding='same', activation='relu',
                     data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (8,8)
    # decoder
    model.add(UpSampling2D(size=(2, 2)))
    # (16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, 3), padding='same', activation='relu',
                     data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))

    model.add(Activation('sigmoid'))
    # model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.summary()
    return model


# model = SegNet()
#############################################
"""
Based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
"""

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
import sys

sys.setrecursionlimit(3000)
# 将默认的递归深度修改为3000
# 将Python解释器堆栈的最大深度设置为所需的限制。此限制可防止任何程序进入无限递归，否则无限递归将导致C堆栈溢出并使Python崩溃。
# 用法：sys.setrecursionlimit(limit)
# 参数：
# limit:它是整数类型的值，表示python解释器堆栈的新限制。
# 返回值：此方法不返回任何内容
kern_init = keras.initializers.he_normal()
# 初始化器基类：所有初始化器继承这个类。
kern_reg = keras.regularizers.l2(1e-5)
# kernel_regularizer 计算的就是层参数的相应值（l1、l2等）


class Scale(Layer):
    # 学习用于缩放输入数据的一组权重和偏差。输出仅由输入的元素和权重相乘加上偏差组成，out = in * gamma + beta，其中“gamma”和“beta”是权重和偏差。
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
                        This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        # 动量:计算数据的平均值和标准差的指数平均值的动量，用于特征的归一化。
        self.axis = axis
        # 轴:整数，在模式0下规范化的轴。例如：如果你的输入张量有形状(样本，通道，行，cols)，将轴设置为1，以规范化每个特征映射(通道轴)。
        self.beta_init = initializers.get(beta_init)
        # beta_init:用于移动参数的初始化函数的名称或者用于权重初始化的Theano/TensorFlow函数。此参数仅在不传递' weights '参数时才相关。
        self.gamma_init = initializers.get(gamma_init)
        # gamma_init:用于缩放参数的初始化函数的名称或者用于权重初始化的Theano/TensorFlow函数。此参数仅在不传递' weights '参数时才相关。
        self.initial_weights = weights
        # 权重:初始化权重。2个Numpy数组的列表，带有形状:((input_shape) (input_shape))
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = tf.Variable(self.gamma_init(shape), trainable=True)  # , name='{}_gamma'.format(self.name)权重
        self.beta = tf.Variable(self.beta_init(shape), trainable=True)  # , name='{}_beta'.format(self.name)偏差
        # self.trainable_weights = [self.gamma, self.beta]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        # broadcast只适用于加减，broadcast执行的时候，如果两个array的shape不一样，会先给“短”的那一个，增加高维度“扩展”（broadcasting），比如，一个2维的array，可以是一个3维size为1的3维array。
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        # 重塑
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    # identity_block恒等残差块，没有卷积层的短路连接
    # 对应于输入激活与输出激活具有相同维度
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    # input_tensor:输入张量
    # Kernel_size:主路径中间卷积层的内核大小
    # 过滤器:整数列表，nb_filters的3 conv层在主路径
    # Stage:整数，当前阶段标签，用于生成层名
    # block:“a”、“b”……，当前块标签，用于生成层名
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, kernel_size,
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = Add(name='res' + str(stage) + block)([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    # conv_block卷积残差块，当输入和输出尺寸不匹配时，可以使用这种类型的块
    # 与identity block恒等残差块不同的地方是：在shortcut路径中是一个CONV2D的层，用于将输入x调整为不同的尺寸
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, kernel_size,
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = Add(name='res' + str(stage) + block)([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def resnet101_model(input_shape, weights_path=None):
    '''Instantiate the ResNet101 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    bn_axis = 3
    img_input = Input(shape=input_shape, name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1', padding='same')(x)

    # Block 1
    x = conv_block(x, (3, 3), [64, 64, 256], stage=2, block='a', strides=(1, 1))  # conv2_1
    x = identity_block(x, (3, 3), [64, 64, 256], stage=2, block='b')  # conv2_2
    block_1_out = identity_block(x, (3, 3), [64, 64, 256], stage=2, block='c')  # conv2_3

    # Block 2
    x = conv_block(block_1_out, (3, 3), [128, 128, 512], stage=3, block='a')  # conv3_1
    for i in range(1, 3):
        x = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='b' + str(i))  # conv3_2-3
    block_2_out = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='b3')  # conv3_4

    # Block 3
    x = conv_block(block_2_out, (3, 3), [256, 256, 1024], stage=4, block='a')  # conv4_1
    for i in range(1, 22):
        x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='b' + str(i))  # conv4_2-22
    block_3_out = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='b22')  # conv4_23

    # Block 4
    x = conv_block(block_3_out, (3, 3), [512, 512, 2048], stage=5, block='a')  # conv5_1
    x = identity_block(x, (3, 3), [512, 512, 2048], stage=5, block='b')  # conv5_2
    block_4_out = identity_block(x, (3, 3), [512, 512, 2048], stage=5, block='c')  # conv5_3

    model = Model(inputs=[img_input], outputs=[block_4_out, block_3_out, block_2_out, block_1_out])

    if weights_path:
        model.load_weights(weights_path, by_name=True)
        print('Frontend weights loaded.')

    return model

# RefineNet Block主要由Residual Convolution Unit(RCU)、Multi-Resolution Fusion(MRF)、Chained Residual Pooling(CRP)组成。
def ResidualConvUnit(inputs, n_filters=256, kernel_size=3, name=''):
    # RCU是从残差网络中提取出来的单元结构，由两组ReLU和3x3卷积构成的块组成。
    # 局部残差单元设计用于微调预先训练的ResNet权重
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel
    Returns:
      Output of local residual block
    """

    net = ReLU(name=name + 'relu1')(inputs)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name + 'conv1', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = ReLU(name=name + 'relu2')(net)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name + 'conv2', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = Add(name=name + 'sum')([net, inputs])

    return net


def ChainedResidualPooling(inputs, n_filters=256, name=''):
    # CRP先进行一次ReLU激活，然后经过多次残差连接，每个残差都由一个5x5卷积和3x3卷积块组成。
    # 链式残差池的目的是从一个大的图像区域捕获背景上下文。
    # 该组件构建为由2个池化块组成的链，每个池化块由一个最大池化层和一个卷积层组成。一个池块接受前一个池块的输出作为输入。
    # 所有池化块的输出特征映射通过剩余连接求和与输入特征映射融合在一起。
    """
    Chained residual pooling aims to capture background
    context from a large image region. This component is
    built as a chain of 2 pooling blocks, each consisting
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are
    fused together with the input feature map through summation
    of residual connections.
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
    Returns:
      Double-pooled feature maps
    """

    net = ReLU(name=name + 'relu')(inputs)
    net_out_1 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv1', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool1', data_format='channels_last')(
        net)
    net_out_2 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv2', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool2', data_format='channels_last')(
        net)
    net_out_3 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv3', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool3', data_format='channels_last')(
        net)
    net_out_4 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv4', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool4', data_format='channels_last')(
        net)
    net_out_5 = net

    net = Add(name=name + 'sum')([net_out_1, net_out_2, net_out_3, net_out_4, net_out_5])

    return net


def MultiResolutionFusion(high_inputs=None, low_inputs=None, n_filters=256, name=''):
    # MRF先对多输入的特征图都用一个卷积层进行自适应再上采样，最后进行element-wise相加。
    # 将所有路径输入融合在一起。
    # 该块首先应用卷积进行输入适应，生成相同特征维度(输入中最小的那个)的特征映射，然后向上采样所有(较小的)特征映射到最大分辨率的输入。最后，对所有特征图进行求和融合。
    """
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
      n_filters: Number of output feature maps for each conv
    Returns:
      Fused feature maps at higher resolution

    """

    if low_inputs is None:  # RefineNet block 4
        return high_inputs

    else:
        conv_low = Conv2D(n_filters, 3, padding='same', name=name + 'conv_lo', kernel_initializer=kern_init,
                          kernel_regularizer=kern_reg)(low_inputs)
        conv_low = BatchNormalization()(conv_low)
        conv_high = Conv2D(n_filters, 3, padding='same', name=name + 'conv_hi', kernel_initializer=kern_init,
                           kernel_regularizer=kern_reg)(high_inputs)
        conv_high = BatchNormalization()(conv_high)

        conv_low_up = UpSampling2D(size=2, interpolation='bilinear', name=name + 'up')(conv_low)

        return Add(name=name + 'sum')([conv_low_up, conv_high])


def RefineBlock(high_inputs=None, low_inputs=None, block=0):
    # 一个RefineNet块，它将ResidualConvUnit结合在一起，使用MultiResolutionFusion融合特征映射，然后使用ResidualConvUnit获得大规模上下文。
    """
    A RefineNet Block which combines together the ResidualConvUnits,
    fuses the feature maps using MultiResolutionFusion, and then gets
    large-scale context with the ResidualConvUnit.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
    Returns:
      RefineNet block for a single path i.e one resolution

    """

    if low_inputs is None:  # block 4
        rcu_high = ResidualConvUnit(high_inputs, n_filters=512, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters=512, name='rb_{}_rcu_h2_'.format(block))

        # nothing happens here
        fuse = MultiResolutionFusion(high_inputs=rcu_high,
                                     low_inputs=None,
                                     n_filters=512,
                                     name='rb_{}_mrf_'.format(block))

        fuse_pooling = ChainedResidualPooling(fuse, n_filters=512, name='rb_{}_crp_'.format(block))

        output = ResidualConvUnit(fuse, n_filters=512, name='rb_{}_rcu_o1_'.format(block))
        return output
    else:
        high_n = K.int_shape(high_inputs)[-1]
        low_n = K.int_shape(low_inputs)[-1]

        rcu_high = ResidualConvUnit(high_inputs, n_filters=high_n, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters=high_n, name='rb_{}_rcu_h2_'.format(block))

        rcu_low = ResidualConvUnit(low_inputs, n_filters=low_n, name='rb_{}_rcu_l1_'.format(block))
        rcu_low = ResidualConvUnit(rcu_low, n_filters=low_n, name='rb_{}_rcu_l2_'.format(block))

        fuse = MultiResolutionFusion(high_inputs=rcu_high,
                                     low_inputs=rcu_low,
                                     n_filters=256,
                                     name='rb_{}_mrf_'.format(block))
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256, name='rb_{}_crp_'.format(block))
        output = ResidualConvUnit(fuse_pooling, n_filters=256, name='rb_{}_rcu_o1_'.format(block))
        return output


def build_refinenet(input_shape, num_class, resnet_weights=None,
                    frontend_trainable=True):
    """
    Builds the RefineNet model.
    Arguments:
      input_shape: Size of input image, including number of channels
      num_classes: Number of classes
      resnet_weights: Path to pre-trained weights for ResNet-101
      frontend_trainable: Whether or not to freeze ResNet layers during training
    Returns:
      RefineNet model
    """

    # Build ResNet-101
    model_base = resnet101_model(input_shape, resnet_weights)

    # Get ResNet block output layers
    high = model_base.output
    low = [None, None, None]

    # Get the feature maps to the proper size with bottleneck
    high[0] = Conv2D(512, 1, padding='same', name='resnet_map1', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[0])
    high[1] = Conv2D(256, 1, padding='same', name='resnet_map2', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[1])
    high[2] = Conv2D(256, 1, padding='same', name='resnet_map3', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[2])
    high[3] = Conv2D(256, 1, padding='same', name='resnet_map4', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[3])
    for h in high:
        h = BatchNormalization()(h)

    # RefineNet
    low[0] = RefineBlock(high_inputs=high[0], low_inputs=None, block=4)  # Only input ResNet 1/32
    low[1] = RefineBlock(high_inputs=high[1], low_inputs=low[0],
                         block=3)  # High input = ResNet 1/16, Low input = Previous 1/16
    low[2] = RefineBlock(high_inputs=high[2], low_inputs=low[1],
                         block=2)  # High input = ResNet 1/8, Low input = Previous 1/8
    net = RefineBlock(high_inputs=high[3], low_inputs=low[2],
                      block=1)  # High input = ResNet 1/4, Low input = Previous 1/4.

    net = ResidualConvUnit(net, name='rf_rcu_o1_')
    net = ResidualConvUnit(net, name='rf_rcu_o2_')

    net = UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')(net)
    net = Conv2D(num_class, 1, activation='sigmoid', name='rf_pred')(net)

    model = Model(model_base.input, net)

    for layer in model.layers:
        if 'rb' in layer.name or 'rf_' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = frontend_trainable
            # frontend_trainable:是否在训练期间冻结ResNet层
    return model

####
# model= build_refinenet((256, 256, 1), 1, resnet_weights=None, frontend_trainable=False)

#############################################
__all__ = ['ENet']

def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged

def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def bottleneck1(inp, output,dilated_kernelsize, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (dilated_kernelsize, dilated_kernelsize), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def  en_build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    enet = BatchNormalization(momentum=0.1)(enet)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = PReLU(shared_axes=[1, 2])(enet)

    # enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    # for _ in range(4):
    #     enet = bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i

    # CRENet
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    enet = bottleneck(enet, 64, dilated=2)  # bottleneck 1.2
    enet = bottleneck(enet, 64, asymmetric=5)  # bottleneck 1.3 enet = bottleneck(enet, 128, asymmetric=5)
    enet = bottleneck(enet, 64, dilated=4)  # bottleneck 1.4
    enet = bottleneck(enet, 64, asymmetric=5)  # bottleneck 1.5

    # enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # # bottleneck 2.x and 3.x
    # for _ in range(2):
    #     enet = bottleneck(enet, 128)  # bottleneck 2.1
    #     enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
    #     enet = bottleneck(enet, 128,)  # bottleneck 2.3 enet = bottleneck(enet, 128, asymmetric=5)
    #     enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
    #     enet = bottleneck(enet, 128)  # bottleneck 2.5
    #     enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
    #     enet = bottleneck(enet, 128)  # bottleneck 2.7 enet = bottleneck(enet, 128, asymmetric=5)
    #     enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8

    # # CENet
    # enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # # for _ in range(2): # bottleneck 2.x and 3.x
    # enet = bottleneck(enet, 128)  # bottleneck 2.1
    # enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
    # enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3 enet = bottleneck(enet, 128, asymmetric=5)
    # enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
    # enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.5
    # enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
    # enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7 enet = bottleneck(enet, 128, asymmetric=5)
    # enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8


    return enet

# decoder
def de_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder

def de_build(encoder, nc):
    enet = de_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = de_bottleneck(enet, 64)  # bottleneck 4.1
    enet = de_bottleneck(enet, 64)  # bottleneck 4.2
    enet = de_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = de_bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet

def ENet(n_classes, input_height=256, input_width=256):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    img_input = Input(shape=(input_height, input_width, 1))
    enet = en_build(img_input)
    enet = de_build(enet, n_classes)
    enet=Activation('sigmoid')(enet)
    o_shape =Model(img_input,enet)
    # plot_model(o_shape, to_file='images/enet.png', show_shapes=True)
    return o_shape
###
model=ENet(1)

#############################################

# coding=utf-8
from keras.models import *
from keras.layers import *
import os
from tensorflow.keras.regularizers import l2
import keras.backend as K
# crop o1 wrt o2
def crop(o1, o2, i):
    # 裁剪
    o_shape2 = Model(i, o2).output_shape

    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape

    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

    return o1, o2

# 设置图像大小
img_w = 256
img_h = 256

# 分类
n_label = 1

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
import sys

sys.setrecursionlimit(3000)
# 将默认的递归深度修改为3000
# 将Python解释器堆栈的最大深度设置为所需的限制。此限制可防止任何程序进入无限递归，否则无限递归将导致C堆栈溢出并使Python崩溃。
# 用法：sys.setrecursionlimit(limit)
# 参数：
# limit:它是整数类型的值，表示python解释器堆栈的新限制。
# 返回值：此方法不返回任何内容
kern_init = keras.initializers.he_normal()
# 初始化器基类：所有初始化器继承这个类。
kern_reg = keras.regularizers.l2(1e-5)
# kernel_regularizer 计算的就是层参数的相应值（l1、l2等）
#############################################

__all__ = ['ENet']


def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged

def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)


    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def bottleneck1(inp, output,dilated_kernelsize, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (dilated_kernelsize, dilated_kernelsize), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def  en_build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    enet = BatchNormalization(momentum=0.1)(enet)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = PReLU(shared_axes=[1, 2])(enet)
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate) # bottleneck 1.0
    enet = bottleneck(enet, 64, dilated=2)  # bottleneck 1.2
    enet = bottleneck(enet, 64, asymmetric=5)  # bottleneck 1.3 enet = bottleneck(enet, 128, asymmetric=5)
    enet = bottleneck(enet, 64, dilated=4)  # bottleneck 1.4
    enet = bottleneck(enet, 64, asymmetric=5)  # bottleneck 1.5



    # for _ in range(4):
         # enet = bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i



    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # for _ in range(1): # bottleneck 2.x and 3.x
    enet = bottleneck(enet, 128)  # bottleneck 2.1
    enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
    enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3 enet = bottleneck(enet, 128, asymmetric=5)
    enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
    enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.5
    enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
    enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7 enet = bottleneck(enet, 128, asymmetric=5)
    enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8

    return enet


# decoder
def de_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder

def de_build(encoder, nc):
    enet = de_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = eca_block(enet)
    enet = spatial_attention(enet)
    enet = de_bottleneck(enet, 64)  # bottleneck 4.1
    enet = de_bottleneck(enet, 64)  # bottleneck 4.2

    enet = de_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = eca_block(enet, name="eca_layer_2")
    enet = spatial_attention(enet)
    enet = de_bottleneck(enet, 16)  # bottleneck 5.1
    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

#（1）通道注意力
def channel_attention(inputs):
    # 定义可训练变量，反向传播可更新
    gama = tf.Variable(tf.ones(1))  # 初始化1

    # 获取输入特征图的shape
    b, h, w, c = inputs.shape

    # 重新排序维度[b,h,w,c]==>[b,c,h,w]
    x = tf.transpose(inputs, perm=[0,3,1,2])  # perm代表重新排序的轴
    # 重塑特征图尺寸[b,c,h,w]==>[b,c,h*w]
    x_reshape = tf.reshape(x, shape=[-1,c,h*w])

    # 重新排序维度[b,c,h*w]==>[b,h*w,c]
    x_reshape_trans = tf.transpose(x_reshape, perm=[0,2,1])  # 指定需要交换的轴
    # 矩阵相乘
    x_mutmul = x_reshape_trans @ x_reshape
    # 经过softmax归一化权重
    x_mutmul = tf.nn.softmax(x_mutmul)

    # reshape后的特征图与归一化权重矩阵相乘[b,x,h*w]
    x = x_reshape @ x_mutmul
    # 重塑形状[b,c,h*w]==>[b,c,h,w]
    x = tf.reshape(x, shape=[-1,c,h,w])
    # 重新排序维度[b,c,h,w]==>[b,h,w,c]
    x = tf.transpose(x, perm=[0,2,3,1])
    # 结果乘以可训练变量
    x = x * gama

    # 输入和输出特征图叠加
    x = layers.add([x, inputs])

    return x

#（2）位置注意力
def position_attention(inputs):
    # 定义可训练变量，反向传播可更新
    gama = tf.Variable(tf.ones(1))  # 初始化1

    # 获取输入特征图的shape
    b, h, w, c = inputs.shape

    # 深度可分离卷积[b,h,w,c]==>[b,h,w,c//8]
    x1 = layers.SeparableConv2D(filters=c//8, kernel_size=(1,1), strides=1, padding='same')(inputs)
    # 调整维度排序[b,h,w,c//8]==>[b,c//8,h,w]
    x1_trans = tf.transpose(x1, perm=[0,3,1,2])
    # 重塑特征图尺寸[b,c//8,h,w]==>[b,c//8,h*w]
    x1_trans_reshape = tf.reshape(x1_trans, shape=[-1,c//8,h*w])
    # 调整维度排序[b,c//8,h*w]==>[b,h*w,c//8]
    x1_trans_reshape_trans = tf.transpose(x1_trans_reshape, perm=[0,2,1])
    # 矩阵相乘
    x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
    # 经过softmax归一化权重
    x1_mutmul = tf.nn.softmax(x1_mutmul)

    # 深度可分离卷积[b,h,w,c]==>[b,h,w,c]
    x2 = layers.SeparableConv2D(filters=c, kernel_size=(1,1), strides=1, padding='same')(inputs)
    # 调整维度排序[b,h,w,c]==>[b,c,h,w]
    x2_trans = tf.transpose(x2, perm=[0,3,1,2])
    # 重塑尺寸[b,c,h,w]==>[b,c,h*w]
    x2_trans_reshape = tf.reshape(x2_trans, shape=[-1,c,h*w])

    # 调整x1_mutmul的轴，和x2矩阵相乘
    x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0,2,1])
    x2_mutmul = x2_trans_reshape @ x1_mutmul_trans

    # 重塑尺寸[b,c,h*w]==>[b,c,h,w]
    x2_mutmul = tf.reshape(x2_mutmul, shape=[-1,c,h,w])
    # 轴变换[b,c,h,w]==>[b,h,w,c]
    x2_mutmul = tf.transpose(x2_mutmul, perm=[0,2,3,1])
    # 结果乘以可训练变量
    x2_mutmul = x2_mutmul * gama

    # 输入和输出叠加
    x = layers.add([x2_mutmul, inputs])
    return x

#（3）DANet网络架构
def danet(inputs):

    # 输入分为两个分支
    x1 = channel_attention(inputs)  # 通道注意力
    x2 = position_attention(inputs)  # 位置注意力

    # 叠加两个注意力的结果
    x = layers.add([x1,x2])
    return x


from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras import backend as K
import math




def channel_attention(input_feature, ratio=8):
        channel = input_feature.shape[-1]

        shared_layer_one = Dense(channel // ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        return multiply([input_feature, cbam_feature])

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

def cbam_block(cbam_feature, ratio=2):
        # https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature)
        return cbam_feature

def eca_block(input_feature, b=1, gamma=2, name=""):
    channel = input_feature.shape[-1]
    # 输入特征图的通道数
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    # 根据公式计算自适应卷积核大小
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    # 如果卷积核大小是奇数就变成偶数

    avg_pool = GlobalAveragePooling2D()(input_feature)
    # [h,w,c]==>[None,c] 全局平均池化

    x = Reshape((-1, 1))(avg_pool)
    # [None,c]==>[c,1]
    x = Conv1D(1, kernel_size=kernel_size, padding="same", name="eca_layer_" + str(name), use_bias=False, )(x)
    # [c,1]==>[c,1]
    x = Activation('sigmoid')(x)
    # sigmoid激活
    x = Reshape((1, 1, -1))(x)
    # [c,1]==>[1,1,c]

    output = multiply([input_feature, x])
    # 结果和输入相乘
    return output



def CRAENet(n_classes, input_height=256, input_width=256):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    def dice_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (
                K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    img_input = Input(shape=(input_height, input_width, 1))
    enet = en_build(img_input)
    enet = de_build(enet, n_classes)
    enet = Activation('sigmoid')(enet)
    o_shape = Model(img_input, enet)
    # plot_model(o_shape, to_file='images/enet.png', show_shapes=True)
    return o_shape
###

model=CRAENet(1)
model.summary()


#############################################

from tensorflow.keras.initializers import RandomNormal
# 用于从“服从指定正态分布的序列”中随机取出指定个数的值，从而生成网络搭建初始的随机权重。
# 所谓正态分布，又叫高斯分布，是一个连续概率密度分布函数。
def conv(x, outsize, kernel_size, strides_=1, padding_='same', activation=None):
    return Conv2D(outsize, kernel_size, strides=strides_, padding=padding_, kernel_initializer=RandomNormal(
        stddev=0.001), use_bias=False, activation=activation)(x)
# stddev标准差


def Bottleneck(x, size, downsampe=False):
    residual = x

    out = conv(x, size, 1, padding_='valid')
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size * 4, 1, padding_='valid')
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)

    if downsampe:
        residual = conv(x, size * 4, 1, padding_='valid')
        residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)

    out = Add()([out, residual])
    out = Activation('relu')(out)

    return out


def BasicBlock(x, size, downsampe=False):
    residual = x

    out = conv(x, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)

    if downsampe:
        residual = conv(x, size, 1, padding_='valid')
        residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)

    out = Add()([out, residual])
    out = Activation('relu')(out)

    return out


def layer1(x):
    x = Bottleneck(x, 64, downsampe=True)
    x = Bottleneck(x, 64)
    x = Bottleneck(x, 64)
    x = Bottleneck(x, 64)

    return x


def transition_layer(x, in_channels, out_channels):
    # 每个Dense Block结束后的输出channel个数很多，需要用1*1的卷积核来降维。
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []

    for i in range(num_out):
        if i < num_in:
            if in_channels[i] != out_channels[i]:
                residual = conv(x[i], out_channels[i], 3)
                residual = BatchNormalization(
                    epsilon=1e-5, momentum=0.1)(residual)
                residual = Activation('relu')(residual)
                out.append(residual)
            else:
                out.append(x[i])
        else:
            residual = conv(x[-1], out_channels[i], 3, strides_=2)
            residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)
            residual = Activation('relu')(residual)
            out.append(residual)

    return out


def branches(x, block_num, channels):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = BasicBlock(residual, channels[i])
        out.append(residual)
    return out


def fuse_layers(x, channels, multi_scale_output=True):
    out = []

    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]
        for j in range(len(channels)):
            if j > i:
                y = conv(x[j], channels[i], 1, padding_='valid')
                y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                y = UpSampling2D(size=2 ** (j - i))(y)
                residual = Add()([residual, y])
            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = conv(y, channels[i], 3, strides_=2)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                    else:
                        y = conv(y, channels[j], 3, strides_=2)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                        y = Activation('relu')(y)
                residual = Add()([residual, y])

        residual = Activation('relu')(residual)
        out.append(residual)

    return out


def HighResolutionModule(x, channels, multi_scale_output=True):
    residual = branches(x, 4, channels)
    out = fuse_layers(residual, channels,
                      multi_scale_output=multi_scale_output)
    return out


def stage(x, num_modules, channels, multi_scale_output=True):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = HighResolutionModule(out, channels, multi_scale_output=False)
        else:
            out = HighResolutionModule(out, channels)

    return out


def hrnet(input_size=(256, 256, 1)):
    channels_2 = [16, 32]
    channels_3 = [16, 32, 64]
    channels_4 = [16, 32, 64, 128]
    num_modules_2 = 1
    num_modules_3 = 4
    num_modules_4 = 3

    inputs = Input(input_size)
    x = conv(inputs, 64, 3, strides_=2)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = conv(x, 64, 3, strides_=2)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)

    la1 = layer1(x)
    tr1 = transition_layer([la1], [256], channels_2)
    st2 = stage(tr1, num_modules_2, channels_2)
    tr2 = transition_layer(st2, channels_2, channels_3)
    st3 = stage(tr2, num_modules_3, channels_3)
    tr3 = transition_layer(st3, channels_3, channels_4)
    st4 = stage(tr3, num_modules_4, channels_4, multi_scale_output=False)
    up1 = UpSampling2D()(st4[0])
    up1 = conv(up1, 32, 3)
    up1 = BatchNormalization(epsilon=1e-5, momentum=0.1)(up1)
    up1 = Activation('relu')(up1)
    up2 = UpSampling2D()(up1)
    up2 = conv(up2, 32, 3)
    up2 = BatchNormalization(epsilon=1e-5, momentum=0.1)(up2)

    final = conv(up2, 1, 1, activation='sigmoid')

    model = Model(inputs=inputs, outputs=final)

    return model

####
# model = hrnet()


#############################################
TOP_DOWN_PYRAMID_SIZE = 256

"""
Implementation of Resnext FPN 
"""


def resnext_fpn(input_shape, nb_labels, depth=(3, 4, 6, 3), cardinality=32, width=4, weight_decay=5e-4, batch_norm=True,
                batch_momentum=0.9):
    """
    TODO: add dilated convolutions as well
    Resnext-50 is defined by (3, 4, 6, 3) [default]
    Resnext-101 is defined by (3, 4, 23, 3)
    Resnext-152 is defined by (3, 8, 23, 3)
    :param input_shape:
    :param nb_labels:
    :param depth:
    :param cardinality:
    :param width:
    :param weight_decay:
    :param batch_norm:
    :param batch_momentum:
    :return:
    """
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)

    bn_axis = 3
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input_tensor)
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    stage_1 = x

    # filters are cardinality * width * 2 for each depth level
    for i in range(depth[0]):
        x = bottleneck_block(x, 128, cardinality, strides=1, weight_decay=weight_decay)
    stage_2 = x

    # this can be done with a for loop but is more explicit this way
    x = bottleneck_block(x, 256, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[1]):
        x = bottleneck_block(x, 256, cardinality, strides=1, weight_decay=weight_decay)
    stage_3 = x

    x = bottleneck_block(x, 512, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[2]):
        x = bottleneck_block(x, 512, cardinality, strides=1, weight_decay=weight_decay)
    stage_4 = x

    x = bottleneck_block(x, 1024, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[3]):
        x = bottleneck_block(x, 1024, cardinality, strides=1, weight_decay=weight_decay)
    stage_5 = x

    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(stage_5)
    P4 = Add(name="fpn_p4add")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4', padding='same')(stage_4)])
    P3 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(stage_3)])
    P2 = Add(name="fpn_p2add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2', padding='same')(stage_2)])
    # Attach 3x3 conv to all P layers to get the final feature maps. --> Reduce aliasing effect of upsampling
    P2 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)

    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv")(P2)
    head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv_2")(head1)

    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv")(P3)
    head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv_2")(head2)

    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv")(P4)
    head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv_2")(head3)

    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv")(P5)
    head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv_2")(head4)

    f_p2 = UpSampling2D(size=(8, 8), name="pre_cat_2")(head4)
    f_p3 = UpSampling2D(size=(4, 4), name="pre_cat_3")(head3)
    f_p4 = UpSampling2D(size=(2, 2), name="pre_cat_4")(head2)
    f_p5 = head1

    x = Concatenate(axis=-1)([f_p2, f_p3, f_p4, f_p5])
    x = Conv2D(nb_labels, (3, 3), padding="SAME", name="final_conv", kernel_initializer='he_normal',
               activation='linear')(x)
    x = UpSampling2D(size=(4, 4), name="final_upsample")(x)
    x = Activation('sigmoid')(x)

    model = Model(input_tensor, x)

    return model


def grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=3)
    x = BatchNormalization(axis=3)(group_merge)
    x = Activation('relu')(x)
    return x


def bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input
    grouped_channels = int(filters / cardinality)

    if init.shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=3)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=3)(x)

    x = add([init, x])
    x = Activation('sigmoid')(x)
    return x

#####
# model=resnext_fpn((256,256,1), 1, depth=(3, 4, 6, 3), cardinality=32, width=2, weight_decay=5e-4, batch_norm=True,
#              batch_momentum=0.9)

#############################################
model.summary()





