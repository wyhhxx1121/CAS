import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import glob
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt


root = "G:/czy/data/malignant/"

# lung = os.path.join(root, "benign/")
img_list = sorted(glob.glob(root + '/*.png'))
print("img_list:", len(img_list))
IMG_SIZE = 256
image = np.empty((len(img_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
# print(x_data.shape)
for i, img_path in enumerate(img_list):
    # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    #     # load image
    img1 = imread(img_path)
    # resize image with 1 channel
    img1 = resize(img1, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    # save to x_data
    image[i] = img1
image = image / 255.0

print("* " * 80)

seg_root = "G:/czy/data/mask/malignant/"

mask_list = sorted(glob.glob(seg_root + '/*.png'))
print("mask_list:", len(mask_list))
IMG_SIZE = 256
image1 = np.empty((len(mask_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
# print(x_data.shape)
for i, img_path in enumerate(mask_list):
    #     # load image
    img1 = imread(img_path)
    # resize image with 1 channel
    img1 = resize(img1, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    # save to x_data
    image1[i] = img1
image1 = image1 / 255.0


plt.imshow(image[3])
plt.show()
plt.imshow(image1[3])
plt.show()
im2 = cv2.multiply(image[3], image1[3])
plt.imshow(im2)
plt.show()

img = cv2.multiply(image, image1)
# img = cv2.add(image, image1)
for i in range(img.shape[0]):
    im = img[i]
    im3 = array_to_img(im)
    im3.save("G:/czy/data/dotresult/malignant/{}.png".format(i))

print("!!!!!finished!!!!!")
