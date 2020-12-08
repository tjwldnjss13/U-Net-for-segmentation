import os
import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


def elastic_distortion(img, label, sigma, alpha):
    img_dst = np.zeros(img.shape)
    label_dst = np.zeros(label.shape)

    # Sampling from Unif(-1, 1)
    dx = np.random.uniform(-1, 1, img.shape[:2])
    dy = np.random.uniform(-1, 1, img.shape[:2])

    # STD of gaussian kernel
    sig = sigma

    dx_gauss = cv.GaussianBlur(dx, (7, 7), sig)
    dy_gauss = cv.GaussianBlur(dy, (7, 7), sig)

    n = np.sqrt(dx_gauss ** 2 + dy_gauss ** 2)  # for normalization

    # Strength of distortion
    alpha = alpha

    ndx = alpha * dx_gauss / n
    ndy = alpha * dy_gauss / n

    indy, indx = np.indices(img.shape[:2], dtype=np.float32)

    map_x = ndx + indx
    map_x = map_x.reshape(img.shape[:2]).astype(np.float32)
    map_y = ndy + indy
    map_y = map_y.reshape(img.shape[:2]).astype(np.float32)

    img_dst = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)
    label_dst = cv.remap(label, map_x, map_y, cv.INTER_LINEAR)

    return img_dst, label_dst


img = 'data/train/0.tif'
img = Image.open(img)
img = transforms.ToTensor()(img)

h, w = img.shape[0], img.shape[1]
M = cv.getRotationMatrix2D((h / 2, w / 2), 90, 1)
dst = cv.warpAffine(img, M, (h, w))
print(dst)

# label = 'data/label/0.tif'
# label = Image.open(label)
# label = np.array(label)
#
# img_dst, label_dst = elastic_distortion(img, label, 10, 3)
#
# print(img.shape)
#
# plt.figure(0)
# plt.imshow(img)
#
# plt.figure(1)
# plt.imshow(img_dst)
#
# plt.figure(2)
# plt.imshow(label)
#
# plt.figure(3)
# plt.imshow(label_dst)
#
# plt.show()