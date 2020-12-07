# import os
# import numpy as np
# import torch
# import cv2 as cv
# import matplotlib.pyplot as plt
#
#
# from PIL import Image
#
#
# def shift_right_image(image, shift_distance):
#     d = shift_distance
#     img_shift = np.zeros(image.shape)
#
#     for i in range(d):
#         img_shift[:, i] = image[:, 0]
#
#     img_temp = image[:, :-d]
#     img_shift[:, d:] = img_temp
#
#     return img_shift
#
#
# def shift_left_image(image, shift_distance):
#     d = shift_distance
#     img_shift = np.zeros(image.shape)
#
#     for i in range(d):
#         img_shift[:, -(i + 1)] = image[:, -1]
#
#     img_temp = image[:, d:]
#     img_shift[:, :-d] = img_temp
#
#     return img_shift
#
#
# def shift_up_image(image, shift_distance):
#     d = shift_distance
#     img_shift = np.zeros(image.shape)
#
#     for i in range(d):
#         img_shift[-(i + 1), :] = image[-1, :]
#
#     img_temp = image[d:, :]
#     img_shift[:-d, :] = img_temp
#
#     return img_shift
#
#
# def shift_down_image(image, shift_distance):
#     d = shift_distance
#     img_shift = np.zeros(image.shape)
#
#     for i in range(d):
#         img_shift[i, :] = image[0, :]
#
#     img_temp = image[:-d, :]
#     img_shift[d:, :] = img_temp
#
#     return img_shift
#
#
# img_pth = 'data/train/0.tif'
# img = Image.open(img_pth)
# img = np.array(img)
#
# M = cv.getRotationMatrix2D((512 / 2, 512 / 2), 270, 1)
# dst = cv.warpAffine(img, M, (512, 512))
#
# # dst = shift_up_image(img, 5)
#
# plt.figure(0)
# plt.imshow(img)
#
# plt.figure(1)
# plt.imshow(dst)
# plt.show()

from dataset.em_dataset import EMDataset

root_dir = 'data/'
dset = EMDataset(root_dir, True, True)
print(len(dset))
