import os
import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

if not torch.cuda.is_available():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def elastic_distortion(img, label, sigma, alpha):
    img = img.numpy()
    label = label.numpy()

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

    img_dst = torch.as_tensor(img_dst)
    label_dst = torch.as_tensor(label_dst)

    return img_dst, label_dst


a = torch.Tensor([1])
print(a.device)

# transform = transforms.Compose([transforms.Resize((388, 388)), transforms.ToTensor()])
#
# img_pth = 'data/train/0.tif'
# img = Image.open(img_pth)
# img = transform(img)
#
# print(img[0, :10])
#
# from pytorch_utils import mirrored_padding
# img = mirrored_padding(img, (572, 572))
#
# img_np = img.numpy()
# img_np = np.transpose(img_np, [1, 2, 0])
# plt.imshow(img_np)
# plt.show()

