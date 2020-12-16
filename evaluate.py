import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchsummary import summary
from model import UNet
from pytorch_utils import mean_iou_seg_argmin_pytorch

if __name__ == '__main__':
    PATH = 'trained models/em dataset/base model/unet_5e-05lr_200epoch_4.34130loss_0.70393iou.pth'
    model = torch.load(PATH).cuda()
    # model = UNet(in_dim=1, out_dim=2).cuda()

    for i in range(30):
        img_dir = 'data/test/'
        img_pth = os.path.join(img_dir, str(i) + '.tif')
        img = Image.open(img_pth)
        img = np.array(img)
        x = torch.as_tensor(img, dtype=torch.float).unsqueeze(dim=0).unsqueeze(0).cuda()
        output = model(x)
        output = output.argmax(dim=1).squeeze(0)

        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(output.cpu())
        plt.show()