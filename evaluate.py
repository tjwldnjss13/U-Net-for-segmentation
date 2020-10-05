import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchsummary import summary
from model import UNet

if __name__ == '__main__':
    PATH = 'saved_models/unet_10epoch_0.4129500046267646loss.pth'
    model = torch.load(PATH).cuda()

    img_pth = 'sample/pomeranian.jpg'
    img = Image.open(img_pth)
    img = np.array(img)
    img = torch.FloatTensor(img).cuda().permute(2, 0, 1).unsqueeze(0)
    y_ = model(img)

    plt.imshow(y_.cpu().detach().squeeze(0).numpy()[1], cmap='gray')
    plt.show()