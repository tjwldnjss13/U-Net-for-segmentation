import torch
import os
from model import UNet

from torchsummary import summary

if __name__ == '__main__':
    PATH = 'saved_models/unet_10epoch_0.4129500046267646loss.pth'
    model = torch.load(PATH)

    