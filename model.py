import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.datasets as dset

from torchsummary import summary
from pytorch_utils import pad_4dim
from utils import crop


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.front1 = self.conv_block(self.in_dim, 64)
        self.front2 = self.conv_block(64, 128)
        self.front3 = self.conv_block(128, 256)
        self.front4 = self.conv_block(256, 512)
        self.front5 = self.conv_block(512, 1024, True)
        self.back4 = self.conv_block(1024, 512, True)
        self.back3 = self.conv_block(512, 256, True)
        self.back2 = self.conv_block(256, 128, True)
        self.back1 = self.conv_block(128, 64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(16, self.out_dim, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = x = self.front1(x)
        x = self.maxpool(x)
        x2 = x = self.front2(x)
        x = self.maxpool(x)
        x3 = x = self.front3(x)
        x = self.maxpool(x)
        x4 = x = self.front4(x)
        x = self.maxpool(x)
        x5 = x = self.front5(x)

        print('x1: {}'.format(x1.shape))
        print('x2: {}'.format(x2.shape))
        print('x3: {}'.format(x3.shape))
        print('x4: {}'.format(x4.shape))
        print('x5: {}'.format(x5.shape))

        # x4_crop = crop(x4, (int(self.in_size[0] / 8), int(self.in_size[1] / 8)))
        # x3_crop = crop(x3, (int(self.in_size[0] / 4), int(self.in_size[1] / 4)))
        # x2_crop = crop(x2, (int(self.in_size[0] / 2), int(self.in_size[1] / 2)))


        x6 = x = torch.cat([x, x4], dim=1)
        x = self.back4(x)

        x7 = x = torch.cat([x, x3], dim=1)
        x = self.back3(x)

        x8 = x = torch.cat([x, x2], dim=1)
        x = self.back2(x)

        x9 = x = torch.cat([x, x1], dim=1)
        x = self.back1(x)
        x9 = x = self.conv1(x)

        x = self.softmax(x)

        return x

    @staticmethod
    def conv_block(in_dim, dim, up=False):
        conv1 = nn.Conv2d(in_dim, dim, 3, 1, 0)
        conv2 = nn.Conv2d(dim, dim, 3, 1, 0)
        relu = nn.ReLU(True)

        layers = [conv1, relu, conv2, relu]
        if up:
            up_conv = nn.ConvTranspose2d(dim, int(dim / 2), 2, 2)
            layers.append(up_conv)

        return nn.Sequential(*layers)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    unet = UNet(in_dim=1, out_dim=2).to(device)

    # summary(unet, (1, 572, 572))
    dummy = torch.zeros((1, 1, 572, 572), dtype=torch.float).to(device)
    output = unet(dummy)
    print(output.dtype)
    print(output.shape)