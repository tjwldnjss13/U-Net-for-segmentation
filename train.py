import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
from model import UNet
from dataset import COCODataset, collate_fn
from utils import crop

if __name__ == '__main__':
    root = 'C://DeepLearningData/COCOdataset2017'
    root_train = os.path.join(root, 'images', 'train')
    root_val = os.path.join(root, 'images', 'val')
    ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
    ann_val = os.path.join(root, 'annotations', 'instances_val2017.json')

    batch_size = 4
    num_epoch = 10
    learning_rate = .001

    dset_train = COCODataset(root_train, ann_train, transforms.Compose([transforms.ToTensor()]))
    dset_val = COCODataset(root_val, ann_val, transforms.Compose([transforms.ToTensor()]))
    train_data_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_data_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = UNet(3, 2).cuda()

    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    for e in range(num_epoch):
        train_loss = 0
        for i, (images, annotations) in enumerate(train_data_loader):
            print('[{}/{}] {}/{} '.format(e + 1, num_epoch, i + 1, len(train_data_loader)))
            ann = annotations

            # print(type(annotations[0]))

            x = crop(images[0], (256, 256)).unsqueeze(0).cuda()
            for b in range(1, batch_size):
                x_add = crop(images[b], (256, 256)).unsqueeze(0).cuda()
                x = torch.cat([x, x_add], dim=0)

            y_ = ann[0]['mask']
            if len(y_) == 0:
                y_ = torch.zeros((1, 1, 256, 256)).cuda()
            else:
                y_ = crop(torch.Tensor(y_).unsqueeze(0), (256, 256)).unsqueeze(0).cuda()
            for m in range(1, batch_size):
                y_add = ann[m]['mask']
                if len(y_add) == 0:
                    y_add = torch.zeros((1, 1, 256, 256)).cuda()
                else:
                    y_add = crop(torch.Tensor(y_add).unsqueeze(0), (256, 256)).unsqueeze(0).cuda()
                y_ = torch.cat([y_, y_add], dim=0).cuda()

            optimizer.zero_grad()
            output = model(x)

            print('x: {}'.format(x.shape))
            print('output: {}'.format(output.shape))
            print('y_: {}'.format(y_.shape))

            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i + 1 == 23:
                print('break')

        train_losses.append(train_loss / (i + 1))
        print('<train_loss> {} '.format(train_losses[-1]), end='')

        val_loss = 0
        with torch.no_grad():
            for j, (images, annotations) in enumerate(val_data_loader):
                x = images[0].cuda()
                x = x.unsqueeze(0)
                y_ = annotations[0]['mask']
                y_ = torch.tensor(y_, dtype=torch.int64).cuda()
                y_ = y_.unsqueeze(0)

                output = model(x)
                loss = loss_func(output, y_)

                val_loss += loss.item()
            val_losses.append(val_loss / (j + 1))
        print('<val_loss> {}'.format(val_losses[-1]))

    save_path = './'
    torch.save(model.state_dict(), save_path)


