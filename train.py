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

if __name__ == '__main__':
    root = 'C://DeepLearningData/COCOdataset2017'
    root_train = os.path.join(root, 'images', 'train')
    root_val = os.path.join(root, 'images', 'val')
    ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
    ann_val = os.path.join(root, 'annotations', 'instances_val2017.json')

    dset_train = COCODataset(root_train, ann_train, transforms.Compose([transforms.ToTensor()]))
    dset_val = COCODataset(root_val, ann_val, transforms.Compose([transforms.ToTensor()]))
    train_data_loader = DataLoader(dset_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(dset_val, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = UNet(3, 1).cuda()
    num_epoch = 10
    learning_rate = .001

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    for e in range(num_epoch):
        print('[{}/{}] '.format(e + 1, num_epoch), end='')
        train_loss = 0
        for i, (images, annotations) in enumerate(train_data_loader):
            if i != 5:
                continue
            x = images[0].cuda()
            x = x.unsqueeze(0)
            y_ = annotations[0]['mask']
            y_ = torch.tensor(y_).cuda()
            y_ = y_.unsqueeze(0)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_losses.append(train_loss / (i + 1))
        print('<train_loss> {} '.format(train_losses[-1]), end='')

        val_loss = 0
        with torch.no_grad():
            for j, (images, annotations) in enumerate(val_data_loader):
                x = images[0].cuda()
                x = x.unsqueeze(0)
                y_ = annotations[0]['mask']
                y_ = torch.tensor(y_).cuda()
                y_ = y_.unsqueeze(0)

                output = model(x)
                loss = loss_func(output, y_)

                val_loss += loss.item()
            val_losses.append(val_loss / (j + 1))
        print('<val_loss> {}'.format(val_losses[-1]))

    save_path = './'
    torch.save(model.state_dict(), save_path)


