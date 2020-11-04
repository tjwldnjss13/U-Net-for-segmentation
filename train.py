import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import time
from model import UNet
from dataset.coco_dataset import COCODataset, collate_fn
from dataset.voc_dataset import VOCDataset, collate_fn
from utils import crop, pad_3dim, pad_2dim, mean_iou, time_calculator

if __name__ == '__main__':
    batch_size = 128
    num_epoch = 20
    learning_rate = .0001

    #################### COCO Datasets ####################
    root = 'C://DeepLearningData/COCOdataset2017'
    root_train = os.path.join(root, 'images', 'train')
    root_val = os.path.join(root, 'images', 'val')
    ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
    ann_val = os.path.join(root, 'annotations', 'instances_val2017.json')

    dset_train = COCODataset(root_train, ann_train, transforms.Compose([transforms.ToTensor()]))
    dset_val = COCODataset(root_val, ann_val, transforms.Compose([transforms.ToTensor()]))

    #################### PASCAL VOC Dataset ####################
    # root = 'C://DeepLearningData/VOC2012/'
    # transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # dset_train = VOCDataset(root, True, transforms=transforms)
    # dset_val = VOCDataset(root, False, transforms=transforms)

    train_data_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_data_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_train_images = dset_train.__len__()
    num_val_images = dset_val.__len__()

    model = UNet(3, 2).cuda()

    # model_load = 'saved_models/unet_0.001lr_30epoch_0.46214loss_0.65161iou.pth'
    # model = torch.load(model_load).cuda()

    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    # train_accs = []
    train_ious = []
    val_losses = []
    # val_accs = []
    val_ious = []
    train_times = []

    for e in range(num_epoch):
        start_time = time.time()
        train_loss = 0
        # train_acc = 0
        train_iou = 0
        n_images = 0
        train_step = 0
        for i, (images, annotations) in enumerate(train_data_loader):
            n_batch = len(images)
            n_images += n_batch
            train_step += 1
            print('[{}/{}] {}/{} '.format(e + 1, num_epoch, n_images, num_train_images), end='')
            ann = annotations

            x_pad = pad_3dim(images[0].cuda(), (256, 256))
            x = crop(x_pad, (256, 256)).unsqueeze(0).cuda()
            for b in range(1, n_batch):
                x_add_pad = pad_3dim(images[b].cuda(), (256, 256))
                x_add = crop(x_add_pad, (256, 256)).unsqueeze(0).cuda()
                x = torch.cat([x, x_add], dim=0)

            y_ = ann[0]['mask']
            if len(y_) == 0:
                y_ = torch.zeros((1, 256, 256), dtype=torch.long).cuda()
            else:
                y_ = torch.LongTensor(y_).cuda()
                y_pad = pad_2dim(y_, (256, 256))
                y_ = crop(y_pad.unsqueeze(0), (256, 256)).cuda()
            for m in range(1, n_batch):
                y_add = ann[m]['mask']
                if len(y_add) == 0:
                    y_add = torch.zeros((1, 256, 256), dtype=torch.long).cuda()
                else:
                    y_add = torch.LongTensor(y_add).cuda()
                    y_add_pad = pad_2dim(y_add, (256, 256))
                    y_add = crop(y_add_pad.unsqueeze(0), (256, 256)).cuda()
                y_ = torch.cat([y_, y_add], dim=0).cuda()

            optimizer.zero_grad()
            output = model(x)

            loss = loss_func(output, y_)
            # acc = (output[:, 1, :, :].squeeze(1) == y_).sum().item() / 256 ** 2
            iou = mean_iou(output, y_)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * n_batch
            # train_acc += acc
            train_iou += iou

            print('<loss> {} <iou> {}'.format(loss.item(), iou))

        train_losses.append(train_loss / num_train_images)
        # train_accs.append(train_acc / num_train_images)
        train_ious.append(train_iou / train_step)
        print('      <train_loss> {} <train_iou> {}'.format(train_losses[-1], train_ious[-1]), end='')

        end_time = time.time()
        train_times.append(end_time - start_time)

        val_loss = 0
        # val_acc = 0
        val_iou = 0
        val_step = 0
        with torch.no_grad():
            for j, (images, annotations) in enumerate(val_data_loader):
                ann = annotations
                n_batch = len(images)
                val_step += 1

                x_pad = pad_3dim(images[0].cuda(), (256, 256))
                x = crop(x_pad, (256, 256)).unsqueeze(0).cuda()
                for b in range(1, n_batch):
                    x_add_pad = pad_3dim(images[b].cuda(), (256, 256))
                    x_add = crop(x_add_pad, (256, 256)).unsqueeze(0).cuda()
                    x = torch.cat([x, x_add], dim=0)

                y_ = ann[0]['mask']
                if len(y_) == 0:
                    y_ = torch.zeros((1, 256, 256), dtype=torch.long).cuda()
                else:
                    y_ = torch.LongTensor(y_).cuda()
                    y_pad = pad_2dim(y_, (256, 256))
                    y_ = crop(y_pad.unsqueeze(0), (256, 256)).cuda()
                for m in range(1, n_batch):
                    y_add = ann[m]['mask']
                    if len(y_add) == 0:
                        y_add = torch.zeros((1, 256, 256), dtype=torch.long).cuda()
                    else:
                        y_add = torch.LongTensor(y_add).cuda()
                        y_add_pad = pad_2dim(y_add, (256, 256))
                        y_add = crop(y_add_pad.unsqueeze(0), (256, 256)).cuda()
                    y_ = torch.cat([y_, y_add], dim=0).cuda()

                output = model(x)
                loss = loss_func(output, y_)
                # acc = (output[:, 1, :, :].squeeze(1) == y_).sum().item() / 256 ** 2
                iou = mean_iou(output, y_)
                val_loss += loss.item() * n_batch
                # val_acc += acc
                val_iou += iou

            val_losses.append(val_loss / num_val_images)
            # val_accs.append(val_acc / num_val_images)
            val_ious.append(val_iou / val_step)
        print('<val_loss> {} <val_iou> {}'.format(val_losses[-1], val_ious[-1]))

        if (e + 1) % 2 == 0:
            PATH = 'saved_models/unet_{}lr_{}epoch_{:.5f}loss_{:.5f}iou.pth'.format(learning_rate, e + 1, val_losses[-1], val_ious[-1])
            torch.save(model, PATH)

    H, M, S = time_calculator(sum(train_times))
    print('Train time : {}H {}M {:.2f}S'.format(H, M, S))

    plt.figure(1)
    plt.title('Train/Validation Loss')
    plt.plot([i for i in range(num_epoch)], train_losses, 'r-', label='train')
    plt.plot([i for i in range(num_epoch)], val_losses, 'b-', label='val')
    plt.legend()

    plt.figure(2)
    plt.title('Train/Validation IoU')
    plt.plot([i for i in range(num_epoch)], train_ious, 'r-', label='train')
    plt.plot([i for i in range(num_epoch)], val_ious, 'b-', label='val')
    plt.legend()
    plt.show()


