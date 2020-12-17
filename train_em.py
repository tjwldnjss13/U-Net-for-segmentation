import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


import os
import time
from model import UNet
from dataset.em_dataset import EMDataset, collate_fn
from utils import time_calculator
from pytorch_utils import mean_iou_seg_argmin_pytorch
from loss import custom_cross_entropy_loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_batch_tensor(tensor_list):
    n_tensor = len(tensor_list)
    t = tensor_list[0].unsqueeze(0)
    for i in range(1, n_tensor):
        temp = tensor_list[i].unsqueeze(0)
        t = torch.cat([t, temp], dim=0)

    return t


if __name__ == '__main__':
    batch_size = 16
    num_epoch = 200
    learning_rate = .00005
    save_term = 5
    in_size = (572, 572)

    #################### EM Segment Challenge Dataset ####################
    dset_name = 'em dataset'
    root = 'data/'
    transforms = transforms.Compose([transforms.Resize((388, 388)), transforms.ToTensor()])
    dset = EMDataset(root, in_size, transforms, True, True)
    n_data = len(dset)
    dset_train, dset_val = random_split(dset, [int(n_data * .7), int(n_data * .3)])

    train_data_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_data_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_train_images = dset_train.__len__()
    num_val_images = dset_val.__len__()

    model = UNet(in_dim=1, out_dim=2).to(device)
    # PATH = 'trained models/em dataset/base model/unet_0.0001lr_135epoch_3.87355loss_0.91230iou.pth'
    # model = torch.load(PATH).cuda()

    # loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.NLLLoss()
    loss_func = custom_cross_entropy_loss
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
        break

        start_time = time.time()
        train_loss = 0
        # train_acc = 0
        train_iou = 0
        n_images = 0
        train_step = 0

        for i, (images, labels) in enumerate(train_data_loader):
            n_batch = len(images)
            n_images += n_batch
            train_step += 1
            print('[{}/{}] {}/{} '.format(e + 1, num_epoch, n_images, num_train_images), end='')

            x = make_batch_tensor(images)
            y = make_batch_tensor(labels)

            x = x.to(device)
            y = y.to(device)

            output = model(x)

            optimizer.zero_grad()
            loss = loss_func(output, y)
            # acc = (output[:, 1, :, :].squeeze(1) == y_).sum().item() / 256 ** 2
            iou = mean_iou_seg_argmin_pytorch(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * n_batch
            # train_acc += acc
            train_iou += iou

            print('<loss> {} <iou> {}'.format(loss.item(), iou))

        train_losses.append(train_loss / num_train_images)
        # train_accs.append(train_acc / num_train_images)
        train_ious.append(train_iou / train_step)
        print('      <train_loss> {} <train_iou> {} '.format(train_losses[-1], train_ious[-1]), end='')

        end_time = time.time()
        train_times.append(end_time - start_time)

        val_loss = 0
        # val_acc = 0
        val_iou = 0
        val_step = 0
        with torch.no_grad():
            for j, (images, labels) in enumerate(val_data_loader):
                n_batch = len(images)
                n_images += n_batch
                val_step += 1

                x = make_batch_tensor(images)
                y = make_batch_tensor(labels)

                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = loss_func(output, y)
                # acc = (output[:, 1, :, :].squeeze(1) == y_).sum().item() / 256 ** 2
                iou = mean_iou_seg_argmin_pytorch(output, y)
                val_loss += loss.item() * n_batch
                # val_acc += acc
                val_iou += iou

            val_losses.append(val_loss / num_val_images)
            # val_accs.append(val_acc / num_val_images)
            val_ious.append(val_iou / val_step)
        print('<val_loss> {} <val_iou> {}'.format(val_losses[-1], val_ious[-1]))

        if (e + 1) % save_term == 0:
            DIR = os.path.join('trained models', dset_name)
            PATH = os.path.join(DIR, 'unet_{}lr_{}epoch_{:.5f}loss_{:.5f}iou.pth'.format(learning_rate, e + 1, val_losses[-1], val_ious[-1]))
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


