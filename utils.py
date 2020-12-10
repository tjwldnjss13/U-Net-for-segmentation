import torch


def crop(in_, out_size):
    in_size = (in_.shape[1], in_.shape[2])
    drop_size = (in_size[0] - out_size[0], in_size[1] - out_size[1])
    drop_size = (drop_size[0] // 2, drop_size[1] // 2)
    out_ = in_[:, drop_size[0]:drop_size[0] + out_size[0], drop_size[1]:drop_size[1] + out_size[1]]

    return out_


def pad_4dim(x, ref, cuda=True):
    zeros = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3])
    if cuda:
        zeros = zeros.cuda()
    while x.shape[2] < ref.shape[2]:
        x = torch.cat([x, zeros], dim=2)
    zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1)
    if cuda:
        zeros = zeros.cuda()
    while x.shape[3] < ref.shape[3]:
        x = torch.cat([x, zeros], dim=3)

    return x


def pad_3dim(x, ref_size):
    zeros = torch.zeros(x.shape[0], 1, x.shape[2]).cuda()
    while x.shape[1] < ref_size[0]:
        x = torch.cat([x, zeros], dim=1)
    zeros = torch.zeros(x.shape[0], x.shape[1], 1).cuda()
    while x.shape[2] < ref_size[1]:
        x = torch.cat([x, zeros], dim=2)

    return x


def pad_2dim(x, ref_size):
    zeros = torch.zeros(1, x.shape[1], dtype=torch.long).cuda()
    while x.shape[0] < ref_size[0]:
        x = torch.cat([x, zeros], dim=0)
    zeros = torch.zeros(x.shape[0], 1, dtype=torch.long).cuda()
    while x.shape[1] < ref_size[1]:
        x = torch.cat([x, zeros], dim=1)

    return x


def mean_iou(output, target):
    a, b = (output > 0), (target > 0)

    a_area = len(a.nonzero())
    b_area = len(b.nonzero())
    union = a_area + b_area
    inter = len((a & b).nonzero())
    iou = inter / (union - inter)

    return iou


def iou_segmentation_multichannel(a, b):
    a = a.argmax(dim=1)
    b = b.argmax(dim=1)

    a_area = len(a.nonzero())
    b_area = len(b.nonzero())
    union = a_area + b_area
    inter = len((a & b).nonzero())
    iou = inter / (union - inter)

    return iou


def time_calculator(sec):
    if sec < 60:
        return 0, 0, sec
    if sec < 3600:
        M = sec // 60
        S = sec % M
        return 0, M, S
    H = sec // 3600
    sec = sec % 3600
    M = sec // 60
    S = sec % 60
    return int(H), int(M), S

