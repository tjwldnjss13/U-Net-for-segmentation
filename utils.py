import torch


def crop(in_, out_size):
    in_size = (in_.shape[1], in_.shape[2])
    drop_size = (in_size[0] - out_size[0], in_size[1] - out_size[1])
    drop_size = (drop_size[0] // 2, drop_size[1] // 2)
    out_ = in_[:, drop_size[0]:drop_size[0] + out_size[0], drop_size[1]:drop_size[1] + out_size[1]]

    return out_


def pad_4dim(x, ref):
    zeros = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3]).cuda()
    while x.shape[2] < ref.shape[2]:
        x = torch.cat([x, zeros], dim=2)
    zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1).cuda()
    while x.shape[3] < ref.shape[3]:
        x = torch.cat([x, zeros], dim=3)

    return x


def pad_3dim(x, ref):
    zeros = torch.zeros(x.shape[0], 1, x.shape[2]).cuda()
    while x.shape[1] < ref.shape[1]:
        x = torch.cat([x, zeros], dim=1)
    zeros = torch.zeros(x.shape[0], x.shape[1], 1).cuda()
    while x.shape[2] < ref.shape[2]:
        x = torch.cat([x, zeros], dim=2)

    return x


def pad_2dim(x, ref):
    zeros = torch.zeros(1, x.shape[1]).cuda()
    while x.shape[0] < ref.shape[0]:
        x = torch.cat([x, zeros], dim=0)
    zeros = torch.zeros(x.shape[0], 1).cuda()
    while x.shape[1] < ref.shape[1]:
        x = torch.cat([x, zeros], dim=1)

    return x