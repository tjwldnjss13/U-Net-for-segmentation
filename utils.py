import torch


def crop(in_, out_size):
    in_size = (in_.shape[1], in_.shape[2])
    drop_size = (in_size[0] - out_size[0], in_size[1] - out_size[1])
    drop_size = (drop_size[0] // 2, drop_size[1] // 2)
    out_ = in_[:, drop_size[0]:drop_size[0] + out_size[0], drop_size[1]:drop_size[1] + out_size[1]]

    return out_


def mean_iou(a, b):
    # a, b = (output > 0), (target > 0)

    a_area = (a > 0).sum()
    b_area = (b > 0).sum()
    bg_area = ((a == 0) & (b == 0)).sum()
    inter = (a == b).sum() - bg_area
    union = a_area + b_area - inter
    iou = inter / union

    print('a_area:', a_area)
    print('b_area:', b_area)
    print('inter:', inter)
    print('union:', union)

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


