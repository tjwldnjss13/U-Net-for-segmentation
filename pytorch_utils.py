import torch


def mirrored_padding(img, out_size):
    c_img, h_img, w_img = img.shape
    h_out, w_out = out_size
    pad_h, pad_w = int((h_out - h_img) / 2), int((w_out - w_img) / 2)

    out_tensor = torch.zeros(c_img, out_size[0], out_size[1])
    out_tensor[:, pad_h:-pad_h, pad_w:-pad_w] = img

    # Up, Down padding
    out_tensor[:, :pad_h, pad_w:-pad_w] = torch.flip(img[:, :pad_h, :], [1])
    out_tensor[:, -pad_h:, pad_w:-pad_w] = torch.flip(img[:, -pad_h:, :], [1])

    # Left, Right padding
    out_tensor[:, pad_h:-pad_h, :pad_w] = torch.flip(img[:, :, :pad_w], [2])
    out_tensor[:, pad_h:-pad_h, -pad_w:] = torch.flip(img[:, :, -pad_w:], [2])

    # Top left, right padding
    out_tensor[:, :pad_h, :pad_w] = torch.flip(img[:, :pad_h, :pad_w], [1, 2])
    out_tensor[:, :pad_h, -pad_w:] = torch.flip(img[:, :pad_h, -pad_w:], [1, 2])

    # Bottom left, right padding
    out_tensor[:, -pad_h:, :pad_w] = torch.flip(img[:, -pad_h:, :pad_w], [1, 2])
    out_tensor[:, -pad_h:, -pad_w:] = torch.flip(img[:, -pad_h:, -pad_w:], [1, 2])

    return out_tensor


def pad_4dim(x, ref):
    device = x.device
    zeros = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3]).to(device)
    while x.shape[2] < ref.shape[2]:
        x = torch.cat([x, zeros], dim=2)
    zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1).to(device)
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


def mean_iou_seg_argmax_pytorch(a, b):
    # a = a.argmax(dim=1)
    # b = b.argmax(dim=1)
    #
    # a_area = (a > 0).sum()
    # b_area = (b > 0).sum()
    # bg_area = ((a == 0) & (b == 0)).sum()
    # inter = (a == b).sum() - bg_area
    # union = a_area + b_area - inter
    # iou = torch.true_divide(inter, union)

    a = a.argmax(dim=1)
    b = b.argmax(dim=1)
    n_batch = a.shape[0]

    iou_sum = 0
    for batch in range(n_batch):
        a_batch, b_batch = a[batch], b[batch]

        a_area = (a_batch > 0).sum()
        b_area = (b_batch > 0).sum()
        bg_area = ((a_batch == 0) & (b_batch == 0)).sum()
        inter = (a_batch == b_batch).sum() - bg_area
        union = a_area + b_area - inter

        iou_temp = torch.true_divide(inter, union)
        iou_sum += iou_temp

    iou = torch.true_divide(iou_sum, n_batch)

    return iou


def mean_iou_seg_argmin_pytorch(a, b):
    a = a.argmin(dim=1)
    b = b.argmin(dim=1)
    n_batch = a.shape[0]

    iou_sum = 0
    for batch in range(n_batch):
        a_batch, b_batch = a[batch], b[batch]

        a_area = (a_batch > 0).sum()
        b_area = (b_batch > 0).sum()
        bg_area = ((a_batch == 0) & (b_batch == 0)).sum()
        inter = (a_batch == b_batch).sum() - bg_area
        union = a_area + b_area - inter

        iou_temp = torch.true_divide(inter, union)
        iou_sum += iou_temp

    iou = torch.true_divide(iou_sum, n_batch)

    return iou